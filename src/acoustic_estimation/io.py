"""Input/output utilities for microphone-array recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile


def _wav_to_float64(data: NDArray) -> NDArray[np.float64]:
    """Convert WAV samples to floating-point values.

    Integer PCM samples are scaled by the maximum representable positive
    value of their dtype. Floating-point WAV samples are converted directly
    to float64.
    """
    if np.issubdtype(data.dtype, np.integer):
        max_value = np.iinfo(data.dtype).max
        return data.astype(np.float64) / max_value

    return data.astype(np.float64)


def load_wav_mono(
    path: str | Path,
) -> tuple[float, NDArray[np.float64]]:
    """Load one WAV file as a mono float64 signal.

    If the WAV contains multiple channels, only the first channel is retained,
    matching the convention used in the original UMA16 analysis.

    Parameters
    ----------
    path
        Path to the WAV file.

    Returns
    -------
    sample_rate
        Sampling frequency in hertz.
    signal
        One-dimensional float64 signal.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")

    sample_rate, data = wavfile.read(path)

    if data.ndim == 2:
        data = data[:, 0]
    elif data.ndim != 1:
        raise ValueError(
            f"unsupported WAV shape {data.shape} in {path}"
        )

    signal = _wav_to_float64(data)

    return float(sample_rate), signal


def load_wav_array(
    directory: str | Path,
    expected_channels: int | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Load a microphone array stored as one WAV file per channel.

    WAV files are sorted by filename. All files must have the same sampling
    frequency. Signals are truncated to the length of the shortest recording.

    Parameters
    ----------
    directory
        Directory containing the WAV files.
    expected_channels
        Optional expected number of microphone channels.

    Returns
    -------
    sample_rate
        Common sampling frequency in hertz.
    signals
        Array with shape ``(n_channels, n_samples)``.
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            f"WAV directory not found: {directory}"
        )

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    if not files:
        raise ValueError(
            f"no WAV files found in {directory}"
        )

    if (
        expected_channels is not None
        and len(files) != expected_channels
    ):
        raise ValueError(
            f"expected {expected_channels} WAV files, "
            f"found {len(files)}"
        )

    signals = []
    sample_rate: float | None = None

    for path in files:
        current_rate, signal = load_wav_mono(path)

        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise ValueError(
                "all WAV files must have the same sample rate"
            )

        signals.append(signal)

    min_length = min(len(signal) for signal in signals)

    stacked = np.stack(
        [signal[:min_length] for signal in signals],
        axis=0,
    ).astype(np.float64)

    assert sample_rate is not None

    return sample_rate, stacked


def load_npy_array(
    path: str | Path,
    expected_channels: int | None = None,
    channels_axis: int = 1,
) -> NDArray[np.float64]:
    """Load a multichannel NumPy recording.

    The returned array always follows the package convention
    ``(n_channels, n_samples)``.

    Parameters
    ----------
    path
        Path to the ``.npy`` recording.
    expected_channels
        Optional expected number of microphone channels.
    channels_axis
        Axis containing microphone channels in the stored array.
        Use ``1`` for recordings produced by ``mmlib.record()``, whose
        shape is ``(n_samples, n_channels)``.

    Returns
    -------
    ndarray
        Recording with shape ``(n_channels, n_samples)``.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"NumPy recording not found: {path}"
        )

    data = np.load(path, allow_pickle=False)

    if data.ndim != 2:
        raise ValueError(
            "recording must be a two-dimensional array"
        )

    if channels_axis not in (0, 1):
        raise ValueError(
            "channels_axis must be either 0 or 1"
        )

    if channels_axis == 1:
        data = data.T

    signals = np.asarray(data, dtype=np.float64)

    if (
        expected_channels is not None
        and signals.shape[0] != expected_channels
    ):
        raise ValueError(
            f"expected {expected_channels} channels, "
            f"found {signals.shape[0]}"
        )

    if not np.all(np.isfinite(signals)):
        raise ValueError(
            "recording must contain only finite values"
        )

    return signals