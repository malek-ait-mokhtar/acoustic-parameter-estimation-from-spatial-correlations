"""Spectral estimation utilities for multichannel acoustic signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window


def _validate_signals(signals: ArrayLike) -> NDArray[np.float64]:
    """Validate and convert a multichannel signal array."""
    signals = np.asarray(signals, dtype=np.float64)

    if signals.ndim != 2:
        raise ValueError(
            "signals must have shape (n_channels, n_samples)"
        )

    if signals.shape[0] == 0:
        raise ValueError("signals must contain at least one channel")

    if signals.shape[1] == 0:
        raise ValueError("signals must contain at least one sample")

    if not np.all(np.isfinite(signals)):
        raise ValueError("signals must contain only finite values")

    return signals


def average_spectrum(
    signals: ArrayLike,
    sample_rate: float,
    nperseg: int = 16384,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the mean single-segment spectrum across microphones.

    Each channel is centred, windowed, transformed with a real FFT, and
    normalised by the root-sum-square of the window. The magnitude spectra are
    then averaged across channels.

    Parameters
    ----------
    signals
        Multichannel signals with shape ``(n_channels, n_samples)``.
    sample_rate
        Sampling frequency in hertz.
    nperseg
        Maximum number of samples used from each channel.
    window
        Window specification accepted by ``scipy.signal.get_window``.

    Returns
    -------
    frequencies
        FFT frequencies in hertz.
    mean_spectrum
        Mean magnitude spectrum across channels.
    """
    signals = _validate_signals(signals)

    if sample_rate <= 0:
        raise ValueError("sample_rate must be strictly positive")

    if nperseg <= 0:
        raise ValueError("nperseg must be strictly positive")

    n_samples = signals.shape[1]
    n = min(nperseg, n_samples)

    win = get_window(window, n)
    window_norm = np.sqrt(np.sum(win**2))

    centred = signals[:, :n] - np.mean(
        signals[:, :n],
        axis=1,
        keepdims=True,
    )

    spectra = np.fft.rfft(
        centred * win[None, :],
        axis=1,
    )

    magnitudes = np.abs(spectra) / window_norm

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate,
    )

    return frequencies, np.mean(magnitudes, axis=0)


def cross_spectral_matrices(
    signals: ArrayLike,
    sample_rate: float,
    nperseg: int = 4096,
    overlap: float = 0.5,
    window: str = "hann",
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    int,
]:
    """Estimate cross-spectral matrices by averaging FFT snapshots.

    Parameters
    ----------
    signals
        Multichannel signals with shape ``(n_channels, n_samples)``.
    sample_rate
        Sampling frequency in hertz.
    nperseg
        Number of samples per FFT snapshot.
    overlap
        Fractional overlap between consecutive snapshots. Must satisfy
        ``0 <= overlap < 1``.
    window
        Window specification accepted by ``scipy.signal.get_window``.

    Returns
    -------
    frequencies
        FFT frequencies in hertz.
    csms
        Cross-spectral matrices with shape
        ``(n_frequencies, n_channels, n_channels)``.
    n_snapshots
        Number of snapshots used in the average.
    """
    signals = _validate_signals(signals)

    if sample_rate <= 0:
        raise ValueError("sample_rate must be strictly positive")

    if nperseg <= 0:
        raise ValueError("nperseg must be strictly positive")

    if signals.shape[1] < nperseg:
        raise ValueError(
            "signals contain fewer samples than nperseg"
        )

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    hop = int(nperseg * (1.0 - overlap))

    if hop < 1:
        raise ValueError("overlap produces a zero-length hop")

    win = get_window(window, nperseg)
    window_power = np.sum(win**2)

    frequencies = np.fft.rfftfreq(
        nperseg,
        d=1.0 / sample_rate,
    )

    starts = np.arange(
        0,
        signals.shape[1] - nperseg + 1,
        hop,
    )

    n_snapshots = len(starts)
    n_channels = signals.shape[0]
    n_frequencies = len(frequencies)

    csms = np.zeros(
        (n_frequencies, n_channels, n_channels),
        dtype=np.complex128,
    )

    for start in starts:
        segment = signals[:, start : start + nperseg]
        segment = segment - np.mean(
            segment,
            axis=1,
            keepdims=True,
        )
        segment = segment * win[None, :]

        fft_values = np.fft.rfft(
            segment,
            axis=1,
        ) / np.sqrt(window_power)

        # For each frequency f:
        #
        # CSM_f = x_f x_f^H
        #
        # einsum computes all frequencies simultaneously.
        csms += np.einsum(
            "cf,df->fcd",
            fft_values,
            np.conj(fft_values),
        )

    csms /= n_snapshots

    return frequencies, csms, n_snapshots


def coherence_matrix(
    csm: ArrayLike,
    eps: float = 1e-15,
) -> NDArray[np.complex128]:
    r"""Normalise a cross-spectral matrix into a coherence matrix.

    The normalised cross-spectrum is

    .. math::

        \Gamma_{ij}
        =
        \frac{S_{ij}}
        {\sqrt{S_{ii} S_{jj}}}.

    Parameters
    ----------
    csm
        Square cross-spectral matrix.
    eps
        Small positive regularisation term preventing division by zero.

    Returns
    -------
    ndarray
        Complex coherence matrix with the same shape as ``csm``.
    """
    csm = np.asarray(csm, dtype=np.complex128)

    if csm.ndim != 2 or csm.shape[0] != csm.shape[1]:
        raise ValueError("csm must be a square 2D matrix")

    if eps <= 0:
        raise ValueError("eps must be strictly positive")

    diagonal = np.real(np.diag(csm))

    if np.any(diagonal < 0):
        raise ValueError(
            "csm diagonal must contain non-negative powers"
        )

    denominator = np.sqrt(
        np.outer(diagonal, diagonal)
    )

    return np.asarray(
        csm / (denominator + eps),
        dtype=np.complex128,
    )