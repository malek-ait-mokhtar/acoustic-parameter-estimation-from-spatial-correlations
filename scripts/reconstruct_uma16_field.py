"""Reconstruct the UMA16 acoustic field with Gaussian-process regression.

By default, this script reproduces the historical CROUS experiment:

- 16 microphones on the 4 x 4 UMA16 array;
- target frequency of 500 Hz;
- theoretical sound speed of 343 m/s;
- first 16384 samples with a Hann window;
- diffuse-field sinc covariance kernel;
- GP regularization of 1e-6;
- 220 x 220 reconstruction grid on [-0.05, 0.17]^2 m;
- leave-one-out validation.

Numerical results are saved to an NPZ file so that diagnostic figures can
be regenerated without recomputing the Gaussian-process reconstruction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

from acoustic_estimation.gaussian_process import (
    extract_complex_pressure,
    leave_one_out_gp,
    reconstruct_gp_field,
)
from acoustic_estimation.geometry import (
    uma16_positions,
)


def load_wav_mono(
    path: Path,
) -> tuple[int, np.ndarray]:
    """Load and normalize one WAV recording."""
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            WavFileWarning,
        )

        sample_rate, data = wavfile.read(
            path
        )

    if data.ndim > 1:
        data = data[:, 0]

    if np.issubdtype(
        data.dtype,
        np.integer,
    ):
        maximum = np.iinfo(
            data.dtype
        ).max

        data = (
            data.astype(
                np.float64
            )
            / maximum
        )

    else:
        data = data.astype(
            np.float64
        )

    return (
        sample_rate,
        data,
    )


def load_microphone_signals(
    directory: Path,
) -> tuple[
    float,
    np.ndarray,
    list[Path],
]:
    """Load all WAV recordings in a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"directory not found: {directory}"
        )

    files = sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".wav"
        )
    )

    if not files:
        raise ValueError(
            f"no WAV files found in {directory}"
        )

    signals: list[np.ndarray] = []

    sample_rate: int | None = None

    for path in files:
        current_rate, signal = (
            load_wav_mono(
                path
            )
        )

        if sample_rate is None:
            sample_rate = current_rate

        elif current_rate != sample_rate:
            raise ValueError(
                "all WAV files must have the same sample rate"
            )

        signals.append(
            signal
        )

    if sample_rate is None:
        raise RuntimeError(
            "sample rate could not be determined"
        )

    minimum_length = min(
        len(signal)
        for signal in signals
    )

    signal_array = np.array(
        [
            signal[:minimum_length]
            for signal in signals
        ],
        dtype=np.float64,
    )

    return (
        float(sample_rate),
        signal_array,
        files,
    )


def reconstruct(
    input_directory: Path,
    output_file: Path,
    target_frequency: float = 500.0,
    sound_speed: float = 343.0,
    nperseg: int = 16384,
    regularization: float = 1e-6,
    grid_size: int = 220,
    x_min: float = -0.05,
    x_max: float = 0.17,
    y_min: float = -0.05,
    y_max: float = 0.17,
) -> None:
    """Run the complete UMA16 GP reconstruction experiment."""
    sample_rate, signals, files = (
        load_microphone_signals(
            input_directory
        )
    )

    positions = uma16_positions(
        spacing=0.04
    )

    if signals.shape[0] != positions.shape[0]:
        raise ValueError(
            "the UMA16 reconstruction requires exactly "
            f"{positions.shape[0]} WAV files, "
            f"but {signals.shape[0]} were found"
        )

    pressures, used_frequency = (
        extract_complex_pressure(
            signals=signals,
            sample_rate=sample_rate,
            target_frequency=target_frequency,
            nperseg=nperseg,
        )
    )

    wavenumber = float(
        2.0
        * np.pi
        * used_frequency
        / sound_speed
    )

    print("=" * 72)
    print("UMA16 GAUSSIAN-PROCESS FIELD RECONSTRUCTION")
    print("=" * 72)

    print(
        f"Input directory     : "
        f"{input_directory}"
    )

    print(
        f"WAV files           : "
        f"{len(files)}"
    )

    print(
        f"Samples per mic     : "
        f"{signals.shape[1]}"
    )

    print(
        f"Sample rate         : "
        f"{sample_rate:.6f} Hz"
    )

    print()
    print("Frequency extraction")
    print("--------------------")

    print(
        f"Requested frequency : "
        f"{target_frequency:.6f} Hz"
    )

    print(
        f"Used frequency      : "
        f"{used_frequency:.12f} Hz"
    )

    print(
        f"Sound speed         : "
        f"{sound_speed:.6f} m/s"
    )

    print(
        f"Wavenumber          : "
        f"{wavenumber:.12f} rad/m"
    )

    print()
    print("Measured pressure")
    print("-----------------")

    print(
        f"Mean |P|            : "
        f"{np.mean(np.abs(pressures)):.12e}"
    )

    print(
        f"Maximum |P|         : "
        f"{np.max(np.abs(pressures)):.12e}"
    )

    # ------------------------------------------------------------------
    # Leave-one-out validation.
    # ------------------------------------------------------------------

    loo = leave_one_out_gp(
        microphone_positions=positions,
        pressures=pressures,
        wavenumber=wavenumber,
        regularization=regularization,
    )

    print()
    print("Leave-one-out validation")
    print("------------------------")

    print(
        f"RMSE                : "
        f"{loo.rmse:.12e}"
    )

    print(
        f"NRMSE               : "
        f"{loo.nrmse:.12e}"
    )

    print(
        f"Correlation         : "
        f"{loo.correlation:.12e}"
    )

    print(
        f"MAE                 : "
        f"{loo.mae:.12e}"
    )

    # ------------------------------------------------------------------
    # Field reconstruction.
    # ------------------------------------------------------------------

    print()
    print("Field reconstruction")
    print("--------------------")

    print(
        f"Grid                : "
        f"{grid_size} x {grid_size}"
    )

    print(
        f"Prediction points   : "
        f"{grid_size * grid_size}"
    )

    reconstruction = reconstruct_gp_field(
        microphone_positions=positions,
        pressures=pressures,
        frequency=used_frequency,
        sound_speed=sound_speed,
        grid_size=grid_size,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        regularization=regularization,
    )

    predictive_std = np.sqrt(
        np.maximum(
            reconstruction.predictive_variance,
            0.0,
        )
    )

    print(
        f"min |P|             : "
        f"{np.min(np.abs(reconstruction.mean_field)):.12e}"
    )

    print(
        f"max |P|             : "
        f"{np.max(np.abs(reconstruction.mean_field)):.12e}"
    )

    print(
        f"max predictive std. : "
        f"{np.max(predictive_std):.12e}"
    )

    # ------------------------------------------------------------------
    # Export.
    # ------------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_file,

        target_frequency_hz=np.asarray(
            target_frequency,
            dtype=np.float64,
        ),

        used_frequency_hz=np.asarray(
            used_frequency,
            dtype=np.float64,
        ),

        sound_speed_m_s=np.asarray(
            sound_speed,
            dtype=np.float64,
        ),

        wavenumber_rad_m=np.asarray(
            wavenumber,
            dtype=np.float64,
        ),

        sample_rate_hz=np.asarray(
            sample_rate,
            dtype=np.float64,
        ),

        nperseg=np.asarray(
            nperseg,
            dtype=np.int64,
        ),

        regularization=np.asarray(
            regularization,
            dtype=np.float64,
        ),

        microphone_positions_m=np.asarray(
            positions,
            dtype=np.float64,
        ),

        measured_pressures=np.asarray(
            pressures,
            dtype=np.complex128,
        ),

        x_coordinates_m=np.asarray(
            reconstruction.x_coordinates_m,
            dtype=np.float64,
        ),

        y_coordinates_m=np.asarray(
            reconstruction.y_coordinates_m,
            dtype=np.float64,
        ),

        mean_field=np.asarray(
            reconstruction.mean_field,
            dtype=np.complex128,
        ),

        predictive_variance=np.asarray(
            reconstruction.predictive_variance,
            dtype=np.float64,
        ),

        predictive_std=np.asarray(
            predictive_std,
            dtype=np.float64,
        ),

        loo_true_values=np.asarray(
            loo.true_values,
            dtype=np.complex128,
        ),

        loo_predicted_values=np.asarray(
            loo.predicted_values,
            dtype=np.complex128,
        ),

        loo_absolute_errors=np.asarray(
            loo.absolute_errors,
            dtype=np.float64,
        ),

        rmse=np.asarray(
            loo.rmse,
            dtype=np.float64,
        ),

        nrmse=np.asarray(
            loo.nrmse,
            dtype=np.float64,
        ),

        correlation=np.asarray(
            loo.correlation,
            dtype=np.float64,
        ),

        mae=np.asarray(
            loo.mae,
            dtype=np.float64,
        ),
    )

    # Historical script also exported a human-readable metrics file.
    metrics_file = (
        output_file.parent
        / "metrics_16mics_500Hz.txt"
    )

    metrics_file.write_text(
        (
            f"used_freq_hz={used_frequency:.6f}\n"
            f"k_rad_per_m={wavenumber:.12f}\n"
            f"rmse={loo.rmse:.12e}\n"
            f"nrmse={loo.nrmse:.12e}\n"
            f"corr={loo.correlation:.12e}\n"
            f"mae={loo.mae:.12e}\n"
        ),
        encoding="utf-8",
    )

    print()
    print("Results saved to")
    print("----------------")

    print(
        f"NPZ     : {output_file}"
    )

    print(
        f"Metrics : {metrics_file}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct an UMA16 acoustic field "
            "using Gaussian-process regression."
        )
    )

    parser.add_argument(
        "input_directory",
        type=Path,
        help=(
            "Directory containing the 16 microphone WAV files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/uma16/crous/gp/reconstruction.npz"
        ),
        help=(
            "Output NPZ file "
            "(default: results/uma16/crous/gp/reconstruction.npz)."
        ),
    )

    parser.add_argument(
        "--frequency",
        type=float,
        default=500.0,
        help=(
            "Target frequency in Hz "
            "(default: 500)."
        ),
    )

    parser.add_argument(
        "--sound-speed",
        type=float,
        default=343.0,
        help=(
            "Reference sound speed in m/s "
            "(default: 343)."
        ),
    )

    parser.add_argument(
        "--nperseg",
        type=int,
        default=16384,
        help=(
            "Number of samples used for the FFT "
            "(default: 16384)."
        ),
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-6,
        help=(
            "GP diagonal regularization "
            "(default: 1e-6)."
        ),
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=220,
        help=(
            "Number of grid points per spatial axis "
            "(default: 220)."
        ),
    )

    parser.add_argument(
        "--x-min",
        type=float,
        default=-0.05,
    )

    parser.add_argument(
        "--x-max",
        type=float,
        default=0.17,
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=-0.05,
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=0.17,
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line reconstruction."""
    args = parse_args()

    reconstruct(
        input_directory=args.input_directory,
        output_file=args.output,
        target_frequency=args.frequency,
        sound_speed=args.sound_speed,
        nperseg=args.nperseg,
        regularization=args.regularization,
        grid_size=args.grid_size,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()