"""Correct high-frequency 24-microphone estimates using local RSS minima."""



from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from acoustic_estimation.plotting import (
    plot_sound_speed_correction,
)
from acoustic_estimation.estimation import (
    correct_wavenumber_with_local_minima,
)


def correct_array24_results(
    results_file: str | Path,
    reference_sound_speed: float = 347.0,
    frequency_switch: float = 1500.0,
    relative_tolerance: float = 0.05,
    n_grid: int = 12000,
) -> dict[str, np.ndarray]:
    """Apply the historical local-minimum correction to processed results."""
    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if frequency_switch < 0:
        raise ValueError(
            "frequency_switch must be non-negative"
        )

    if relative_tolerance < 0:
        raise ValueError(
            "relative_tolerance must be non-negative"
        )

    if n_grid < 3:
        raise ValueError(
            "n_grid must be at least 3"
        )

    data = np.load(
        results_file
    )

    frequencies = np.asarray(
        data["frequency_hz"],
        dtype=np.float64,
    )

    baseline_wavenumbers = np.asarray(
        data["wavenumber_rad_m"],
        dtype=np.float64,
    )

    baseline_sound_speeds = np.asarray(
        data["sound_speed_m_s"],
        dtype=np.float64,
    )

    distances = np.asarray(
        data["distances_m"],
        dtype=np.float64,
    )

    observed_coherence = np.asarray(
        data["observed_coherence"],
        dtype=np.float64,
    )

    n_frequencies = frequencies.size

    if not (
        baseline_wavenumbers.size
        == baseline_sound_speeds.size
        == distances.shape[0]
        == observed_coherence.shape[0]
        == n_frequencies
    ):
        raise ValueError(
            "inconsistent number of frequencies in analysis results"
        )

    corrected_wavenumbers = np.empty(
        n_frequencies,
        dtype=np.float64,
    )

    corrected_sound_speeds = np.empty(
        n_frequencies,
        dtype=np.float64,
    )

    theoretical_wavenumbers = np.empty(
        n_frequencies,
        dtype=np.float64,
    )

    corrected_mask = np.zeros(
        n_frequencies,
        dtype=bool,
    )

    above_switch_mask = (
        frequencies > frequency_switch
    )

    outside_tolerance_mask = np.zeros(
        n_frequencies,
        dtype=bool,
    )

    for index in range(
        n_frequencies
    ):
        frequency = float(
            frequencies[index]
        )

        baseline_wavenumber = float(
            baseline_wavenumbers[index]
        )

        result = correct_wavenumber_with_local_minima(
            frequency=frequency,
            baseline_wavenumber=baseline_wavenumber,
            distances=distances[index],
            observed_coherence=observed_coherence[index],
            reference_sound_speed=reference_sound_speed,
            frequency_switch=frequency_switch,
            relative_tolerance=relative_tolerance,
            n_grid=n_grid,
        )

        corrected_wavenumbers[index] = (
            result.corrected_wavenumber_rad_m
        )

        corrected_sound_speeds[index] = (
            result.corrected_sound_speed_m_s
        )

        theoretical_wavenumbers[index] = (
            result.theoretical_wavenumber_rad_m
        )

        corrected_mask[index] = (
            result.was_corrected
        )

        if frequency > frequency_switch:
            relative_error = (
                abs(
                    baseline_wavenumber
                    - result.theoretical_wavenumber_rad_m
                )
                / result.theoretical_wavenumber_rad_m
            )

            outside_tolerance_mask[index] = (
                relative_error > relative_tolerance
            )

    return {
        "frequency_hz": frequencies,
        "baseline_wavenumber_rad_m": baseline_wavenumbers,
        "corrected_wavenumber_rad_m": corrected_wavenumbers,
        "baseline_sound_speed_m_s": baseline_sound_speeds,
        "corrected_sound_speed_m_s": corrected_sound_speeds,
        "theoretical_wavenumber_rad_m": theoretical_wavenumbers,
        "above_switch_mask": above_switch_mask,
        "outside_tolerance_mask": outside_tolerance_mask,
        "corrected_mask": corrected_mask,
    }


def print_summary(
    results: dict[str, np.ndarray],
    frequency_switch: float,
    relative_tolerance: float,
) -> None:
    """Print a compact summary of the correction."""
    frequencies = results[
        "frequency_hz"
    ]

    above_switch = results[
        "above_switch_mask"
    ]

    outside_tolerance = results[
        "outside_tolerance_mask"
    ]

    corrected = results[
        "corrected_mask"
    ]

    baseline_sound_speeds = results[
        "baseline_sound_speed_m_s"
    ]

    corrected_sound_speeds = results[
        "corrected_sound_speed_m_s"
    ]

    print("=" * 72)
    print("24-MICROPHONE LOCAL-MINIMUM CORRECTION")
    print("=" * 72)

    print(
        f"Total frequencies             : "
        f"{frequencies.size}"
    )

    print(
        f"Frequencies above "
        f"{frequency_switch:g} Hz     : "
        f"{np.count_nonzero(above_switch)}"
    )

    print(
        f"Outside {100 * relative_tolerance:g}% "
        f"k tolerance         : "
        f"{np.count_nonzero(outside_tolerance)}"
    )

    print(
        f"Corrected frequencies         : "
        f"{np.count_nonzero(corrected)}"
    )

    if np.any(
        above_switch
    ):
        print()
        print("High-frequency sound-speed summary")
        print("----------------------------------")

        print(
            f"Baseline mean  : "
            f"{np.mean(
                baseline_sound_speeds[above_switch]
            ):.2f} m/s"
        )

        print(
            f"Corrected mean : "
            f"{np.mean(
                corrected_sound_speeds[above_switch]
            ):.2f} m/s"
        )

    corrected_indices = np.flatnonzero(
        corrected
    )

    if corrected_indices.size == 0:
        return

    print()
    print("Corrected estimates")
    print("-------------------")

    print(
        f"{'f (Hz)':>10} "
        f"{'c baseline':>12} "
        f"{'c corrected':>13}"
    )

    print("-" * 38)

    for index in corrected_indices:
        print(
            f"{frequencies[index]:10.2f} "
            f"{baseline_sound_speeds[index]:12.2f} "
            f"{corrected_sound_speeds[index]:13.2f}"
        )


def save_corrected_results(
    results: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save corrected estimates to an NPZ archive."""
    path = Path(
        path
    )

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(
            ".npz"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        path,
        **results,
    )

    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Correct high-frequency 24-microphone "
            "sound-speed estimates using local RSS minima."
        )
    )

    parser.add_argument(
        "results_file",
        type=Path,
        help="Baseline 24-microphone .npz results.",
    )

    parser.add_argument(
        "--reference-sound-speed",
        type=float,
        default=347.0,
        help=(
            "Reference sound speed in m/s "
            "(default: 347)."
        ),
    )

    parser.add_argument(
        "--frequency-switch",
        type=float,
        default=1500.0,
        help=(
            "Minimum frequency for local-minimum correction "
            "(default: 1500 Hz)."
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help=(
            "Relative wavenumber tolerance before correction "
            "(default: 0.05)."
        ),
    )

    parser.add_argument(
        "--n-grid",
        type=int,
        default=12000,
        help=(
            "Number of RSS grid points "
            "(default: 12000)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/array24/local_minima_corrected.npz"
        ),
        help=(
            "Output .npz file "
            "(default: results/array24/"
            "local_minima_corrected.npz)."
        ),
    )
    
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Optional path for the correction figure.",
    )

    return parser.parse_args()


def main() -> None:
    """Apply the local-minimum correction."""
    args = parse_args()

    results = correct_array24_results(
        results_file=args.results_file,
        reference_sound_speed=args.reference_sound_speed,
        frequency_switch=args.frequency_switch,
        relative_tolerance=args.tolerance,
        n_grid=args.n_grid,
    )

    print_summary(
        results,
        frequency_switch=args.frequency_switch,
        relative_tolerance=args.tolerance,
    )

    output_path = save_corrected_results(
        results,
        args.output,
    )

    print()
    print(
        f"Corrected results saved to: "
        f"{output_path}"
    )
    
    if args.figure is not None:
        figure = plot_sound_speed_correction(
            frequencies=results["frequency_hz"],
            baseline_sound_speeds=results[
                "baseline_sound_speed_m_s"
            ],
            corrected_sound_speeds=results[
                "corrected_sound_speed_m_s"
            ],
            reference_sound_speed=args.reference_sound_speed,
            frequency_switch=args.frequency_switch,
            relative_tolerance=args.tolerance,
            path=args.figure,
        )

        plt.close(
            figure
        )

        print(
            f"Figure saved to: "
            f"{args.figure}"
        )


if __name__ == "__main__":
    main()