"""Apply the RSS second-derivative estimator to UMA16 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_estimation.estimation import (
    estimate_second_derivative_wavenumber,
)


def apply_second_derivative_estimator(
    results_file: str | Path,
    reference_sound_speed: float = 343.0,
    frequency_switch: float = 1500.0,
    n_grid: int = 12000,
    lower_factor: float = 0.05,
    upper_factor: float = 15.0,
    edge_offset_min: int = 20,
) -> dict[str, np.ndarray]:
    """Apply the RSS-curvature estimator above a frequency threshold."""
    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if frequency_switch < 0:
        raise ValueError(
            "frequency_switch must be non-negative"
        )

    if n_grid < 3:
        raise ValueError(
            "n_grid must be at least 3"
        )

    if lower_factor <= 0:
        raise ValueError(
            "lower_factor must be strictly positive"
        )

    if upper_factor <= lower_factor:
        raise ValueError(
            "upper_factor must be greater than lower_factor"
        )

    if edge_offset_min < 0:
        raise ValueError(
            "edge_offset_min must be non-negative"
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

    second_derivative_wavenumbers = (
        baseline_wavenumbers.copy()
    )

    second_derivative_sound_speeds = (
        baseline_sound_speeds.copy()
    )

    second_derivative_mask = (
        frequencies >= frequency_switch
    )

    for index in np.flatnonzero(
        second_derivative_mask
    ):
        estimate = (
            estimate_second_derivative_wavenumber(
                distances=distances[index],
                observed_coherence=observed_coherence[index],
                frequency=float(
                    frequencies[index]
                ),
                reference_sound_speed=reference_sound_speed,
                lower_factor=lower_factor,
                upper_factor=upper_factor,
                n_grid=n_grid,
                edge_offset_min=edge_offset_min,
            )
        )

        second_derivative_wavenumbers[index] = (
            estimate.wavenumber_rad_m
        )

        second_derivative_sound_speeds[index] = (
            estimate.sound_speed_m_s
        )

    return {
        "frequency_hz": frequencies,
        "baseline_wavenumber_rad_m": baseline_wavenumbers,
        "second_derivative_wavenumber_rad_m": (
            second_derivative_wavenumbers
        ),
        "baseline_sound_speed_m_s": baseline_sound_speeds,
        "second_derivative_sound_speed_m_s": (
            second_derivative_sound_speeds
        ),
        "second_derivative_mask": second_derivative_mask,
    }


def save_second_derivative_results(
    results: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save second-derivative estimates to an NPZ archive."""
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


def print_summary(
    results: dict[str, np.ndarray],
) -> None:
    """Print a compact summary of the second-derivative estimation."""
    frequencies = results[
        "frequency_hz"
    ]

    mask = results[
        "second_derivative_mask"
    ]

    print("=" * 72)
    print("UMA16 SECOND-DERIVATIVE RSS ESTIMATION")
    print("=" * 72)

    print(
        f"Total frequencies                 : "
        f"{len(frequencies)}"
    )

    print(
        f"Second-derivative frequencies     : "
        f"{np.count_nonzero(mask)}"
    )

    if not np.any(
        mask
    ):
        return

    selected_frequencies = frequencies[
        mask
    ]

    baseline = results[
        "baseline_sound_speed_m_s"
    ][mask]

    adjusted = results[
        "second_derivative_sound_speed_m_s"
    ][mask]

    print(
        f"First second-derivative frequency : "
        f"{selected_frequencies[0]:.2f} Hz"
    )

    print(
        f"Last second-derivative frequency  : "
        f"{selected_frequencies[-1]:.2f} Hz"
    )

    print()
    print("High-frequency sound-speed summary")
    print("----------------------------------")

    print(
        f"Baseline mean          : "
        f"{np.mean(baseline):.6f} m/s"
    )

    print(
        f"Second-derivative mean : "
        f"{np.mean(adjusted):.6f} m/s"
    )

    print(
        f"Minimum adjusted speed : "
        f"{np.min(adjusted):.6f} m/s"
    )

    print(
        f"Maximum adjusted speed : "
        f"{np.max(adjusted):.6f} m/s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply the RSS second-derivative estimator "
            "to processed UMA16 results."
        )
    )

    parser.add_argument(
        "results_file",
        type=Path,
        help="Processed UMA16 .npz results.",
    )

    parser.add_argument(
        "--reference-sound-speed",
        type=float,
        default=343.0,
        help=(
            "Reference sound speed in m/s "
            "(default: 343)."
        ),
    )

    parser.add_argument(
        "--frequency-switch",
        type=float,
        default=1500.0,
        help=(
            "Frequency above which the second-derivative "
            "estimator is used (default: 1500 Hz)."
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
            "results/uma16/crous/second_derivative.npz"
        ),
        help=(
            "Output .npz path "
            "(default: results/uma16/crous/"
            "second_derivative.npz)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the UMA16 second-derivative estimator."""
    args = parse_args()

    results = apply_second_derivative_estimator(
        results_file=args.results_file,
        reference_sound_speed=args.reference_sound_speed,
        frequency_switch=args.frequency_switch,
        n_grid=args.n_grid,
    )

    print_summary(
        results
    )

    output_path = save_second_derivative_results(
        results,
        args.output,
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()