"""Apply the piecewise-affine RSS estimator to UMA16 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_estimation.estimation import (
    estimate_piecewise_affine_break,
)


def apply_piecewise_estimator(
    results_file: str | Path,
    reference_sound_speed: float = 343.0,
    frequency_switch: float = 1500.0,
    n_k_grid: int = 800,
    n_break_grid: int = 500,
) -> dict[str, np.ndarray]:
    """Apply the piecewise-affine estimator above a frequency threshold."""
    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if frequency_switch < 0:
        raise ValueError(
            "frequency_switch must be non-negative"
        )

    if n_k_grid < 4:
        raise ValueError(
            "n_k_grid must be at least 4"
        )

    if n_break_grid < 1:
        raise ValueError(
            "n_break_grid must be strictly positive"
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

    piecewise_wavenumbers = (
        baseline_wavenumbers.copy()
    )

    piecewise_sound_speeds = (
        baseline_sound_speeds.copy()
    )

    piecewise_mask = (
        frequencies >= frequency_switch
    )

    for index in np.flatnonzero(
        piecewise_mask
    ):
        estimate = estimate_piecewise_affine_break(
            distances=distances[index],
            observed_coherence=observed_coherence[index],
            frequency=float(
                frequencies[index]
            ),
            reference_sound_speed=reference_sound_speed,
            lower_factor=0.5,
            upper_factor=2.0,
            n_k_grid=n_k_grid,
            n_break_grid=n_break_grid,
        )

        piecewise_wavenumbers[index] = (
            estimate.break_wavenumber_rad_m
        )

        piecewise_sound_speeds[index] = (
            estimate.sound_speed_m_s
        )

    return {
        "frequency_hz": frequencies,
        "baseline_wavenumber_rad_m": baseline_wavenumbers,
        "piecewise_wavenumber_rad_m": piecewise_wavenumbers,
        "baseline_sound_speed_m_s": baseline_sound_speeds,
        "piecewise_sound_speed_m_s": piecewise_sound_speeds,
        "piecewise_mask": piecewise_mask,
    }


def save_piecewise_results(
    results: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """Save piecewise-affine estimates to an NPZ archive."""
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
    """Print a compact summary of the piecewise-affine estimation."""
    frequencies = results[
        "frequency_hz"
    ]

    mask = results[
        "piecewise_mask"
    ]

    print("=" * 72)
    print("UMA16 PIECEWISE-AFFINE RSS ESTIMATION")
    print("=" * 72)

    print(
        f"Total frequencies            : "
        f"{len(frequencies)}"
    )

    print(
        f"Piecewise-affine frequencies : "
        f"{np.count_nonzero(mask)}"
    )

    if np.any(
        mask
    ):
        print(
            f"First piecewise frequency    : "
            f"{frequencies[mask][0]:.2f} Hz"
        )

        print(
            f"Last piecewise frequency     : "
            f"{frequencies[mask][-1]:.2f} Hz"
        )

        baseline = results[
            "baseline_sound_speed_m_s"
        ][mask]

        piecewise = results[
            "piecewise_sound_speed_m_s"
        ][mask]

        print()
        print("High-frequency sound-speed summary")
        print("----------------------------------")

        print(
            f"Baseline mean  : "
            f"{np.mean(baseline):.2f} m/s"
        )

        print(
            f"Piecewise mean : "
            f"{np.mean(piecewise):.2f} m/s"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply the piecewise-affine RSS estimator "
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
            "Frequency above which the piecewise-affine "
            "estimator is used (default: 1500 Hz)."
        ),
    )

    parser.add_argument(
        "--n-k-grid",
        type=int,
        default=800,
        help=(
            "Number of RSS grid points "
            "(default: 800)."
        ),
    )

    parser.add_argument(
        "--n-break-grid",
        type=int,
        default=500,
        help=(
            "Maximum number of breakpoint candidates "
            "(default: 500)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/uma16/piecewise_affine.npz"
        ),
        help=(
            "Output .npz path "
            "(default: results/uma16/piecewise_affine.npz)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the piecewise-affine UMA16 estimator."""
    args = parse_args()

    results = apply_piecewise_estimator(
        results_file=args.results_file,
        reference_sound_speed=args.reference_sound_speed,
        frequency_switch=args.frequency_switch,
        n_k_grid=args.n_k_grid,
        n_break_grid=args.n_break_grid,
    )

    print_summary(
        results
    )

    output_path = save_piecewise_results(
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