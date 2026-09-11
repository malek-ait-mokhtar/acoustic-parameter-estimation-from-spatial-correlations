"""Generate the historical UMA16 high-frequency estimator figures.

This script reproduces the final sound-speed figures associated with:

- R6: second-derivative RSS estimator;
- R5: piecewise-affine RSS estimator.

The numerical estimates are loaded from previously computed and validated
NPZ files. No RSS landscape or wavenumber estimation is recomputed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustic_estimation.plotting import (
    plot_piecewise_affine_sound_speed,
    plot_second_derivative_sound_speed,
)


def generate_figures(
    second_derivative_file: str | Path,
    piecewise_affine_file: str | Path,
    output_directory: str | Path,
    reference_sound_speed: float = 343.0,
    frequency_switch: float = 1500.0,
) -> tuple[Path, Path]:
    """Generate the R6 and R5 high-frequency sound-speed figures."""
    second_derivative_file = Path(
        second_derivative_file
    )

    piecewise_affine_file = Path(
        piecewise_affine_file
    )

    output_directory = Path(
        output_directory
    )

    if not second_derivative_file.is_file():
        raise FileNotFoundError(
            f"Second-derivative results not found: "
            f"{second_derivative_file}"
        )

    if not piecewise_affine_file.is_file():
        raise FileNotFoundError(
            f"Piecewise-affine results not found: "
            f"{piecewise_affine_file}"
        )

    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if frequency_switch < 0:
        raise ValueError(
            "frequency_switch must be non-negative"
        )

    second = np.load(
        second_derivative_file
    )

    piecewise = np.load(
        piecewise_affine_file
    )

    second_frequencies = np.asarray(
        second["frequency_hz"],
        dtype=np.float64,
    )

    piecewise_frequencies = np.asarray(
        piecewise["frequency_hz"],
        dtype=np.float64,
    )

    if (
        second_frequencies.shape
        != piecewise_frequencies.shape
    ):
        raise ValueError(
            "R5 and R6 results contain different "
            "numbers of frequencies"
        )

    if not np.allclose(
        second_frequencies,
        piecewise_frequencies,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "R5 and R6 results use different "
            "frequency bins"
        )

    second_baseline = np.asarray(
        second["baseline_sound_speed_m_s"],
        dtype=np.float64,
    )

    piecewise_baseline = np.asarray(
        piecewise["baseline_sound_speed_m_s"],
        dtype=np.float64,
    )

    if not np.allclose(
        second_baseline,
        piecewise_baseline,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "R5 and R6 files contain different "
            "baseline sound-speed estimates"
        )

    second_adjusted = np.asarray(
        second["second_derivative_sound_speed_m_s"],
        dtype=np.float64,
    )

    piecewise_adjusted = np.asarray(
        piecewise["piecewise_sound_speed_m_s"],
        dtype=np.float64,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    second_path = (
        output_directory
        / "sound_speed_second_derivative.png"
    )

    piecewise_path = (
        output_directory
        / "sound_speed_piecewise_affine.png"
    )

    second_figure = (
        plot_second_derivative_sound_speed(
            frequencies_hz=second_frequencies,
            baseline_sound_speed_m_s=second_baseline,
            second_derivative_sound_speed_m_s=second_adjusted,
            reference_sound_speed=reference_sound_speed,
            path=second_path,
        )
    )

    piecewise_figure = (
        plot_piecewise_affine_sound_speed(
            frequencies_hz=piecewise_frequencies,
            baseline_sound_speed_m_s=piecewise_baseline,
            piecewise_sound_speed_m_s=piecewise_adjusted,
            frequency_switch=frequency_switch,
            reference_sound_speed=reference_sound_speed,
            path=piecewise_path,
        )
    )

    plt.close(
        second_figure
    )

    plt.close(
        piecewise_figure
    )

    return (
        second_path,
        piecewise_path,
    )


def print_summary(
    second_derivative_file: str | Path,
    piecewise_affine_file: str | Path,
) -> None:
    """Print a compact summary of the two validated estimators."""
    second = np.load(
        second_derivative_file
    )

    piecewise = np.load(
        piecewise_affine_file
    )

    frequencies = np.asarray(
        second["frequency_hz"],
        dtype=np.float64,
    )

    second_mask = np.asarray(
        second["second_derivative_mask"],
        dtype=bool,
    )

    piecewise_mask = np.asarray(
        piecewise["piecewise_mask"],
        dtype=bool,
    )

    second_speed = np.asarray(
        second["second_derivative_sound_speed_m_s"],
        dtype=np.float64,
    )

    piecewise_speed = np.asarray(
        piecewise["piecewise_sound_speed_m_s"],
        dtype=np.float64,
    )

    print("=" * 72)
    print("UMA16 HIGH-FREQUENCY ESTIMATOR FIGURES")
    print("=" * 72)

    print(
        f"Total frequencies             : "
        f"{len(frequencies)}"
    )

    print(
        f"Second-derivative frequencies : "
        f"{np.count_nonzero(second_mask)}"
    )

    print(
        f"Piecewise-affine frequencies  : "
        f"{np.count_nonzero(piecewise_mask)}"
    )

    print()

    print("High-frequency sound-speed summary")
    print("----------------------------------")

    print(
        f"Second-derivative mean : "
        f"{np.mean(second_speed[second_mask]):.6f} m/s"
    )

    print(
        f"Piecewise-affine mean  : "
        f"{np.mean(piecewise_speed[piecewise_mask]):.6f} m/s"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the historical R5/R6 UMA16 "
            "high-frequency sound-speed figures."
        )
    )

    parser.add_argument(
        "--second-derivative",
        type=Path,
        default=Path(
            "results/uma16/crous/second_derivative.npz"
        ),
        help=(
            "Validated R6 results file "
            "(default: results/uma16/crous/"
            "second_derivative.npz)."
        ),
    )

    parser.add_argument(
        "--piecewise-affine",
        type=Path,
        default=Path(
            "results/uma16/crous/piecewise_affine.npz"
        ),
        help=(
            "Validated R5 results file "
            "(default: results/uma16/crous/"
            "piecewise_affine.npz)."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/uma16/crous/figures"
        ),
        help=(
            "Figure output directory "
            "(default: results/uma16/crous/figures)."
        ),
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
            "High-frequency estimator threshold in Hz "
            "(default: 1500)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate both historical high-frequency estimator figures."""
    args = parse_args()

    print_summary(
        second_derivative_file=args.second_derivative,
        piecewise_affine_file=args.piecewise_affine,
    )

    second_path, piecewise_path = generate_figures(
        second_derivative_file=args.second_derivative,
        piecewise_affine_file=args.piecewise_affine,
        output_directory=args.output_directory,
        reference_sound_speed=args.reference_sound_speed,
        frequency_switch=args.frequency_switch,
    )

    print()
    print("Figures saved to:")

    print(
        f"  {second_path}"
    )

    print(
        f"  {piecewise_path}"
    )


if __name__ == "__main__":
    main()