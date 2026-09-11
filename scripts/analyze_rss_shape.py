"""Analyse the shape of an RSS landscape for one frequency."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustic_estimation.estimation import (
    analyze_rss_shape,
)
from acoustic_estimation.plotting import (
    plot_rss_piecewise_affine,
    plot_rss_second_derivative,
    plot_rss_shape_estimators,
)


def analyze_frequency(
    results_file: str | Path,
    target_frequency: float,
    reference_sound_speed: float = 343.0,
    output_directory: str | Path | None = None,
) -> None:
    """Analyse RSS shape at the available frequency nearest the target."""
    data = np.load(
        results_file
    )

    frequencies = np.asarray(
        data["frequency_hz"],
        dtype=np.float64,
    )

    wavenumbers = np.asarray(
        data["wavenumber_rad_m"],
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

    if frequencies.size == 0:
        raise ValueError(
            "analysis results contain no frequencies"
        )

    index = int(
        np.argmin(
            np.abs(
                frequencies
                - target_frequency
            )
        )
    )

    frequency = float(
        frequencies[index]
    )

    baseline_wavenumber = float(
        wavenumbers[index]
    )

    analysis = analyze_rss_shape(
        distances=distances[index],
        observed_coherence=observed_coherence[index],
        frequency=frequency,
        baseline_wavenumber=baseline_wavenumber,
        reference_sound_speed=reference_sound_speed,
    )

    print("=" * 72)
    print("RSS SHAPE ANALYSIS")
    print("=" * 72)

    print(
        f"Requested frequency       : "
        f"{target_frequency:.2f} Hz"
    )

    print(
        f"Available frequency       : "
        f"{frequency:.2f} Hz"
    )

    print(
        f"Theoretical wavenumber    : "
        f"{analysis.theoretical_wavenumber_rad_m:.6f} rad/m"
    )

    print(
        f"Baseline wavenumber       : "
        f"{analysis.baseline_wavenumber_rad_m:.6f} rad/m"
    )

    print(
        f"Second-derivative estimate: "
        f"{analysis.second_derivative_wavenumber_rad_m:.6f} rad/m"
    )

    print(
        f"Piecewise-affine break    : "
        f"{analysis.piecewise_break_wavenumber_rad_m:.6f} rad/m"
    )

    if output_directory is None:
        figure_second = plot_rss_second_derivative(
            analysis
        )

        figure_piecewise = plot_rss_piecewise_affine(
            analysis
        )

        figure_comparison = plot_rss_shape_estimators(
            analysis
        )

        plt.show()

        plt.close(
            figure_second
        )

        plt.close(
            figure_piecewise
        )

        plt.close(
            figure_comparison
        )

        return

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frequency_label = int(
        round(
            target_frequency
        )
    )

    second_path = (
        output_directory
        / f"rss_second_derivative_{frequency_label}hz.png"
    )

    piecewise_path = (
        output_directory
        / f"rss_piecewise_affine_{frequency_label}hz.png"
    )

    comparison_path = (
        output_directory
        / f"rss_shape_estimators_{frequency_label}hz.png"
    )

    figure_second = plot_rss_second_derivative(
        analysis,
        path=second_path,
    )

    figure_piecewise = plot_rss_piecewise_affine(
        analysis,
        path=piecewise_path,
    )

    figure_comparison = plot_rss_shape_estimators(
        analysis,
        path=comparison_path,
    )

    plt.close(
        figure_second
    )

    plt.close(
        figure_piecewise
    )

    plt.close(
        figure_comparison
    )

    print()
    print("Figures saved to:")

    print(
        f"  {second_path}"
    )

    print(
        f"  {piecewise_path}"
    )

    print(
        f"  {comparison_path}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyse the RSS shape and alternative "
            "wavenumber estimators at one frequency."
        )
    )

    parser.add_argument(
        "results_file",
        type=Path,
        help="Processed analysis .npz results.",
    )

    parser.add_argument(
        "frequency",
        type=float,
        help="Target frequency in Hz.",
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
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional directory in which the three "
            "diagnostic figures are saved."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the RSS-shape diagnostic."""
    args = parse_args()

    analyze_frequency(
        results_file=args.results_file,
        target_frequency=args.frequency,
        reference_sound_speed=args.reference_sound_speed,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()