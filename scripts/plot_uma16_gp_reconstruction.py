"""Generate the historical UMA16 Gaussian-process reconstruction figures.

The script reads an already computed GP reconstruction NPZ file.
No WAV loading, FFT, GP prediction, or leave-one-out calculation is
performed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustic_estimation.plotting import (
    plot_gp_measured_magnitude,
    plot_gp_prediction_vs_measurement,
    plot_gp_predictive_uncertainty,
    plot_gp_reconstructed_magnitude,
    plot_gp_reconstructed_real,
    plot_gp_spatial_error_map,
)


def generate_figures(
    results_file: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    """Generate all historical GP diagnostic figures."""
    results_file = Path(
        results_file
    )

    output_directory = Path(
        output_directory
    )

    if not results_file.is_file():
        raise FileNotFoundError(
            f"GP results not found: {results_file}"
        )

    data = np.load(
        results_file
    )

    positions = np.asarray(
        data["microphone_positions_m"],
        dtype=np.float64,
    )

    pressures = np.asarray(
        data["measured_pressures"],
        dtype=np.complex128,
    )

    frequency = float(
        data["used_frequency_hz"]
    )

    x_coordinates = np.asarray(
        data["x_coordinates_m"],
        dtype=np.float64,
    )

    y_coordinates = np.asarray(
        data["y_coordinates_m"],
        dtype=np.float64,
    )

    mean_field = np.asarray(
        data["mean_field"],
        dtype=np.complex128,
    )

    variance = np.asarray(
        data["predictive_variance"],
        dtype=np.float64,
    )

    loo_true = np.asarray(
        data["loo_true_values"],
        dtype=np.complex128,
    )

    loo_predicted = np.asarray(
        data["loo_predicted_values"],
        dtype=np.complex128,
    )

    loo_errors = np.asarray(
        data["loo_absolute_errors"],
        dtype=np.float64,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "measured": (
            output_directory
            / "figure23_measured_magnitude.png"
        ),
        "field": (
            output_directory
            / "figure24_reconstructed_field.png"
        ),
        "real": (
            output_directory
            / "reconstructed_real_part.png"
        ),
        "uncertainty": (
            output_directory
            / "figure25_gp_uncertainty.png"
        ),
        "loo": (
            output_directory
            / "figure26_loo_prediction.png"
        ),
        "errors": (
            output_directory
            / "figure27_spatial_loo_error.png"
        ),
    }

    figures = [
        plot_gp_measured_magnitude(
            positions,
            pressures,
            frequency,
            path=paths["measured"],
        ),

        plot_gp_reconstructed_magnitude(
            mean_field,
            positions,
            x_coordinates,
            y_coordinates,
            frequency,
            path=paths["field"],
        ),

        plot_gp_reconstructed_real(
            mean_field,
            positions,
            x_coordinates,
            y_coordinates,
            path=paths["real"],
        ),

        plot_gp_predictive_uncertainty(
            variance,
            positions,
            x_coordinates,
            y_coordinates,
            path=paths["uncertainty"],
        ),

        plot_gp_prediction_vs_measurement(
            loo_true,
            loo_predicted,
            frequency,
            path=paths["loo"],
        ),

        plot_gp_spatial_error_map(
            positions,
            loo_errors,
            frequency,
            path=paths["errors"],
        ),
    ]

    for figure in figures:
        plt.close(
            figure
        )

    return list(
        paths.values()
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate figures from an existing "
            "UMA16 GP reconstruction."
        )
    )

    parser.add_argument(
        "results_file",
        type=Path,
        nargs="?",
        default=Path(
            "results/uma16/crous/gp/reconstruction.npz"
        ),
        help=(
            "GP reconstruction NPZ file."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "results/uma16/crous/gp/figures"
        ),
        help=(
            "Directory in which the figures are saved."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Generate all GP figures."""
    args = parse_args()

    paths = generate_figures(
        results_file=args.results_file,
        output_directory=args.output_directory,
    )

    print("=" * 72)
    print("UMA16 GP FIGURES")
    print("=" * 72)

    print(
        f"Results file : "
        f"{args.results_file}"
    )

    print()
    print("Figures saved to:")

    for path in paths:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()