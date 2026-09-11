"""Analyse local minima of the RSS wavenumber landscape."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_estimation.estimation import (
    closest_minimum_to_wavenumber,
    detect_local_minima,
    evaluate_rss_landscape,
    refine_local_minima,
    sound_speed_from_wavenumber,
    theoretical_wavenumber,
)


def closest_frequency_index(
    frequencies: np.ndarray,
    target_frequency: float,
) -> int:
    """Return the index of the available frequency closest to a target."""
    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    if frequencies.size == 0:
        raise ValueError(
            "analysis result contains no frequencies"
        )

    return int(
        np.argmin(
            np.abs(
                frequencies
                - target_frequency
            )
        )
    )


def analyze_rss_landscape(
    results_file: str | Path,
    target_frequency: float,
    reference_sound_speed: float = 347.0,
    n_grid: int = 12000,
    top_n: int = 20,
) -> None:
    """Analyse local RSS minima near a target frequency.

    Parameters
    ----------
    results_file
        Processed ``.npz`` analysis results containing the pairwise
        distances and observed spatial coherence.
    target_frequency
        Target frequency in hertz. The closest available retained
        frequency is analysed.
    reference_sound_speed
        Reference sound speed used to compute the theoretical wavenumber.
    n_grid
        Number of points used to evaluate the RSS landscape.
    top_n
        Number of lowest-RSS local minima displayed in the summary.
        The minimum closest to the theoretical wavenumber is always shown,
        even when it does not belong to these lowest-RSS minima.
    """
    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if n_grid < 3:
        raise ValueError(
            "n_grid must be at least 3"
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be strictly positive"
        )

    data = np.load(
        results_file
    )

    frequencies = np.asarray(
        data["frequency_hz"],
        dtype=np.float64,
    )

    index = closest_frequency_index(
        frequencies,
        target_frequency,
    )

    frequency = float(
        frequencies[index]
    )

    default_wavenumber = float(
        data["wavenumber_rad_m"][index]
    )

    default_sound_speed = float(
        data["sound_speed_m_s"][index]
    )

    default_rss = float(
        data["rss"][index]
    )

    distances = np.asarray(
        data["distances_m"][index],
        dtype=np.float64,
    )

    observed_coherence = np.asarray(
        data["observed_coherence"][index],
        dtype=np.float64,
    )

    theoretical_k = theoretical_wavenumber(
        frequency,
        reference_sound_speed,
    )

    # Historical RSS-analysis interval:
    #
    # k_min = 0.05 * k_theory
    # k_max = max(60 rad/m, 15 * k_theory)
    #
    # The very wide interval is required to expose the multiple local
    # minima that appear in the high-frequency RSS landscape.
    k_min = max(
        1e-6,
        0.05 * theoretical_k,
    )

    k_max = max(
        60.0,
        15.0 * theoretical_k,
    )

    k_grid, rss_grid = evaluate_rss_landscape(
        distances,
        observed_coherence,
        k_min=k_min,
        k_max=k_max,
        n_grid=n_grid,
    )

    minimum_indices = detect_local_minima(
        k_grid,
        rss_grid,
    )

    minima = refine_local_minima(
        k_grid,
        rss_grid,
        minimum_indices,
        distances,
        observed_coherence,
    )

    # ------------------------------------------------------------------
    # General information.
    # ------------------------------------------------------------------

    print("=" * 76)
    print("RSS WAVENUMBER LANDSCAPE")
    print("=" * 76)

    print(
        f"Requested frequency : "
        f"{target_frequency:.2f} Hz"
    )

    print(
        f"Available frequency : "
        f"{frequency:.2f} Hz"
    )

    print(
        f"Reference sound speed: "
        f"{reference_sound_speed:.2f} m/s"
    )

    print(
        f"Theoretical k       : "
        f"{theoretical_k:.6f} rad/m"
    )

    print(
        f"RSS search interval : "
        f"[{k_min:.6f}, {k_max:.6f}] rad/m"
    )

    print(
        f"RSS grid points     : "
        f"{n_grid}"
    )

    # ------------------------------------------------------------------
    # Baseline bounded estimate.
    # ------------------------------------------------------------------

    print()
    print("Baseline bounded estimate")
    print("-------------------------")

    print(
        f"k   : "
        f"{default_wavenumber:.6f} rad/m"
    )

    print(
        f"c   : "
        f"{default_sound_speed:.2f} m/s"
    )

    print(
        f"RSS : "
        f"{default_rss:.6e}"
    )

    # ------------------------------------------------------------------
    # Local minima.
    # ------------------------------------------------------------------

    print()
    print(
        f"Detected local minima: "
        f"{len(minima)}"
    )

    if not minima:
        print(
            "No local minima were detected "
            "inside the RSS search interval."
        )
        return

    closest = closest_minimum_to_wavenumber(
        minima,
        theoretical_k,
    )

    closest_sound_speed = (
        sound_speed_from_wavenumber(
            frequency,
            closest.wavenumber_rad_m,
        )
    )

    closest_relative_k_error = (
        abs(
            closest.wavenumber_rad_m
            - theoretical_k
        )
        / theoretical_k
    )

    print(
        f"Showing best minima by RSS: "
        f"{min(top_n, len(minima))}"
    )

    # Keep the lowest-RSS minima for compact terminal output.
    displayed_minima = list(
        minima[:top_n]
    )

    # The physically guided candidate must remain visible even when its
    # RSS rank is greater than top_n.
    closest_is_displayed = any(
        minimum is closest
        for minimum in displayed_minima
    )

    if not closest_is_displayed:
        displayed_minima.append(
            closest
        )

    print()
    print(
        f"{'#':>3} "
        f"{'k (rad/m)':>12} "
        f"{'c (m/s)':>12} "
        f"{'RSS':>14} "
        f"{'|k-k_th|/k_th':>16}"
    )

    print("-" * 63)

    for minimum in displayed_minima:
        # Since `minima` is sorted by increasing RSS, recover the actual
        # RSS rank rather than renumbering an appended physical candidate
        # as if it belonged to the top-N minima.
        rank = next(
            index + 1
            for index, candidate in enumerate(minima)
            if candidate is minimum
        )

        sound_speed = (
            sound_speed_from_wavenumber(
                frequency,
                minimum.wavenumber_rad_m,
            )
        )

        relative_error = (
            abs(
                minimum.wavenumber_rad_m
                - theoretical_k
            )
            / theoretical_k
        )

        marker = (
            "  <-- closest to theory"
            if minimum is closest
            else ""
        )

        print(
            f"{rank:3d} "
            f"{minimum.wavenumber_rad_m:12.6f} "
            f"{sound_speed:12.2f} "
            f"{minimum.rss:14.6e} "
            f"{100 * relative_error:15.2f}%"
            f"{marker}"
        )

    # ------------------------------------------------------------------
    # Physically guided candidate.
    # ------------------------------------------------------------------

    print()
    print("Minimum closest to theoretical wavenumber")
    print("-----------------------------------------")

    print(
        f"k   : "
        f"{closest.wavenumber_rad_m:.6f} rad/m"
    )

    print(
        f"c   : "
        f"{closest_sound_speed:.2f} m/s"
    )

    print(
        f"RSS : "
        f"{closest.rss:.6e}"
    )

    print(
        f"Relative k error : "
        f"{100 * closest_relative_k_error:.2f}%"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyse the local minima of RSS(k) from "
            "previously processed acoustic results."
        )
    )

    parser.add_argument(
        "results_file",
        type=Path,
        help="Processed .npz analysis results.",
    )

    parser.add_argument(
        "frequency",
        type=float,
        help="Target frequency in Hz.",
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
        "--n-grid",
        type=int,
        default=12000,
        help=(
            "Number of RSS grid points "
            "(default: 12000)."
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help=(
            "Number of lowest-RSS local minima displayed "
            "(default: 20)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run RSS-landscape analysis from the command line."""
    args = parse_args()

    analyze_rss_landscape(
        results_file=args.results_file,
        target_frequency=args.frequency,
        reference_sound_speed=args.reference_sound_speed,
        n_grid=args.n_grid,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()