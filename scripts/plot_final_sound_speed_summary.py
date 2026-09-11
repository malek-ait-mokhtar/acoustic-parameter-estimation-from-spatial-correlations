"""Generate the final UMA16 and array24 sound-speed summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustic_estimation.plotting import (
    plot_array24_corrected_summary,
    plot_uma16_retained_band_summary,
)


UMA16_RESULTS = Path(
    "results/uma16/crous/analysis.npz"
)

ARRAY24_RESULTS = Path(
    "results/array24/global_local_minima_corrected.npz"
)

OUTPUT_DIRECTORY = Path(
    "results/final_summary"
)


def main() -> None:
    """Generate the historical final sound-speed summary figures."""
    if not UMA16_RESULTS.is_file():
        raise FileNotFoundError(
            f"UMA16 results not found: {UMA16_RESULTS}"
        )

    if not ARRAY24_RESULTS.is_file():
        raise FileNotFoundError(
            f"Array24 results not found: {ARRAY24_RESULTS}"
        )

    uma16 = np.load(
        UMA16_RESULTS
    )

    array24 = np.load(
        ARRAY24_RESULTS
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Figure 21 — UMA16 retained band.
    # ---------------------------------------------------------

    uma_frequencies = np.asarray(
        uma16["frequency_hz"],
        dtype=np.float64,
    )

    uma_speeds = np.asarray(
        uma16["sound_speed_m_s"],
        dtype=np.float64,
    )

    order = np.argsort(
        uma_frequencies
    )

    sorted_frequencies = uma_frequencies[
        order
    ]

    sorted_speeds = uma_speeds[
        order
    ]

    above_threshold = (
        sorted_speeds > 300.0
    )

    if not np.any(
        above_threshold
    ):
        raise RuntimeError(
            "no UMA16 estimate exceeds 300 m/s"
        )

    first_index = int(
        np.argmax(
            above_threshold
        )
    )

    uma_min_frequency = float(
        sorted_frequencies[
            first_index
        ]
    )

    uma_mask = (
        (sorted_frequencies >= uma_min_frequency)
        & (sorted_frequencies <= 1500.0)
    )

    uma_mean = float(
        np.mean(
            sorted_speeds[
                uma_mask
            ]
        )
    )

    # ---------------------------------------------------------
    # Figure 22 — globally corrected array24.
    # ---------------------------------------------------------

    array_frequencies = np.asarray(
        array24["frequency_hz"],
        dtype=np.float64,
    )

    array_speeds = np.asarray(
        array24["corrected_sound_speed_m_s"],
        dtype=np.float64,
    )

    array_mask = (
        array_frequencies >= 140.0
    )

    array_mean = float(
        np.mean(
            array_speeds[
                array_mask
            ]
        )
    )

    # ---------------------------------------------------------
    # Numerical summary.
    # ---------------------------------------------------------

    print("=" * 72)
    print("FINAL SOUND-SPEED SUMMARY")
    print("=" * 72)

    print()
    print("UMA16 — Figure 21")
    print("-----------------")

    print(
        f"First retained frequency : "
        f"{uma_min_frequency:.6f} Hz"
    )

    print(
        f"Retained frequencies     : "
        f"{np.count_nonzero(uma_mask)}"
    )

    print(
        f"Mean sound speed         : "
        f"{uma_mean:.6f} m/s"
    )

    print(
        f"±5% interval             : "
        f"[{0.95 * uma_mean:.6f}, "
        f"{1.05 * uma_mean:.6f}] m/s"
    )

    print()
    print("Array24 — Figure 22")
    print("-------------------")

    print(
        f"First averaged frequency : "
        f"{np.min(array_frequencies[array_mask]):.6f} Hz"
    )

    print(
        f"Averaged frequencies     : "
        f"{np.count_nonzero(array_mask)}"
    )

    print(
        f"Mean sound speed         : "
        f"{array_mean:.6f} m/s"
    )

    print(
        f"±5% interval             : "
        f"[{0.95 * array_mean:.6f}, "
        f"{1.05 * array_mean:.6f}] m/s"
    )

    # ---------------------------------------------------------
    # Figures.
    # ---------------------------------------------------------

    uma_figure = plot_uma16_retained_band_summary(
        frequencies_hz=uma_frequencies,
        sound_speed_m_s=uma_speeds,
        max_frequency=1500.0,
        speed_threshold=300.0,
        reference_sound_speed=343.0,
        confidence_ratio=0.05,
        path=(
            OUTPUT_DIRECTORY
            / "figure21_uma16_retained_band.png"
        ),
    )

    array_figure = plot_array24_corrected_summary(
        frequencies_hz=array_frequencies,
        corrected_sound_speed_m_s=array_speeds,
        min_average_frequency=140.0,
        reference_sound_speed=347.0,
        confidence_ratio=0.05,
        path=(
            OUTPUT_DIRECTORY
            / "figure22_array24_corrected.png"
        ),
    )

    plt.close(
        uma_figure
    )

    plt.close(
        array_figure
    )

    print()
    print(
        f"Figures saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    main()