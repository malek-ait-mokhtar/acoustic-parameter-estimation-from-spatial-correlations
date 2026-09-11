"""Visualisation utilities for acoustic parameter estimation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
    LocalMinimum,
)
from acoustic_estimation.models import spherical_sinc


def _save_figure(
    figure: Figure,
    path: str | Path | None,
) -> None:
    """Save a figure when an output path is provided."""
    if path is None:
        return

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )


def _frequency_arrays(
    result: AnalysisResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sorted frequency, sound-speed, and RSS arrays."""
    frequencies = np.array(
        [estimate.frequency_hz for estimate in result.estimates]
    )
    sound_speeds = np.array(
        [estimate.sound_speed_m_s for estimate in result.estimates]
    )
    rss = np.array(
        [estimate.rss for estimate in result.estimates]
    )

    order = np.argsort(frequencies)

    return (
        frequencies[order],
        sound_speeds[order],
        rss[order],
    )


# ======================================================================
# Primary figures
# ======================================================================


def plot_sound_speed(
    result: AnalysisResult,
    reference_sound_speed: float = 343.0,
    path: str | Path | None = None,
) -> Figure:
    """Plot estimated sound speed over the complete analysed band."""
    frequencies, sound_speeds, _ = _frequency_arrays(result)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        frequencies,
        sound_speeds,
        "o-",
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        label=f"{reference_sound_speed:g} m/s",
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Estimated sound speed (m/s)")
    axis.set_title("Sound speed versus frequency")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_sound_speed_retained_band(
    result: AnalysisResult,
    min_frequency: float = 290.0,
    max_frequency: float = 1500.0,
    reference_sound_speed: float = 343.0,
    interval_ratio: float = 0.05,
    path: str | Path | None = None,
) -> Figure:
    """Plot sound-speed estimates over the retained UMA16 frequency band."""
    frequencies, sound_speeds, _ = _frequency_arrays(result)

    mask = (
        (frequencies >= min_frequency)
        & (frequencies <= max_frequency)
    )

    if not np.any(mask):
        raise ValueError(
            "no frequency estimates lie inside the requested frequency band"
        )

    selected_frequencies = frequencies[mask]
    selected_speeds = sound_speeds[mask]

    mean_speed = float(
        np.mean(selected_speeds)
    )

    lower_bound = mean_speed * (
        1.0 - interval_ratio
    )
    upper_bound = mean_speed * (
        1.0 + interval_ratio
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        selected_frequencies,
        selected_speeds,
        "o-",
        label="Estimates",
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=(
            f"Theoretical sound speed = "
            f"{reference_sound_speed:g} m/s"
        ),
    )

    axis.axhline(
        mean_speed,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_speed:.2f} m/s",
    )

    axis.fill_between(
        selected_frequencies,
        lower_bound,
        upper_bound,
        color="limegreen",
        alpha=0.20,
        label=(
            f"±{100 * interval_ratio:g}% interval "
            f"({lower_bound:.2f}–{upper_bound:.2f} m/s)"
        ),
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Estimated sound speed (m/s)")
    axis.set_title(
        "Estimated sound speed "
        f"from {min_frequency:g} Hz to {max_frequency:g} Hz"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def closest_frequency_estimate(
    result: AnalysisResult,
    target_frequency: float,
) -> FrequencyEstimate:
    """Return the retained estimate closest to a target frequency."""
    if not result.estimates:
        raise ValueError(
            "analysis result contains no frequency estimates"
        )

    return min(
        result.estimates,
        key=lambda estimate: abs(
            estimate.frequency_hz - target_frequency
        ),
    )


def plot_sinc_comparison(
    result: AnalysisResult,
    target_frequency: float = 500.0,
    reference_sound_speed: float = 343.0,
    path: str | Path | None = None,
) -> Figure:
    """Compare measured, theoretical, and fitted coherence near 500 Hz."""
    estimate = closest_frequency_estimate(
        result,
        target_frequency,
    )

    frequency = estimate.frequency_hz
    distances = estimate.distances_m
    observed = estimate.observed_coherence

    theoretical_wavenumber = (
        2.0
        * np.pi
        * frequency
        / reference_sound_speed
    )

    distance_grid = np.linspace(
        0.0,
        np.max(distances) * 1.05,
        2000,
    )

    theoretical_model = spherical_sinc(
        distance_grid,
        theoretical_wavenumber,
    )

    fitted_model = spherical_sinc(
        distance_grid,
        estimate.wavenumber_rad_m,
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.scatter(
        distances,
        observed,
        s=35,
        alpha=0.8,
        label="Measurements",
    )

    axis.plot(
        distance_grid,
        theoretical_model,
        linewidth=2,
        label=(
            "Theoretical sin(kr)/(kr) "
            f"(c={reference_sound_speed:g} m/s)"
        ),
    )

    axis.plot(
        distance_grid,
        fitted_model,
        "--",
        linewidth=2,
        label="Fitted sin(kr)/(kr)",
    )

    axis.axhline(
        0,
        color="gray",
        linestyle=":",
    )

    axis.set_xlabel("Distance r (m)")
    axis.set_ylabel(r"$\mathrm{Re}(\Gamma_{ij})$")
    axis.set_title(
        f"Spatial coherence at {frequency:.2f} Hz"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


# ======================================================================
# Secondary diagnostic figures
# ======================================================================


def plot_rss(
    result: AnalysisResult,
    path: str | Path | None = None,
) -> Figure:
    """Plot minimum RSS versus frequency on a logarithmic scale."""
    frequencies, _, rss = _frequency_arrays(result)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.semilogy(
        frequencies,
        rss,
        "o-",
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("RSS")
    axis.set_title("Sinc model fit quality")

    axis.grid(True)

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_coherence_mean(
    result: AnalysisResult,
    path: str | Path | None = None,
) -> Figure:
    """Plot mean spatial coherence magnitude versus frequency."""
    frequencies = np.array(
        [estimate.frequency_hz for estimate in result.estimates]
    )

    coherence = np.array(
        [estimate.coherence_mean for estimate in result.estimates]
    )

    order = np.argsort(frequencies)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        frequencies[order],
        coherence[order],
        "o-",
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Mean coherence")
    axis.set_title("Mean spatial coherence")

    axis.grid(True)

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_mean_spectrum(
    result: AnalysisResult,
    relative_threshold: float = 0.03,
    max_frequency: float | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Plot the mean microphone spectrum and selection threshold."""
    frequencies = result.spectrum_frequencies_hz
    spectrum = result.mean_spectrum

    threshold = (
        relative_threshold
        * np.max(spectrum)
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        frequencies,
        spectrum,
        label="Mean spectrum",
    )

    axis.axhline(
        threshold,
        linestyle="--",
        label=(
            f"Selection threshold "
            f"({100 * relative_threshold:g}% of maximum)"
        ),
    )

    if max_frequency is not None:
        if max_frequency <= 0:
            raise ValueError(
                "max_frequency must be strictly positive"
            )

        axis.set_xlim(
            0.0,
            max_frequency,
        )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Mean spectral amplitude")
    axis.set_title("Mean microphone spectrum")

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_sinc_fit(
    result: AnalysisResult,
    target_frequency: float = 500.0,
    path: str | Path | None = None,
) -> Figure:
    """Plot measured coherence and the fitted sinc model."""
    estimate = closest_frequency_estimate(
        result,
        target_frequency,
    )

    order = np.argsort(
        estimate.distances_m
    )

    distances = estimate.distances_m[order]
    observed = estimate.observed_coherence[order]

    distance_grid = np.linspace(
        0.0,
        np.max(distances) * 1.05,
        1000,
    )

    fitted_model = spherical_sinc(
        distance_grid,
        estimate.wavenumber_rad_m,
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.scatter(
        distances,
        observed,
        s=30,
        alpha=0.8,
        label="Measurements",
    )

    axis.plot(
        distance_grid,
        fitted_model,
        linewidth=2,
        label="Sinc model",
    )

    axis.axhline(
        0,
        color="gray",
        linestyle="--",
    )

    axis.set_xlabel("Microphone separation (m)")
    axis.set_ylabel(r"$\mathrm{Re}(\Gamma_{ij})$")
    axis.set_title(
        f"Fit at {estimate.frequency_hz:.2f} Hz"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_rss_landscape(
    k_grid: np.ndarray,
    rss_grid: np.ndarray,
    theoretical_wavenumber: float,
    baseline_wavenumber: float,
    local_minima: list[LocalMinimum],
    frequency: float,
    top_n: int = 20,
    path: str | Path | None = None,
) -> Figure:
    """Plot an RSS wavenumber landscape and its local minima."""
    if top_n <= 0:
        raise ValueError(
            "top_n must be strictly positive"
        )

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        k_grid,
        rss_grid,
        linewidth=1.5,
        label="RSS(k)",
    )

    axis.axvline(
        theoretical_wavenumber,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical k",
    )

    axis.axvline(
        baseline_wavenumber,
        linestyle=":",
        linewidth=2,
        label="Baseline estimate",
    )

    for rank, minimum in enumerate(
        local_minima[:top_n],
        start=1,
    ):
        axis.scatter(
            minimum.wavenumber_rad_m,
            minimum.rss,
            s=40,
        )

        axis.annotate(
            str(rank),
            (
                minimum.wavenumber_rad_m,
                minimum.rss,
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    axis.set_xlabel("k (rad/m)")
    axis.set_ylabel("RSS")

    axis.set_title(
        "RSS objective versus wavenumber "
        f"({frequency:.2f} Hz)"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure