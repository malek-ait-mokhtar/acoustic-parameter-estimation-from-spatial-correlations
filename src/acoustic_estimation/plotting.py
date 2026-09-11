"""Visualisation utilities for acoustic parameter estimation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
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
    path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )


def plot_sound_speed(
    result: AnalysisResult,
    reference_sound_speed: float = 343.0,
    path: str | Path | None = None,
) -> Figure:
    """Plot estimated sound speed as a function of frequency."""
    frequencies = np.array(
        [estimate.frequency_hz for estimate in result.estimates]
    )
    sound_speeds = np.array(
        [estimate.sound_speed_m_s for estimate in result.estimates]
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        frequencies,
        sound_speeds,
        "o-",
        markersize=3,
        linewidth=1,
        label="Estimated sound speed",
    )

    axis.axhline(
        reference_sound_speed,
        linestyle="--",
        label=f"Reference: {reference_sound_speed:.0f} m/s",
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Sound speed (m/s)")
    axis.set_title("Estimated sound speed")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_rss(
    result: AnalysisResult,
    path: str | Path | None = None,
) -> Figure:
    """Plot minimum RSS as a function of frequency."""
    frequencies = np.array(
        [estimate.frequency_hz for estimate in result.estimates]
    )
    rss = np.array(
        [estimate.rss for estimate in result.estimates]
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        frequencies,
        rss,
        "o-",
        markersize=3,
        linewidth=1,
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("RSS")
    axis.set_title("Model residual error")
    axis.grid(alpha=0.3)

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def closest_frequency_estimate(
    result: AnalysisResult,
    target_frequency: float,
) -> FrequencyEstimate:
    """Return the retained estimate closest to a target frequency."""
    if not result.estimates:
        raise ValueError("analysis result contains no frequency estimates")

    return min(
        result.estimates,
        key=lambda estimate: abs(
            estimate.frequency_hz - target_frequency
        ),
    )


def plot_sinc_fit(
    result: AnalysisResult,
    target_frequency: float = 500.0,
    path: str | Path | None = None,
) -> Figure:
    """Plot measured spatial coherence and fitted sinc model."""
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
        max(distances) * 1.05,
        500,
    )

    fitted = spherical_sinc(
        distance_grid,
        estimate.wavenumber_rad_m,
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.scatter(
        distances,
        observed,
        s=25,
        alpha=0.7,
        label="Measured coherence",
    )

    axis.plot(
        distance_grid,
        fitted,
        linewidth=2,
        label="Fitted sinc model",
    )

    axis.set_xlabel("Microphone separation (m)")
    axis.set_ylabel(r"$\mathrm{Re}(\Gamma_{ij})$")
    axis.set_title(
        f"Spatial coherence fit at "
        f"{estimate.frequency_hz:.1f} Hz"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure


def plot_mean_spectrum(
    result: AnalysisResult,
    relative_threshold: float = 0.03,
    path: str | Path | None = None,
) -> Figure:
    """Plot the mean spectrum and the frequency-selection threshold."""
    frequencies = result.spectrum_frequencies_hz
    spectrum = result.mean_spectrum

    threshold = (
        relative_threshold
        * np.max(spectrum)
    )

    figure, axis = plt.subplots(figsize=(9, 5))

    axis.plot(
        frequencies,
        spectrum,
        linewidth=1,
        label="Mean spectrum",
    )

    axis.axhline(
        threshold,
        linestyle="--",
        label=(
            f"Selection threshold "
            f"({100 * relative_threshold:.0f}% of maximum)"
        ),
    )

    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Mean spectral amplitude")
    axis.set_title("Mean microphone spectrum")
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    _save_figure(figure, path)

    return figure