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
    RssShapeAnalysis,
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


def plot_sound_speed_correction(
    frequencies: np.ndarray,
    baseline_sound_speeds: np.ndarray,
    corrected_sound_speeds: np.ndarray,
    reference_sound_speed: float = 347.0,
    frequency_switch: float = 1500.0,
    relative_tolerance: float = 0.05,
    path: str | Path | None = None,
) -> Figure:
    """Plot baseline and local-minimum-corrected sound-speed estimates."""
    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    baseline_sound_speeds = np.asarray(
        baseline_sound_speeds,
        dtype=np.float64,
    )

    corrected_sound_speeds = np.asarray(
        corrected_sound_speeds,
        dtype=np.float64,
    )

    if not (
        frequencies.shape
        == baseline_sound_speeds.shape
        == corrected_sound_speeds.shape
    ):
        raise ValueError(
            "frequencies and sound-speed arrays must have the same shape"
        )

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies and sound-speed arrays must be one-dimensional"
        )

    if frequencies.size == 0:
        raise ValueError(
            "at least one frequency estimate is required"
        )

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

    order = np.argsort(
        frequencies
    )

    frequencies = frequencies[
        order
    ]

    baseline_sound_speeds = baseline_sound_speeds[
        order
    ]

    corrected_sound_speeds = corrected_sound_speeds[
        order
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        frequencies,
        baseline_sound_speeds,
        "o-",
        linewidth=1.8,
        markersize=4,
        label="Initial estimate",
    )

    axis.plot(
        frequencies,
        corrected_sound_speeds,
        "o-",
        linewidth=2.2,
        markersize=4,
        label=(
            "Corrected estimate "
            f"(f > {frequency_switch:.0f} Hz and "
            f"|k-k_th|/k_th > "
            f"{100 * relative_tolerance:.0f}%)"
        ),
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=(
            f"Theoretical sound speed "
            f"({reference_sound_speed:.0f} m/s)"
        ),
    )

    axis.axvline(
        frequency_switch,
        color="gray",
        linestyle=":",
        linewidth=2,
        label=(
            f"Threshold {frequency_switch:.0f} Hz"
        ),
    )

    axis.set_xlabel(
        "Frequency (Hz)"
    )

    axis.set_ylabel(
        "Estimated sound speed (m/s)"
    )

    axis.set_title(
        "Estimated sound speed versus frequency"
    )

    axis.grid(True)

    axis.legend(
        loc="lower left"
    )

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure





def plot_rss_second_derivative(
    analysis: RssShapeAnalysis,
    top_n: int = 20,
    path: str | Path | None = None,
) -> Figure:
    """Plot RSS, its local minima, and its numerical second derivative."""
    if top_n <= 0:
        raise ValueError(
            "top_n must be strictly positive"
        )

    figure, axis_rss = plt.subplots(
        figsize=(11, 5)
    )

    axis_rss.plot(
        analysis.k_grid,
        analysis.rss_grid,
        linewidth=1.5,
        label="RSS(k)",
    )

    axis_rss.axvline(
        analysis.theoretical_wavenumber_rad_m,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical k",
    )

    axis_rss.axvline(
        analysis.baseline_wavenumber_rad_m,
        color="blue",
        linestyle=":",
        linewidth=2,
        label="RSS-minimization estimate",
    )

    for rank, (k_minimum, rss_minimum) in enumerate(
        zip(
            analysis.local_minima_wavenumbers_rad_m[:top_n],
            analysis.local_minima_rss[:top_n],
        ),
        start=1,
    ):
        axis_rss.scatter(
            k_minimum,
            rss_minimum,
            s=40,
        )

        axis_rss.annotate(
            str(rank),
            (
                k_minimum,
                rss_minimum,
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    k_second = (
        analysis.second_derivative_wavenumber_rad_m
    )

    rss_at_second = float(
        np.interp(
            k_second,
            analysis.k_grid,
            analysis.rss_grid,
        )
    )

    axis_rss.scatter(
        [k_second],
        [rss_at_second],
        s=90,
        marker="X",
        color="green",
        zorder=8,
        label=r"Abscissa of max $RSS''(k)$",
    )

    axis_rss.set_xlabel(
        "k (rad/m)"
    )

    axis_rss.set_ylabel(
        "RSS"
    )

    axis_rss.set_title(
        "RSS objective versus wavenumber "
        f"({analysis.frequency_hz:.2f} Hz)"
    )

    axis_rss.grid(True)

    axis_second = axis_rss.twinx()

    axis_second.plot(
        analysis.k_grid,
        analysis.second_derivative,
        color="green",
        linewidth=1.5,
        alpha=0.85,
        label=r"$RSS''(k)$",
    )

    axis_second.set_ylabel(
        r"Second derivative of $RSS(k)$"
    )

    handles_rss, labels_rss = (
        axis_rss.get_legend_handles_labels()
    )

    handles_second, labels_second = (
        axis_second.get_legend_handles_labels()
    )

    axis_rss.legend(
        handles_rss + handles_second,
        labels_rss + labels_second,
        loc="upper right",
    )

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure

def plot_rss_piecewise_affine(
    analysis: RssShapeAnalysis,
    top_n: int = 20,
    path: str | Path | None = None,
) -> Figure:
    """Plot RSS and its historical R4 piecewise-affine approximation."""
    if top_n <= 0:
        raise ValueError(
            "top_n must be strictly positive"
        )

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.plot(
        analysis.k_grid,
        analysis.rss_grid,
        linewidth=1.5,
        label="RSS(k)",
    )

    axis.axvline(
        analysis.theoretical_wavenumber_rad_m,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical k",
    )

    axis.axvline(
        analysis.baseline_wavenumber_rad_m,
        color="blue",
        linestyle=":",
        linewidth=2,
        label="RSS-minimization estimate",
    )

    for rank, (k_minimum, rss_minimum) in enumerate(
        zip(
            analysis.local_minima_wavenumbers_rad_m[:top_n],
            analysis.local_minima_rss[:top_n],
        ),
        start=1,
    ):
        axis.scatter(
            k_minimum,
            rss_minimum,
            s=40,
        )

        axis.annotate(
            str(rank),
            (
                k_minimum,
                rss_minimum,
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    axis.plot(
        analysis.k_grid,
        analysis.piecewise_fit,
        color="darkorange",
        linewidth=2.5,
        label="Piecewise-affine fit",
    )

    k_break = (
        analysis.piecewise_break_wavenumber_rad_m
    )

    # First cross: breakpoint on the fitted piecewise-affine model.
    axis.scatter(
        [k_break],
        [analysis.piecewise_fit_at_break],
        s=110,
        marker="X",
        color="darkorange",
        zorder=8,
        label="Breakpoint",
    )

    # Second cross: same abscissa projected onto the actual RSS curve.
    rss_at_break = float(
        np.interp(
            k_break,
            analysis.k_grid,
            analysis.rss_grid,
        )
    )

    axis.scatter(
        [k_break],
        [rss_at_break],
        s=110,
        marker="X",
        color="gold",
        zorder=9,
        label="Breakpoint abscissa",
    )

    axis.set_xlabel(
        "k (rad/m)"
    )

    axis.set_ylabel(
        "RSS"
    )

    axis.set_title(
        "RSS objective versus wavenumber "
        f"({analysis.frequency_hz:.2f} Hz)"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure


def plot_rss_shape_estimators(
    analysis: RssShapeAnalysis,
    top_n: int = 20,
    path: str | Path | None = None,
) -> Figure:
    """Compare the historical R4 alternative RSS estimators."""
    if top_n <= 0:
        raise ValueError(
            "top_n must be strictly positive"
        )

    figure, axis_rss = plt.subplots(
        figsize=(11, 5)
    )

    axis_rss.plot(
        analysis.k_grid,
        analysis.rss_grid,
        linewidth=1.5,
        label="RSS(k)",
    )

    axis_rss.axvline(
        analysis.theoretical_wavenumber_rad_m,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical k",
    )

    axis_rss.axvline(
        analysis.baseline_wavenumber_rad_m,
        color="blue",
        linestyle=":",
        linewidth=2,
        label="RSS-minimization estimate",
    )

    for rank, (k_minimum, rss_minimum) in enumerate(
        zip(
            analysis.local_minima_wavenumbers_rad_m[:top_n],
            analysis.local_minima_rss[:top_n],
        ),
        start=1,
    ):
        axis_rss.scatter(
            k_minimum,
            rss_minimum,
            s=40,
        )

        axis_rss.annotate(
            str(rank),
            (
                k_minimum,
                rss_minimum,
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    axis_rss.plot(
        analysis.k_grid,
        analysis.piecewise_fit,
        color="darkorange",
        linewidth=2.5,
        label="Piecewise-affine fit",
    )

    # R4 Figure 17 only projects the breakpoint abscissa onto RSS.
    k_break = (
        analysis.piecewise_break_wavenumber_rad_m
    )

    axis_rss.scatter(
        [k_break],
        [
            np.interp(
                k_break,
                analysis.k_grid,
                analysis.rss_grid,
            )
        ],
        s=110,
        marker="X",
        color="darkorange",
        zorder=9,
        label="Breakpoint abscissa",
    )

    k_second = (
        analysis.second_derivative_wavenumber_rad_m
    )

    axis_rss.scatter(
        [k_second],
        [
            np.interp(
                k_second,
                analysis.k_grid,
                analysis.rss_grid,
            )
        ],
        s=110,
        marker="X",
        color="green",
        zorder=10,
        label=r"Abscissa of max $RSS''(k)$",
    )

    axis_rss.set_xlabel(
        "k (rad/m)"
    )

    axis_rss.set_ylabel(
        "RSS"
    )

    axis_rss.set_title(
        "RSS objective versus wavenumber "
        f"({analysis.frequency_hz:.2f} Hz)"
    )

    axis_rss.grid(True)

    axis_second = axis_rss.twinx()

    axis_second.plot(
        analysis.k_grid,
        analysis.second_derivative,
        color="green",
        linewidth=1.5,
        alpha=0.85,
        label=r"$RSS''(k)$",
    )

    axis_second.set_ylabel(
        r"Second derivative of $RSS(k)$"
    )

    handles_rss, labels_rss = (
        axis_rss.get_legend_handles_labels()
    )

    handles_second, labels_second = (
        axis_second.get_legend_handles_labels()
    )

    axis_rss.legend(
        handles_rss + handles_second,
        labels_rss + labels_second,
        loc="upper right",
    )

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure


def plot_second_derivative_sound_speed(
    frequencies_hz: np.ndarray,
    baseline_sound_speed_m_s: np.ndarray,
    second_derivative_sound_speed_m_s: np.ndarray,
    reference_sound_speed: float = 343.0,
    path: str | Path | None = None,
) -> Figure:
    """Plot the historical R6 second-derivative sound-speed result."""
    frequencies_hz = np.asarray(
        frequencies_hz,
        dtype=np.float64,
    )

    baseline_sound_speed_m_s = np.asarray(
        baseline_sound_speed_m_s,
        dtype=np.float64,
    )

    second_derivative_sound_speed_m_s = np.asarray(
        second_derivative_sound_speed_m_s,
        dtype=np.float64,
    )

    if not (
        frequencies_hz.shape
        == baseline_sound_speed_m_s.shape
        == second_derivative_sound_speed_m_s.shape
    ):
        raise ValueError(
            "frequency and sound-speed arrays must have the same shape"
        )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        frequencies_hz,
        baseline_sound_speed_m_s,
        "o-",
        linewidth=1.8,
        markersize=4,
        label="Initial estimate",
    )

    axis.plot(
        frequencies_hz,
        second_derivative_sound_speed_m_s,
        "o-",
        linewidth=1.8,
        markersize=4,
        color="green",
        label=(
            r"Second-derivative estimate "
            r"(maximum of $RSS''$, $f \geq 1500$ Hz)"
        ),
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{reference_sound_speed:.0f} m/s",
    )

    axis.set_xlabel(
        "Frequency (Hz)"
    )

    axis.set_ylabel(
        "Estimated sound speed (m/s)"
    )

    axis.set_title(
        "Estimated sound speed versus frequency"
    )

    axis.grid(True)

    axis.legend(
        loc="lower left"
    )

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure

def plot_piecewise_affine_sound_speed(
    frequencies_hz: np.ndarray,
    baseline_sound_speed_m_s: np.ndarray,
    piecewise_sound_speed_m_s: np.ndarray,
    frequency_switch: float = 1500.0,
    reference_sound_speed: float = 343.0,
    path: str | Path | None = None,
) -> Figure:
    """Plot the historical R5 piecewise-affine sound-speed result."""
    frequencies_hz = np.asarray(
        frequencies_hz,
        dtype=np.float64,
    )

    baseline_sound_speed_m_s = np.asarray(
        baseline_sound_speed_m_s,
        dtype=np.float64,
    )

    piecewise_sound_speed_m_s = np.asarray(
        piecewise_sound_speed_m_s,
        dtype=np.float64,
    )

    if not (
        frequencies_hz.shape
        == baseline_sound_speed_m_s.shape
        == piecewise_sound_speed_m_s.shape
    ):
        raise ValueError(
            "frequency and sound-speed arrays must have the same shape"
        )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        frequencies_hz,
        baseline_sound_speed_m_s,
        "o-",
        markersize=4,
        linewidth=1.5,
        label="Initial estimate",
    )

    axis.plot(
        frequencies_hz,
        piecewise_sound_speed_m_s,
        "o-",
        markersize=4,
        linewidth=1.8,
        color="green",
        label=(
            "Piecewise-affine estimate "
            f"(for f >= {frequency_switch:.0f} Hz)"
        ),
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=(
            "Theoretical sound speed "
            f"({reference_sound_speed:.0f} m/s)"
        ),
    )

    axis.axvline(
        frequency_switch,
        color="gray",
        linestyle=":",
        linewidth=1.4,
        alpha=0.8,
        label=(
            f"Threshold {frequency_switch:.0f} Hz"
        ),
    )

    axis.set_xlabel(
        "Frequency (Hz)"
    )

    axis.set_ylabel(
        "Estimated sound speed (m/s)"
    )

    axis.set_title(
        "Estimated sound speed versus frequency"
    )

    axis.grid(True)

    axis.legend(
        loc="lower left"
    )

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure




def plot_uma16_retained_band_summary(
    frequencies_hz: np.ndarray,
    sound_speed_m_s: np.ndarray,
    max_frequency: float = 1500.0,
    speed_threshold: float = 300.0,
    reference_sound_speed: float = 343.0,
    confidence_ratio: float = 0.05,
    path: str | Path | None = None,
) -> Figure:
    """Plot the historical UMA16 retained-band sound-speed summary."""
    frequencies_hz = np.asarray(
        frequencies_hz,
        dtype=np.float64,
    )

    sound_speed_m_s = np.asarray(
        sound_speed_m_s,
        dtype=np.float64,
    )

    if frequencies_hz.shape != sound_speed_m_s.shape:
        raise ValueError(
            "frequencies_hz and sound_speed_m_s "
            "must have the same shape"
        )

    if frequencies_hz.ndim != 1:
        raise ValueError(
            "inputs must be one-dimensional"
        )

    if frequencies_hz.size == 0:
        raise ValueError(
            "at least one estimate is required"
        )

    if max_frequency <= 0:
        raise ValueError(
            "max_frequency must be strictly positive"
        )

    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if not 0.0 < confidence_ratio < 1.0:
        raise ValueError(
            "confidence_ratio must lie strictly between 0 and 1"
        )

    order = np.argsort(
        frequencies_hz
    )

    frequencies = frequencies_hz[
        order
    ]

    speeds = sound_speed_m_s[
        order
    ]

    above_threshold = (
        speeds > speed_threshold
    )

    if not np.any(
        above_threshold
    ):
        raise ValueError(
            "no sound-speed estimate exceeds speed_threshold"
        )

    first_index = int(
        np.argmax(
            above_threshold
        )
    )

    min_frequency = float(
        frequencies[first_index]
    )

    retained_mask = (
        (frequencies >= min_frequency)
        & (frequencies <= max_frequency)
    )

    if not np.any(
        retained_mask
    ):
        raise ValueError(
            "the retained frequency band is empty"
        )

    retained_frequencies = frequencies[
        retained_mask
    ]

    retained_speeds = speeds[
        retained_mask
    ]

    mean_speed = float(
        np.mean(
            retained_speeds
        )
    )

    confidence_low = (
        mean_speed
        * (1.0 - confidence_ratio)
    )

    confidence_high = (
        mean_speed
        * (1.0 + confidence_ratio)
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        retained_frequencies,
        retained_speeds,
        "o-",
        label="Estimates",
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=(
            "Theoretical sound speed "
            f"= {reference_sound_speed:.0f} m/s"
        ),
    )

    axis.axhline(
        mean_speed,
        color="green",
        linestyle="--",
        linewidth=2,
        label=(
            f"Mean = {mean_speed:.2f} m/s"
        ),
    )

    axis.fill_between(
        retained_frequencies,
        confidence_low,
        confidence_high,
        color="limegreen",
        alpha=0.20,
        label=(
            f"±{int(confidence_ratio * 100)}% interval "
            f"({confidence_low:.2f}–"
            f"{confidence_high:.2f} m/s)"
        ),
    )

    axis.set_xlabel(
        "Frequency (Hz)"
    )

    axis.set_ylabel(
        "Estimated sound speed (m/s)"
    )

    axis.set_title(
        "Estimated sound speed "
        f"from {min_frequency:.2f} Hz "
        f"to {max_frequency:.0f} Hz"
    )

    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure

def plot_array24_corrected_summary(
    frequencies_hz: np.ndarray,
    corrected_sound_speed_m_s: np.ndarray,
    min_average_frequency: float = 140.0,
    reference_sound_speed: float = 347.0,
    confidence_ratio: float = 0.05,
    path: str | Path | None = None,
) -> Figure:
    """Plot the historical array24 globally corrected sound-speed summary."""
    frequencies_hz = np.asarray(
        frequencies_hz,
        dtype=np.float64,
    )

    corrected_sound_speed_m_s = np.asarray(
        corrected_sound_speed_m_s,
        dtype=np.float64,
    )

    if (
        frequencies_hz.shape
        != corrected_sound_speed_m_s.shape
    ):
        raise ValueError(
            "frequencies_hz and corrected_sound_speed_m_s "
            "must have the same shape"
        )

    if frequencies_hz.ndim != 1:
        raise ValueError(
            "inputs must be one-dimensional"
        )

    if frequencies_hz.size == 0:
        raise ValueError(
            "at least one estimate is required"
        )

    if min_average_frequency < 0:
        raise ValueError(
            "min_average_frequency must be non-negative"
        )

    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if not 0.0 < confidence_ratio < 1.0:
        raise ValueError(
            "confidence_ratio must lie strictly between 0 and 1"
        )

    order = np.argsort(
        frequencies_hz
    )

    frequencies = frequencies_hz[
        order
    ]

    corrected_speeds = corrected_sound_speed_m_s[
        order
    ]

    average_mask = (
        frequencies
        >= min_average_frequency
    )

    if not np.any(
        average_mask
    ):
        raise ValueError(
            "no frequencies lie inside the averaging band"
        )

    mean_speed = float(
        np.mean(
            corrected_speeds[
                average_mask
            ]
        )
    )

    confidence_low = (
        mean_speed
        * (1.0 - confidence_ratio)
    )

    confidence_high = (
        mean_speed
        * (1.0 + confidence_ratio)
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    # Historical Figure 22 keeps the complete corrected curve visible.
    axis.plot(
        frequencies,
        corrected_speeds,
        "o-",
        linewidth=2.0,
        markersize=4,
        label="Corrected sound speed",
    )

    axis.axhline(
        reference_sound_speed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=(
            "Theoretical sound speed "
            f"({reference_sound_speed:.0f} m/s)"
        ),
    )

    axis.axhline(
        mean_speed,
        color="green",
        linestyle="--",
        linewidth=2,
        label=(
            f"Mean for f ≥ "
            f"{min_average_frequency:.0f} Hz: "
            f"{mean_speed:.2f} m/s"
        ),
    )

    axis.fill_between(
        frequencies,
        confidence_low,
        confidence_high,
        where=average_mask,
        color="green",
        alpha=0.18,
        label=(
            f"±{int(confidence_ratio * 100)}% interval: "
            f"[{confidence_low:.2f}, "
            f"{confidence_high:.2f}] m/s"
        ),
    )

    axis.set_xlabel(
        "Frequency (Hz)"
    )

    axis.set_ylabel(
        "Estimated sound speed (m/s)"
    )

    axis.set_title(
        "Estimated sound speed versus frequency"
    )

    axis.grid(True)

    axis.legend(
        loc="lower left"
    )

    figure.tight_layout()

    _save_figure(
        figure,
        path,
    )

    return figure