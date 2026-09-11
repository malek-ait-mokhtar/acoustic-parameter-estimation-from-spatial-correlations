import pytest
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
    LocalMinimum,
    RssShapeAnalysis,
)
from acoustic_estimation.plotting import (
    closest_frequency_estimate,
    plot_coherence_mean,
    plot_mean_spectrum,
    plot_rss,
    plot_rss_landscape,
    plot_sinc_comparison,
    plot_sinc_fit,
    plot_sound_speed,
    plot_sound_speed_retained_band,
    plot_sound_speed_correction,
    plot_rss_piecewise_affine,
    plot_rss_second_derivative,
    plot_rss_shape_estimators,
    plot_piecewise_affine_sound_speed,
    plot_second_derivative_sound_speed,
)


def make_result() -> AnalysisResult:
    """Create a small synthetic analysis result for plotting tests."""
    estimates = [
        FrequencyEstimate(
            frequency_hz=490.0,
            wavenumber_rad_m=9.0,
            sound_speed_m_s=342.1,
            rss=0.3,
            coherence_mean=0.85,
            distances_m=np.array(
                [0.04, 0.08, 0.12]
            ),
            observed_coherence=np.array(
                [0.98, 0.91, 0.80]
            ),
        ),
        FrequencyEstimate(
            frequency_hz=510.0,
            wavenumber_rad_m=9.4,
            sound_speed_m_s=340.9,
            rss=0.4,
            coherence_mean=0.82,
            distances_m=np.array(
                [0.04, 0.08, 0.12]
            ),
            observed_coherence=np.array(
                [0.97, 0.90, 0.78]
            ),
        ),
    ]

    return AnalysisResult(
        estimates=estimates,
        sample_rate_hz=44100.0,
        n_channels=16,
        n_samples=1000,
        n_snapshots=10,
        spectrum_frequencies_hz=np.array(
            [0.0, 250.0, 500.0, 750.0]
        ),
        mean_spectrum=np.array(
            [0.1, 0.5, 2.0, 0.3]
        ),
    )


def test_closest_frequency_estimate():
    result = make_result()

    estimate = closest_frequency_estimate(
        result,
        505.0,
    )

    assert estimate.frequency_hz == 510.0


def test_plot_sound_speed_returns_figure():
    figure = plot_sound_speed(
        make_result()
    )

    assert figure is not None

    plt.close(figure)


def test_plot_sound_speed_retained_band_returns_figure():
    figure = plot_sound_speed_retained_band(
        make_result(),
        min_frequency=400.0,
        max_frequency=600.0,
    )

    assert figure is not None

    plt.close(figure)


def test_plot_rss_returns_figure():
    figure = plot_rss(
        make_result()
    )

    assert figure is not None

    plt.close(figure)


def test_plot_sinc_comparison_returns_figure():
    figure = plot_sinc_comparison(
        make_result(),
        target_frequency=500.0,
    )

    assert figure is not None

    plt.close(figure)


def test_plot_sinc_fit_returns_figure():
    figure = plot_sinc_fit(
        make_result(),
        target_frequency=500.0,
    )

    assert figure is not None

    plt.close(figure)


def test_plot_coherence_mean_returns_figure():
    figure = plot_coherence_mean(
        make_result()
    )

    assert figure is not None

    plt.close(figure)


def test_plot_mean_spectrum_returns_figure():
    figure = plot_mean_spectrum(
        make_result()
    )

    assert figure is not None

    plt.close(figure)


def test_plot_can_save_figure(tmp_path):
    path = tmp_path / "sound_speed.png"

    figure = plot_sound_speed(
        make_result(),
        path=path,
    )

    assert path.is_file()

    plt.close(figure)


def test_retained_band_plot_rejects_empty_band():
    with pytest.raises(ValueError):
        plot_sound_speed_retained_band(
            make_result(),
            min_frequency=1000.0,
            max_frequency=1500.0,
        )


def test_plot_mean_spectrum_can_limit_frequency_range():
    figure = plot_mean_spectrum(
        make_result(),
        max_frequency=600.0,
    )

    axis = figure.axes[0]

    assert np.allclose(
        axis.get_xlim(),
        (0.0, 600.0),
    )

    plt.close(figure)


def test_plot_rss_landscape_returns_figure():
    k_grid = np.linspace(
        1.0,
        10.0,
        100,
    )

    rss_grid = (
        k_grid - 5.0
    ) ** 2

    minima = [
        LocalMinimum(
            wavenumber_rad_m=5.0,
            rss=0.0,
        )
    ]

    figure = plot_rss_landscape(
        k_grid=k_grid,
        rss_grid=rss_grid,
        theoretical_wavenumber=5.2,
        baseline_wavenumber=5.0,
        local_minima=minima,
        frequency=1500.0,
    )

    assert figure is not None

    plt.close(figure)
    
def test_plot_sound_speed_correction_returns_figure():
    frequencies = np.array(
        [1000.0, 1600.0, 2000.0]
    )

    baseline = np.array(
        [345.0, 180.0, 200.0]
    )

    corrected = np.array(
        [345.0, 350.0, 348.0]
    )

    figure = plot_sound_speed_correction(
        frequencies=frequencies,
        baseline_sound_speeds=baseline,
        corrected_sound_speeds=corrected,
    )

    assert figure is not None

    plt.close(
        figure
    )
    
def make_rss_shape_analysis() -> RssShapeAnalysis:
    """Create a small synthetic RSS-shape analysis for plotting tests."""
    k_grid = np.linspace(
        10.0,
        40.0,
        100,
    )

    rss_grid = (
        5.0
        - 0.1 * k_grid
        + 0.08
        * np.maximum(
            0.0,
            k_grid - 25.0,
        )
    )

    second_derivative = np.gradient(
        np.gradient(
            rss_grid,
            k_grid,
        ),
        k_grid,
    )

    return RssShapeAnalysis(
        frequency_hz=1528.86,
        theoretical_wavenumber_rad_m=28.0,
        baseline_wavenumber_rad_m=35.0,
        k_grid=k_grid,
        rss_grid=rss_grid,
        local_minima_wavenumbers_rad_m=np.array(
            [18.0, 32.0],
            dtype=np.float64,
        ),
        local_minima_rss=np.array(
            [3.2, 2.8],
            dtype=np.float64,
        ),
        second_derivative=second_derivative,
        second_derivative_wavenumber_rad_m=25.0,
        piecewise_break_wavenumber_rad_m=25.0,
        piecewise_fit=rss_grid.copy(),
        piecewise_fit_at_break=float(
            np.interp(
                25.0,
                k_grid,
                rss_grid,
            )
        ),
    )


def test_plot_rss_second_derivative_returns_figure():
    figure = plot_rss_second_derivative(
        make_rss_shape_analysis()
    )

    assert figure is not None
    assert len(figure.axes) == 2

    plt.close(
        figure
    )


def test_plot_rss_piecewise_affine_returns_figure():
    figure = plot_rss_piecewise_affine(
        make_rss_shape_analysis()
    )

    assert figure is not None
    assert len(figure.axes) == 1

    plt.close(
        figure
    )


def test_plot_rss_shape_estimators_returns_figure():
    figure = plot_rss_shape_estimators(
        make_rss_shape_analysis()
    )

    assert figure is not None
    assert len(figure.axes) == 2

    plt.close(
        figure
    )
    
def test_plot_second_derivative_sound_speed_returns_figure():
    frequencies = np.array(
        [1000.0, 1500.0, 2000.0]
    )

    baseline = np.array(
        [340.0, 180.0, 190.0]
    )

    adjusted = np.array(
        [340.0, 600.0, 700.0]
    )

    figure = plot_second_derivative_sound_speed(
        frequencies,
        baseline,
        adjusted,
    )

    assert figure is not None
    assert len(figure.axes) == 1

    plt.close(
        figure
    )


def test_plot_piecewise_affine_sound_speed_returns_figure():
    frequencies = np.array(
        [1000.0, 1500.0, 2000.0]
    )

    baseline = np.array(
        [340.0, 180.0, 190.0]
    )

    adjusted = np.array(
        [340.0, 360.0, 350.0]
    )

    figure = plot_piecewise_affine_sound_speed(
        frequencies,
        baseline,
        adjusted,
    )

    assert figure is not None
    assert len(figure.axes) == 1

    plt.close(
        figure
    )
    
def test_piecewise_sound_speed_plot_has_threshold_line():
    figure = plot_piecewise_affine_sound_speed(
        np.array([1000.0, 2000.0]),
        np.array([340.0, 180.0]),
        np.array([340.0, 350.0]),
    )

    axis = figure.axes[0]

    vertical_lines = [
        line
        for line in axis.lines
        if np.allclose(
            np.asarray(line.get_xdata()),
            1500.0,
        )
    ]

    assert len(vertical_lines) == 1

    plt.close(figure)


def test_piecewise_sound_speed_plot_has_threshold_line():
    figure = plot_piecewise_affine_sound_speed(
        np.array([1000.0, 2000.0]),
        np.array([340.0, 180.0]),
        np.array([340.0, 350.0]),
    )

    axis = figure.axes[0]

    vertical_lines = [
        line
        for line in axis.lines
        if (
            len(np.asarray(line.get_xdata())) == 2
            and np.allclose(
                np.asarray(line.get_xdata()),
                [1500.0, 1500.0],
            )
        )
    ]

    assert len(vertical_lines) == 1

    plt.close(figure)