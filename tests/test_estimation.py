import numpy as np
import pytest

from acoustic_estimation.estimation import (
    LocalMinimum,
    build_pairwise_dataset,
    closest_minimum_to_wavenumber,
    detect_local_minima,
    estimate_wavenumber,
    evaluate_rss_landscape,
    refine_local_minima,
    sinc_rss,
    sound_speed_from_wavenumber,
    theoretical_wavenumber,
    CorrectedFrequencyEstimate,
    correct_wavenumber_with_local_minima,
)
from acoustic_estimation.models import spherical_sinc


def test_theoretical_wavenumber():
    frequency = 500.0
    sound_speed = 343.0

    expected = (
        2.0
        * np.pi
        * frequency
        / sound_speed
    )

    result = theoretical_wavenumber(
        frequency,
        sound_speed,
    )

    assert np.isclose(
        result,
        expected,
    )


def test_theoretical_wavenumber_rejects_negative_frequency():
    with pytest.raises(ValueError):
        theoretical_wavenumber(
            -1.0,
            343.0,
        )


def test_theoretical_wavenumber_rejects_non_positive_sound_speed():
    with pytest.raises(ValueError):
        theoretical_wavenumber(
            500.0,
            0.0,
        )


def test_sound_speed_from_wavenumber():
    frequency = 500.0
    wavenumber = (
        2.0
        * np.pi
        * frequency
        / 343.0
    )

    result = sound_speed_from_wavenumber(
        frequency,
        wavenumber,
    )

    assert np.isclose(
        result,
        343.0,
    )


def test_sound_speed_from_wavenumber_rejects_non_positive_wavenumber():
    with pytest.raises(ValueError):
        sound_speed_from_wavenumber(
            500.0,
            0.0,
        )


def test_build_pairwise_dataset():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    coherence = np.array(
        [
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.4],
            [0.6, 0.4, 1.0],
        ],
        dtype=np.complex128,
    )

    distances, observed = build_pairwise_dataset(
        coherence,
        positions,
    )

    expected_distances = np.array(
        [
            1.0,
            1.0,
            np.sqrt(2.0),
        ]
    )

    expected_observed = np.array(
        [
            0.8,
            0.6,
            0.4,
        ]
    )

    assert np.allclose(
        distances,
        expected_distances,
    )

    assert np.allclose(
        observed,
        expected_observed,
    )


def test_build_pairwise_dataset_uses_real_coherence():
    positions = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    coherence = np.array(
        [
            [1.0, 0.5 + 0.3j],
            [0.5 - 0.3j, 1.0],
        ]
    )

    _, observed = build_pairwise_dataset(
        coherence,
        positions,
    )

    assert np.allclose(
        observed,
        [0.5],
    )


def test_build_pairwise_dataset_supports_3d_geometry():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    coherence = np.eye(
        3,
        dtype=np.complex128,
    )

    distances, _ = build_pairwise_dataset(
        coherence,
        positions,
    )

    assert np.allclose(
        distances,
        [
            1.0,
            1.0,
            np.sqrt(2.0),
        ],
    )


def test_sinc_rss_is_zero_for_exact_model():
    distances = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
        ]
    )

    wavenumber = 8.0

    observed = spherical_sinc(
        distances,
        wavenumber,
    )

    rss = sinc_rss(
        wavenumber,
        distances,
        observed,
    )

    assert np.isclose(
        rss,
        0.0,
        atol=1e-15,
    )


def test_sinc_rss_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        sinc_rss(
            5.0,
            np.array([0.1, 0.2]),
            np.array([0.5]),
        )


def test_estimate_wavenumber_recovers_synthetic_value():
    frequency = 500.0
    true_sound_speed = 343.0

    true_wavenumber = theoretical_wavenumber(
        frequency,
        true_sound_speed,
    )

    distances = np.linspace(
        0.01,
        0.4,
        100,
    )

    observed = spherical_sinc(
        distances,
        true_wavenumber,
    )

    estimated_wavenumber, rss = estimate_wavenumber(
        distances,
        observed,
        frequency,
        reference_sound_speed=true_sound_speed,
    )

    assert np.isclose(
        estimated_wavenumber,
        true_wavenumber,
        rtol=1e-5,
    )

    assert rss < 1e-10


def test_estimate_wavenumber_recovers_sound_speed():
    frequency = 500.0
    true_sound_speed = 343.0

    true_wavenumber = theoretical_wavenumber(
        frequency,
        true_sound_speed,
    )

    distances = np.linspace(
        0.01,
        0.4,
        100,
    )

    observed = spherical_sinc(
        distances,
        true_wavenumber,
    )

    estimated_wavenumber, _ = estimate_wavenumber(
        distances,
        observed,
        frequency,
        reference_sound_speed=true_sound_speed,
    )

    estimated_sound_speed = sound_speed_from_wavenumber(
        frequency,
        estimated_wavenumber,
    )

    assert np.isclose(
        estimated_sound_speed,
        true_sound_speed,
        rtol=1e-5,
    )


def test_estimate_wavenumber_rejects_empty_dataset():
    with pytest.raises(ValueError):
        estimate_wavenumber(
            np.array([]),
            np.array([]),
            frequency=500.0,
        )


def test_estimate_wavenumber_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        estimate_wavenumber(
            np.array([0.1, 0.2]),
            np.array([0.5]),
            frequency=500.0,
        )


def test_estimate_wavenumber_rejects_invalid_bounds():
    distances = np.array(
        [0.1, 0.2]
    )

    observed = np.array(
        [0.9, 0.7]
    )

    with pytest.raises(ValueError):
        estimate_wavenumber(
            distances,
            observed,
            frequency=500.0,
            lower_factor=2.0,
            upper_factor=1.0,
        )


# ======================================================================
# RSS landscape and local-minimum analysis
# ======================================================================


def test_detect_local_minima():
    k_grid = np.arange(
        7,
        dtype=float,
    )

    rss_grid = np.array(
        [
            5.0,
            2.0,
            4.0,
            1.0,
            4.0,
            2.0,
            5.0,
        ]
    )

    indices = detect_local_minima(
        k_grid,
        rss_grid,
    )

    assert np.array_equal(
        indices,
        np.array(
            [1, 3, 5]
        ),
    )


def test_evaluate_rss_landscape():
    distances = np.array(
        [
            0.1,
            0.2,
            0.3,
        ]
    )

    observed = np.array(
        [
            0.9,
            0.7,
            0.5,
        ]
    )

    k_grid, rss_grid = evaluate_rss_landscape(
        distances,
        observed,
        k_min=1.0,
        k_max=10.0,
        n_grid=100,
    )

    assert k_grid.shape == (100,)
    assert rss_grid.shape == (100,)

    assert np.isclose(
        k_grid[0],
        1.0,
    )

    assert np.isclose(
        k_grid[-1],
        10.0,
    )

    assert np.all(
        rss_grid >= 0.0
    )


def test_refine_local_minimum_recovers_synthetic_wavenumber():
    true_wavenumber = 12.0

    distances = np.linspace(
        0.03,
        0.8,
        100,
    )

    observed = spherical_sinc(
        distances,
        true_wavenumber,
    )

    k_grid, rss_grid = evaluate_rss_landscape(
        distances,
        observed,
        k_min=5.0,
        k_max=20.0,
        n_grid=1000,
    )

    indices = detect_local_minima(
        k_grid,
        rss_grid,
    )

    minima = refine_local_minima(
        k_grid,
        rss_grid,
        indices,
        distances,
        observed,
    )

    best = min(
        minima,
        key=lambda minimum: minimum.rss,
    )

    assert np.isclose(
        best.wavenumber_rad_m,
        true_wavenumber,
        atol=1e-6,
    )

    assert best.rss < 1e-12


def test_closest_minimum_to_wavenumber():
    minima = [
        LocalMinimum(
            wavenumber_rad_m=10.0,
            rss=1.0,
        ),
        LocalMinimum(
            wavenumber_rad_m=20.0,
            rss=0.5,
        ),
        LocalMinimum(
            wavenumber_rad_m=30.0,
            rss=0.2,
        ),
    ]

    selected = closest_minimum_to_wavenumber(
        minima,
        target_wavenumber=22.0,
    )

    assert (
        selected.wavenumber_rad_m
        == 20.0
    )
    

def test_local_minimum_correction_keeps_low_frequency_estimate():
    frequency = 1000.0
    baseline_wavenumber = 30.0

    result = correct_wavenumber_with_local_minima(
        frequency=frequency,
        baseline_wavenumber=baseline_wavenumber,
        distances=np.array([0.1, 0.2]),
        observed_coherence=np.array([0.8, 0.5]),
    )

    assert isinstance(
        result,
        CorrectedFrequencyEstimate,
    )

    assert not result.was_corrected

    assert (
        result.corrected_wavenumber_rad_m
        == baseline_wavenumber
    )

def test_local_minimum_correction_keeps_close_high_frequency_estimate():
    frequency = 2000.0

    theoretical_k = theoretical_wavenumber(
        frequency,
        347.0,
    )

    baseline_wavenumber = (
        1.02 * theoretical_k
    )

    result = correct_wavenumber_with_local_minima(
        frequency=frequency,
        baseline_wavenumber=baseline_wavenumber,
        distances=np.array([0.1, 0.2]),
        observed_coherence=np.array([0.8, 0.5]),
    )

    assert not result.was_corrected

    assert np.isclose(
        result.corrected_wavenumber_rad_m,
        baseline_wavenumber,
    )
    
def test_local_minimum_correction_recovers_physical_high_frequency_minimum():
    frequency = 2000.0
    reference_sound_speed = 347.0

    true_wavenumber = theoretical_wavenumber(
        frequency,
        reference_sound_speed,
    )

    distances = np.linspace(
        0.03,
        1.0,
        200,
    )

    observed = spherical_sinc(
        distances,
        true_wavenumber,
    )

    # Deliberately provide a strongly incorrect baseline estimate.
    baseline_wavenumber = (
        2.0 * true_wavenumber
    )

    result = correct_wavenumber_with_local_minima(
        frequency=frequency,
        baseline_wavenumber=baseline_wavenumber,
        distances=distances,
        observed_coherence=observed,
        reference_sound_speed=reference_sound_speed,
        n_grid=4000,
    )

    assert result.was_corrected

    assert np.isclose(
        result.corrected_wavenumber_rad_m,
        true_wavenumber,
        rtol=1e-5,
    )

    assert np.isclose(
        result.corrected_sound_speed_m_s,
        reference_sound_speed,
        rtol=1e-5,
    )