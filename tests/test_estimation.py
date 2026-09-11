import numpy as np
import pytest

from acoustic_estimation.estimation import (
    build_pairwise_dataset,
    estimate_wavenumber,
    sinc_rss,
    sound_speed_from_wavenumber,
    theoretical_wavenumber,
)
from acoustic_estimation.models import spherical_sinc


def test_build_pairwise_dataset_simple_geometry():
    positions = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ]
    )

    coherence = np.array(
        [
            [1.0, 0.2 + 0.1j, 0.3 - 0.2j],
            [0.2 - 0.1j, 1.0, -0.4 + 0.5j],
            [0.3 + 0.2j, -0.4 - 0.5j, 1.0],
        ]
    )

    distances, observed = build_pairwise_dataset(
        coherence,
        positions,
    )

    assert np.allclose(distances, [3.0, 4.0, 5.0])
    assert np.allclose(observed, [0.2, 0.3, -0.4])


def test_build_pairwise_dataset_rejects_incompatible_shapes():
    coherence = np.eye(3)
    positions = np.zeros((4, 2))

    with pytest.raises(ValueError):
        build_pairwise_dataset(coherence, positions)


def test_sinc_rss_is_zero_for_exact_model():
    distances = np.linspace(0.01, 1.0, 100)
    k = 8.0

    observed = spherical_sinc(distances, k)

    rss = sinc_rss(k, distances, observed)

    assert np.isclose(rss, 0.0, atol=1e-14)


def test_sinc_rss_increases_away_from_exact_solution():
    distances = np.linspace(0.01, 1.0, 100)
    k_true = 8.0

    observed = spherical_sinc(distances, k_true)

    rss_true = sinc_rss(
        k_true,
        distances,
        observed,
    )
    rss_wrong = sinc_rss(
        12.0,
        distances,
        observed,
    )

    assert rss_true < rss_wrong


def test_theoretical_wavenumber():
    frequency = 500.0
    sound_speed = 343.0

    expected = 2.0 * np.pi * frequency / sound_speed

    assert np.isclose(
        theoretical_wavenumber(frequency, sound_speed),
        expected,
    )


def test_sound_speed_and_wavenumber_are_inverse_operations():
    frequency = 750.0
    sound_speed = 347.0

    k = theoretical_wavenumber(
        frequency,
        sound_speed,
    )

    recovered = sound_speed_from_wavenumber(
        frequency,
        k,
    )

    assert np.isclose(recovered, sound_speed)


def test_estimate_wavenumber_recovers_exact_synthetic_value():
    frequency = 500.0
    sound_speed = 343.0

    k_true = theoretical_wavenumber(
        frequency,
        sound_speed,
    )

    distances = np.linspace(
        0.02,
        1.20,
        200,
    )

    observed = spherical_sinc(
        distances,
        k_true,
    )

    k_estimated, rss = estimate_wavenumber(
        distances,
        observed,
        frequency,
        reference_sound_speed=sound_speed,
    )

    assert np.isclose(
        k_estimated,
        k_true,
        rtol=1e-5,
    )

    assert rss < 1e-10


def test_estimated_wavenumber_recovers_sound_speed():
    frequency = 1000.0
    true_sound_speed = 347.0

    k_true = theoretical_wavenumber(
        frequency,
        true_sound_speed,
    )

    distances = np.linspace(
        0.02,
        1.20,
        200,
    )

    observed = spherical_sinc(
        distances,
        k_true,
    )

    k_estimated, _ = estimate_wavenumber(
        distances,
        observed,
        frequency,
        reference_sound_speed=true_sound_speed,
    )

    c_estimated = sound_speed_from_wavenumber(
        frequency,
        k_estimated,
    )

    assert np.isclose(
        c_estimated,
        true_sound_speed,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("frequency", "sound_speed"),
    [
        (0.0, 343.0),
        (-100.0, 343.0),
        (500.0, 0.0),
        (500.0, -343.0),
    ],
)
def test_theoretical_wavenumber_rejects_nonpositive_inputs(
    frequency,
    sound_speed,
):
    with pytest.raises(ValueError):
        theoretical_wavenumber(
            frequency,
            sound_speed,
        )


def test_estimate_wavenumber_rejects_empty_dataset():
    with pytest.raises(ValueError):
        estimate_wavenumber(
            np.array([]),
            np.array([]),
            frequency=500.0,
        )