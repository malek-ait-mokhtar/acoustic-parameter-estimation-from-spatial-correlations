import numpy as np

from acoustic_estimation.models import (
    plane_wave_coherence,
    spherical_sinc,
)


def test_spherical_sinc_at_zero():
    result = spherical_sinc(np.array([0.0]), wavenumber=10.0)

    assert np.allclose(result, [1.0])


def test_spherical_sinc_matches_analytical_formula():
    distances = np.array([0.02, 0.05, 0.10])
    k = 7.5

    expected = np.sin(k * distances) / (k * distances)

    assert np.allclose(spherical_sinc(distances, k), expected)


def test_spherical_sinc_is_even_in_distance():
    distances = np.array([0.02, 0.05, 0.10])
    k = 12.0

    assert np.allclose(
        spherical_sinc(distances, k),
        spherical_sinc(-distances, k),
    )


def test_spherical_sinc_is_even_in_wavenumber():
    distances = np.array([0.02, 0.05, 0.10])

    assert np.allclose(
        spherical_sinc(distances, 8.0),
        spherical_sinc(distances, -8.0),
    )


def test_plane_wave_coherence_at_zero_displacement():
    result = plane_wave_coherence(np.array([0.0]), wavenumber=10.0)

    assert np.allclose(result, [1.0 + 0.0j])


def test_plane_wave_coherence_has_unit_magnitude():
    displacements = np.linspace(-1.0, 1.0, 20)

    coherence = plane_wave_coherence(
        displacements,
        wavenumber=5.0,
    )

    assert np.allclose(np.abs(coherence), 1.0)


def test_plane_wave_coherence_matches_analytical_formula():
    displacements = np.array([-0.4, 0.0, 0.3])
    k = 4.5

    expected = np.exp(-1j * k * displacements)

    assert np.allclose(
        plane_wave_coherence(displacements, k),
        expected,
    )


def test_plane_wave_coherence_conjugate_symmetry():
    displacements = np.array([0.1, 0.4, 0.8])
    k = 6.0

    positive = plane_wave_coherence(displacements, k)
    negative = plane_wave_coherence(-displacements, k)

    assert np.allclose(negative, np.conj(positive))