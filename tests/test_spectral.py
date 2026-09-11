import numpy as np
import pytest

from acoustic_estimation.spectral import (
    average_spectrum,
    coherence_matrix,
    cross_spectral_matrices,
    select_frequency_bins,
)


def test_average_spectrum_detects_sinusoid():
    sample_rate = 1000.0
    frequency = 125.0

    t = np.arange(1000) / sample_rate
    signal = np.sin(2.0 * np.pi * frequency * t)

    signals = np.vstack([signal, signal])

    frequencies, spectrum = average_spectrum(
        signals,
        sample_rate,
        nperseg=1000,
    )

    peak_frequency = frequencies[np.argmax(spectrum)]

    assert np.isclose(peak_frequency, frequency)


def test_average_spectrum_output_shape():
    rng = np.random.default_rng(0)
    signals = rng.normal(size=(4, 1000))

    frequencies, spectrum = average_spectrum(
        signals,
        sample_rate=1000.0,
        nperseg=512,
    )

    assert frequencies.shape == spectrum.shape
    assert len(frequencies) == 512 // 2 + 1


def test_cross_spectral_matrix_shape():
    rng = np.random.default_rng(0)
    signals = rng.normal(size=(3, 2048))

    frequencies, csms, n_snapshots = cross_spectral_matrices(
        signals,
        sample_rate=1000.0,
        nperseg=256,
        overlap=0.5,
    )

    assert csms.shape == (
        len(frequencies),
        3,
        3,
    )

    assert n_snapshots > 0


def test_cross_spectral_matrices_are_hermitian():
    rng = np.random.default_rng(1)
    signals = rng.normal(size=(4, 2048))

    _, csms, _ = cross_spectral_matrices(
        signals,
        sample_rate=1000.0,
        nperseg=256,
    )

    assert np.allclose(
        csms,
        np.swapaxes(np.conj(csms), 1, 2),
    )


def test_cross_spectral_diagonal_is_nonnegative():
    rng = np.random.default_rng(2)
    signals = rng.normal(size=(3, 2048))

    _, csms, _ = cross_spectral_matrices(
        signals,
        sample_rate=1000.0,
        nperseg=256,
    )

    diagonal = np.real(
        np.diagonal(csms, axis1=1, axis2=2)
    )

    assert np.all(diagonal >= 0.0)


def test_identical_channels_have_unit_coherence():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=2048)

    signals = np.vstack(
        [
            signal,
            signal,
        ]
    )

    _, csms, _ = cross_spectral_matrices(
        signals,
        sample_rate=1000.0,
        nperseg=256,
    )

    gamma = coherence_matrix(csms[20])

    assert np.allclose(
        np.abs(gamma),
        np.ones((2, 2)),
    )


def test_coherence_diagonal_is_one_for_positive_power():
    csm = np.array(
        [
            [4.0, 1.0 + 1.0j],
            [1.0 - 1.0j, 9.0],
        ],
        dtype=np.complex128,
    )

    gamma = coherence_matrix(csm)

    assert np.allclose(
        np.diag(gamma),
        [1.0, 1.0],
    )


def test_coherence_preserves_hermitian_symmetry():
    csm = np.array(
        [
            [4.0, 1.0 + 2.0j],
            [1.0 - 2.0j, 9.0],
        ],
        dtype=np.complex128,
    )

    gamma = coherence_matrix(csm)

    assert np.allclose(
        gamma,
        gamma.conj().T,
    )


def test_cross_spectral_rejects_too_short_signal():
    signals = np.zeros((4, 100))

    with pytest.raises(ValueError):
        cross_spectral_matrices(
            signals,
            sample_rate=1000.0,
            nperseg=256,
        )


@pytest.mark.parametrize(
    "overlap",
    [-0.1, 1.0, 1.5],
)
def test_cross_spectral_rejects_invalid_overlap(overlap):
    signals = np.zeros((4, 1000))

    with pytest.raises(ValueError):
        cross_spectral_matrices(
            signals,
            sample_rate=1000.0,
            nperseg=256,
            overlap=overlap,
        )


def test_select_frequency_bins():
    spectrum_frequencies = np.array(
        [0.0, 100.0, 200.0, 300.0, 400.0]
    )

    mean_spectrum = np.array(
        [0.0, 1.0, 10.0, 2.0, 0.0]
    )

    csm_frequencies = spectrum_frequencies.copy()

    indices = select_frequency_bins(
        spectrum_frequencies,
        mean_spectrum,
        csm_frequencies,
        min_frequency=50.0,
        max_frequency=350.0,
        relative_threshold=0.15,
    )

    assert np.array_equal(
        indices,
        np.array([2, 3]),
    )


def test_select_frequency_bins_respects_frequency_range():
    frequencies = np.array(
        [0.0, 100.0, 200.0, 300.0, 400.0]
    )

    spectrum = np.ones(5)

    indices = select_frequency_bins(
        frequencies,
        spectrum,
        frequencies,
        min_frequency=150.0,
        max_frequency=350.0,
        relative_threshold=0.0,
    )

    assert np.array_equal(
        indices,
        np.array([2, 3]),
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_select_frequency_bins_rejects_invalid_threshold(
    threshold,
):
    frequencies = np.arange(5, dtype=float)

    with pytest.raises(ValueError):
        select_frequency_bins(
            frequencies,
            np.ones(5),
            frequencies,
            relative_threshold=threshold,
        )