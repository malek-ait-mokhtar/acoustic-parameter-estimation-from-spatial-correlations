import numpy as np
import pytest

from scripts.analyze_uma16 import select_frequency_bins


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