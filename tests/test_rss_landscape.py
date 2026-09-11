import numpy as np
import pytest

from scripts.analyze_rss_landscape import (
    closest_frequency_index,
)


def test_closest_frequency_index():
    frequencies = np.array(
        [100.0, 200.0, 300.0]
    )

    index = closest_frequency_index(
        frequencies,
        220.0,
    )

    assert index == 1


def test_closest_frequency_index_rejects_empty_array():
    with pytest.raises(ValueError):
        closest_frequency_index(
            np.array([]),
            500.0,
        )