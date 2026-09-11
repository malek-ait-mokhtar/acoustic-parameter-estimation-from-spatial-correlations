import numpy as np
import pytest

from acoustic_estimation.geometry import (
    array24_positions,
    distance_multiplicities,
    pairwise_distances,
    uma16_positions,
)


def test_uma16_shape():
    positions = uma16_positions()
    assert positions.shape == (16, 2)


def test_uma16_default_spacing():
    positions = uma16_positions()

    assert np.isclose(np.linalg.norm(positions[1] - positions[0]), 0.04)
    assert np.isclose(np.linalg.norm(positions[4] - positions[0]), 0.04)


def test_uma16_extent():
    positions = uma16_positions()

    assert np.isclose(positions[:, 0].min(), 0.0)
    assert np.isclose(positions[:, 0].max(), 0.12)
    assert np.isclose(positions[:, 1].min(), 0.0)
    assert np.isclose(positions[:, 1].max(), 0.12)


def test_uma16_rejects_nonpositive_spacing():
    with pytest.raises(ValueError):
        uma16_positions(spacing=0.0)

    with pytest.raises(ValueError):
        uma16_positions(spacing=-0.04)


def test_array24_shape():
    positions = array24_positions()
    assert positions.shape == (24, 3)


def test_array24_branch_ordering():
    positions = array24_positions()

    assert np.allclose(positions[:8, 1:], 0.0)
    assert np.allclose(positions[8:16, [0, 2]], 0.0)
    assert np.allclose(positions[16:, :2], 0.0)


def test_array24_known_endpoints():
    positions = array24_positions()

    assert np.allclose(positions[0], [0.15, 0.0, 0.0])
    assert np.allclose(positions[7], [1.05, 0.0, 0.0])
    assert np.allclose(positions[8], [0.0, 0.14, 0.0])
    assert np.allclose(positions[15], [0.0, 1.08, 0.0])
    assert np.allclose(positions[16], [0.0, 0.0, 0.13])
    assert np.allclose(positions[23], [0.0, 0.0, 1.06])


@pytest.mark.parametrize(
    ("positions", "expected_pairs"),
    [
        (uma16_positions(), 120),
        (array24_positions(), 276),
    ],
)
def test_pairwise_distance_count(positions, expected_pairs):
    distances = pairwise_distances(positions)
    assert distances.shape == (expected_pairs,)


def test_pairwise_distances_simple_geometry():
    positions = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ]
    )

    distances = pairwise_distances(positions)

    assert np.allclose(distances, [3.0, 4.0, 5.0])


def test_distance_multiplicities_account_for_all_pairs():
    positions = uma16_positions()

    _, counts = distance_multiplicities(positions)

    assert counts.sum() == 120