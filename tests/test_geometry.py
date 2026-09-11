import numpy as np
import pytest

from acoustic_estimation.geometry import (
    array24_positions,
    distance_multiplicities,
    pairwise_distances,
    uma16_positions,
)

from acoustic_estimation.geometry import (
    distinct_distance_count,
    optimize_array24_geometry,
    orthogonal_array_distances,
    orthogonal_array_positions,
    random_axis_positions,
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
    
    
def test_random_axis_positions_respects_constraints():
    rng = np.random.default_rng(
        42
    )

    positions = random_axis_positions(
        length_cm=120,
        rng=rng,
    )

    assert positions.shape == (8,)

    spacing = np.diff(
        positions
    )

    assert np.all(
        spacing >= 2
    )

    assert np.all(
        spacing <= 15
    )

    assert positions[0] <= 15

    assert (
        120 - positions[-1]
        <= 15
    )
    
def test_orthogonal_array_has_276_pairwise_distances():
    x = np.array(
        [15, 30, 38, 48, 60, 75, 90, 105]
    )

    y = np.array(
        [14, 28, 43, 58, 73, 88, 98, 108]
    )

    z = np.array(
        [13, 26, 37, 52, 64, 79, 93, 106]
    )

    distances = (
        orthogonal_array_distances(
            x,
            y,
            z,
        )
    )

    assert distances.shape == (
        276,
    )
    
def test_historical_array24_geometry_has_expected_distinct_distances():
    x = np.array(
        [15, 30, 38, 48, 60, 75, 90, 105]
    )

    y = np.array(
        [14, 28, 43, 58, 73, 88, 98, 108]
    )

    z = np.array(
        [13, 26, 37, 52, 64, 79, 93, 106]
    )

    count = distinct_distance_count(
        x,
        y,
        z,
    )

    assert count == 239
    
def test_orthogonal_positions_match_array24_geometry():
    x = np.array(
        [15, 30, 38, 48, 60, 75, 90, 105]
    )

    y = np.array(
        [14, 28, 43, 58, 73, 88, 98, 108]
    )

    z = np.array(
        [13, 26, 37, 52, 64, 79, 93, 106]
    )

    positions = orthogonal_array_positions(
        x,
        y,
        z,
        scale=0.01,
    )

    assert np.allclose(
        positions,
        array24_positions(),
    )
    
def test_array24_geometry_search_is_reproducible():
    first = optimize_array24_geometry(
        n_iterations=20,
        seed=1234,
    )

    second = optimize_array24_geometry(
        n_iterations=20,
        seed=1234,
    )

    assert (
        first.distinct_distance_count
        == second.distinct_distance_count
    )

    assert np.array_equal(
        first.x_cm,
        second.x_cm,
    )

    assert np.array_equal(
        first.y_cm,
        second.y_cm,
    )

    assert np.array_equal(
        first.z_cm,
        second.z_cm,
    )