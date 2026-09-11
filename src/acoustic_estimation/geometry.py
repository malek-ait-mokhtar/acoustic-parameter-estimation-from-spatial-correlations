"""Microphone-array geometries and pairwise-distance utilities."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Final 24-microphone 3D array coordinates, measured from the common origin.
# Values are stored in metres.
_ARRAY24_X = np.array([0.15, 0.30, 0.38, 0.48, 0.60, 0.75, 0.90, 1.05])
_ARRAY24_Y = np.array([0.14, 0.28, 0.43, 0.58, 0.73, 0.88, 0.98, 1.08])
_ARRAY24_Z = np.array([0.13, 0.26, 0.37, 0.52, 0.64, 0.79, 0.93, 1.06])

@dataclass(frozen=True)
class Array24GeometrySearchResult:
    """Best geometry found by the random array24 search."""

    x_cm: NDArray[np.int64]
    y_cm: NDArray[np.int64]
    z_cm: NDArray[np.int64]
    distinct_distance_count: int
    iterations_completed: int


def uma16_positions(spacing: float = 0.04) -> NDArray[np.float64]:
    """Return the positions of the 4x4 UMA16 microphone array.

    The microphones form a square planar grid with uniform spacing.

    Parameters
    ----------
    spacing
        Distance between adjacent microphones, in metres.

    Returns
    -------
    ndarray, shape (16, 2)
        Microphone coordinates in metres, ordered row by row.
    """
    if spacing <= 0:
        raise ValueError("spacing must be strictly positive")

    return np.array(
        [[col * spacing, row * spacing] for row in range(4) for col in range(4)],
        dtype=np.float64,
    )


def array24_positions() -> NDArray[np.float64]:
    """Return the positions of the 24-microphone 3D array.

    The array consists of three orthogonal branches originating from a common
    reference point. Eight microphones are placed along each Cartesian axis.

    Returns
    -------
    ndarray, shape (24, 3)
        Microphone coordinates in metres. Microphones 0-7 lie on the x-axis,
        8-15 on the y-axis, and 16-23 on the z-axis.
    """
    x_branch = np.column_stack(
        (_ARRAY24_X, np.zeros_like(_ARRAY24_X), np.zeros_like(_ARRAY24_X))
    )
    y_branch = np.column_stack(
        (np.zeros_like(_ARRAY24_Y), _ARRAY24_Y, np.zeros_like(_ARRAY24_Y))
    )
    z_branch = np.column_stack(
        (np.zeros_like(_ARRAY24_Z), np.zeros_like(_ARRAY24_Z), _ARRAY24_Z)
    )

    return np.vstack((x_branch, y_branch, z_branch)).astype(np.float64)


def pairwise_distances(
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return Euclidean distances between all unique microphone pairs.

    Parameters
    ----------
    positions
        Array of microphone coordinates with shape (n_microphones, dimension).

    Returns
    -------
    ndarray, shape (n_microphones * (n_microphones - 1) / 2,)
        Distances for pairs (i, j) with i < j.
    """
    positions = np.asarray(positions, dtype=np.float64)

    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array")

    n_microphones = positions.shape[0]

    if n_microphones < 2:
        return np.empty(0, dtype=np.float64)

    i, j = np.triu_indices(n_microphones, k=1)
    return np.linalg.norm(positions[i] - positions[j], axis=1)


def distance_multiplicities(
    positions: NDArray[np.float64],
    decimals: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return distinct pairwise distances and their multiplicities.

    Distances are rounded before grouping to avoid floating-point differences
    between geometrically identical distances.

    Parameters
    ----------
    positions
        Array of microphone coordinates.
    decimals
        Number of decimal places used when grouping distances.

    Returns
    -------
    unique_distances
        Sorted distinct distances.
    counts
        Number of microphone pairs associated with each distance.
    """
    distances = pairwise_distances(positions)

    unique_distances, counts = np.unique(
        np.round(distances, decimals=decimals),
        return_counts=True,
    )

    return unique_distances, counts


def random_axis_positions(
    length_cm: int,
    rng: np.random.Generator,
    n_microphones: int = 8,
    min_spacing_cm: int = 2,
    max_spacing_cm: int = 15,
) -> NDArray[np.int64]:
    """Generate one valid microphone branch for the historical array24 search.

    Positions are integer centimetres measured from the common origin.
    Adjacent microphones are separated by between ``min_spacing_cm`` and
    ``max_spacing_cm``. The first and last microphones must also lie within
    ``max_spacing_cm`` of the corresponding ends of the branch.
    """
    if length_cm <= 0:
        raise ValueError(
            "length_cm must be strictly positive"
        )

    if n_microphones < 2:
        raise ValueError(
            "n_microphones must be at least 2"
        )

    if min_spacing_cm <= 0:
        raise ValueError(
            "min_spacing_cm must be strictly positive"
        )

    if max_spacing_cm < min_spacing_cm:
        raise ValueError(
            "max_spacing_cm must be greater than or equal to "
            "min_spacing_cm"
        )

    minimum_required_length = (
        (n_microphones - 1)
        * min_spacing_cm
    )

    if (
        minimum_required_length
        > length_cm
    ):
        raise ValueError(
            "branch is too short for the requested microphone spacing"
        )

    while True:
        gaps = rng.integers(
            min_spacing_cm,
            max_spacing_cm + 1,
            size=n_microphones - 1,
        )

        span = int(
            np.sum(gaps)
        )

        min_start = max(
            0,
            length_cm
            - span
            - max_spacing_cm,
        )

        max_start = min(
            max_spacing_cm,
            length_cm - span,
        )

        if min_start > max_start:
            continue

        start = int(
            rng.integers(
                min_start,
                max_start + 1,
            )
        )

        positions = np.empty(
            n_microphones,
            dtype=np.int64,
        )

        positions[0] = start
        positions[1:] = (
            start
            + np.cumsum(
                gaps,
                dtype=np.int64,
            )
        )

        spacing = np.diff(
            positions
        )

        if np.any(
            spacing < min_spacing_cm
        ):
            continue

        if np.any(
            spacing > max_spacing_cm
        ):
            continue

        if (
            positions[0]
            > max_spacing_cm
        ):
            continue

        if (
            length_cm
            - positions[-1]
            > max_spacing_cm
        ):
            continue

        return positions
    
def orthogonal_array_positions(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Build Cartesian coordinates for three orthogonal microphone branches.

    Parameters
    ----------
    x, y, z
        Coordinates along the three branches.
    scale
        Multiplicative conversion factor applied to the coordinates.
        For centimetres to metres, use ``scale=0.01``.
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    z = np.asarray(
        z,
        dtype=np.float64,
    )

    if (
        x.ndim != 1
        or y.ndim != 1
        or z.ndim != 1
    ):
        raise ValueError(
            "x, y and z must be one-dimensional"
        )

    if scale <= 0:
        raise ValueError(
            "scale must be strictly positive"
        )

    x_branch = np.column_stack(
        (
            x,
            np.zeros_like(x),
            np.zeros_like(x),
        )
    )

    y_branch = np.column_stack(
        (
            np.zeros_like(y),
            y,
            np.zeros_like(y),
        )
    )

    z_branch = np.column_stack(
        (
            np.zeros_like(z),
            np.zeros_like(z),
            z,
        )
    )

    return (
        scale
        * np.vstack(
            (
                x_branch,
                y_branch,
                z_branch,
            )
        )
    ).astype(
        np.float64
    )
    
def orthogonal_array_distances(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
) -> NDArray[np.float64]:
    """Return all pairwise distances of a three-branch orthogonal array."""
    positions = orthogonal_array_positions(
        x,
        y,
        z,
    )

    return pairwise_distances(
        positions
    )
    
def distinct_distance_count(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    decimals: int = 10,
) -> int:
    """Return the number of distinct pairwise distances."""
    if decimals < 0:
        raise ValueError(
            "decimals must be non-negative"
        )

    distances = (
        orthogonal_array_distances(
            x,
            y,
            z,
        )
    )

    unique_distances = np.unique(
        np.round(
            distances,
            decimals,
        )
    )

    return int(
        unique_distances.size
    )
    
def optimize_array24_geometry(
    n_iterations: int = 10_000,
    seed: int | None = None,
    length_x_cm: int = 120,
    length_y_cm: int = 120,
    length_z_cm: int = 120,
    n_microphones_per_axis: int = 8,
    min_spacing_cm: int = 2,
    max_spacing_cm: int = 15,
    decimals: int = 10,
) -> Array24GeometrySearchResult:
    """Search randomly for a 24-microphone geometry with diverse distances.

    The objective is the number of distinct pairwise distances among the
    24 microphones. The theoretical maximum is C(24, 2) = 276.
    """
    if n_iterations <= 0:
        raise ValueError(
            "n_iterations must be strictly positive"
        )

    rng = np.random.default_rng(
        seed
    )

    best_score = -1
    best_x = None
    best_y = None
    best_z = None

    total_microphones = (
        3
        * n_microphones_per_axis
    )

    theoretical_maximum = (
        total_microphones
        * (total_microphones - 1)
        // 2
    )

    iterations_completed = 0

    for iteration in range(
        1,
        n_iterations + 1,
    ):
        x = random_axis_positions(
            length_cm=length_x_cm,
            rng=rng,
            n_microphones=n_microphones_per_axis,
            min_spacing_cm=min_spacing_cm,
            max_spacing_cm=max_spacing_cm,
        )

        y = random_axis_positions(
            length_cm=length_y_cm,
            rng=rng,
            n_microphones=n_microphones_per_axis,
            min_spacing_cm=min_spacing_cm,
            max_spacing_cm=max_spacing_cm,
        )

        z = random_axis_positions(
            length_cm=length_z_cm,
            rng=rng,
            n_microphones=n_microphones_per_axis,
            min_spacing_cm=min_spacing_cm,
            max_spacing_cm=max_spacing_cm,
        )

        current_score = (
            distinct_distance_count(
                x,
                y,
                z,
                decimals=decimals,
            )
        )

        iterations_completed = (
            iteration
        )

        if current_score > best_score:
            best_score = (
                current_score
            )

            best_x = x.copy()
            best_y = y.copy()
            best_z = z.copy()

            if (
                best_score
                == theoretical_maximum
            ):
                break

    if (
        best_x is None
        or best_y is None
        or best_z is None
    ):
        raise RuntimeError(
            "geometry search produced no candidate"
        )

    return Array24GeometrySearchResult(
        x_cm=best_x,
        y_cm=best_y,
        z_cm=best_z,
        distinct_distance_count=best_score,
        iterations_completed=iterations_completed,
    )