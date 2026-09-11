"""Microphone-array geometries and pairwise-distance utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# Final 24-microphone 3D array coordinates, measured from the common origin.
# Values are stored in metres.
_ARRAY24_X = np.array([0.15, 0.30, 0.38, 0.48, 0.60, 0.75, 0.90, 1.05])
_ARRAY24_Y = np.array([0.14, 0.28, 0.43, 0.58, 0.73, 0.88, 0.98, 1.08])
_ARRAY24_Z = np.array([0.13, 0.26, 0.37, 0.52, 0.64, 0.79, 0.93, 1.06])


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