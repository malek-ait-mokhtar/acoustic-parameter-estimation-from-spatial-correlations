"""Estimation utilities for acoustic parameter inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from acoustic_estimation.models import spherical_sinc


@dataclass
class FrequencyEstimate:
    """Estimated acoustic parameters and pairwise fit data at one frequency."""

    frequency_hz: float
    wavenumber_rad_m: float
    sound_speed_m_s: float
    rss: float
    coherence_mean: float
    distances_m: NDArray[np.float64]
    observed_coherence: NDArray[np.float64]


@dataclass
class AnalysisResult:
    """Results of a frequency-domain acoustic parameter analysis."""

    estimates: list[FrequencyEstimate]
    sample_rate_hz: float
    n_channels: int
    n_samples: int
    n_snapshots: int
    spectrum_frequencies_hz: NDArray[np.float64]
    mean_spectrum: NDArray[np.float64]


@dataclass
class LocalMinimum:
    """Refined local minimum of an RSS wavenumber landscape."""

    wavenumber_rad_m: float
    rss: float

@dataclass
class CorrectedFrequencyEstimate:
    """Result of the local-minimum correction at one frequency."""

    frequency_hz: float
    baseline_wavenumber_rad_m: float
    corrected_wavenumber_rad_m: float
    baseline_sound_speed_m_s: float
    corrected_sound_speed_m_s: float
    theoretical_wavenumber_rad_m: float
    was_corrected: bool

def theoretical_wavenumber(
    frequency: float,
    sound_speed: float,
) -> float:
    """Return the theoretical acoustic wavenumber.

    Parameters
    ----------
    frequency
        Acoustic frequency in hertz.
    sound_speed
        Sound speed in metres per second.

    Returns
    -------
    float
        Wavenumber in radians per metre.
    """
    if frequency < 0:
        raise ValueError("frequency must be non-negative")

    if sound_speed <= 0:
        raise ValueError("sound_speed must be strictly positive")

    return float(
        2.0
        * np.pi
        * frequency
        / sound_speed
    )


def sound_speed_from_wavenumber(
    frequency: float,
    wavenumber: float,
) -> float:
    """Return sound speed from frequency and wavenumber.

    Parameters
    ----------
    frequency
        Acoustic frequency in hertz.
    wavenumber
        Acoustic wavenumber in radians per metre.

    Returns
    -------
    float
        Sound speed in metres per second.
    """
    if frequency < 0:
        raise ValueError("frequency must be non-negative")

    if wavenumber <= 0:
        raise ValueError("wavenumber must be strictly positive")

    return float(
        2.0
        * np.pi
        * frequency
        / wavenumber
    )


def build_pairwise_dataset(
    coherence: ArrayLike,
    positions: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build pairwise distances and observed real coherence values.

    Parameters
    ----------
    coherence
        Complex coherence matrix with shape ``(n_microphones, n_microphones)``.
    positions
        Microphone coordinates with shape
        ``(n_microphones, spatial_dimension)``.

    Returns
    -------
    distances
        Euclidean distance for each unique microphone pair.
    observed_coherence
        Real part of the measured coherence for each unique pair.
    """
    coherence = np.asarray(
        coherence,
        dtype=np.complex128,
    )

    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    if coherence.ndim != 2:
        raise ValueError("coherence must be a 2D array")

    if coherence.shape[0] != coherence.shape[1]:
        raise ValueError("coherence must be square")

    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array")

    if positions.shape[0] != coherence.shape[0]:
        raise ValueError(
            "positions and coherence must contain "
            "the same number of microphones"
        )

    n_microphones = coherence.shape[0]

    if n_microphones < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    i, j = np.triu_indices(
        n_microphones,
        k=1,
    )

    distances = np.linalg.norm(
        positions[i] - positions[j],
        axis=1,
    )

    observed_coherence = np.real(
        coherence[i, j]
    )

    return (
        distances.astype(np.float64),
        observed_coherence.astype(np.float64),
    )


def sinc_rss(
    wavenumber: float,
    distances: ArrayLike,
    observed_coherence: ArrayLike,
) -> float:
    """Return the RSS between observed coherence and the spherical sinc model.

    Parameters
    ----------
    wavenumber
        Candidate wavenumber in radians per metre.
    distances
        Pairwise microphone distances in metres.
    observed_coherence
        Measured real spatial coherence values.

    Returns
    -------
    float
        Residual sum of squares.
    """
    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    observed_coherence = np.asarray(
        observed_coherence,
        dtype=np.float64,
    )

    if distances.shape != observed_coherence.shape:
        raise ValueError(
            "distances and observed_coherence must have the same shape"
        )

    model = spherical_sinc(
        distances,
        wavenumber,
    )

    residuals = (
        observed_coherence
        - model
    )

    return float(
        np.sum(
            residuals**2
        )
    )


def estimate_wavenumber(
    distances: ArrayLike,
    observed_coherence: ArrayLike,
    frequency: float,
    reference_sound_speed: float = 343.0,
    lower_factor: float = 0.4,
    upper_factor: float = 2.5,
) -> tuple[float, float]:
    """Estimate the wavenumber by bounded RSS minimisation.

    The search interval is defined relative to the theoretical wavenumber

    ``k_ref = 2*pi*frequency/reference_sound_speed``.

    Parameters
    ----------
    distances
        Pairwise microphone distances in metres.
    observed_coherence
        Real part of the measured spatial coherence.
    frequency
        Acoustic frequency in hertz.
    reference_sound_speed
        Reference sound speed used only to construct the search interval.
    lower_factor
        Lower bound as a multiple of ``k_ref``.
    upper_factor
        Upper bound as a multiple of ``k_ref``.

    Returns
    -------
    wavenumber
        Estimated wavenumber in radians per metre.
    rss
        Residual sum of squares at the optimum.
    """
    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    observed_coherence = np.asarray(
        observed_coherence,
        dtype=np.float64,
    )

    if distances.shape != observed_coherence.shape:
        raise ValueError(
            "distances and observed_coherence must have the same shape"
        )

    if distances.ndim != 1:
        raise ValueError(
            "distances and observed_coherence must be one-dimensional"
        )

    if distances.size == 0:
        raise ValueError(
            "at least one microphone pair is required"
        )

    if not np.all(
        np.isfinite(distances)
    ):
        raise ValueError(
            "distances must contain only finite values"
        )

    if not np.all(
        np.isfinite(observed_coherence)
    ):
        raise ValueError(
            "observed_coherence must contain only finite values"
        )

    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if lower_factor <= 0:
        raise ValueError(
            "lower_factor must be strictly positive"
        )

    if upper_factor <= lower_factor:
        raise ValueError(
            "upper_factor must be greater than lower_factor"
        )

    k_reference = theoretical_wavenumber(
        frequency,
        reference_sound_speed,
    )

    result = minimize_scalar(
        sinc_rss,
        args=(
            distances,
            observed_coherence,
        ),
        bounds=(
            lower_factor * k_reference,
            upper_factor * k_reference,
        ),
        method="bounded",
    )

    return (
        float(result.x),
        float(result.fun),
    )


def evaluate_rss_landscape(
    distances: ArrayLike,
    observed_coherence: ArrayLike,
    k_min: float,
    k_max: float,
    n_grid: int = 12000,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Evaluate the sinc RSS objective on a regular wavenumber grid.

    Parameters
    ----------
    distances
        Pairwise microphone distances in metres.
    observed_coherence
        Measured real spatial coherence values.
    k_min
        Lower wavenumber bound in radians per metre.
    k_max
        Upper wavenumber bound in radians per metre.
    n_grid
        Number of regularly spaced grid points.

    Returns
    -------
    k_grid
        Wavenumber grid.
    rss_grid
        RSS objective evaluated at each grid point.
    """
    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    observed_coherence = np.asarray(
        observed_coherence,
        dtype=np.float64,
    )

    if distances.shape != observed_coherence.shape:
        raise ValueError(
            "distances and observed_coherence must have the same shape"
        )

    if distances.ndim != 1:
        raise ValueError(
            "distances and observed_coherence must be one-dimensional"
        )

    if distances.size == 0:
        raise ValueError(
            "at least one microphone pair is required"
        )

    if not np.all(
        np.isfinite(distances)
    ):
        raise ValueError(
            "distances must contain only finite values"
        )

    if not np.all(
        np.isfinite(observed_coherence)
    ):
        raise ValueError(
            "observed_coherence must contain only finite values"
        )

    if k_min <= 0:
        raise ValueError(
            "k_min must be strictly positive"
        )

    if k_max <= k_min:
        raise ValueError(
            "k_max must be greater than k_min"
        )

    if n_grid < 3:
        raise ValueError(
            "n_grid must be at least 3"
        )

    k_grid = np.linspace(
        k_min,
        k_max,
        n_grid,
        dtype=np.float64,
    )

    rss_grid = np.array(
        [
            sinc_rss(
                k,
                distances,
                observed_coherence,
            )
            for k in k_grid
        ],
        dtype=np.float64,
    )

    return (
        k_grid,
        rss_grid,
    )


def detect_local_minima(
    k_grid: ArrayLike,
    rss_grid: ArrayLike,
    relative_prominence: float = 1e-4,
) -> NDArray[np.int64]:
    """Return grid indices corresponding to significant local RSS minima.

    Parameters
    ----------
    k_grid
        Wavenumber grid.
    rss_grid
        RSS objective evaluated on ``k_grid``.
    relative_prominence
        Minimum peak prominence relative to the total RSS range.

    Returns
    -------
    ndarray
        Integer indices of detected local minima.
    """
    k_grid = np.asarray(
        k_grid,
        dtype=np.float64,
    )

    rss_grid = np.asarray(
        rss_grid,
        dtype=np.float64,
    )

    if k_grid.ndim != 1 or rss_grid.ndim != 1:
        raise ValueError(
            "k_grid and rss_grid must be one-dimensional"
        )

    if k_grid.shape != rss_grid.shape:
        raise ValueError(
            "k_grid and rss_grid must have the same shape"
        )

    if k_grid.size < 3:
        raise ValueError(
            "at least three grid points are required"
        )

    if not np.all(
        np.isfinite(k_grid)
    ):
        raise ValueError(
            "k_grid must contain only finite values"
        )

    if not np.all(
        np.isfinite(rss_grid)
    ):
        raise ValueError(
            "rss_grid must contain only finite values"
        )

    if relative_prominence < 0:
        raise ValueError(
            "relative_prominence must be non-negative"
        )

    span = float(
        np.max(rss_grid)
        - np.min(rss_grid)
    )

    prominence = max(
        1e-12,
        relative_prominence * span,
    )

    indices, _ = find_peaks(
        -rss_grid,
        prominence=prominence,
    )

    return indices.astype(
        np.int64
    )


def refine_local_minima(
    k_grid: ArrayLike,
    rss_grid: ArrayLike,
    minimum_indices: ArrayLike,
    distances: ArrayLike,
    observed_coherence: ArrayLike,
) -> list[LocalMinimum]:
    """Refine grid-detected RSS minima with bounded scalar optimisation.

    Each detected minimum is refined inside an interval bounded by the
    midpoints between neighbouring grid-detected minima. This reproduces the
    local-refinement strategy used in the historical RSS analysis.

    Parameters
    ----------
    k_grid
        Wavenumber grid.
    rss_grid
        RSS objective evaluated on ``k_grid``.
    minimum_indices
        Grid indices of detected local minima.
    distances
        Pairwise microphone distances in metres.
    observed_coherence
        Measured real spatial coherence values.

    Returns
    -------
    list of LocalMinimum
        Refined minima sorted by increasing RSS.
    """
    k_grid = np.asarray(
        k_grid,
        dtype=np.float64,
    )

    rss_grid = np.asarray(
        rss_grid,
        dtype=np.float64,
    )

    minimum_indices = np.asarray(
        minimum_indices,
        dtype=np.int64,
    )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    observed_coherence = np.asarray(
        observed_coherence,
        dtype=np.float64,
    )

    if k_grid.ndim != 1 or rss_grid.ndim != 1:
        raise ValueError(
            "k_grid and rss_grid must be one-dimensional"
        )

    if k_grid.shape != rss_grid.shape:
        raise ValueError(
            "k_grid and rss_grid must have the same shape"
        )

    if distances.ndim != 1:
        raise ValueError(
            "distances and observed_coherence must be one-dimensional"
        )

    if distances.shape != observed_coherence.shape:
        raise ValueError(
            "distances and observed_coherence must have the same shape"
        )

    if minimum_indices.ndim != 1:
        raise ValueError(
            "minimum_indices must be one-dimensional"
        )

    if np.any(
        (minimum_indices < 0)
        | (minimum_indices >= k_grid.size)
    ):
        raise ValueError(
            "minimum_indices contains an invalid grid index"
        )

    if minimum_indices.size == 0:
        return []

    minimum_indices = np.sort(
        minimum_indices
    )

    refined: list[LocalMinimum] = []

    for position, index in enumerate(
        minimum_indices
    ):
        if position == 0:
            left = float(
                k_grid[0]
            )
        else:
            previous_index = minimum_indices[
                position - 1
            ]

            left = float(
                0.5
                * (
                    k_grid[previous_index]
                    + k_grid[index]
                )
            )

        if position == minimum_indices.size - 1:
            right = float(
                k_grid[-1]
            )
        else:
            next_index = minimum_indices[
                position + 1
            ]

            right = float(
                0.5
                * (
                    k_grid[index]
                    + k_grid[next_index]
                )
            )

        if right <= left:
            continue

        result = minimize_scalar(
            sinc_rss,
            args=(
                distances,
                observed_coherence,
            ),
            bounds=(
                left,
                right,
            ),
            method="bounded",
            options={
                "xatol": 1e-10,
                "maxiter": 5000,
            },
        )

        refined.append(
            LocalMinimum(
                wavenumber_rad_m=float(
                    result.x
                ),
                rss=float(
                    result.fun
                ),
            )
        )

    refined.sort(
        key=lambda minimum: minimum.rss
    )

    return refined


def closest_minimum_to_wavenumber(
    minima: list[LocalMinimum],
    target_wavenumber: float,
) -> LocalMinimum:
    """Return the local minimum closest to a target wavenumber.

    Parameters
    ----------
    minima
        Refined local minima.
    target_wavenumber
        Target wavenumber in radians per metre.

    Returns
    -------
    LocalMinimum
        Minimum whose wavenumber is closest to the target.
    """
    if target_wavenumber <= 0:
        raise ValueError(
            "target_wavenumber must be strictly positive"
        )

    if not minima:
        raise ValueError(
            "at least one local minimum is required"
        )

    return min(
        minima,
        key=lambda minimum: abs(
            minimum.wavenumber_rad_m
            - target_wavenumber
        ),
    )
    

def correct_wavenumber_with_local_minima(
    frequency: float,
    baseline_wavenumber: float,
    distances: ArrayLike,
    observed_coherence: ArrayLike,
    reference_sound_speed: float = 347.0,
    frequency_switch: float = 1500.0,
    relative_tolerance: float = 0.05,
    n_grid: int = 12000,
) -> CorrectedFrequencyEstimate:
    """Correct a high-frequency estimate using local RSS minima.

    Frequencies at or below ``frequency_switch`` are left unchanged.
    Above the switch, the baseline estimate is retained when its relative
    wavenumber error is within ``relative_tolerance``. Otherwise, the local
    RSS minimum closest to the theoretical wavenumber is selected.

    This reproduces the correction rule used in the historical 24-microphone
    analysis.
    """
    if frequency < 0:
        raise ValueError(
            "frequency must be non-negative"
        )

    if baseline_wavenumber <= 0:
        raise ValueError(
            "baseline_wavenumber must be strictly positive"
        )

    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if frequency_switch < 0:
        raise ValueError(
            "frequency_switch must be non-negative"
        )

    if relative_tolerance < 0:
        raise ValueError(
            "relative_tolerance must be non-negative"
        )

    if n_grid < 3:
        raise ValueError(
            "n_grid must be at least 3"
        )

    theoretical_k = theoretical_wavenumber(
        frequency,
        reference_sound_speed,
    )

    baseline_sound_speed = sound_speed_from_wavenumber(
        frequency,
        baseline_wavenumber,
    )

    # Historical rule: frequencies up to 1500 Hz are never corrected.
    if frequency <= frequency_switch:
        return CorrectedFrequencyEstimate(
            frequency_hz=frequency,
            baseline_wavenumber_rad_m=baseline_wavenumber,
            corrected_wavenumber_rad_m=baseline_wavenumber,
            baseline_sound_speed_m_s=baseline_sound_speed,
            corrected_sound_speed_m_s=baseline_sound_speed,
            theoretical_wavenumber_rad_m=theoretical_k,
            was_corrected=False,
        )

    relative_error = (
        abs(
            baseline_wavenumber
            - theoretical_k
        )
        / theoretical_k
    )

    # Keep estimates already sufficiently close to the physical reference.
    if relative_error <= relative_tolerance:
        return CorrectedFrequencyEstimate(
            frequency_hz=frequency,
            baseline_wavenumber_rad_m=baseline_wavenumber,
            corrected_wavenumber_rad_m=baseline_wavenumber,
            baseline_sound_speed_m_s=baseline_sound_speed,
            corrected_sound_speed_m_s=baseline_sound_speed,
            theoretical_wavenumber_rad_m=theoretical_k,
            was_corrected=False,
        )

    k_min = max(
        1e-6,
        0.05 * theoretical_k,
    )

    k_max = max(
        60.0,
        15.0 * theoretical_k,
    )

    k_grid, rss_grid = evaluate_rss_landscape(
        distances,
        observed_coherence,
        k_min=k_min,
        k_max=k_max,
        n_grid=n_grid,
    )

    minimum_indices = detect_local_minima(
        k_grid,
        rss_grid,
    )

    minima = refine_local_minima(
        k_grid,
        rss_grid,
        minimum_indices,
        distances,
        observed_coherence,
    )

    # Historical behaviour: if no alternative minimum is detected,
    # retain the baseline estimate.
    if not minima:
        return CorrectedFrequencyEstimate(
            frequency_hz=frequency,
            baseline_wavenumber_rad_m=baseline_wavenumber,
            corrected_wavenumber_rad_m=baseline_wavenumber,
            baseline_sound_speed_m_s=baseline_sound_speed,
            corrected_sound_speed_m_s=baseline_sound_speed,
            theoretical_wavenumber_rad_m=theoretical_k,
            was_corrected=False,
        )

    selected = closest_minimum_to_wavenumber(
        minima,
        theoretical_k,
    )

    corrected_wavenumber = (
        selected.wavenumber_rad_m
    )

    corrected_sound_speed = (
        sound_speed_from_wavenumber(
            frequency,
            corrected_wavenumber,
        )
    )

    return CorrectedFrequencyEstimate(
        frequency_hz=frequency,
        baseline_wavenumber_rad_m=baseline_wavenumber,
        corrected_wavenumber_rad_m=corrected_wavenumber,
        baseline_sound_speed_m_s=baseline_sound_speed,
        corrected_sound_speed_m_s=corrected_sound_speed,
        theoretical_wavenumber_rad_m=theoretical_k,
        was_corrected=True,
    )