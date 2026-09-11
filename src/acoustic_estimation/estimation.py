"""Wavenumber and sound-speed estimation from spatial coherence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

from acoustic_estimation.models import spherical_sinc


@dataclass
class FrequencyEstimate:
    """Estimated acoustic parameters and pairwise fit data at one frequency."""

    frequency_hz: float
    wavenumber_rad_m: float
    sound_speed_m_s: float
    rss: float
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

def build_pairwise_dataset(
    coherence: ArrayLike,
    positions: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the distance/coherence dataset used for model fitting.

    For every unique microphone pair ``(i, j)`` with ``i < j``, this function
    extracts

    - the Euclidean microphone separation ``r_ij``;
    - the real part of the measured spatial coherence ``Gamma_ij``.

    Parameters
    ----------
    coherence
        Complex coherence matrix with shape ``(n_microphones, n_microphones)``.
    positions
        Microphone coordinates with shape ``(n_microphones, dimension)``.

    Returns
    -------
    distances
        Pairwise microphone distances in metres.
    observed_coherence
        Real part of the measured coherence for the corresponding pairs.
    """
    coherence = np.asarray(coherence, dtype=np.complex128)
    positions = np.asarray(positions, dtype=np.float64)

    if coherence.ndim != 2 or coherence.shape[0] != coherence.shape[1]:
        raise ValueError("coherence must be a square 2D matrix")

    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array")

    if coherence.shape[0] != positions.shape[0]:
        raise ValueError(
            "coherence and positions must contain the same number "
            "of microphones"
        )

    n_microphones = positions.shape[0]

    if n_microphones < 2:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    i, j = np.triu_indices(n_microphones, k=1)

    distances = np.linalg.norm(
        positions[i] - positions[j],
        axis=1,
    )

    observed_coherence = np.real(
        coherence[i, j]
    ).astype(np.float64)

    return distances, observed_coherence


def sinc_rss(
    wavenumber: float,
    distances: ArrayLike,
    observed_coherence: ArrayLike,
) -> float:
    r"""Return the residual sum of squares for the spherical sinc model.

    The objective function is

    .. math::

        RSS(k)
        =
        \sum_{i<j}
        \left[
        \operatorname{Re}(\Gamma_{ij})
        -
        \frac{\sin(k r_{ij})}{k r_{ij}}
        \right]^2.

    Parameters
    ----------
    wavenumber
        Candidate acoustic wavenumber in radians per metre.
    distances
        Pairwise microphone distances in metres.
    observed_coherence
        Observed real spatial coherence values.

    Returns
    -------
    float
        Residual sum of squares.
    """
    distances = np.asarray(distances, dtype=np.float64)
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

    model = spherical_sinc(
        distances,
        wavenumber,
    )

    residuals = observed_coherence - model

    return float(np.sum(residuals**2))


def theoretical_wavenumber(
    frequency: float,
    sound_speed: float,
) -> float:
    r"""Return the acoustic wavenumber ``k = 2*pi*f/c``."""
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive")

    if sound_speed <= 0:
        raise ValueError("sound_speed must be strictly positive")

    return float(2.0 * np.pi * frequency / sound_speed)


def sound_speed_from_wavenumber(
    frequency: float,
    wavenumber: float,
) -> float:
    r"""Return the sound speed ``c = 2*pi*f/k``."""
    if frequency <= 0:
        raise ValueError("frequency must be strictly positive")

    if wavenumber <= 0:
        raise ValueError("wavenumber must be strictly positive")

    return float(2.0 * np.pi * frequency / wavenumber)


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
    distances = np.asarray(distances, dtype=np.float64)
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
        raise ValueError("at least one microphone pair is required")

    if not np.all(np.isfinite(distances)):
        raise ValueError("distances must contain only finite values")

    if not np.all(np.isfinite(observed_coherence)):
        raise ValueError(
            "observed_coherence must contain only finite values"
        )

    if reference_sound_speed <= 0:
        raise ValueError(
            "reference_sound_speed must be strictly positive"
        )

    if lower_factor <= 0:
        raise ValueError("lower_factor must be strictly positive")

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
        args=(distances, observed_coherence),
        bounds=(
            lower_factor * k_reference,
            upper_factor * k_reference,
        ),
        method="bounded",
    )

    return float(result.x), float(result.fun)