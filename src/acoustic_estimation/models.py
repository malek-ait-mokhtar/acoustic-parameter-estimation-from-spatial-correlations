"""Theoretical spatial-coherence models used for acoustic estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def spherical_sinc(
    distance: ArrayLike,
    wavenumber: float,
) -> NDArray[np.float64]:
    r"""Evaluate the diffuse-field spatial coherence model.

    The theoretical coherence between two omnidirectional microphones
    separated by a distance ``r`` in a three-dimensional diffuse field is

    .. math::

        \Gamma(r; k) = \frac{\sin(kr)}{kr},

    where ``k`` is the acoustic wavenumber.

    Parameters
    ----------
    distance
        Microphone separation distance(s), in metres.
    wavenumber
        Acoustic wavenumber, in radians per metre.

    Returns
    -------
    ndarray
        Model coherence evaluated at each input distance.
    """
    distance = np.asarray(distance, dtype=np.float64)
    kr = wavenumber * distance

    return np.sinc(kr / np.pi)


def plane_wave_coherence(
    displacement: ArrayLike,
    wavenumber: float,
) -> NDArray[np.complex128]:
    r"""Evaluate the coherence model for a plane wave along one axis.

    The model is

    .. math::

        \Gamma(\Delta x; k) = \exp(-i k \Delta x).

    Parameters
    ----------
    displacement
        Signed microphone separation(s) along the propagation axis, in metres.
    wavenumber
        Acoustic wavenumber, in radians per metre.

    Returns
    -------
    ndarray
        Complex spatial coherence for each displacement.
    """
    displacement = np.asarray(displacement, dtype=np.float64)

    return np.asarray(
        np.exp(-1j * wavenumber * displacement),
        dtype=np.complex128,
    )