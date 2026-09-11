"""Gaussian-process reconstruction of diffuse acoustic fields.

This module contains the numerical components used by the historical
UMA16 Gaussian-process reconstruction experiment.

The historical experiment uses a diffuse-field sinc covariance kernel,
complex Fourier coefficients as observations, and a small diagonal
regularization term for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window


@dataclass(frozen=True)
class GPLeaveOneOutResult:
    """Results of Gaussian-process leave-one-out validation."""

    true_values: NDArray[np.complex128]
    predicted_values: NDArray[np.complex128]
    absolute_errors: NDArray[np.float64]

    rmse: float
    nrmse: float
    correlation: float
    mae: float


@dataclass(frozen=True)
class GPReconstructionResult:
    """Gaussian-process reconstruction on a regular two-dimensional grid."""

    used_frequency_hz: float
    wavenumber_rad_m: float

    microphone_positions_m: NDArray[np.float64]
    measured_pressures: NDArray[np.complex128]

    x_coordinates_m: NDArray[np.float64]
    y_coordinates_m: NDArray[np.float64]

    mean_field: NDArray[np.complex128]
    predictive_variance: NDArray[np.float64]


def extract_complex_pressure(
    signals: ArrayLike,
    sample_rate: float,
    target_frequency: float,
    nperseg: int = 16384,
) -> tuple[
    NDArray[np.complex128],
    float,
]:
    """Extract the complex Fourier coefficient nearest a target frequency.

    This intentionally reproduces the historical GP experiment:

    - only the first ``nperseg`` samples are used;
    - the temporal mean is removed independently from each microphone;
    - a Hann window is applied;
    - the raw real FFT coefficient is retained without amplitude
      normalization;
    - the FFT bin nearest ``target_frequency`` is selected.
    """
    signals = np.asarray(
        signals,
        dtype=np.float64,
    )

    if signals.ndim != 2:
        raise ValueError(
            "signals must have shape (n_microphones, n_samples)"
        )

    if signals.shape[0] == 0:
        raise ValueError(
            "at least one microphone signal is required"
        )

    if signals.shape[1] == 0:
        raise ValueError(
            "signals must contain at least one sample"
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be strictly positive"
        )

    if target_frequency < 0:
        raise ValueError(
            "target_frequency must be non-negative"
        )

    if nperseg <= 0:
        raise ValueError(
            "nperseg must be strictly positive"
        )

    n_samples = signals.shape[1]

    n = min(
        nperseg,
        n_samples,
    )

    window = get_window(
        "hann",
        n,
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate,
    )

    frequency_index = int(
        np.argmin(
            np.abs(
                frequencies
                - target_frequency
            )
        )
    )

    used_frequency = float(
        frequencies[
            frequency_index
        ]
    )

    pressures = np.empty(
        signals.shape[0],
        dtype=np.complex128,
    )

    for microphone_index in range(
        signals.shape[0]
    ):
        signal = signals[
            microphone_index,
            :n,
        ]

        signal = (
            signal
            - np.mean(signal)
        )

        spectrum = np.fft.rfft(
            signal * window
        )

        pressures[
            microphone_index
        ] = spectrum[
            frequency_index
        ]

    return (
        pressures,
        used_frequency,
    )


def sinc_kernel(
    distance: ArrayLike,
    wavenumber: float,
) -> NDArray[np.float64]:
    """Evaluate the isotropic diffuse-field sinc covariance kernel."""
    distance = np.asarray(
        distance,
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(distance)
    ):
        raise ValueError(
            "distance must contain only finite values"
        )

    if np.any(
        distance < 0
    ):
        raise ValueError(
            "distance must be non-negative"
        )

    if not np.isfinite(
        wavenumber
    ):
        raise ValueError(
            "wavenumber must be finite"
        )

    if wavenumber < 0:
        raise ValueError(
            "wavenumber must be non-negative"
        )

    kr = (
        wavenumber
        * distance
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        values = np.where(
            np.abs(kr) < 1e-12,
            1.0,
            np.sin(kr) / kr,
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def build_kernel_matrix(
    positions: ArrayLike,
    wavenumber: float,
) -> NDArray[np.float64]:
    """Build the covariance matrix between spatial positions."""
    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    if positions.ndim != 2:
        raise ValueError(
            "positions must be a 2D array"
        )

    if positions.shape[0] == 0:
        raise ValueError(
            "at least one position is required"
        )

    if not np.all(
        np.isfinite(positions)
    ):
        raise ValueError(
            "positions must contain only finite values"
        )

    differences = (
        positions[:, None, :]
        - positions[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    return sinc_kernel(
        distances,
        wavenumber,
    )



def gp_predict(
    microphone_positions: ArrayLike,
    pressures: ArrayLike,
    prediction_points: ArrayLike,
    wavenumber: float,
    regularization: float = 1e-6,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
]:
    """Predict complex pressure and variance using batched GP algebra."""
    microphone_positions = np.asarray(
        microphone_positions,
        dtype=np.float64,
    )

    pressures = np.asarray(
        pressures,
        dtype=np.complex128,
    )

    prediction_points = np.asarray(
        prediction_points,
        dtype=np.float64,
    )

    if microphone_positions.ndim != 2:
        raise ValueError(
            "microphone_positions must be a 2D array"
        )

    if prediction_points.ndim != 2:
        raise ValueError(
            "prediction_points must be a 2D array"
        )

    if pressures.ndim != 1:
        raise ValueError(
            "pressures must be one-dimensional"
        )

    if microphone_positions.shape[0] == 0:
        raise ValueError(
            "at least one microphone is required"
        )

    if (
        microphone_positions.shape[0]
        != pressures.size
    ):
        raise ValueError(
            "one pressure value is required per microphone"
        )

    if (
        microphone_positions.shape[1]
        != prediction_points.shape[1]
    ):
        raise ValueError(
            "microphone positions and prediction points "
            "must have the same dimension"
        )

    if not np.all(
        np.isfinite(microphone_positions)
    ):
        raise ValueError(
            "microphone_positions must contain only finite values"
        )

    if not np.all(
        np.isfinite(pressures)
    ):
        raise ValueError(
            "pressures must contain only finite values"
        )

    if not np.all(
        np.isfinite(prediction_points)
    ):
        raise ValueError(
            "prediction_points must contain only finite values"
        )

    if regularization <= 0:
        raise ValueError(
            "regularization must be strictly positive"
        )

    # Training covariance K.
    kernel = build_kernel_matrix(
        microphone_positions,
        wavenumber,
    ).astype(
        np.complex128
    )

    kernel += (
        regularization
        * np.eye(
            kernel.shape[0],
            dtype=np.complex128,
        )
    )

    # alpha = K^-1 P
    alpha = np.linalg.solve(
        kernel,
        pressures,
    )

    # Distances from every prediction point to every microphone.
    differences = (
        prediction_points[:, None, :]
        - microphone_positions[None, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    # Shape:
    # (n_prediction_points, n_microphones)
    cross_kernel = sinc_kernel(
        distances,
        wavenumber,
    ).astype(
        np.complex128
    )

    # mu_* = k_* alpha
    means = (
        cross_kernel
        @ alpha
    )

    # Solve all K^-1 k_* vectors simultaneously.
    #
    # cross_kernel.T has shape
    # (n_microphones, n_prediction_points).
    solutions = np.linalg.solve(
        kernel,
        cross_kernel.T,
    )

    # sigma² = 1 - k_* K^-1 k_*
    predictive_variance_complex = (
        1.0
        - np.sum(
            cross_kernel.T
            * solutions,
            axis=0,
        )
    )

    predictive_variance = np.real_if_close(
        predictive_variance_complex
    )

    return (
        np.asarray(
            means,
            dtype=np.complex128,
        ),
        np.asarray(
            predictive_variance,
            dtype=np.float64,
        ),
    )
    
    
def leave_one_out_gp(
    microphone_positions: ArrayLike,
    pressures: ArrayLike,
    wavenumber: float,
    regularization: float = 1e-6,
) -> GPLeaveOneOutResult:
    """Perform leave-one-out validation of the GP reconstruction."""
    microphone_positions = np.asarray(
        microphone_positions,
        dtype=np.float64,
    )

    pressures = np.asarray(
        pressures,
        dtype=np.complex128,
    )

    if microphone_positions.ndim != 2:
        raise ValueError(
            "microphone_positions must be a 2D array"
        )

    if pressures.ndim != 1:
        raise ValueError(
            "pressures must be one-dimensional"
        )

    if (
        microphone_positions.shape[0]
        != pressures.size
    ):
        raise ValueError(
            "one pressure value is required per microphone"
        )

    if pressures.size < 2:
        raise ValueError(
            "leave-one-out validation requires at least two microphones"
        )

    predicted_values = np.empty(
        pressures.shape,
        dtype=np.complex128,
    )

    for left_out in range(
        pressures.size
    ):
        mask = np.ones(
            pressures.size,
            dtype=bool,
        )

        mask[left_out] = False

        prediction, _ = gp_predict(
            microphone_positions[
                mask
            ],
            pressures[
                mask
            ],
            microphone_positions[
                left_out
            ][None, :],
            wavenumber,
            regularization=regularization,
        )

        predicted_values[
            left_out
        ] = prediction[0]

    absolute_errors = np.abs(
        predicted_values
        - pressures
    )

    rmse = float(
        np.sqrt(
            np.mean(
                absolute_errors**2
            )
        )
    )

    signal_rms = float(
        np.sqrt(
            np.mean(
                np.abs(pressures) ** 2
            )
        )
    )

    nrmse = float(
        rmse
        / (
            signal_rms
            + 1e-12
        )
    )

    measured_magnitudes = np.abs(
        pressures
    )

    predicted_magnitudes = np.abs(
        predicted_values
    )

    correlation = float(
        np.corrcoef(
            measured_magnitudes,
            predicted_magnitudes,
        )[0, 1]
    )

    mae = float(
        np.mean(
            absolute_errors
        )
    )

    return GPLeaveOneOutResult(
        true_values=pressures.copy(),
        predicted_values=predicted_values,
        absolute_errors=np.asarray(
            absolute_errors,
            dtype=np.float64,
        ),
        rmse=rmse,
        nrmse=nrmse,
        correlation=correlation,
        mae=mae,
    )


def reconstruct_gp_field(
    microphone_positions: ArrayLike,
    pressures: ArrayLike,
    frequency: float,
    sound_speed: float = 343.0,
    grid_size: int = 220,
    x_min: float = -0.05,
    x_max: float = 0.17,
    y_min: float = -0.05,
    y_max: float = 0.17,
    regularization: float = 1e-6,
) -> GPReconstructionResult:
    """Reconstruct a two-dimensional acoustic field on a regular grid."""
    microphone_positions = np.asarray(
        microphone_positions,
        dtype=np.float64,
    )

    pressures = np.asarray(
        pressures,
        dtype=np.complex128,
    )

    if frequency <= 0:
        raise ValueError(
            "frequency must be strictly positive"
        )

    if sound_speed <= 0:
        raise ValueError(
            "sound_speed must be strictly positive"
        )

    if grid_size < 2:
        raise ValueError(
            "grid_size must be at least 2"
        )

    if x_max <= x_min:
        raise ValueError(
            "x_max must be greater than x_min"
        )

    if y_max <= y_min:
        raise ValueError(
            "y_max must be greater than y_min"
        )

    wavenumber = float(
        2.0
        * np.pi
        * frequency
        / sound_speed
    )

    x_coordinates = np.linspace(
        x_min,
        x_max,
        grid_size,
    )

    y_coordinates = np.linspace(
        y_min,
        y_max,
        grid_size,
    )

    xx, yy = np.meshgrid(
        x_coordinates,
        y_coordinates,
    )

    prediction_points = np.column_stack(
        (
            xx.ravel(),
            yy.ravel(),
        )
    )

    mean, variance = gp_predict(
        microphone_positions,
        pressures,
        prediction_points,
        wavenumber,
        regularization=regularization,
    )

    mean_field = mean.reshape(
        grid_size,
        grid_size,
    )

    predictive_variance = variance.reshape(
        grid_size,
        grid_size,
    )

    return GPReconstructionResult(
        used_frequency_hz=float(
            frequency
        ),
        wavenumber_rad_m=wavenumber,
        microphone_positions_m=(
            microphone_positions.copy()
        ),
        measured_pressures=(
            pressures.copy()
        ),
        x_coordinates_m=np.asarray(
            x_coordinates,
            dtype=np.float64,
        ),
        y_coordinates_m=np.asarray(
            y_coordinates,
            dtype=np.float64,
        ),
        mean_field=np.asarray(
            mean_field,
            dtype=np.complex128,
        ),
        predictive_variance=np.asarray(
            predictive_variance,
            dtype=np.float64,
        ),
    )