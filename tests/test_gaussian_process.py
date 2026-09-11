import numpy as np

from acoustic_estimation.gaussian_process import (
    build_kernel_matrix,
    extract_complex_pressure,
    gp_predict,
    leave_one_out_gp,
    reconstruct_gp_field,
    sinc_kernel,
)


def test_extract_complex_pressure_selects_nearest_bin():
    sample_rate = 1000.0
    n_samples = 1000

    time = np.arange(
        n_samples
    ) / sample_rate

    signals = np.vstack(
        (
            np.sin(
                2.0 * np.pi * 100.0 * time
            ),
            0.5
            * np.sin(
                2.0 * np.pi * 100.0 * time
            ),
        )
    )

    pressures, used_frequency = (
        extract_complex_pressure(
            signals,
            sample_rate=sample_rate,
            target_frequency=100.2,
            nperseg=n_samples,
        )
    )

    assert pressures.shape == (2,)

    assert np.isclose(
        used_frequency,
        100.0,
    )

    assert np.all(
        np.isfinite(pressures)
    )


def test_sinc_kernel_at_zero():
    values = sinc_kernel(
        np.array(
            [0.0]
        ),
        10.0,
    )

    assert np.allclose(
        values,
        [1.0],
    )


def test_kernel_matrix_is_symmetric():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.04, 0.0],
            [0.0, 0.04],
        ]
    )

    kernel = build_kernel_matrix(
        positions,
        9.0,
    )

    assert kernel.shape == (
        3,
        3,
    )

    assert np.allclose(
        kernel,
        kernel.T,
    )

    assert np.allclose(
        np.diag(kernel),
        1.0,
    )


def test_gp_predict_returns_expected_shapes():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.04, 0.0],
            [0.0, 0.04],
        ]
    )

    pressures = np.array(
        [
            1.0 + 0.0j,
            0.8 + 0.1j,
            0.7 - 0.1j,
        ]
    )

    prediction_points = np.array(
        [
            [0.02, 0.02],
            [0.08, 0.08],
        ]
    )

    mean, variance = gp_predict(
        positions,
        pressures,
        prediction_points,
        9.0,
    )

    assert mean.shape == (
        2,
    )

    assert variance.shape == (
        2,
    )

    assert np.all(
        np.isfinite(mean)
    )

    assert np.all(
        np.isfinite(variance)
    )


def test_gp_predict_reconstructs_training_points():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.04, 0.0],
            [0.0, 0.04],
        ]
    )

    pressures = np.array(
        [
            1.0 + 0.2j,
            0.8 - 0.1j,
            0.6 + 0.3j,
        ]
    )

    mean, _ = gp_predict(
        positions,
        pressures,
        positions,
        9.0,
        regularization=1e-10,
    )

    assert np.allclose(
        mean,
        pressures,
        rtol=1e-6,
        atol=1e-6,
    )


def test_leave_one_out_gp_returns_complete_result():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.04, 0.0],
            [0.08, 0.0],
            [0.0, 0.04],
        ]
    )

    pressures = np.array(
        [
            1.0 + 0.0j,
            0.9 + 0.1j,
            0.8 + 0.2j,
            0.7 - 0.1j,
        ]
    )

    result = leave_one_out_gp(
        positions,
        pressures,
        9.0,
    )

    assert result.true_values.shape == (
        4,
    )

    assert result.predicted_values.shape == (
        4,
    )

    assert result.absolute_errors.shape == (
        4,
    )

    assert np.isfinite(
        result.rmse
    )

    assert np.isfinite(
        result.nrmse
    )

    assert np.isfinite(
        result.correlation
    )

    assert np.isfinite(
        result.mae
    )


def test_reconstruct_gp_field_returns_regular_grid():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.04, 0.0],
            [0.0, 0.04],
        ]
    )

    pressures = np.array(
        [
            1.0 + 0.0j,
            0.8 + 0.1j,
            0.7 - 0.1j,
        ]
    )

    result = reconstruct_gp_field(
        microphone_positions=positions,
        pressures=pressures,
        frequency=500.0,
        sound_speed=343.0,
        grid_size=10,
    )

    assert result.mean_field.shape == (
        10,
        10,
    )

    assert (
        result.predictive_variance.shape
        == (
            10,
            10,
        )
    )

    assert result.x_coordinates_m.shape == (
        10,
    )

    assert result.y_coordinates_m.shape == (
        10,
    )

    assert np.isclose(
        result.wavenumber_rad_m,
        2.0 * np.pi * 500.0 / 343.0,
    )
    