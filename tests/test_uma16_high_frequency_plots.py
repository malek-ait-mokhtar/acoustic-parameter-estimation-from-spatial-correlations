import numpy as np

from scripts.plot_uma16_high_frequency_estimators import (
    generate_figures,
)


def test_generate_high_frequency_figures(
    tmp_path,
):
    frequencies = np.array(
        [1000.0, 1600.0, 2000.0],
        dtype=np.float64,
    )

    baseline = np.array(
        [340.0, 180.0, 190.0],
        dtype=np.float64,
    )

    second_path = (
        tmp_path
        / "second.npz"
    )

    piecewise_path = (
        tmp_path
        / "piecewise.npz"
    )

    np.savez(
        second_path,
        frequency_hz=frequencies,
        baseline_sound_speed_m_s=baseline,
        second_derivative_sound_speed_m_s=np.array(
            [340.0, 600.0, 700.0]
        ),
        second_derivative_mask=np.array(
            [False, True, True]
        ),
    )

    np.savez(
        piecewise_path,
        frequency_hz=frequencies,
        baseline_sound_speed_m_s=baseline,
        piecewise_sound_speed_m_s=np.array(
            [340.0, 360.0, 350.0]
        ),
        piecewise_mask=np.array(
            [False, True, True]
        ),
    )

    output_directory = (
        tmp_path
        / "figures"
    )

    r6_figure, r5_figure = generate_figures(
        second_derivative_file=second_path,
        piecewise_affine_file=piecewise_path,
        output_directory=output_directory,
    )

    assert r6_figure.is_file()
    assert r5_figure.is_file()

    assert (
        output_directory
        / "sound_speed_second_derivative.png"
    ).is_file()

    assert (
        output_directory
        / "sound_speed_piecewise_affine.png"
    ).is_file()