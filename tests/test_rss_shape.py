import numpy as np

from scripts.analyze_rss_shape import (
    analyze_frequency,
)


def test_analyze_rss_shape_can_save_figures(
    tmp_path,
):
    frequency = 1500.0

    distances = np.linspace(
        0.02,
        0.18,
        40,
    )

    observed = np.sinc(
        20.0
        * distances
        / np.pi
    )

    results_path = (
        tmp_path
        / "analysis.npz"
    )

    np.savez(
        results_path,
        frequency_hz=np.array(
            [frequency]
        ),
        wavenumber_rad_m=np.array(
            [20.0]
        ),
        distances_m=distances[
            None,
            :
        ],
        observed_coherence=observed[
            None,
            :
        ],
    )

    output_directory = (
        tmp_path
        / "figures"
    )

    analyze_frequency(
        results_file=results_path,
        target_frequency=frequency,
        output_directory=output_directory,
    )

    assert (
        output_directory
        / "rss_second_derivative_1500hz.png"
    ).is_file()

    assert (
        output_directory
        / "rss_piecewise_affine_1500hz.png"
    ).is_file()

    assert (
        output_directory
        / "rss_shape_estimators_1500hz.png"
    ).is_file()