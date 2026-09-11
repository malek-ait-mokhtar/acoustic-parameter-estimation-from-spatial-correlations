import numpy as np

from scripts.apply_uma16_second_derivative import (
    save_second_derivative_results,
)


def test_save_second_derivative_results(tmp_path):
    results = {
        "frequency_hz": np.array(
            [1000.0, 1600.0]
        ),
        "second_derivative_sound_speed_m_s": np.array(
            [343.0, 700.0]
        ),
    }

    path = save_second_derivative_results(
        results,
        tmp_path / "second_derivative",
    )

    assert path.is_file()
    assert path.suffix == ".npz"

    data = np.load(
        path
    )

    assert np.allclose(
        data["frequency_hz"],
        [1000.0, 1600.0],
    )