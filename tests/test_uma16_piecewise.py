import numpy as np

from scripts.apply_uma16_piecewise import (
    save_piecewise_results,
)


def test_save_piecewise_results(tmp_path):
    results = {
        "frequency_hz": np.array(
            [1000.0, 1600.0]
        ),
        "piecewise_sound_speed_m_s": np.array(
            [343.0, 345.0]
        ),
    }

    path = save_piecewise_results(
        results,
        tmp_path / "piecewise",
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