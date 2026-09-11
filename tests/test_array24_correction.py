import numpy as np

from scripts.correct_array24_minima import (
    save_corrected_results,
)


def test_save_corrected_results(tmp_path):
    results = {
        "frequency_hz": np.array(
            [1000.0, 2000.0]
        ),
        "corrected_sound_speed_m_s": np.array(
            [343.0, 347.0]
        ),
    }

    path = save_corrected_results(
        results,
        tmp_path / "corrected",
    )

    assert path.is_file()
    assert path.suffix == ".npz"

    data = np.load(
        path
    )

    assert np.allclose(
        data["frequency_hz"],
        [1000.0, 2000.0],
    )