import numpy as np

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
)
from acoustic_estimation.results import save_analysis_result


def test_save_analysis_result(tmp_path):
    estimate = FrequencyEstimate(
        frequency_hz=500.0,
        wavenumber_rad_m=9.0,
        sound_speed_m_s=349.0,
        rss=0.25,
        distances_m=np.array([0.04, 0.08]),
        observed_coherence=np.array([0.9, 0.7]),
    )

    result = AnalysisResult(
        estimates=[estimate],
        sample_rate_hz=44100.0,
        n_channels=16,
        n_samples=1000,
        n_snapshots=10,
    )

    path = tmp_path / "analysis.npz"

    save_analysis_result(
        result,
        path,
    )

    data = np.load(path)

    assert np.allclose(
        data["frequency_hz"],
        [500.0],
    )

    assert np.allclose(
        data["wavenumber_rad_m"],
        [9.0],
    )

    assert np.allclose(
        data["sound_speed_m_s"],
        [349.0],
    )

    assert np.allclose(
        data["rss"],
        [0.25],
    )

    assert np.allclose(
        data["distances_m"],
        [[0.04, 0.08]],
    )

    assert np.allclose(
        data["observed_coherence"],
        [[0.9, 0.7]],
    )

    assert data["sample_rate_hz"] == 44100.0
    assert data["n_channels"] == 16
    assert data["n_samples"] == 1000
    assert data["n_snapshots"] == 10