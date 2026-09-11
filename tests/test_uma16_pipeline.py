import matplotlib

matplotlib.use("Agg")

import numpy as np

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
)
from scripts.analyze_uma16 import save_standard_figures


def test_save_standard_figures(tmp_path):
    estimate = FrequencyEstimate(
        frequency_hz=500.0,
        wavenumber_rad_m=9.0,
        sound_speed_m_s=349.0,
        rss=0.25,
        distances_m=np.array([0.04, 0.08, 0.12]),
        observed_coherence=np.array([0.98, 0.90, 0.78]),
    )

    result = AnalysisResult(
        estimates=[estimate],
        sample_rate_hz=44100.0,
        n_channels=16,
        n_samples=1000,
        n_snapshots=10,
        spectrum_frequencies_hz=np.array(
            [0.0, 250.0, 500.0, 750.0]
        ),
        mean_spectrum=np.array(
            [0.1, 0.5, 2.0, 0.3]
        ),
    )

    save_standard_figures(
        result,
        tmp_path,
    )

    assert (tmp_path / "sound_speed.png").is_file()
    assert (tmp_path / "rss.png").is_file()
    assert (tmp_path / "sinc_fit_500hz.png").is_file()
    assert (tmp_path / "mean_spectrum.png").is_file()