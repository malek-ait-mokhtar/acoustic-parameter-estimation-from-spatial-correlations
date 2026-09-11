import matplotlib

matplotlib.use("Agg")

import numpy as np

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
)
from scripts.analyze_uma16 import (
    save_diagnostic_figures,
    save_primary_figures,
)


def make_result() -> AnalysisResult:
    estimate = FrequencyEstimate(
        frequency_hz=500.0,
        wavenumber_rad_m=9.0,
        sound_speed_m_s=349.0,
        rss=0.25,
        coherence_mean=0.85,
        distances_m=np.array(
            [0.04, 0.08, 0.12]
        ),
        observed_coherence=np.array(
            [0.98, 0.90, 0.78]
        ),
    )

    return AnalysisResult(
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


def test_save_primary_figures(tmp_path):
    save_primary_figures(
        make_result(),
        tmp_path,
    )

    assert (
        tmp_path / "sound_speed_full.png"
    ).is_file()

    assert (
        tmp_path / "sound_speed_retained_band.png"
    ).is_file()

    assert (
        tmp_path / "sinc_comparison_500hz.png"
    ).is_file()


def test_save_diagnostic_figures(tmp_path):
    save_diagnostic_figures(
        make_result(),
        tmp_path,
    )

    assert (
        tmp_path / "rss_vs_frequency.png"
    ).is_file()

    assert (
        tmp_path / "coherence_vs_frequency.png"
    ).is_file()

    assert (
        tmp_path / "mean_spectrum.png"
    ).is_file()

    assert not (
        tmp_path / "sinc_fit_500hz.png"
    ).exists()