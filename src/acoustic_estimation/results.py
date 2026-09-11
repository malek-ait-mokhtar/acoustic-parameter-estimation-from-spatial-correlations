"""Persistence utilities for acoustic analysis results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acoustic_estimation.estimation import AnalysisResult


def save_analysis_result(
    result: AnalysisResult,
    path: str | Path,
) -> None:
    """Save an acoustic analysis result as a compressed NumPy archive.

    The archive contains scalar acquisition metadata, estimated acoustic
    parameters for every retained frequency, and the pairwise fit data needed
    to reconstruct the RSS objective without reprocessing the raw recordings.

    Parameters
    ----------
    result
        Complete acoustic analysis result.
    path
        Destination path. The ``.npz`` extension is added automatically when
        omitted.
    """
    path = Path(path)

    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimates = result.estimates

    frequencies = np.array(
        [estimate.frequency_hz for estimate in estimates],
        dtype=np.float64,
    )

    wavenumbers = np.array(
        [estimate.wavenumber_rad_m for estimate in estimates],
        dtype=np.float64,
    )

    sound_speeds = np.array(
        [estimate.sound_speed_m_s for estimate in estimates],
        dtype=np.float64,
    )

    rss = np.array(
        [estimate.rss for estimate in estimates],
        dtype=np.float64,
    )

    if estimates:
        distances = np.stack(
            [estimate.distances_m for estimate in estimates],
        )

        observed_coherence = np.stack(
            [
                estimate.observed_coherence
                for estimate in estimates
            ],
        )
    else:
        distances = np.empty(
            (0, 0),
            dtype=np.float64,
        )

        observed_coherence = np.empty(
            (0, 0),
            dtype=np.float64,
        )

    np.savez_compressed(
        path,
        frequency_hz=frequencies,
        wavenumber_rad_m=wavenumbers,
        sound_speed_m_s=sound_speeds,
        rss=rss,
        distances_m=distances,
        observed_coherence=observed_coherence,
        sample_rate_hz=np.float64(result.sample_rate_hz),
        n_channels=np.int64(result.n_channels),
        n_samples=np.int64(result.n_samples),
        n_snapshots=np.int64(result.n_snapshots),
    )