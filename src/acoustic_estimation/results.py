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

    The archive contains acquisition metadata, estimated acoustic parameters,
    mean coherence values, the mean spectrum, and the pairwise fit data needed
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

    # ------------------------------------------------------------------
    # Frequency-dependent estimated quantities.
    # ------------------------------------------------------------------

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

    coherence_mean = np.array(
        [estimate.coherence_mean for estimate in estimates],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Pairwise data required to reconstruct RSS(k).
    # ------------------------------------------------------------------

    if estimates:
        distances = np.stack(
            [
                estimate.distances_m
                for estimate in estimates
            ]
        )

        observed_coherence = np.stack(
            [
                estimate.observed_coherence
                for estimate in estimates
            ]
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

    # ------------------------------------------------------------------
    # Persist the complete processed result without Python pickling.
    # ------------------------------------------------------------------

    np.savez_compressed(
        path,
        frequency_hz=frequencies,
        wavenumber_rad_m=wavenumbers,
        sound_speed_m_s=sound_speeds,
        rss=rss,
        coherence_mean=coherence_mean,
        distances_m=distances,
        observed_coherence=observed_coherence,
        spectrum_frequencies_hz=result.spectrum_frequencies_hz,
        mean_spectrum=result.mean_spectrum,
        sample_rate_hz=np.float64(result.sample_rate_hz),
        n_channels=np.int64(result.n_channels),
        n_samples=np.int64(result.n_samples),
        n_snapshots=np.int64(result.n_snapshots),
    )