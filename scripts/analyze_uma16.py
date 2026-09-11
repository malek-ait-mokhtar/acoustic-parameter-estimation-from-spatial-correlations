"""Estimate sound speed from an UMA16 microphone-array recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from acoustic_estimation.estimation import (
    AnalysisResult,
    FrequencyEstimate,
    build_pairwise_dataset,
    estimate_wavenumber,
    sound_speed_from_wavenumber,
)
from acoustic_estimation.geometry import uma16_positions
from acoustic_estimation.io import load_wav_array
from acoustic_estimation.plotting import (
    plot_coherence_mean,
    plot_mean_spectrum,
    plot_rss,
    plot_sinc_comparison,
    plot_sound_speed,
    plot_sound_speed_retained_band,
)
from acoustic_estimation.results import save_analysis_result
from acoustic_estimation.spectral import (
    average_spectrum,
    coherence_matrix,
    cross_spectral_matrices,
    select_frequency_bins,
)


def analyze_uma16(
    data_directory: str | Path,
    min_frequency: float = 50.0,
    max_frequency: float = 3000.0,
    relative_threshold: float = 0.03,
    reference_sound_speed: float = 343.0,
) -> AnalysisResult:
    """Run the sound-speed estimation pipeline on one UMA16 acquisition.

    Parameters
    ----------
    data_directory
        Directory containing the 16 UMA16 WAV recordings.
    min_frequency
        Minimum analysed frequency in hertz.
    max_frequency
        Maximum analysed frequency in hertz.
    relative_threshold
        Minimum spectral amplitude relative to the maximum mean spectrum.
    reference_sound_speed
        Reference sound speed used to define the wavenumber search interval.

    Returns
    -------
    AnalysisResult
        Estimated acoustic parameters, pairwise fit data, spectrum, and
        acquisition metadata.
    """
    sample_rate, signals = load_wav_array(
        data_directory,
        expected_channels=16,
    )

    print(f"Loaded {signals.shape[0]} microphones")
    print(f"Samples per microphone: {signals.shape[1]}")
    print(f"Sample rate: {sample_rate:.1f} Hz")

    # ------------------------------------------------------------------
    # Mean spectrum used to identify sufficiently energetic frequencies.
    # ------------------------------------------------------------------

    spectrum_frequencies, mean_spectrum = average_spectrum(
        signals,
        sample_rate,
        nperseg=16384,
    )

    # ------------------------------------------------------------------
    # Cross-spectral matrices estimated from overlapping FFT snapshots.
    # ------------------------------------------------------------------

    csm_frequencies, csms, n_snapshots = cross_spectral_matrices(
        signals,
        sample_rate,
        nperseg=4096,
        overlap=0.5,
    )

    # ------------------------------------------------------------------
    # Frequency selection.
    # ------------------------------------------------------------------

    selected_indices = select_frequency_bins(
        spectrum_frequencies,
        mean_spectrum,
        csm_frequencies,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        relative_threshold=relative_threshold,
    )

    print(f"CSM snapshots: {n_snapshots}")
    print(f"Selected frequency bins: {len(selected_indices)}")

    # ------------------------------------------------------------------
    # Spatial-coherence model fitting.
    # ------------------------------------------------------------------

    positions = uma16_positions()

    estimates: list[FrequencyEstimate] = []

    for index in selected_indices:
        frequency = float(
            csm_frequencies[index]
        )

        gamma = coherence_matrix(
            csms[index]
        )

        # Mean magnitude of the off-diagonal spatial coherence terms.
        off_diagonal = ~np.eye(
            gamma.shape[0],
            dtype=bool,
        )

        coherence_mean = float(
            np.mean(
                np.abs(
                    gamma[off_diagonal]
                )
            )
        )

        distances, observed_coherence = build_pairwise_dataset(
            gamma,
            positions,
        )

        wavenumber, rss = estimate_wavenumber(
            distances,
            observed_coherence,
            frequency,
            reference_sound_speed=reference_sound_speed,
        )

        sound_speed = sound_speed_from_wavenumber(
            frequency,
            wavenumber,
        )

        estimates.append(
            FrequencyEstimate(
                frequency_hz=frequency,
                wavenumber_rad_m=wavenumber,
                sound_speed_m_s=sound_speed,
                rss=rss,
                coherence_mean=coherence_mean,
                distances_m=distances,
                observed_coherence=observed_coherence,
            )
        )

    return AnalysisResult(
        estimates=estimates,
        sample_rate_hz=sample_rate,
        n_channels=signals.shape[0],
        n_samples=signals.shape[1],
        n_snapshots=n_snapshots,
        spectrum_frequencies_hz=spectrum_frequencies,
        mean_spectrum=mean_spectrum,
    )


def print_summary(
    result: AnalysisResult,
) -> None:
    """Print estimated acoustic parameters for all retained frequencies."""
    estimates = result.estimates

    if not estimates:
        print(
            "No frequency bins satisfied the selection criteria."
        )
        return

    print()
    print(
        f"{'Frequency (Hz)':>14} "
        f"{'k (rad/m)':>12} "
        f"{'c (m/s)':>12} "
        f"{'RSS':>14}"
    )
    print("-" * 56)

    for estimate in estimates:
        print(
            f"{estimate.frequency_hz:14.2f} "
            f"{estimate.wavenumber_rad_m:12.4f} "
            f"{estimate.sound_speed_m_s:12.2f} "
            f"{estimate.rss:14.4e}"
        )


def save_primary_figures(
    result: AnalysisResult,
    directory: str | Path,
    reference_sound_speed: float = 343.0,
) -> None:
    """Generate and save the primary UMA16 result figures.

    The primary figures correspond to the main scientific results retained
    in the final analysis:

    - sound-speed estimates over the complete analysed frequency range;
    - sound-speed estimates over the retained 290-1500 Hz frequency band;
    - comparison between measured, theoretical, and fitted spatial
      coherence near 500 Hz.
    """
    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures = [
        plot_sound_speed(
            result,
            reference_sound_speed=reference_sound_speed,
            path=directory / "sound_speed_full.png",
        ),
        plot_sound_speed_retained_band(
            result,
            min_frequency=290.0,
            max_frequency=1500.0,
            reference_sound_speed=reference_sound_speed,
            path=directory / "sound_speed_retained_band.png",
        ),
        plot_sinc_comparison(
            result,
            target_frequency=500.0,
            reference_sound_speed=reference_sound_speed,
            path=directory / "sinc_comparison_500hz.png",
        ),
    ]

    for figure in figures:
        plt.close(figure)


def save_diagnostic_figures(
    result: AnalysisResult,
    directory: str | Path,
    relative_threshold: float = 0.03,
    max_frequency: float = 3000.0,
) -> None:
    """Generate and save secondary UMA16 diagnostic figures."""
    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures = [
        plot_rss(
            result,
            path=directory / "rss_vs_frequency.png",
        ),
        plot_coherence_mean(
            result,
            path=directory / "coherence_vs_frequency.png",
        ),
        plot_mean_spectrum(
            result,
            relative_threshold=relative_threshold,
            max_frequency=max_frequency,
            path=directory / "mean_spectrum.png",
        ),
    ]

    for figure in figures:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate sound speed from an UMA16 recording "
            "using spatial coherence."
        )
    )

    parser.add_argument(
        "data_directory",
        type=Path,
        help="Directory containing the 16 UMA16 WAV recordings.",
    )

    parser.add_argument(
        "--min-frequency",
        type=float,
        default=50.0,
        help="Minimum analysed frequency in Hz (default: 50).",
    )

    parser.add_argument(
        "--max-frequency",
        type=float,
        default=3000.0,
        help="Maximum analysed frequency in Hz (default: 3000).",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help=(
            "Relative spectrum threshold used for frequency selection "
            "(default: 0.03)."
        ),
    )

    parser.add_argument(
        "--reference-sound-speed",
        type=float,
        default=343.0,
        help=(
            "Reference sound speed used to define the wavenumber "
            "search interval (default: 343 m/s)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the processed .npz analysis results.",
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory in which the primary analysis "
            "figures are saved."
        ),
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Also generate secondary diagnostic figures. "
            "Requires --figures-dir."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the UMA16 analysis from the command line."""
    args = parse_args()

    if args.diagnostics and args.figures_dir is None:
        raise ValueError(
            "--diagnostics requires --figures-dir"
        )

    result = analyze_uma16(
        data_directory=args.data_directory,
        min_frequency=args.min_frequency,
        max_frequency=args.max_frequency,
        relative_threshold=args.threshold,
        reference_sound_speed=args.reference_sound_speed,
    )

    print_summary(result)

    # ------------------------------------------------------------------
    # Processed numerical results.
    # ------------------------------------------------------------------

    if args.output is not None:
        save_analysis_result(
            result,
            args.output,
        )

        output_path = args.output

        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        print()
        print(f"Results saved to: {output_path}")

    # ------------------------------------------------------------------
    # Primary and optional diagnostic figures.
    # ------------------------------------------------------------------

    if args.figures_dir is not None:
        save_primary_figures(
            result,
            directory=args.figures_dir,
            reference_sound_speed=args.reference_sound_speed,
        )

        print(
            f"Primary figures saved to: "
            f"{args.figures_dir}"
        )

        if args.diagnostics:
            diagnostics_directory = (
                args.figures_dir / "diagnostics"
            )

            save_diagnostic_figures(
                result,
                directory=diagnostics_directory,
                relative_threshold=args.threshold,
            )

            print(
                f"Diagnostic figures saved to: "
                f"{diagnostics_directory}"
            )


if __name__ == "__main__":
    main()