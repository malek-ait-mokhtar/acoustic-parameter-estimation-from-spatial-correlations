import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_estimation.io import (
    load_npy_array,
    load_wav_array,
    load_wav_mono,
)


def test_load_wav_mono(tmp_path):
    sample_rate = 8000

    data = np.array(
        [0, 1000, -1000, 2000],
        dtype=np.int16,
    )

    path = tmp_path / "mic.wav"
    wavfile.write(path, sample_rate, data)

    loaded_rate, signal = load_wav_mono(path)

    assert loaded_rate == sample_rate
    assert signal.dtype == np.float64
    assert signal.shape == (4,)

    expected = data.astype(np.float64) / np.iinfo(np.int16).max

    assert np.allclose(signal, expected)


def test_load_wav_mono_keeps_first_channel(tmp_path):
    sample_rate = 8000

    data = np.array(
        [
            [100, 200],
            [300, 400],
            [500, 600],
        ],
        dtype=np.int16,
    )

    path = tmp_path / "stereo.wav"
    wavfile.write(path, sample_rate, data)

    _, signal = load_wav_mono(path)

    expected = data[:, 0].astype(np.float64)
    expected /= np.iinfo(np.int16).max

    assert np.allclose(signal, expected)


def test_load_wav_array_stacks_channels(tmp_path):
    sample_rate = 8000

    for index in range(3):
        data = np.full(
            100,
            index + 1,
            dtype=np.int16,
        )

        wavfile.write(
            tmp_path / f"mic_{index:02d}.wav",
            sample_rate,
            data,
        )

    loaded_rate, signals = load_wav_array(
        tmp_path,
        expected_channels=3,
    )

    assert loaded_rate == sample_rate
    assert signals.shape == (3, 100)


def test_load_wav_array_truncates_to_shortest_signal(tmp_path):
    sample_rate = 8000

    wavfile.write(
        tmp_path / "mic_00.wav",
        sample_rate,
        np.zeros(100, dtype=np.int16),
    )

    wavfile.write(
        tmp_path / "mic_01.wav",
        sample_rate,
        np.zeros(80, dtype=np.int16),
    )

    _, signals = load_wav_array(tmp_path)

    assert signals.shape == (2, 80)


def test_load_wav_array_rejects_mismatched_sample_rates(tmp_path):
    wavfile.write(
        tmp_path / "mic_00.wav",
        8000,
        np.zeros(100, dtype=np.int16),
    )

    wavfile.write(
        tmp_path / "mic_01.wav",
        16000,
        np.zeros(100, dtype=np.int16),
    )

    with pytest.raises(ValueError):
        load_wav_array(tmp_path)


def test_load_wav_array_checks_expected_channel_count(tmp_path):
    wavfile.write(
        tmp_path / "mic_00.wav",
        8000,
        np.zeros(100, dtype=np.int16),
    )

    with pytest.raises(ValueError):
        load_wav_array(
            tmp_path,
            expected_channels=16,
        )


def test_load_npy_array_transposes_mmlib_layout(tmp_path):
    recording = np.arange(
        100 * 24,
        dtype=np.float64,
    ).reshape(100, 24)

    path = tmp_path / "recording.npy"
    np.save(path, recording)

    signals = load_npy_array(
        path,
        expected_channels=24,
        channels_axis=1,
    )

    assert signals.shape == (24, 100)
    assert np.allclose(signals, recording.T)


def test_load_npy_array_preserves_channel_first_layout(tmp_path):
    recording = np.arange(
        24 * 100,
        dtype=np.float64,
    ).reshape(24, 100)

    path = tmp_path / "recording.npy"
    np.save(path, recording)

    signals = load_npy_array(
        path,
        expected_channels=24,
        channels_axis=0,
    )

    assert signals.shape == (24, 100)
    assert np.allclose(signals, recording)


def test_load_npy_array_checks_expected_channels(tmp_path):
    recording = np.zeros((100, 8))

    path = tmp_path / "recording.npy"
    np.save(path, recording)

    with pytest.raises(ValueError):
        load_npy_array(
            path,
            expected_channels=24,
            channels_axis=1,
        )


def test_load_npy_array_rejects_invalid_channel_axis(tmp_path):
    path = tmp_path / "recording.npy"
    np.save(path, np.zeros((100, 24)))

    with pytest.raises(ValueError):
        load_npy_array(
            path,
            channels_axis=2,
        )