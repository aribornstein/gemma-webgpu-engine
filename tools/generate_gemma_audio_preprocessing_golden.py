import json

import numpy as np


SAMPLE_RATE = 16_000
FRAME_LENGTH = 320
HOP_LENGTH = 160
FFT_LENGTH = 512
FEATURE_SIZE = 128
MEL_FLOOR = 0.001


def hz_to_mel(frequency):
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def mel_to_hz(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filters():
    mel_frequencies = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), FEATURE_SIZE + 2)
    filter_frequencies = mel_to_hz(mel_frequencies)
    fft_frequencies = np.linspace(0.0, SAMPLE_RATE // 2, FFT_LENGTH // 2 + 1)
    filters = np.zeros((FEATURE_SIZE, len(fft_frequencies)), dtype=np.float64)
    for mel in range(FEATURE_SIZE):
        down = (fft_frequencies - filter_frequencies[mel]) / (
            filter_frequencies[mel + 1] - filter_frequencies[mel]
        )
        up = (filter_frequencies[mel + 2] - fft_frequencies) / (
            filter_frequencies[mel + 2] - filter_frequencies[mel + 1]
        )
        filters[mel] = np.maximum(0.0, np.minimum(down, up))
    return filters


def extract(waveform):
    padded_length = ((len(waveform) + 127) // 128) * 128
    padded = np.pad(waveform, (FRAME_LENGTH // 2, padded_length - len(waveform)))
    frame_size = FRAME_LENGTH + 1
    frame_count = (len(padded) - frame_size) // HOP_LENGTH + 1
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_size)[
        ::HOP_LENGTH
    ][:frame_count, :-1]
    window = 0.5 - 0.5 * np.cos(
        2.0 * np.pi * np.arange(FRAME_LENGTH, dtype=np.float64) / FRAME_LENGTH
    )
    magnitudes = np.abs(np.fft.rfft(frames * window, n=FFT_LENGTH, axis=-1))
    features = np.log(magnitudes @ mel_filters().T + MEL_FLOOR).astype(np.float32)
    frame_end_indices = np.arange(frame_count) * HOP_LENGTH + frame_size - 1
    sample_mask = np.pad(
        np.ones(len(waveform), dtype=np.uint8),
        (FRAME_LENGTH // 2, padded_length - len(waveform)),
    )
    mask = sample_mask[frame_end_indices].astype(bool)
    features *= mask[:, None]
    return features, mask


indices = [(0, 0), (0, 17), (1, 7), (10, 23), (50, 40), (98, 127)]
sample_indices = np.arange(SAMPLE_RATE, dtype=np.float64)
waveform = (
    0.4 * np.sin(2.0 * np.pi * 440.0 * sample_indices / SAMPLE_RATE)
    + 0.1 * np.cos(2.0 * np.pi * 880.0 * sample_indices / SAMPLE_RATE)
).astype(np.float32)
features, mask = extract(waveform)
print(json.dumps({
    "sampleRate": SAMPLE_RATE,
    "sampleCount": len(waveform),
    "frameCount": len(features),
    "validFrameCount": int(mask.sum()),
    "values": [
        {"frame": frame, "bin": bin_index, "value": float(features[frame, bin_index])}
        for frame, bin_index in indices
    ],
}, indent=2))