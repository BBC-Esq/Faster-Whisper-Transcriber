from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _linear_resample_mono(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


def try_load_wav_for_faster_whisper(
    path: str | Path,
    *,
    target_sr: int,
) -> np.ndarray | None:
    """Load 16-bit PCM WAV (mono/stereo) for faster-whisper; float32 mono at target_sr.

    Returns None if the file is not a supported WAV (other formats keep file-path fallback).
    """
    path = Path(path)
    if path.suffix.lower() != ".wav":
        return None

    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getcomptype() != "NONE" or wf.getsampwidth() != 2:
                return None
            n_channels = wf.getnchannels()
            if n_channels not in (1, 2):
                return None
            framerate = wf.getframerate()
            if framerate <= 0:
                return None
            raw = wf.readframes(wf.getnframes())
    except (OSError, EOFError, wave.Error):
        return None

    if not raw:
        return np.zeros(0, dtype=np.float32)

    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size % n_channels != 0:
        return None
    pcm = pcm.reshape(-1, n_channels)
    if n_channels == 2:
        # Match PyAV mono path: mean of channels, same scale as decode_audio (s16 -> f32)
        audio_f = pcm.astype(np.float32).mean(axis=1) / 32768.0
    else:
        audio_f = pcm[:, 0].astype(np.float32) / 32768.0

    return _linear_resample_mono(audio_f, framerate, target_sr)
