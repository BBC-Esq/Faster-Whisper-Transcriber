from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from core.logging_config import get_logger
from core.output.writers import SegmentData

logger = get_logger(__name__)

DEFAULT_SPEAKER_LABELS = ("Lawyer", "Client")
USER_PROFILE_KEY = "speaker_1"
_EPS = 1e-8
_PROFILE_MAX_DISTANCE = 0.35
_PROFILE_MIN_MARGIN = 0.10
_PROFILE_STRONG_DISTANCE = 0.12
_PROFILE_STRONG_MARGIN = 0.04
_PROFILE_STRONG_RATIO = 0.55


def apply_client_call_speakers(
    audio_file: str | Path,
    segments: list[SegmentData],
    labels: list[str] | tuple[str, ...] | None = None,
    voice_profiles: dict[str, list[float]] | None = None,
) -> bool:
    """Assign lightweight speaker labels to transcript segments.

    It first tries very fast stereo channel separation, then falls back to
    simple acoustic clustering for mono or mixed-channel recordings.
    """
    if not segments:
        return False

    labels = _normalize_labels(labels)
    profiles = _normalize_voice_profiles(voice_profiles)

    loaded = _load_pcm_wav(audio_file)
    if loaded is None:
        logger.info("Client call speaker labels skipped: audio could not be decoded")
        return False

    audio, sample_rate = loaded
    assigned = _assign_from_stereo_channels(audio, sample_rate, segments, labels, profiles)
    if assigned is None:
        assigned = _assign_from_mono_features(audio, sample_rate, segments, labels, profiles)

    if not assigned:
        return False

    for segment, speaker in zip(segments, assigned):
        segment.speaker = speaker

    return True


def build_voice_profile_from_samples(samples: np.ndarray, sample_rate: int) -> list[float]:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    feature = _segment_feature(samples, sample_rate, min_seconds=1.0)
    if feature is None:
        raise ValueError("Not enough clear speech was captured for a voice profile")
    return [round(float(value), 6) for value in feature.tolist()]


def format_client_call_transcript(segments: list[SegmentData]) -> str:
    blocks: list[str] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if not current_parts:
            return
        speaker = current_speaker or "Speaker"
        text = " ".join(part.strip() for part in current_parts if part.strip()).strip()
        if text:
            blocks.append(f"{speaker}: {text}")

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        speaker = segment.speaker
        if speaker != current_speaker:
            flush()
            current_speaker = speaker
            current_parts = [text]
        else:
            current_parts.append(text)

    flush()
    return "\n\n".join(blocks)


def _load_pcm_wav(path: str | Path) -> tuple[np.ndarray, int] | None:
    path = Path(path)
    if path.suffix.lower() != ".wav":
        return _load_with_audio_decoder(path)

    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getcomptype() != "NONE" or wf.getsampwidth() != 2:
                return _load_with_audio_decoder(path)
            channels = wf.getnchannels()
            if channels not in (1, 2):
                return _load_with_audio_decoder(path)
            sample_rate = wf.getframerate()
            if sample_rate <= 0:
                return _load_with_audio_decoder(path)
            raw = wf.readframes(wf.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        logger.debug(f"Unable to read WAV for client call labels: {exc}")
        return _load_with_audio_decoder(path)

    if not raw:
        return np.zeros((0, channels), dtype=np.float32), sample_rate

    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size % channels != 0:
        return _load_with_audio_decoder(path)
    audio = pcm.reshape(-1, channels).astype(np.float32) / 32768.0
    return audio, sample_rate


def _load_with_audio_decoder(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        from faster_whisper.audio import decode_audio

        sample_rate = 16000
        decoded = decode_audio(str(path), sampling_rate=sample_rate, split_stereo=True)
        if isinstance(decoded, tuple) and len(decoded) == 2:
            left, right = decoded
            size = min(left.size, right.size)
            if size <= 0:
                return np.zeros((0, 2), dtype=np.float32), sample_rate
            audio = np.column_stack([left[:size], right[:size]]).astype(np.float32)
            return audio, sample_rate

        mono = np.asarray(decoded, dtype=np.float32).reshape(-1, 1)
        return mono, sample_rate
    except Exception as exc:
        logger.debug(f"Unable to decode audio for speaker labels: {exc}")
        return None


def _assign_from_stereo_channels(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SegmentData],
    labels: list[str],
    voice_profiles: dict[str, np.ndarray],
) -> list[str] | None:
    if audio.ndim != 2 or audio.shape[1] < 2:
        return None
    if len(labels) != 2:
        return None

    dominant_channels: list[int | None] = []
    channel_features: dict[int, list[np.ndarray]] = {0: [], 1: []}
    confident = 0
    for segment in segments:
        chunk = _segment_audio(audio, sample_rate, segment)
        if chunk.size == 0:
            dominant_channels.append(None)
            continue

        left = _rms(chunk[:, 0])
        right = _rms(chunk[:, 1])
        louder = max(left, right)
        quieter = max(min(left, right), _EPS)
        if louder < 0.002 or louder / quieter < 1.35:
            dominant_channels.append(None)
            continue

        dominant_channel = 0 if left > right else 1
        dominant_channels.append(dominant_channel)
        confident += 1
        feature = _segment_feature(chunk[:, dominant_channel], sample_rate)
        if feature is not None:
            channel_features[dominant_channel].append(feature)

    if confident < max(2, len(segments) // 3):
        return None

    channel_profiles = {
        channel: np.median(np.vstack(features), axis=0)
        for channel, features in channel_features.items()
        if features
    }
    if channel_profiles:
        feature_size = next(iter(channel_profiles.values())).size
        voice_profiles = _matching_voice_profiles(voice_profiles, feature_size)
    channel_to_label = _map_feature_groups_to_labels(channel_profiles, labels, voice_profiles)
    if not channel_to_label:
        first_channel = next((ch for ch in dominant_channels if ch is not None), 0)
        channel_to_label = {
            first_channel: labels[0],
            1 - first_channel: labels[1],
        }

    assigned: list[str] = []
    previous = labels[0]
    for channel in dominant_channels:
        if channel is None:
            assigned.append(previous)
            continue
        previous = channel_to_label[channel]
        assigned.append(previous)

    logger.info("Client call speaker labels assigned from stereo channel energy")
    return assigned


def _assign_from_mono_features(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SegmentData],
    labels: list[str],
    voice_profiles: dict[str, np.ndarray],
) -> list[str]:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio

    indexed_features: list[tuple[int, np.ndarray]] = []
    for idx, segment in enumerate(segments):
        chunk = _segment_audio(mono, sample_rate, segment)
        feature = _segment_feature(chunk, sample_rate)
        if feature is not None:
            indexed_features.append((idx, feature))

    if len(indexed_features) < 2:
        return [labels[0] for _ in segments]

    feature_matrix = np.vstack([feature for _, feature in indexed_features])
    speaker_count = max(2, min(len(labels), len(indexed_features)))
    voice_profiles = _matching_voice_profiles(voice_profiles, feature_matrix.shape[1])
    cluster_ids = _kmeans(feature_matrix, speaker_count)
    cluster_profiles = {
        cluster_id: np.median(feature_matrix[cluster_ids == cluster_id], axis=0)
        for cluster_id in range(speaker_count)
        if np.any(cluster_ids == cluster_id)
    }
    cluster_order = _ordered_clusters_by_first_appearance(indexed_features, cluster_ids)
    cluster_to_label = _map_feature_groups_to_labels(
        cluster_profiles, labels, voice_profiles, cluster_order
    )
    if not cluster_to_label:
        cluster_to_label = _cluster_labels_by_first_appearance(
            indexed_features, cluster_ids, labels
        )

    assigned: list[str | None] = [None] * len(segments)
    for (idx, _), cluster_id in zip(indexed_features, cluster_ids):
        assigned[idx] = cluster_to_label[int(cluster_id)]

    logger.info("Client call speaker labels assigned from acoustic clustering")
    return _fill_missing_labels(assigned, labels[0])


def _segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    segment: SegmentData,
) -> np.ndarray:
    start = max(0, int(segment.start * sample_rate))
    end = min(audio.shape[0], int(segment.end * sample_rate))
    if end <= start:
        return audio[:0]
    return audio[start:end]


def _segment_feature(
    chunk: np.ndarray,
    sample_rate: int,
    min_seconds: float = 0.35,
) -> np.ndarray | None:
    if chunk.ndim == 2:
        chunk = chunk.mean(axis=1)
    if chunk.size < int(sample_rate * min_seconds):
        return None

    frame_len = max(512, int(sample_rate * 0.05))
    hop = max(160, frame_len // 2)
    if chunk.size < frame_len:
        return None

    window = np.hanning(frame_len).astype(np.float32)
    frames = []
    for start in range(0, chunk.size - frame_len + 1, hop):
        frame = chunk[start:start + frame_len].astype(np.float32, copy=False)
        energy = _rms(frame)
        if energy > 0.003:
            frames.append(frame)

    if not frames:
        return None

    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
    nyquist = max(freqs[-1], 1.0)
    band_edges = np.array(
        [80, 150, 250, 400, 650, 1000, 1600, 2600, 4000, 6000, min(8000, nyquist)],
        dtype=np.float32,
    )
    band_edges = band_edges[band_edges <= nyquist]
    if band_edges.size < 4:
        band_edges = np.linspace(80, nyquist, 6, dtype=np.float32)

    feature_rows = []
    for frame in frames:
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        total = float(np.sum(spectrum)) + _EPS
        centroid = float(np.sum(freqs * spectrum) / total) / nyquist
        bandwidth = float(
            np.sqrt(np.sum(((freqs / nyquist - centroid) ** 2) * spectrum) / total)
        )
        zcr = float(np.mean(np.abs(np.diff(np.signbit(frame)))))
        rms = _rms(frame)

        band_values = []
        for low, high in zip(band_edges[:-1], band_edges[1:]):
            mask = (freqs >= low) & (freqs < high)
            band_energy = float(np.sum(spectrum[mask])) / total if np.any(mask) else 0.0
            band_values.append(np.log1p(band_energy * 100.0))

        feature_rows.append([np.log1p(rms * 100.0), centroid, bandwidth, zcr, *band_values])

    rows = np.asarray(feature_rows, dtype=np.float32)
    return np.median(rows, axis=0)


def _kmeans(features: np.ndarray, k: int, iterations: int = 30) -> np.ndarray:
    features = features.astype(np.float32, copy=True)
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    features = (features - means) / np.where(stds < 1e-5, 1.0, stds)

    if k <= 1 or features.shape[0] <= 1:
        return np.zeros(features.shape[0], dtype=np.int32)

    k = min(k, features.shape[0])
    centroid_indices = [0]
    while len(centroid_indices) < k:
        selected = features[centroid_indices]
        distances = _pairwise_squared_distances(features, selected)
        min_distances = np.min(distances, axis=1)
        for idx in centroid_indices:
            min_distances[idx] = -1
        next_idx = int(np.argmax(min_distances))
        if next_idx in centroid_indices:
            break
        centroid_indices.append(next_idx)

    centroids = features[centroid_indices].copy()
    labels = np.zeros(features.shape[0], dtype=np.int32)

    for _ in range(iterations):
        dists = _pairwise_squared_distances(features, centroids)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(centroids.shape[0]):
            cluster = features[labels == cluster_id]
            if cluster.size:
                centroids[cluster_id] = cluster.mean(axis=0)

    return labels


def _pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        np.sum(a * a, axis=1, keepdims=True)
        - 2 * a @ b.T
        + np.sum(b * b, axis=1, keepdims=True).T
    )


def _cluster_labels_by_first_appearance(
    indexed_features: list[tuple[int, np.ndarray]],
    cluster_ids: np.ndarray,
    labels: list[str],
) -> dict[int, str]:
    ordered_clusters = _ordered_clusters_by_first_appearance(indexed_features, cluster_ids)
    return {
        cluster: labels[min(idx, len(labels) - 1)]
        for idx, cluster in enumerate(ordered_clusters)
    }


def _ordered_clusters_by_first_appearance(
    indexed_features: list[tuple[int, np.ndarray]],
    cluster_ids: np.ndarray,
) -> list[int]:
    first_seen: dict[int, int] = {}
    for (idx, _), cluster_id in zip(indexed_features, cluster_ids):
        first_seen.setdefault(int(cluster_id), idx)
    return sorted(first_seen, key=first_seen.get)


def _normalize_labels(labels: list[str] | tuple[str, ...] | None) -> list[str]:
    defaults = DEFAULT_SPEAKER_LABELS
    if not labels:
        return list(defaults)
    values = [str(label).strip() for label in labels[:8]]
    while len(values) < 2:
        values.append(defaults[len(values)])
    return [
        value or (defaults[idx] if idx < len(defaults) else f"Speaker {idx + 1}")
        for idx, value in enumerate(values)
    ]


def _normalize_voice_profiles(
    voice_profiles: dict[str, list[float]] | None,
) -> dict[str, np.ndarray]:
    if not isinstance(voice_profiles, dict):
        return {}

    profiles: dict[str, np.ndarray] = {}
    values = voice_profiles.get(USER_PROFILE_KEY)
    if isinstance(values, list):
        try:
            profile = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            profile = np.asarray([], dtype=np.float32)
        if profile.ndim == 1 and profile.size:
            profiles[USER_PROFILE_KEY] = profile
    return profiles


def _map_feature_groups_to_labels(
    group_features: dict[int, np.ndarray],
    labels: list[str],
    voice_profiles: dict[str, np.ndarray],
    group_order: list[int] | None = None,
) -> dict[int, str]:
    if not group_features:
        return {}

    ordered = group_order or sorted(group_features)
    groups_by_time = [group for group in ordered if group in group_features]
    groups_by_time.extend(group for group in group_features if group not in groups_by_time)
    mapping: dict[int, str] = {}
    remaining_groups = list(groups_by_time)

    user_profile = voice_profiles.get(USER_PROFILE_KEY)
    if user_profile is not None:
        user_group = _confident_profile_group(group_features, user_profile)
        if user_group is not None:
            mapping[user_group] = labels[0]
            remaining_groups = [group for group in remaining_groups if group != user_group]
        else:
            logger.info("Voice enrollment ignored for this audio: match was not distinct enough")

    next_label_idx = 1 if mapping else 0
    for group in remaining_groups:
        mapping[group] = labels[min(next_label_idx, len(labels) - 1)]
        next_label_idx += 1

    return mapping


def _confident_profile_group(
    group_features: dict[int, np.ndarray],
    user_profile: np.ndarray,
) -> int | None:
    distances = [
        (group, _normalized_distance(feature, user_profile))
        for group, feature in group_features.items()
    ]
    distances = [(group, dist) for group, dist in distances if np.isfinite(dist)]
    if not distances:
        return None

    distances.sort(key=lambda item: item[1])
    best_group, best_distance = distances[0]

    if len(distances) == 1:
        return best_group if best_distance <= _PROFILE_MAX_DISTANCE else None

    second_distance = distances[1][1]
    margin = second_distance - best_distance
    ratio = best_distance / max(second_distance, _EPS)

    if (
        best_distance <= _PROFILE_MAX_DISTANCE
        and (
            margin >= _PROFILE_MIN_MARGIN
            or (
                best_distance <= _PROFILE_STRONG_DISTANCE
                and margin >= _PROFILE_STRONG_MARGIN
                and ratio <= _PROFILE_STRONG_RATIO
            )
        )
    ):
        return best_group

    logger.debug(
        "Voice profile match rejected: "
        f"best={best_distance:.4f}, second={second_distance:.4f}, "
        f"margin={margin:.4f}, ratio={ratio:.4f}"
    )
    return None


def _fill_missing_labels(assigned: list[str | None], fallback_label: str) -> list[str]:
    first_label = next((label for label in assigned if label is not None), fallback_label)
    previous = first_label
    filled: list[str] = []
    for label in assigned:
        if label is None:
            filled.append(previous)
        else:
            previous = label
            filled.append(label)
    return filled


def _normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm <= _EPS or b_norm <= _EPS:
        return float("inf")
    cosine_distance = 1.0 - float(np.dot(a, b) / (a_norm * b_norm))
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    scaled_diff = (a - b) / scale
    return cosine_distance + 0.05 * float(np.mean(scaled_diff * scaled_diff))


def _matching_voice_profiles(
    voice_profiles: dict[str, np.ndarray],
    feature_size: int,
) -> dict[str, np.ndarray]:
    return {
        key: profile
        for key, profile in voice_profiles.items()
        if profile.ndim == 1 and profile.size == feature_size
    }


def _zscore(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32, copy=False)
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    return (vectors - mean) / np.where(std < 1e-5, 1.0, std)


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    samples = samples.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(samples * samples)))
