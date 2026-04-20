from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentData:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[SegmentData] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None
    source_file: Path | None = None


def format_timestamp(seconds: float, delimiter: str = ",") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{delimiter}{millis:03d}"


def write_txt(segments: list[SegmentData], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(segment.text.strip() + "\n")


def write_srt(segments: list[SegmentData], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(
                f"{format_timestamp(segment.start, ',')} --> "
                f"{format_timestamp(segment.end, ',')}\n"
            )
            f.write(f"{segment.text.strip()}\n\n")


def write_vtt(segments: list[SegmentData], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in segments:
            f.write(
                f"{format_timestamp(segment.start, '.')} --> "
                f"{format_timestamp(segment.end, '.')}\n"
            )
            f.write(f"{segment.text.strip()}\n\n")


def write_json(result: TranscriptionResult, output_file: Path) -> None:
    output = {
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            for seg in result.segments
        ],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def write_output(
    result: TranscriptionResult, output_file: Path, fmt: str
) -> None:
    writers = {
        "txt": lambda: write_txt(result.segments, output_file),
        "srt": lambda: write_srt(result.segments, output_file),
        "vtt": lambda: write_vtt(result.segments, output_file),
        "json": lambda: write_json(result, output_file),
    }
    writer = writers.get(fmt)
    if writer:
        writer()
        logger.info(f"Output written to {output_file}")
    else:
        logger.error(f"Unknown output format: {fmt}")
