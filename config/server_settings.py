from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TranscriptionSettings:
    """Settings passed to the server worker on each /transcribe call."""
    model_key: str
    device: str
    task_mode: str
    language: str | None
    output_format: str
    include_timestamps: bool = False
    word_timestamps: bool = False
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    batch_size: int = 1
    recursive: bool = False
    selected_extensions: List[str] = field(default_factory=list)
