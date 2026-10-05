"""Provider-neutral STT contract."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Transcript:
    text: str


class STTUnavailable(RuntimeError):
    """Speech recognition is unavailable for this utterance."""


class SpeechToText(Protocol):
    def transcribe(self, pcm: np.ndarray) -> Transcript:
        ...
