"""Transcript grading utilities."""
from __future__ import annotations
from typing import Optional


def compute_transcript_score(
    *,
    wer: float,
    multilingual_accuracy: float,
    speaker_f1: Optional[float] = None,
    multilingual_support: bool = True,
) -> float:
    """Return a 0-1000 score summarising transcript quality."""

    # Base weights
    phrase_weight = 0.8
    multilingual_weight = 0.1
    diarisation_weight = 0.1

    if speaker_f1 is None:
        # Reallocate diarisation weight to phrase accuracy
        phrase_weight += diarisation_weight
        diarisation_weight = 0.0

    phrase_score = max(0.0, 1.0 - wer) * (phrase_weight * 1000)
    multilingual_score = multilingual_accuracy * (multilingual_weight * 1000)
    speaker_score = (speaker_f1 or 0.0) * (diarisation_weight * 1000)

    overall = phrase_score + multilingual_score + speaker_score

    # Cap if multilingual accuracy is poor or system lacks support
    if not multilingual_support or multilingual_accuracy < 0.5:
        overall = min(overall, 500)

    # Clamp final score
    return max(0.0, min(1000.0, overall))
