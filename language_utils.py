import re
from typing import List

# Minimal Spanish word list for heuristics. This is intentionally short and
# only meant to approximate real language identification without external
# models.  The words were chosen from common stop words and fillers that are
# unlikely to appear in German.
SPANISH_WORDS = {
    "el",
    "la",
    "que",
    "de",
    "y",
    "a",
    "en",
    "un",
    "ser",
    "se",
    "no",
    "me",
    "lo",
    "tener",
    "hacer",
    "esque",
    "porque",
    "tengo",
    "mi",
    "quien",
    "por",
    "este",
    "para",
    "con",
    "pero",
    "como",
    "esta",
    "usted",
    "yo",
    "hola",
    "gracias",
    "muy",
    "bien",
    "donde",
    "cuando",
    "quien",
    "dime",
    "porque",
    "perdona",
    "pues",
    "entonces",
}

ACCENT_RE = re.compile(r"[áéíóúñ¡¿]")


def detect_language(sentence: str) -> str:
    """Return ``"es"`` if the sentence appears to be Spanish.

    The heuristic checks for two signals:
    1. Proportion of tokens present in ``SPANISH_WORDS``.
    2. Presence of typical Spanish accented characters (``áéíóúñ``).
    """

    tokens = re.findall(r"[\w']+", sentence.lower())
    if not tokens:
        return "unknown"

    spanish_hits = sum(token in SPANISH_WORDS for token in tokens)
    ratio = spanish_hits / len(tokens)
    has_accent = bool(ACCENT_RE.search(sentence))

    if ratio >= 0.2 or (spanish_hits >= 2 and has_accent) or has_accent:
        return "es"
    return "other"


def identify_spanish_lines(lines: List[str]) -> List[int]:
    """Return indices of lines identified as Spanish."""
    return [i for i, line in enumerate(lines) if detect_language(line) == "es"]

