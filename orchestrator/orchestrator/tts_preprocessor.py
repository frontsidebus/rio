"""Aviation text preprocessor for TTS output.

Converts LLM output into clean, speakable text following ICAO phraseology
conventions. Handles flight levels, headings, frequencies, runway designators,
squawk codes, speeds, altimeter settings, distances, and general markdown cleanup.

Usage:
    from orchestrator.tts_preprocessor import preprocess_for_tts

    clean = preprocess_for_tts("Descend to FL350 and contact tower on 118.30")
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Digit-to-word mappings
# ---------------------------------------------------------------------------

_DIGIT_WORDS: dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "tree",
    "4": "four",
    "5": "fife",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "niner",
}

_DIGIT_WORDS_PLAIN: dict[str, str] = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# Standard English number words for natural readback (speeds, altitudes).
_ONES = [
    "", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Runway suffix mapping
_RWY_SUFFIX: dict[str, str] = {
    "L": " left",
    "R": " right",
    "C": " center",
}

# Aviation acronyms that TTS engines mangle if not expanded.
# Map uppercase acronym → spaced-out letters or spoken form.
_AVIATION_ACRONYMS: dict[str, str] = {
    "NOTAM": "NOTAM",  # already a pronounceable word
    "SIGMET": "SIGMET",  # already pronounceable
    "PIREP": "pilot report",
    "ATIS": "ATIS",  # already pronounceable
    "UNICOM": "UNICOM",  # already pronounceable
    "CTAF": "C TAF",
    "IFR": "I F R",
    "VFR": "V F R",
    "AGL": "A G L",
    "MSL": "M S L",
    "RNAV": "R NAV",
    "TCAS": "T CAS",
    "GPWS": "G P W S",
    "EGPWS": "E G P W S",
    "ELT": "E L T",
    "NDB": "N D B",
    "VOR": "V O R",
    "ILS": "I L S",
    "GPS": "G P S",
    "FMS": "F M S",
    "CDU": "C D U",
    "MDA": "M D A",
    "DH": "D H",
    "DA": "D A",
    "MEA": "M E A",
    "MOCA": "M O C A",
    "SID": "SID",  # pronounceable
    "STAR": "STAR",  # pronounceable
    "METAR": "METAR",  # pronounceable
    "TAF": "TAF",  # pronounceable
}


# ---------------------------------------------------------------------------
# Number helpers
# ---------------------------------------------------------------------------


def _digits_to_words(digits: str, *, aviation: bool = True) -> str:
    """Convert a string of digits to individual spoken words.

    Args:
        digits: A string of digit characters (e.g. "350").
        aviation: If True, use ICAO pronunciation (tree, fife, niner).
    """
    table = _DIGIT_WORDS if aviation else _DIGIT_WORDS_PLAIN
    return " ".join(table[d] for d in digits if d in table)


def _number_to_words(n: int) -> str:
    """Convert an integer (0-99999) to spoken English words.

    Used for altitudes and speeds where natural number readback is expected.
    """
    if n < 0:
        return "minus " + _number_to_words(-n)
    if n == 0:
        return "zero"

    parts: list[str] = []
    remaining = n

    if remaining >= 1000:
        thousands = remaining // 1000
        if thousands >= 20:
            parts.append(_TENS[thousands // 10])
            if thousands % 10:
                parts.append(_ONES[thousands % 10])
        elif thousands >= 10:
            parts.append(_ONES[thousands])
        else:
            parts.append(_ONES[thousands])
        parts.append("thousand")
        remaining %= 1000

    if remaining >= 100:
        parts.append(_ONES[remaining // 100])
        parts.append("hundred")
        remaining %= 100

    if remaining >= 20:
        parts.append(_TENS[remaining // 10])
        remaining %= 10

    if remaining > 0:
        parts.append(_ONES[remaining])

    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Aviation-specific transformers
# ---------------------------------------------------------------------------


def _expand_flight_level(text: str) -> str:
    """FL350 → flight level tree fife zero."""

    def _repl(m: re.Match[str]) -> str:
        digits = m.group(1)
        return "flight level " + _digits_to_words(digits)

    return re.sub(r"\bFL\s*(\d{2,3})\b", _repl, text)


def _expand_altitude(text: str) -> str:
    """3500ft or 3,500 feet → three thousand five hundred feet."""

    def _repl(m: re.Match[str]) -> str:
        raw = m.group(1).replace(",", "")
        n = int(raw)
        unit = m.group(2).strip().lower()
        spoken_unit = "feet" if unit in ("ft", "feet") else unit
        return _number_to_words(n) + " " + spoken_unit

    return re.sub(
        r"\b(\d{1,3}(?:,?\d{3})?)\s*(ft|feet)\b",
        _repl,
        text,
        flags=re.IGNORECASE,
    )


def _expand_heading(text: str) -> str:
    """HDG 270 or heading 090 → heading two seven zero."""

    def _repl(m: re.Match[str]) -> str:
        digits = m.group(1)
        return "heading " + _digits_to_words(digits)

    return re.sub(
        r"\b(?:HDG|heading)\s+(\d{3})\b",
        _repl,
        text,
        flags=re.IGNORECASE,
    )


def _expand_frequency(text: str) -> str:
    """Expand radio frequencies read digit-by-digit with 'point'.

    Matches patterns like 121.7, 118.30 when preceded by frequency-context
    words or standing alone in a likely frequency range (108-137 MHz comm/nav).
    """

    def _repl(m: re.Match[str]) -> str:
        prefix = m.group(1) or ""
        integer_part = m.group(2)
        decimal_part = m.group(3)
        spoken = (
            _digits_to_words(integer_part)
            + " point "
            + _digits_to_words(decimal_part)
        )
        if prefix:
            return prefix + spoken
        return spoken

    # Context-triggered: preceded by frequency-related words
    text = re.sub(
        r"\b((?:on|frequency|freq|contact|monitor|tune|tower|ground|approach|"
        r"departure|center|unicom|CTAF|ATIS|clearance)\s+)"
        r"(\d{2,3})\.(\d{1,3})\b",
        _repl,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _expand_runway(text: str) -> str:
    """RWY 27L → runway two seven left."""

    def _repl(m: re.Match[str]) -> str:
        digits = m.group(1)
        suffix = m.group(2).upper() if m.group(2) else ""
        spoken = "runway " + _digits_to_words(digits)
        if suffix in _RWY_SUFFIX:
            spoken += _RWY_SUFFIX[suffix]
        return spoken

    return re.sub(
        r"\b(?:RWY|runway)\s*(\d{2})([LRC])?\b",
        _repl,
        text,
        flags=re.IGNORECASE,
    )


def _expand_squawk(text: str) -> str:
    """squawk 7700 → squawk seven seven zero zero."""

    def _repl(m: re.Match[str]) -> str:
        digits = m.group(1)
        return "squawk " + _digits_to_words(digits)

    return re.sub(
        r"\bsquawk\s+(\d{4})\b",
        _repl,
        text,
        flags=re.IGNORECASE,
    )


def _expand_speed(text: str) -> str:
    """250kt → two hundred fifty knots; V1/Vr/V2 → spoken form."""
    # V-speeds first (before general number handling)
    text = re.sub(r"\bV1\b", "V one", text)
    text = re.sub(r"\bVr\b", "V R", text)
    text = re.sub(r"\bV2\b", "V two", text)

    def _repl(m: re.Match[str]) -> str:
        raw = m.group(1).replace(",", "")
        n = int(raw)
        unit = m.group(2).strip().lower()
        spoken_unit = "knots" if unit in ("kt", "kts", "knots", "knot") else unit
        return _number_to_words(n) + " " + spoken_unit

    text = re.sub(
        r"\b(\d{1,3}(?:,\d{3})?)\s*(kt|kts|knots?)\b",
        _repl,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _expand_qnh(text: str) -> str:
    """QNH 1013 → Q N H one zero one tree; 29.92 inHg → two niner niner two inches."""

    def _repl_qnh(m: re.Match[str]) -> str:
        digits = m.group(1)
        return "Q N H " + _digits_to_words(digits)

    text = re.sub(r"\bQNH\s+(\d{4})\b", _repl_qnh, text, flags=re.IGNORECASE)

    def _repl_inhg(m: re.Match[str]) -> str:
        integer_part = m.group(1)
        decimal_part = m.group(2)
        return _digits_to_words(integer_part + decimal_part) + " inches"

    text = re.sub(
        r"\b(\d{2})\.(\d{2})\s*inHg\b",
        _repl_inhg,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _expand_aviation_acronyms(text: str) -> str:
    """Expand aviation acronyms that TTS engines mispronounce.

    Only expands standalone uppercase acronyms (word boundaries) to avoid
    mangling words that happen to contain the same letters.
    """
    for acronym, spoken in _AVIATION_ACRONYMS.items():
        if acronym != spoken:
            text = re.sub(rf"\b{acronym}\b", spoken, text)
    return text


def _expand_distance(text: str) -> str:
    """5nm → five nautical miles; DME 12.3 → D M E one two point tree."""
    # NM / nm distances
    def _repl_nm(m: re.Match[str]) -> str:
        raw = m.group(1)
        n = int(raw)
        suffix = "nautical mile" if n == 1 else "nautical miles"
        return _number_to_words(n) + " " + suffix

    text = re.sub(r"\b(\d+)\s*(?:NM|nm)\b", _repl_nm, text)

    # DME readings
    def _repl_dme(m: re.Match[str]) -> str:
        integer_part = m.group(1)
        decimal_part = m.group(2)
        spoken = "D M E " + _digits_to_words(integer_part)
        if decimal_part:
            spoken += " point " + _digits_to_words(decimal_part)
        return spoken

    text = re.sub(
        r"\bDME\s+(\d+)(?:\.(\d+))?\b",
        _repl_dme,
        text,
        flags=re.IGNORECASE,
    )
    return text


# ---------------------------------------------------------------------------
# Markdown and general cleanup (ported from server.py _sanitize_for_tts)
# ---------------------------------------------------------------------------


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting, preserving the underlying text."""
    # Code blocks (``` ... ```) → just the content
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)

    # Inline code `text` → just text
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Markdown links [text](url) → just the link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Bold+italic ***text*** or ___text___
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)

    # Bold **text** or __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)

    # Italic *text* or _text_ (not mid-word underscores like pre_flight)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    # Strikethrough ~~text~~
    text = re.sub(r"~~(.+?)~~", r"\1", text)

    # Headings: ### text → text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Blockquotes: > text → text
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # Horizontal rules (---, ***, ___) → pause
    text = re.sub(r"^[-*_]{3,}\s*$", ".", text, flags=re.MULTILINE)

    # Bullet points (-, *, bullet) at line start → natural pause
    text = re.sub(r"^\s*[-*\u2022]\s+", ". ", text, flags=re.MULTILINE)

    # Numbered lists: 1. or 1) → natural pause
    text = re.sub(r"^\s*\d+[.)]\s+", ". ", text, flags=re.MULTILINE)

    # Any remaining stray asterisks
    text = text.replace("*", "")

    return text


def _replace_special_chars(text: str) -> str:
    """Convert special characters to speakable equivalents."""
    text = text.replace("\u2014", ", ")      # em dash
    text = text.replace("\u2013", " to ")    # en dash
    text = text.replace("\u2026", "...")      # ellipsis
    text = text.replace("\u00b0", " degrees")  # degree sign
    text = text.replace("\u00b1", " plus or minus ")  # plus-minus
    text = text.replace("&", " and ")
    text = text.replace("|", ", ")
    text = text.replace("~", "approximately ")

    # Slash: preserve in frequencies like 121.7/118.3, expand otherwise
    text = re.sub(r"(\d)\s*/\s*(\d)", r"\1 slash \2", text)
    text = re.sub(r"(?<!\d)/(?!\d)", " ", text)

    return text


def _clean_whitespace(text: str) -> str:
    """Collapse excess whitespace and normalize sentence breaks."""
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"[.,]{2,}", ".", text)
    text = re.sub(r"\.\s*,", ".", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_for_tts(text: str) -> str:
    """Convert LLM output into clean, speakable text for TTS engines.

    Applies aviation-specific transformations (flight levels, headings,
    frequencies, runway designators, squawk codes, speeds, altimeter
    settings, distances) followed by markdown stripping and whitespace
    cleanup.

    Args:
        text: Raw text from the LLM, possibly containing markdown and
              aviation shorthand.

    Returns:
        Plain speakable text suitable for ElevenLabs or similar TTS.
    """
    if not text:
        return ""

    # --- Aviation transformations (order matters) ---
    # Flight levels before general altitude (FL350 vs 3500ft)
    text = _expand_flight_level(text)
    # Headings before general number handling
    text = _expand_heading(text)
    # Squawk before general digit handling
    text = _expand_squawk(text)
    # QNH / altimeter before general number handling
    text = _expand_qnh(text)
    # Frequencies (context-sensitive)
    text = _expand_frequency(text)
    # DME / distance before stripping NM
    text = _expand_distance(text)
    # Runway designators
    text = _expand_runway(text)
    # Speeds (including V-speeds)
    text = _expand_speed(text)
    # Altitudes (3500ft, 3,500 feet)
    text = _expand_altitude(text)

    # Aviation acronyms (after specific patterns to avoid interfering)
    text = _expand_aviation_acronyms(text)

    # --- General cleanup ---
    text = _strip_markdown(text)
    text = _replace_special_chars(text)
    text = _clean_whitespace(text)

    return text
