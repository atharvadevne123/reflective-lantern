"""Input validation helpers used across the Social-Pulse API."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_PRINTABLE_RE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e -�]")
_MAX_CONSECUTIVE_CHARS = 30


def sanitise_text(text: str) -> str:
    """Remove control characters and normalise Unicode in *text*.

    Args:
        text: Raw input text from the API request.

    Returns:
        Sanitised string safe for storage and feature extraction.

    Raises:
        ValueError: If *text* is blank after sanitisation.
    """
    normalised = unicodedata.normalize("NFC", text)
    cleaned = _PRINTABLE_RE.sub("", normalised).strip()
    if not cleaned:
        raise ValueError("text is empty after sanitisation")
    return cleaned


def validate_hashtag_consistency(text: str, hashtag_count: int) -> int:
    """Return the actual hashtag count, ignoring the client-provided value.

    This prevents inflated counts from bypassing the model's feature
    extraction.  The caller should use the returned value instead of the
    raw ``hashtag_count`` field.

    Args:
        text: Raw post text.
        hashtag_count: Client-provided hashtag count (used only as a hint).

    Returns:
        The number of #hashtag tokens found in *text*.
    """
    actual = len(re.findall(r"#\w+", text))
    return actual


def validate_mention_consistency(text: str, mention_count: int) -> int:
    """Return the actual @mention count found in *text*.

    Args:
        text: Raw post text.
        mention_count: Client-provided mention count (used only as a hint).

    Returns:
        The number of @mention tokens found in *text*.
    """
    return len(re.findall(r"@\w+", text))


def check_repetitive_content(text: str, threshold: int = _MAX_CONSECUTIVE_CHARS) -> bool:
    """Return True if *text* contains a suspicious run of identical characters.

    Consecutive character runs longer than *threshold* suggest spam or
    adversarial input that could distort TF-IDF features.

    Args:
        text: Raw post text.
        threshold: Maximum allowed consecutive identical character run.

    Returns:
        True if a repetitive run is detected.
    """
    current_char: str = ""
    run_length: int = 0
    for ch in text:
        if ch == current_char:
            run_length += 1
            if run_length >= threshold:
                return True
        else:
            current_char = ch
            run_length = 1
    return False


def validate_url(text: str) -> bool:
    """Return True if *text* contains at least one well-formed HTTP/HTTPS URL.

    Args:
        text: Raw string to scan.

    Returns:
        True when a URL starting with http:// or https:// is found.
    """
    return bool(re.search(r"https?://[^\s]{4,}", text))


def estimate_reading_time_seconds(text: str, wpm: int = 238) -> float:
    """Estimate how long it would take to read *text* at *wpm* words per minute.

    Args:
        text: Post content.
        wpm: Average reading speed in words per minute.

    Returns:
        Estimated reading duration in seconds (minimum 1 s).
    """
    word_count = max(1, len(text.split()))
    return max(1.0, word_count / wpm * 60.0)


def is_alpha_text(text: str) -> bool:
    """Return True if *text* contains at least one alphabetic character.

    Args:
        text: String to test.

    Returns:
        True when at least one letter (any script) is present.
    """
    return any(c.isalpha() for c in text)


def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*.

    Args:
        text: Input string.

    Returns:
        Number of words; 0 for an empty or whitespace-only string.
    """
    return len(text.split())


def validate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise a feature dict before inference.

    Sanitises the text field, corrects hashtag/mention counts, and
    bounds-checks numeric fields.

    Args:
        sample: Raw feature dict from the API request.

    Returns:
        Validated and potentially corrected feature dict.

    Raises:
        ValueError: If any field fails validation.
    """
    cleaned = dict(sample)
    cleaned["text"] = sanitise_text(str(sample.get("text", "")))
    cleaned["hashtag_count"] = validate_hashtag_consistency(
        cleaned["text"], int(sample.get("hashtag_count", 0))
    )
    cleaned["mention_count"] = validate_mention_consistency(
        cleaned["text"], int(sample.get("mention_count", 0))
    )
    cleaned["author_followers"] = max(0, int(sample.get("author_followers", 0)))
    cleaned["hour_of_day"] = max(0, min(23, int(sample.get("hour_of_day", 12))))
    cleaned["day_of_week"] = max(0, min(6, int(sample.get("day_of_week", 0))))
    return cleaned
