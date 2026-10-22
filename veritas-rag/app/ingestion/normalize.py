"""Text normalization applied to every extracted document."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAIL_QUOTE_PREFIX = re.compile(r"^\s*(>\s?)+", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Normalize unicode, whitespace, and control characters.

    Applied uniformly so hashing, dedup, and indexing all see the same bytes
    for equivalent content.

    Args:
        text: Raw extracted text.

    Returns:
        Normalized text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def strip_email_quotes(text: str) -> str:
    """Remove quoted-reply prefixes ('>') from email bodies.

    Quoted history is a major source of near-duplicate content in email
    corpora; stripping it before dedup dramatically improves precision.
    """
    return _EMAIL_QUOTE_PREFIX.sub("", text)


def truncate_to_bytes(text: str, max_bytes: int = 1_000_000) -> str:
    """Truncate *text* so its UTF-8 encoding does not exceed *max_bytes*.

    Truncation happens on a character boundary, so no partial multi-byte
    sequence is ever emitted.

    Args:
        text: Input text.
        max_bytes: Maximum number of UTF-8 bytes to retain.

    Returns:
        Text whose UTF-8 encoding is at most *max_bytes* bytes.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
