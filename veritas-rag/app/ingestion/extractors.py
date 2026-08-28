"""Text extraction for PDF, email (.eml), HTML, and plain-text sources."""

from __future__ import annotations

import email
import email.policy
import logging
import re
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping script/style bodies."""

    _SKIP = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._parts)


def extract_pdf(data: bytes) -> tuple[list[str], dict[str, Any]]:
    """Extract per-page text from PDF bytes.

    Args:
        data: Raw PDF file bytes.

    Returns:
        Tuple of (list of per-page texts, metadata dict).
    """
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    meta: dict[str, Any] = {"page_count": len(pages)}
    if reader.metadata:
        for key in ("title", "author", "creation_date"):
            value = getattr(reader.metadata, key, None)
            if value:
                meta[key] = str(value)
    return pages, meta


def extract_email(data: bytes) -> tuple[list[str], dict[str, Any]]:
    """Extract body text and headers from an RFC 822 (.eml) message.

    Args:
        data: Raw .eml bytes.

    Returns:
        Tuple of ([body text], metadata with subject/from/to/date headers).
    """
    msg = email.message_from_bytes(data, policy=email.policy.default)
    meta: dict[str, Any] = {
        "subject": str(msg.get("Subject", "")),
        "from": str(msg.get("From", "")),
        "to": str(msg.get("To", "")),
        "date": str(msg.get("Date", "")),
    }

    body_part = msg.get_body(preferencelist=("plain", "html"))
    body = ""
    if body_part is not None:
        content = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            body = extract_html(content)
        else:
            body = content

    # Prepend subject so retrieval can match on it
    text = f"{meta['subject']}\n\n{body}".strip()
    return [text], meta


def extract_html(html: str) -> str:
    """Return visible text from an HTML string."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.text()


_MARKDOWN_SYNTAX = re.compile(r"[#*_`>\[\]()!-]{1,}")


def extract_text(data: bytes, source_type: str) -> tuple[list[str], dict[str, Any]]:
    """Dispatch extraction by source type.

    Args:
        data: Raw file bytes.
        source_type: pdf / email / html / markdown / text.

    Returns:
        Tuple of (per-page texts, metadata).

    Raises:
        ValueError: For unsupported source types.
    """
    if source_type == "pdf":
        return extract_pdf(data)
    if source_type == "email":
        return extract_email(data)
    if source_type == "html":
        return [extract_html(data.decode("utf-8", errors="replace"))], {}
    if source_type in ("text", "markdown"):
        return [data.decode("utf-8", errors="replace")], {}
    raise ValueError(f"Unsupported source type: {source_type}")


_EXTENSION_MAP = {
    ".pdf": "pdf",
    ".eml": "email",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".txt": "text",
}


def detect_source_type(filename: str) -> str:
    """Infer source type from a file extension (defaults to text)."""
    return _EXTENSION_MAP.get(Path(filename).suffix.lower(), "text")
