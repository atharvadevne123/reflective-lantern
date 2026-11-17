"""Tests for veritas-rag ingestion extractors module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.extractors import detect_source_type, extract_html, extract_text


def test_extract_html_returns_visible_text():
    html = "<html><body><p>Hello world</p></body></html>"
    result = extract_html(html)
    assert "Hello world" in result


def test_extract_html_skips_script_tags():
    html = "<html><body><script>alert('x')</script><p>Visible</p></body></html>"
    result = extract_html(html)
    assert "Visible" in result
    assert "alert" not in result


def test_extract_html_skips_style_tags():
    html = "<html><head><style>body{color:red}</style></head><body><p>Content</p></body></html>"
    result = extract_html(html)
    assert "Content" in result
    assert "color:red" not in result


def test_extract_html_empty_string():
    result = extract_html("")
    assert isinstance(result, str)


def test_extract_text_plain_text():
    data = b"Hello plain text content."
    pages, meta = extract_text(data, "text")
    assert isinstance(pages, list)
    assert len(pages) == 1
    assert "Hello plain text content." in pages[0]


def test_extract_text_markdown():
    data = b"# Heading\n\nSome content here."
    pages, meta = extract_text(data, "markdown")
    assert isinstance(pages, list)
    assert len(pages) == 1


def test_extract_text_html():
    data = b"<p>HTML content</p>"
    pages, meta = extract_text(data, "html")
    assert isinstance(pages, list)
    assert "HTML content" in pages[0]


def test_extract_text_unsupported_type_raises():
    with pytest.raises((ValueError, KeyError, Exception)):
        extract_text(b"data", "unknown_type_xyz")


def test_detect_source_type_pdf():
    assert detect_source_type("document.pdf") == "pdf"


def test_detect_source_type_html():
    assert detect_source_type("page.html") in ("html", "text")


def test_detect_source_type_txt():
    result = detect_source_type("readme.txt")
    assert isinstance(result, str)


def test_detect_source_type_eml():
    result = detect_source_type("message.eml")
    assert isinstance(result, str)


def test_detect_source_type_unknown_defaults_to_text():
    result = detect_source_type("file.xyz123unknown")
    assert result == "text"


def test_detect_source_type_case_insensitive():
    result = detect_source_type("DOC.PDF")
    assert result == "pdf"


@pytest.mark.parametrize("ext,expected", [
    ("file.pdf", "pdf"),
    ("file.txt", "text"),
    ("file.md", "markdown"),
])
def test_detect_source_type_common_extensions(ext, expected):
    result = detect_source_type(ext)
    assert result == expected
