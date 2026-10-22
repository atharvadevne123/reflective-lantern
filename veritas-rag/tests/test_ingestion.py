"""Stage 1 tests: extraction, normalization, dedup, chunking, versioning."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.dedup import (
    DuplicateDetector,
    content_hash,
    minhash_signature,
    signature_similarity,
)
from app.ingestion.extractors import detect_source_type, extract_email, extract_html
from app.ingestion.normalize import normalize_text, strip_email_quotes, truncate_to_bytes
from app.ingestion.versioning import VersionStore, make_doc_id
from app.schemas import Document


class TestNormalize:
    def test_collapses_whitespace(self) -> None:
        assert normalize_text("a   b\t\tc") == "a b c"

    def test_normalizes_unicode(self) -> None:
        # Full-width characters fold to ASCII under NFKC
        assert normalize_text("ｈｅｌｌｏ") == "hello"

    def test_strips_control_chars(self) -> None:
        assert normalize_text("a\x00b\x08c") == "abc"

    def test_collapses_newlines(self) -> None:
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_strip_email_quotes(self) -> None:
        body = "Thanks!\n> quoted line one\n>> nested quote\nRegards"
        cleaned = strip_email_quotes(body)
        assert "> " not in cleaned
        assert "Thanks!" in cleaned and "Regards" in cleaned


class TestExtractors:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("report.pdf", "pdf"),
            ("mail.eml", "email"),
            ("page.html", "html"),
            ("notes.md", "markdown"),
            ("data.txt", "text"),
            ("unknown.xyz", "text"),
        ],
    )
    def test_detect_source_type(self, filename, expected) -> None:
        assert detect_source_type(filename) == expected

    def test_extract_html_drops_script(self) -> None:
        html = "<html><script>evil()</script><body><p>Visible text</p></body></html>"
        assert extract_html(html) == "Visible text"

    def test_extract_email_body_and_headers(self) -> None:
        raw = (
            b"From: alice@example.com\r\n"
            b"To: bob@example.com\r\n"
            b"Subject: Quarterly report\r\n"
            b"Date: Mon, 01 Jul 2024 10:00:00 +0000\r\n"
            b"Content-Type: text/plain\r\n\r\n"
            b"The quarterly numbers are attached.\r\n"
        )
        pages, meta = extract_email(raw)
        assert meta["subject"] == "Quarterly report"
        assert meta["from"] == "alice@example.com"
        assert "quarterly numbers" in pages[0]
        assert "Quarterly report" in pages[0]  # subject prepended for retrieval

    def test_extract_pdf_roundtrip(self) -> None:
        import io

        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)

        from app.ingestion.extractors import extract_pdf

        pages, meta = extract_pdf(buf.getvalue())
        assert meta["page_count"] == 1
        assert len(pages) == 1


class TestDedup:
    def test_exact_duplicate_detected(self) -> None:
        det = DuplicateDetector()
        det.register("doc-1", "the quick brown fox jumps over the lazy dog")
        dup, sim = det.check("the quick brown fox jumps over the lazy dog")
        assert dup == "doc-1"
        assert sim == 1.0

    def test_near_duplicate_detected(self) -> None:
        det = DuplicateDetector(near_threshold=0.6)
        base = " ".join(f"sentence number {i} about solar policy rebates" for i in range(30))
        det.register("doc-1", base)
        variant = base + " one extra trailing clause"
        dup, sim = det.check(variant)
        assert dup == "doc-1"
        assert sim >= 0.6

    def test_distinct_documents_pass(self) -> None:
        det = DuplicateDetector()
        det.register("doc-1", "alpha beta gamma delta epsilon zeta eta theta")
        dup, _ = det.check("completely different content about turnips and weather")
        assert dup is None

    def test_signature_similarity_bounds(self) -> None:
        sig = minhash_signature("hello world example text")
        assert signature_similarity(sig, sig) == 1.0

    def test_content_hash_stability(self) -> None:
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")


class TestChunker:
    def _doc(self, pages: list[str]) -> Document:
        return Document(
            doc_id="doc-test-v1",
            source="test.txt",
            source_type="text",
            text="\n".join(pages),
            pages=pages,
            created_at=datetime.now(UTC).isoformat(),
        )

    def test_single_small_page_one_chunk(self) -> None:
        chunks = chunk_document(self._doc(["One short sentence."]))
        assert len(chunks) == 1
        assert chunks[0].page == 1

    def test_long_page_multiple_chunks_with_overlap(self) -> None:
        sentences = " ".join(f"This is sentence number {i} in a long document." for i in range(60))
        chunks = chunk_document(self._doc([sentences]), chunk_size=300, overlap=100)
        assert len(chunks) > 3
        # Overlap: some sentence from chunk N also appears in chunk N+1
        assert any(
            c1.text.split(". ")[-1] in c2.text for c1, c2 in zip(chunks, chunks[1:], strict=False)
        )

    def test_page_numbers_tracked(self) -> None:
        chunks = chunk_document(self._doc(["Page one text here.", "Page two text here."]))
        assert {c.page for c in chunks} == {1, 2}

    def test_chunk_ids_unique(self) -> None:
        sentences = " ".join(f"Sentence {i}." for i in range(100))
        chunks = chunk_document(self._doc([sentences, sentences]), chunk_size=200)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestVersioning:
    def _doc(self, source: str, version: int, digest: str) -> Document:
        return Document(
            doc_id=make_doc_id(source, version),
            source=source,
            source_type="text",
            text="x",
            pages=["x"],
            created_at=datetime.now(UTC).isoformat(),
            version=version,
            content_hash=digest,
        )

    def test_version_numbers_increment(self) -> None:
        store = VersionStore()
        assert store.next_version("a.txt") == 1
        store.add(self._doc("a.txt", 1, "h1"))
        assert store.next_version("a.txt") == 2

    def test_unchanged_detection(self) -> None:
        store = VersionStore()
        store.add(self._doc("a.txt", 1, "h1"))
        assert store.is_unchanged("a.txt", "h1")
        assert not store.is_unchanged("a.txt", "h2")

    def test_superseded_ids(self) -> None:
        store = VersionStore()
        store.add(self._doc("a.txt", 1, "h1"))
        store.add(self._doc("a.txt", 2, "h2"))
        assert store.superseded_doc_ids("a.txt") == [make_doc_id("a.txt", 1)]

    def test_history_order(self) -> None:
        store = VersionStore()
        store.add(self._doc("a.txt", 1, "h1"))
        store.add(self._doc("a.txt", 2, "h2"))
        assert [d.version for d in store.history("a.txt")] == [1, 2]


class TestPipelineIngest:
    def test_ingest_reports_chunks(self, empty_pipeline) -> None:
        report = empty_pipeline.ingest(b"Solar rebates pay 900 dollars per kilowatt.", "a.txt")
        assert report["status"] == "ingested"
        assert report["chunks"] >= 1

    def test_duplicate_skipped(self, empty_pipeline) -> None:
        text = b"The battery incentive is capped at 3500 dollars per property."
        empty_pipeline.ingest(text, "orig.txt")
        report = empty_pipeline.ingest(text, "copy.txt")
        assert report["status"] == "duplicate"
        assert empty_pipeline.stats["duplicates_skipped"] == 1

    def test_unchanged_reingest_is_noop(self, empty_pipeline) -> None:
        text = b"Grid export limits are reviewed annually."
        first = empty_pipeline.ingest(text, "a.txt")
        second = empty_pipeline.ingest(text, "a.txt")
        assert second["status"] == "unchanged"
        assert second["doc_id"] == first["doc_id"]

    def test_new_version_supersedes_old(self, empty_pipeline) -> None:
        empty_pipeline.ingest(b"The rebate is 900 dollars per kilowatt.", "policy.txt")
        report = empty_pipeline.ingest(b"The rebate is 750 dollars per kilowatt.", "policy.txt")
        assert report["status"] == "ingested"
        assert report["version"] == 2
        # Old version's chunks removed from the live index
        docs = {c.doc_id for c in empty_pipeline.chunks.values()}
        assert all(d.endswith("-v2") for d in docs)

    def test_empty_document_rejected(self, empty_pipeline) -> None:
        report = empty_pipeline.ingest(b"   \n  ", "blank.txt")
        assert report["status"] == "empty"


class TestIngestionEdgeCases:
    def test_multiple_documents_indexed(self, empty_pipeline) -> None:
        """Ingesting several distinct documents indexes all of them."""
        texts = [
            b"Alpha document about solar policy.",
            b"Beta document about battery storage.",
            b"Gamma document about grid connections.",
        ]
        for i, text in enumerate(texts):
            report = empty_pipeline.ingest(text, f"doc_{i}.txt")
            assert report["status"] == "ingested"
        assert len(empty_pipeline.versions) == 3

    def test_ingest_with_explicit_created_at(self, empty_pipeline) -> None:
        """created_at is stored with the document and used in trust scoring."""
        ts = "2024-06-01T00:00:00+00:00"
        report = empty_pipeline.ingest(b"Dated document content here.", "dated.txt", created_at=ts)
        assert report["status"] == "ingested"
        doc_id = report["doc_id"]
        chunks = [c for c in empty_pipeline.chunks.values() if c.doc_id == doc_id]
        assert all(c.created_at == ts for c in chunks)

    def test_ingest_large_document_creates_multiple_chunks(self, empty_pipeline) -> None:
        """A document larger than chunk_size is split into multiple chunks."""
        long_text = b"This is a complete sentence about solar energy policy. " * 30
        report = empty_pipeline.ingest(long_text, "long.txt")
        assert report["status"] == "ingested"
        assert report["chunks"] > 1

    def test_version_chain_grows_on_update(self, empty_pipeline) -> None:
        """Re-ingesting an updated source increments the version counter."""
        for v, text in enumerate(
            [
                b"Version one content for the policy document.",
                b"Version two content for the policy document.",
                b"Version three content for the policy document.",
            ],
            start=1,
        ):
            report = empty_pipeline.ingest(text, "policy.txt")
            assert report["version"] == v

    def test_metrics_reflect_ingested_counts(self, empty_pipeline) -> None:
        """pipeline.metrics() documents count matches ingested files."""
        empty_pipeline.ingest(b"First document content sentence.", "a.txt")
        empty_pipeline.ingest(b"Second document content sentence.", "b.txt")
        m = empty_pipeline.metrics()
        assert m["documents"] == 2
        assert m["ingested"] == 2
        assert m["chunks"] >= 2


class TestTruncateToBytes:
    def test_short_text_unchanged(self) -> None:
        text = "hello world"
        assert truncate_to_bytes(text, max_bytes=1000) == text

    def test_truncates_to_limit(self) -> None:
        text = "a" * 200
        result = truncate_to_bytes(text, max_bytes=100)
        assert len(result.encode("utf-8")) <= 100

    def test_exact_limit_unchanged(self) -> None:
        text = "x" * 50
        assert truncate_to_bytes(text, max_bytes=50) == text

    def test_multibyte_truncation_is_safe(self) -> None:
        # "é" encodes as 2 bytes; truncating mid-char must not produce garbage
        text = "é" * 100
        result = truncate_to_bytes(text, max_bytes=101)
        result.encode("utf-8")  # must not raise UnicodeEncodeError

    def test_empty_string(self) -> None:
        assert truncate_to_bytes("", max_bytes=10) == ""
