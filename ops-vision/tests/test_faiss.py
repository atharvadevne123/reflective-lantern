"""Tests for Ops-Vision FAISS runbook index."""

import pytest

from app.faiss_index import Runbook, RunbookIndex

SAMPLE_RUNBOOKS = [
    Runbook(
        title="High CPU Mitigation",
        content="Identify top CPU consumers with top -c. Restart runaway processes. Scale horizontally.",
        category="cpu",
    ),
    Runbook(
        title="Memory Leak Remediation",
        content="Take heap dump. Restart service. Increase JVM memory if needed. Check for leak with profiler.",
        category="memory",
    ),
    Runbook(
        title="Network Latency Investigation",
        content="Check DNS resolution times. Inspect TCP retransmissions. Verify load balancer health.",
        category="network",
    ),
    Runbook(
        title="Disk I/O Saturation",
        content="Identify I/O-heavy processes with iotop. Move logs to separate volume. Tune fsync frequency.",
        category="disk",
    ),
    Runbook(
        title="Database Connection Pool Exhaustion",
        content="Increase max pool size. Kill idle connections. Check for connection leaks in application code.",
        category="database",
    ),
]


class TestRunbookIndexBuild:
    """Tests for RunbookIndex.build()."""

    def test_build_stores_runbooks(self):
        """After build(), the index holds all provided runbooks."""
        index = RunbookIndex()
        index.build(SAMPLE_RUNBOOKS)
        assert len(index.runbooks) == len(SAMPLE_RUNBOOKS)

    def test_build_creates_vocab(self):
        """build() populates the vocabulary."""
        index = RunbookIndex()
        index.build(SAMPLE_RUNBOOKS)
        assert len(index._vocab) > 0

    def test_build_creates_embeddings(self):
        """build() creates embedding matrix."""
        index = RunbookIndex()
        index.build(SAMPLE_RUNBOOKS)
        assert index._embeddings is not None
        assert index._embeddings.shape[0] == len(SAMPLE_RUNBOOKS)


class TestRunbookIndexSearch:
    """Tests for RunbookIndex.search()."""

    @pytest.fixture
    def built_index(self) -> RunbookIndex:
        """Return a RunbookIndex pre-built with sample runbooks."""
        index = RunbookIndex()
        index.build(SAMPLE_RUNBOOKS)
        return index

    def test_search_returns_list(self, built_index):
        """search() returns a list."""
        results = built_index.search("high cpu usage", top_k=3)
        assert isinstance(results, list)

    def test_search_returns_top_k(self, built_index):
        """search() returns at most top_k results."""
        results = built_index.search("memory leak", top_k=2)
        assert len(results) <= 2

    def test_search_result_is_runbook_score_tuple(self, built_index):
        """Each result is a (Runbook, float) tuple."""
        results = built_index.search("disk io", top_k=1)
        assert len(results) >= 1
        runbook, score = results[0]
        assert isinstance(runbook, Runbook)
        assert isinstance(score, float)

    def test_search_empty_index(self):
        """search() on an empty index returns empty list."""
        index = RunbookIndex()
        results = index.search("cpu", top_k=3)
        assert results == []

    def test_search_cpu_returns_cpu_runbook(self, built_index):
        """Searching 'cpu usage high' should surface the CPU runbook."""
        results = built_index.search("cpu usage high", top_k=3)
        titles = [r.title for r, _ in results]
        assert any("CPU" in t or "cpu" in t.lower() for t in titles)

    @pytest.mark.parametrize(
        "query,category",
        [
            ("cpu high consumer", "cpu"),
            ("memory heap profiler", "memory"),
            ("network dns latency", "network"),
            ("disk io iotop", "disk"),
            ("database pool connection", "database"),
        ],
    )
    def test_search_category_relevance(self, built_index, query, category):
        """Top result for each query should match the expected category."""
        results = built_index.search(query, top_k=1)
        if results:
            top_runbook, _ = results[0]
            assert top_runbook.category == category


class TestRunbookIndexPersistence:
    """Tests for RunbookIndex save/load roundtrip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved index can be reloaded and produces search results."""
        index = RunbookIndex()
        index.build(SAMPLE_RUNBOOKS)
        save_path = tmp_path / "runbook_index.json"
        index.save(save_path)

        loaded = RunbookIndex.load(save_path)
        results = loaded.search("cpu memory", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_saved_runbooks_preserved(self, tmp_path):
        """Loaded index has the same number of runbooks as original."""
        index = RunbookIndex()
        index.build(SAMPLE_RUNBOOKS)
        path = tmp_path / "test.json"
        index.save(path)
        loaded = RunbookIndex.load(path)
        assert len(loaded.runbooks) == len(SAMPLE_RUNBOOKS)
