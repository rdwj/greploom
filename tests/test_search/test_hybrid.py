from __future__ import annotations

import pytest

from greploom.index.store import SearchResult
from greploom.search.hybrid import SearchHit, hybrid_search  # noqa: E402

# ---------------------------------------------------------------------------
# Fake store
# ---------------------------------------------------------------------------


class FakeStore:
    def __init__(self, vec_results: list[SearchResult], bm25_results: list[SearchResult]) -> None:
        self._vec = vec_results
        self._bm25 = bm25_results

    def vector_search(self, query_embedding: list[float], limit: int = 10) -> list[SearchResult]:
        return self._vec[:limit]

    def bm25_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return self._bm25[:limit]


def sr(node_id: str, score: float = -0.1) -> SearchResult:
    return SearchResult(
        node_id=node_id,
        score=score,
        name=f"name_{node_id}",
        file=f"{node_id}.py",
        line=1,
        summary=f"summary_{node_id}",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_both_empty():
    store = FakeStore([], [])
    result = hybrid_search("foo", [0.1], store)
    assert result == []


@pytest.mark.parametrize(
    "vec,bm25",
    [
        ([sr("a"), sr("b")], []),
        ([], [sr("a"), sr("b")]),
    ],
)
def test_single_source_empty(vec, bm25):
    store = FakeStore(vec, bm25)
    hits = hybrid_search("foo", [0.1], store)
    assert len(hits) == 2
    assert {h.node_id for h in hits} == {"a", "b"}


def test_top_k_limits_results():
    nodes = [sr(str(i)) for i in range(20)]
    store = FakeStore(nodes, [])
    hits = hybrid_search("foo", [0.1], store, top_k=5)
    assert len(hits) == 5


def test_shared_nodes_score_higher_than_single_list():
    """A node in both lists should outscore nodes in only one list."""
    shared = sr("shared")
    vec_only = sr("vec_only")
    bm25_only = sr("bm25_only")

    store = FakeStore(
        vec_results=[shared, vec_only],
        bm25_results=[shared, bm25_only],
    )
    hits = hybrid_search("foo", [0.1], store, top_k=10)
    by_id = {h.node_id: h.score for h in hits}

    assert by_id["shared"] > by_id["vec_only"]
    assert by_id["shared"] > by_id["bm25_only"]


def test_rrf_fusion_ordering():
    """Verify RRF scores are consistent with manual calculation."""
    # vec: [a, b], bm25: [b, a]
    # RRF(a): 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # RRF(b): 1/(60+2) + 1/(60+1) = same — they tie, both appear in both lists at rank 1 and 2
    store = FakeStore(
        vec_results=[sr("a"), sr("b")],
        bm25_results=[sr("b"), sr("a")],
    )
    hits = hybrid_search("foo", [0.1], store, rrf_k=60)
    # Both score identically; just verify both present and scores sum correctly
    by_id = {h.node_id: h.score for h in hits}
    expected_a = 1 / (60 + 1) + 1 / (60 + 2)
    expected_b = 1 / (60 + 2) + 1 / (60 + 1)
    assert by_id["a"] == pytest.approx(expected_a)
    assert by_id["b"] == pytest.approx(expected_b)


def test_rrf_k_affects_scores():
    """Higher rrf_k → smaller individual scores."""
    store = FakeStore([sr("a")], [])
    hit_low_k = hybrid_search("foo", [0.1], store, rrf_k=1)[0]
    hit_high_k = hybrid_search("foo", [0.1], store, rrf_k=1000)[0]
    assert hit_low_k.score > hit_high_k.score


def test_metadata_from_higher_ranked_source():
    """When a node appears in both lists, metadata comes from the better-ranked source."""
    # In vec_results 'x' is rank 1; in bm25_results 'x' is rank 2.
    # Rank 1 is better, so we expect metadata from the vec result.
    x_vec = SearchResult(
        node_id="x", score=-0.1, name="x_vec", file="vec.py", line=10, summary="vec"
    )
    x_bm25 = SearchResult(
        node_id="x", score=-5.0, name="x_bm25", file="bm25.py", line=99, summary="bm25"
    )
    padding = sr("pad")

    store = FakeStore(
        vec_results=[x_vec, padding],
        bm25_results=[padding, x_bm25],
    )
    hits = hybrid_search("foo", [0.1], store)
    x_hit = next(h for h in hits if h.node_id == "x")
    assert x_hit.name == "x_vec"
    assert x_hit.file == "vec.py"


def test_returns_search_hit_type():
    store = FakeStore([sr("a")], [sr("b")])
    hits = hybrid_search("foo", [0.1], store)
    assert all(isinstance(h, SearchHit) for h in hits)
