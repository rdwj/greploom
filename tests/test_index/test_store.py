from __future__ import annotations

from greploom.index.store import IndexStore, NodeRecord, SearchResult

DIM = 4


def _vec(v: list[float]) -> list[float]:
    """Pad or return a DIM-length float vector."""
    assert len(v) == DIM
    return v


def _store() -> IndexStore:
    return IndexStore(db_path=":memory:", embedding_dim=DIM)


def _insert_node(store: IndexStore, node_id: str = "n1", **overrides: object) -> bool:
    defaults = dict(
        kind="function",
        name="my_func",
        file="src/foo.py",
        line=10,
        summary="does something useful",
        content_hash="abc123",
        embedding=_vec([0.1, 0.2, 0.3, 0.4]),
    )
    defaults.update(overrides)
    return store.upsert(node_id=node_id, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema / upsert
# ---------------------------------------------------------------------------


def test_upsert_new_node_returns_true() -> None:
    with _store() as store:
        assert _insert_node(store) is True


def test_get_content_hash_after_upsert() -> None:
    with _store() as store:
        _insert_node(store, content_hash="deadbeef")
        assert store.get_content_hash("n1") == "deadbeef"


def test_get_content_hash_missing_node_returns_none() -> None:
    with _store() as store:
        assert store.get_content_hash("nonexistent") is None


def test_upsert_same_hash_returns_false() -> None:
    with _store() as store:
        _insert_node(store, content_hash="same")
        assert _insert_node(store, content_hash="same") is False


def test_upsert_changed_hash_returns_true() -> None:
    with _store() as store:
        _insert_node(store, content_hash="v1")
        assert _insert_node(store, content_hash="v2") is True


def test_upsert_update_changes_stored_hash() -> None:
    with _store() as store:
        _insert_node(store, content_hash="v1")
        _insert_node(store, content_hash="v2")
        assert store.get_content_hash("n1") == "v2"


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------


def test_get_node_returns_correct_record() -> None:
    with _store() as store:
        _insert_node(store, node_id="n42", kind="class", name="MyClass",
                     file="src/bar.py", line=5, summary="a class", content_hash="h1")
        rec = store.get_node("n42")
        assert isinstance(rec, NodeRecord)
        assert rec.node_id == "n42"
        assert rec.kind == "class"
        assert rec.name == "MyClass"
        assert rec.file == "src/bar.py"
        assert rec.line == 5
        assert rec.summary == "a class"
        assert rec.content_hash == "h1"


def test_get_node_returns_none_for_missing() -> None:
    with _store() as store:
        assert store.get_node("missing") is None


def test_get_node_with_null_file_and_line() -> None:
    with _store() as store:
        _insert_node(store, file=None, line=None)
        rec = store.get_node("n1")
        assert rec is not None
        assert rec.file is None
        assert rec.line is None


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------


def test_vector_search_returns_results() -> None:
    with _store() as store:
        _insert_node(store, node_id="n1", embedding=[0.1, 0.2, 0.3, 0.4])
        _insert_node(store, node_id="n2", embedding=[0.9, 0.8, 0.7, 0.6])
        results = store.vector_search([0.1, 0.2, 0.3, 0.4], limit=10)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)


def test_vector_search_closest_first() -> None:
    with _store() as store:
        _insert_node(store, node_id="near", embedding=[1.0, 0.0, 0.0, 0.0])
        _insert_node(store, node_id="far", embedding=[0.0, 1.0, 0.0, 0.0])
        results = store.vector_search([1.0, 0.0, 0.0, 0.0], limit=10)
        # "near" should rank first (highest score = least distance)
        assert results[0].node_id == "near"


def test_vector_search_score_is_negated_distance() -> None:
    with _store() as store:
        _insert_node(store, node_id="n1", embedding=[1.0, 0.0, 0.0, 0.0])
        results = store.vector_search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert results[0].score <= 0.0  # distance ≥ 0, so score ≤ 0


def test_vector_search_limit_respected() -> None:
    with _store() as store:
        for i in range(5):
            _insert_node(store, node_id=f"n{i}", embedding=[float(i), 0.0, 0.0, 0.0])
        results = store.vector_search([0.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------


def test_bm25_search_matches_name() -> None:
    with _store() as store:
        _insert_node(store, node_id="n1", name="authenticate_user",
                     summary="checks credentials")
        results = store.bm25_search("authenticate_user", limit=10)
        assert any(r.node_id == "n1" for r in results)


def test_bm25_search_matches_summary() -> None:
    with _store() as store:
        _insert_node(store, node_id="n1", name="foo", summary="parses JWT tokens carefully")
        results = store.bm25_search("JWT tokens", limit=10)
        assert any(r.node_id == "n1" for r in results)


def test_bm25_search_no_match_returns_empty() -> None:
    with _store() as store:
        _insert_node(store, node_id="n1", name="foo", summary="bar")
        results = store.bm25_search("zzz_nonexistent_term", limit=10)
        assert results == []


def test_bm25_search_result_fields_populated() -> None:
    with _store() as store:
        _insert_node(store, node_id="n1", name="process_data",
                     file="src/proc.py", line=99, summary="processes data quickly")
        results = store.bm25_search("process_data", limit=1)
        assert len(results) == 1
        r = results[0]
        assert r.node_id == "n1"
        assert r.name == "process_data"
        assert r.file == "src/proc.py"
        assert r.line == 99


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_works() -> None:
    with IndexStore(db_path=":memory:", embedding_dim=DIM) as store:
        assert _insert_node(store) is True
        assert store.get_content_hash("n1") is not None
