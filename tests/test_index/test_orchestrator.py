"""Tests for greploom.index.run_index orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from greploom.config import GrepLoomConfig
from greploom.index import run_index

# ---------------------------------------------------------------------------
# Fixture: minimal CPG JSON with 3 nodes (2 indexable: module, function)
# and one non-indexable variable node.
# ---------------------------------------------------------------------------

_CPG_DATA = {
    "treeloom_version": "0.1.0",
    "nodes": [
        {
            "id": "mod-1",
            "kind": "module",
            "name": "src/app.py",
            "location": {"file": "src/app.py", "line": 1, "column": 0},
            "attrs": {},
        },
        {
            "id": "fn-1",
            "kind": "function",
            "name": "do_work",
            "location": {"file": "src/app.py", "line": 5, "column": 0},
            "attrs": {},
        },
        {
            "id": "var-1",
            "kind": "variable",
            "name": "MY_CONST",
            "location": {"file": "src/app.py", "line": 3, "column": 0},
            "attrs": {},
        },
    ],
    "edges": [
        {"source": "mod-1", "target": "fn-1", "kind": "contains", "attrs": {}},
    ],
    "annotations": {},
    "edge_annotations": [],
}

_EMBED_DIM = 768


class _FakeEmbeddingClient:
    """Returns deterministic 768-dim zero-vectors; never hits the network."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.0] * _EMBED_DIM for _ in texts]

    def close(self) -> None:
        pass

    def __enter__(self) -> _FakeEmbeddingClient:
        return self

    def __exit__(self, *_: object) -> None:
        pass


@pytest.fixture()
def cpg_file(tmp_path: Path) -> Path:
    p = tmp_path / "cpg.json"
    p.write_text(json.dumps(_CPG_DATA), encoding="utf-8")
    return p


@pytest.fixture()
def config(tmp_path: Path) -> GrepLoomConfig:
    return GrepLoomConfig(db_path=str(tmp_path / "index.db"))


@pytest.fixture(autouse=True)
def patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("greploom.index.EmbeddingClient", _FakeEmbeddingClient)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path(cpg_file: Path, config: GrepLoomConfig) -> None:
    """Two indexable nodes (module + function) should be indexed; variable skipped."""
    result = run_index(cpg_file, config)

    assert result.total == 2, f"expected 2 indexable nodes, got total={result.total}"
    assert result.indexed == 2, f"expected 2 indexed, got {result.indexed}"
    assert result.skipped == 0
    assert result.errors == 0


def test_incremental_skips_unchanged(cpg_file: Path, config: GrepLoomConfig) -> None:
    """Second run on identical data must skip all previously indexed nodes."""
    first = run_index(cpg_file, config)
    assert first.indexed == 2

    second = run_index(cpg_file, config)
    assert second.skipped == 2, f"expected 2 skipped on second run, got {second.skipped}"
    assert second.indexed == 0
    assert second.errors == 0


def test_progress_callback(cpg_file: Path, config: GrepLoomConfig) -> None:
    """Progress callback must be invoked for each node processed."""
    messages: list[str] = []
    run_index(cpg_file, config, progress=messages.append)

    # Expect one "index <name>" message per indexed node
    index_msgs = [m for m in messages if m.startswith("index ")]
    assert len(index_msgs) == 2, f"got progress messages: {messages}"

    names = {m.removeprefix("index ") for m in index_msgs}
    assert "do_work" in names
    assert "src/app.py" in names


def test_incremental_progress_shows_skips(cpg_file: Path, config: GrepLoomConfig) -> None:
    """On a second identical run, progress callback reports skips not indexes."""
    run_index(cpg_file, config)

    messages: list[str] = []
    run_index(cpg_file, config, progress=messages.append)

    skip_msgs = [m for m in messages if m.startswith("skip ")]
    index_msgs = [m for m in messages if m.startswith("index ")]
    assert len(skip_msgs) == 2, f"expected 2 skip messages, got: {messages}"
    assert len(index_msgs) == 0


def test_empty_cpg(tmp_path: Path, config: GrepLoomConfig) -> None:
    """A CPG with no indexable nodes returns total=0 and leaves other counts at 0."""
    cpg_path = tmp_path / "empty.json"
    cpg_path.write_text(
        json.dumps(
            {
                "treeloom_version": "0.1.0",
                "nodes": [
                    {
                        "id": "var-1",
                        "kind": "variable",
                        "name": "x",
                        "attrs": {},
                    }
                ],
                "edges": [],
                "annotations": {},
                "edge_annotations": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_index(cpg_path, config)

    assert result.total == 0
    assert result.indexed == 0
    assert result.skipped == 0
    assert result.errors == 0
