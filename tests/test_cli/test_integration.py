"""End-to-end CLI integration tests.

Uses click's CliRunner with a fake EmbeddingClient so no ollama instance is needed.
All tests index from the fixture CPG JSON files and exercise the full CLI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from greploom.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CPG = Path(__file__).parent.parent / "fixtures" / "small_cpg.json"
MEDIUM_CPG = Path(__file__).parent.parent / "fixtures" / "medium_cpg.json"


class FakeEmbeddingClient:
    """Deterministic 768-dim embedder that never touches the network."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [[0.0] * 768] * len(texts)

    def embed_one(self, text: str) -> list[float]:
        return [0.0] * 768

    def close(self) -> None:
        pass

    def __enter__(self) -> FakeEmbeddingClient:
        return self

    def __exit__(self, *_: object) -> None:
        pass


@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the class where the index orchestrator imports it
    monkeypatch.setattr("greploom.index.EmbeddingClient", FakeEmbeddingClient)
    # Patch the class where the query command imports it
    monkeypatch.setattr("greploom.cli.query_cmd.EmbeddingClient", FakeEmbeddingClient)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def indexed_db(tmp_path: Path, runner: CliRunner) -> Path:
    """Return a tmp DB path that has already been indexed from small_cpg.json."""
    db = tmp_path / "test.db"
    result = runner.invoke(main, ["index", str(SMALL_CPG), "--db", str(db)])
    assert result.exit_code == 0, result.output
    return db


# ---------------------------------------------------------------------------
# Index command tests
# ---------------------------------------------------------------------------


def test_index_command_succeeds(tmp_path: Path, runner: CliRunner) -> None:
    db = tmp_path / "test.db"
    result = runner.invoke(main, ["index", str(SMALL_CPG), "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "Indexed" in result.output


def test_index_command_force_reindex(tmp_path: Path, runner: CliRunner) -> None:
    db = tmp_path / "test.db"

    first = runner.invoke(main, ["index", str(SMALL_CPG), "--db", str(db)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(main, ["index", str(SMALL_CPG), "--db", str(db), "--force"])
    assert second.exit_code == 0, second.output
    # --force deletes and re-creates the DB, so all nodes are indexed again — none skipped
    assert "Indexed" in second.output
    assert "skipped, 0" in second.output


def test_index_command_incremental_skips(tmp_path: Path, runner: CliRunner) -> None:
    db = tmp_path / "test.db"

    first = runner.invoke(main, ["index", str(SMALL_CPG), "--db", str(db)])
    assert first.exit_code == 0, first.output

    second = runner.invoke(main, ["index", str(SMALL_CPG), "--db", str(db)])
    assert second.exit_code == 0, second.output
    assert "skipped" in second.output


def test_index_command_missing_file(tmp_path: Path, runner: CliRunner) -> None:
    db = tmp_path / "test.db"
    result = runner.invoke(
        main, ["index", str(tmp_path / "nonexistent.json"), "--db", str(db)]
    )

    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Query command tests
# ---------------------------------------------------------------------------


def test_query_command_no_expansion(indexed_db: Path, runner: CliRunner) -> None:
    result = runner.invoke(
        main,
        ["query", "handle request", "--db", str(indexed_db), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    # Output should be non-empty JSON
    payload = json.loads(result.output)
    assert isinstance(payload, list)


def test_query_command_with_expansion(indexed_db: Path, runner: CliRunner) -> None:
    result = runner.invoke(
        main,
        [
            "query",
            "handle request",
            "--db",
            str(indexed_db),
            "--cpg",
            str(SMALL_CPG),
        ],
    )

    assert result.exit_code == 0, result.output
    # The default (context) format emits the text of each context block
    assert len(result.output.strip()) > 0


def test_query_command_json_format(indexed_db: Path, runner: CliRunner) -> None:
    result = runner.invoke(
        main,
        ["query", "validate input", "--db", str(indexed_db), "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    # Each hit record should have the expected keys
    if payload:
        hit = payload[0]
        assert "node_id" in hit
        assert "score" in hit
        assert "name" in hit
