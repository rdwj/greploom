"""CLI surface tests — verify help text and argument validation."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from greploom.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("subcommand", ["index", "query", "serve"])
def test_help(runner: CliRunner, subcommand: str) -> None:
    result = runner.invoke(main, [subcommand, "--help"])
    assert result.exit_code == 0, result.output
    assert "--help" in result.output


def test_index_missing_cpg_argument(runner: CliRunner) -> None:
    result = runner.invoke(main, ["index"])
    assert result.exit_code != 0
    assert "CPG_JSON" in result.output or "Missing argument" in result.output


def test_index_nonexistent_file(runner: CliRunner) -> None:
    result = runner.invoke(main, ["index", "/nonexistent/path/cpg.json"])
    assert result.exit_code != 0
    assert "does not exist" in result.output or "Invalid value" in result.output


def test_query_node_without_cpg_shows_error(runner: CliRunner) -> None:
    result = runner.invoke(main, ["query", "--node", "function:src/app.py:4:0:2"])
    assert result.exit_code != 0
    assert "--cpg is required" in result.output


def test_query_node_and_query_text_are_mutually_exclusive(runner: CliRunner) -> None:
    result = runner.invoke(
        main,
        ["query", "some text", "--node", "function:src/app.py:4:0:2"],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_query_no_args_shows_error(runner: CliRunner) -> None:
    result = runner.invoke(main, ["query"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# index --embedding-url / --ollama-url mutual exclusion
# ---------------------------------------------------------------------------


def test_embedding_url_and_ollama_url_mutual_exclusion(runner: CliRunner, tmp_path) -> None:
    """Providing both --ollama-url and --embedding-url must be rejected."""
    # We pass a dummy CPG file path; the error should fire before file I/O.
    cpg = tmp_path / "fake.json"
    cpg.write_text("{}")
    result = runner.invoke(
        main,
        [
            "index", str(cpg),
            "--ollama-url", "http://localhost:11434",
            "--embedding-url", "http://vllm:8000",
        ],
    )
    assert result.exit_code != 0
    assert "Cannot use both" in result.output or "mutually exclusive" in result.output


def test_embedding_url_sets_openai_provider(runner: CliRunner, tmp_path, monkeypatch) -> None:
    """--embedding-url should set embedding_provider to 'openai' on the resulting config."""
    captured_configs: list = []

    # Intercept run_index to capture the config without actually indexing.
    def fake_run_index(cpg_path, config, progress=None):
        captured_configs.append(config)

        class _Result:
            indexed = 0
            skipped = 0
            errors = 0
            total = 0

        return _Result()

    monkeypatch.setattr("greploom.cli.index_cmd.run_index", fake_run_index)

    cpg = tmp_path / "fake.json"
    cpg.write_text('{"nodes": [], "edges": []}')

    result = runner.invoke(
        main,
        ["index", str(cpg), "--embedding-url", "http://vllm:8000"],
    )
    assert result.exit_code == 0, result.output
    assert len(captured_configs) == 1
    cfg = captured_configs[0]
    assert cfg.embedding_provider == "openai", (
        f"Expected provider 'openai', got {cfg.embedding_provider!r}"
    )
    assert cfg.embedding_url == "http://vllm:8000", (
        f"Expected URL 'http://vllm:8000', got {cfg.embedding_url!r}"
    )
