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
