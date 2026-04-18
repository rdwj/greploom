"""greploom index command — build the search index from a treeloom CPG JSON file."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import click

from greploom.config import GrepLoomConfig
from greploom.index import run_index


@click.command()
@click.argument("cpg_json", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", "db_path", default=None, help="SQLite database path.")
@click.option(
    "--tier",
    default=None,
    type=click.Choice(["fast", "enhanced"]),
    help="Summary tier.",
)
@click.option("--model", "embedding_model", default=None, help="Embedding model name.")
@click.option("--ollama-url", "embedding_url", default=None, help="Ollama server URL.")
@click.option(
    "--embedding-url",
    "openai_embedding_url",
    default=None,
    help="OpenAI-compatible embedding endpoint (e.g. vLLM). Sets provider to 'openai'.",
)
@click.option("--force", is_flag=True, help="Re-index all nodes, ignoring content hashes.")
def index(
    cpg_json: str,
    db_path: str | None,
    tier: str | None,
    embedding_model: str | None,
    embedding_url: str | None,
    openai_embedding_url: str | None,
    force: bool,
) -> None:
    """Build the search index from a treeloom CPG JSON file."""
    if embedding_url is not None and openai_embedding_url is not None:
        raise click.UsageError("Cannot use both --ollama-url and --embedding-url")

    config = GrepLoomConfig.from_env()

    if openai_embedding_url is not None:
        embedding_url = openai_embedding_url

    overrides = {k: v for k, v in [
        ("db_path", db_path), ("summary_tier", tier),
        ("embedding_model", embedding_model), ("embedding_url", embedding_url),
    ] if v is not None}
    if openai_embedding_url is not None:
        overrides["embedding_provider"] = "openai"
    elif embedding_url is not None:
        overrides["embedding_provider"] = "ollama"
    if overrides:
        config = dataclasses.replace(config, **overrides)

    if force:
        Path(config.db_path).unlink(missing_ok=True)

    Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        result = run_index(Path(cpg_json), config, progress=click.echo)
    except (ConnectionError, RuntimeError) as exc:
        click.echo(f"Error: embedding service — {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"Indexed {result.indexed} nodes "
        f"({result.skipped} skipped, {result.errors} errors) "
        f"of {result.total} total."
    )
