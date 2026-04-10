"""greploom query command — search the index and return graph-aware context."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from greploom.config import GrepLoomConfig
from greploom.cpg_types import load_cpg
from greploom.index.embedder import EmbeddingClient
from greploom.index.store import IndexStore
from greploom.search.budget import assemble_context
from greploom.search.expand import expand_hits
from greploom.search.hybrid import hybrid_search


def _echo_human_header(metadata: dict[str, str]) -> None:
    """Print a one-line metadata banner for human-readable output."""
    model = metadata.get("embedding_model", "unknown")
    indexed_at = metadata.get("indexed_at", "unknown")
    click.echo(f"Model: {model}  |  Indexed: {indexed_at}")


@click.command()
@click.argument("query_text", required=False, default=None)
@click.option("--db", "db_path", default=None, help="SQLite database path.")
@click.option(
    "--cpg",
    "cpg_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="CPG JSON path for graph expansion.",
)
@click.option("--budget", default=None, type=int, help="Token budget.")
@click.option("--top-k", default=5, type=int, help="Number of search results.")
@click.option(
    "--format",
    "output_format",
    default="context",
    type=click.Choice(["context", "json"]),
    help="Output format.",
)
@click.option("--model", "embedding_model", default=None, help="Embedding model name.")
@click.option("--ollama-url", "embedding_url", default=None, help="Ollama server URL.")
@click.option(
    "--node",
    "node_ids",
    multiple=True,
    help="CPG node ID(s) for direct lookup (bypasses search).",
)
@click.option(
    "--include-source",
    is_flag=True,
    default=False,
    help="Include raw source text from the CPG in results when available.",
)
def query(
    query_text: str | None,
    db_path: str | None,
    cpg_path: str | None,
    budget: int | None,
    top_k: int,
    output_format: str,
    embedding_model: str | None,
    embedding_url: str | None,
    node_ids: tuple[str, ...],
    include_source: bool,
) -> None:
    """Search the index and return graph-aware context.

    Results include structural context (callers, callees, parameters) assembled
    from the CPG graph. Use --include-source to include raw source text from the
    CPG when available; by default source text is omitted.
    """
    if node_ids and query_text:
        click.echo("Error: --node and query_text are mutually exclusive.", err=True)
        sys.exit(1)
    if not node_ids and not query_text:
        click.echo("Error: provide either query_text or --node.", err=True)
        sys.exit(1)

    cfg = GrepLoomConfig.from_env()
    if db_path:
        cfg.db_path = db_path
    if budget is not None:
        cfg.token_budget = budget
    if embedding_model:
        cfg.embedding_model = embedding_model
    if embedding_url:
        cfg.embedding_url = embedding_url

    if node_ids:
        if not cpg_path:
            click.echo("Error: --cpg is required when using --node.", err=True)
            sys.exit(1)

        # Open store to retrieve index metadata for the output envelope.
        store = IndexStore(cfg.db_path)
        metadata = store.get_all_metadata()
        store.close()

        cpg = load_cpg(Path(cpg_path))
        expanded = expand_hits(list(node_ids), cpg)
        ctx = assemble_context(expanded, cfg.token_budget, include_source=include_source)
        if output_format == "json":
            results = [
                {
                    "node_id": b.node_id,
                    "name": b.name,
                    "kind": b.kind,
                    "file": b.file,
                    "line": b.line,
                    "relationship": b.relationship,
                    "tokens": b.tokens,
                    "text": b.text,
                }
                for b in ctx.blocks
            ]
            payload = {"metadata": metadata, "results": results}
            click.echo(json.dumps(payload, indent=2))
        else:
            _echo_human_header(metadata)
            click.echo(("\n\n").join(b.text for b in ctx.blocks))
        return

    store = IndexStore(cfg.db_path)
    stored_model = store.get_metadata("embedding_model")
    if stored_model and stored_model != cfg.embedding_model:
        click.echo(
            f"Warning: query model '{cfg.embedding_model}' differs from "
            f"index model '{stored_model}'. Results may be degraded.",
            err=True,
        )
    client = EmbeddingClient(cfg.embedding_url, cfg.embedding_model)
    try:
        query_embedding = client.embed_one(query_text)
        hits = hybrid_search(query_text, query_embedding, store, top_k=top_k)

        if cpg_path:
            cpg = load_cpg(Path(cpg_path))
            expanded = expand_hits([h.node_id for h in hits], cpg)
            ctx = assemble_context(expanded, cfg.token_budget, include_source=include_source)

            metadata = store.get_all_metadata()
            if output_format == "json":
                blocks = [
                    {
                        "node_id": b.node_id,
                        "name": b.name,
                        "kind": b.kind,
                        "file": b.file,
                        "line": b.line,
                        "relationship": b.relationship,
                        "tokens": b.tokens,
                        "text": b.text,
                    }
                    for b in ctx.blocks
                ]
                payload = {"metadata": metadata, "results": blocks}
                click.echo(json.dumps(payload, indent=2))
            else:
                _echo_human_header(metadata)
                click.echo(("\n\n").join(b.text for b in ctx.blocks))
        else:
            if output_format == "json":
                results = [
                    {
                        "node_id": h.node_id,
                        "score": h.score,
                        "name": h.name,
                        "file": h.file,
                        "line": h.line,
                        "summary": h.summary,
                    }
                    for h in hits
                ]
                payload = {"metadata": store.get_all_metadata(), "results": results}
                click.echo(json.dumps(payload, indent=2))
            else:
                _echo_human_header(store.get_all_metadata())
                for h in hits:
                    loc = f"{h.file}:{h.line}" if h.file and h.line else (h.file or "")
                    click.echo(f"[{h.score:.4f}] {h.name}  {loc}\n  {h.summary}")
    except ConnectionError as exc:
        click.echo(f"Error: could not connect to embedding service — {exc}", err=True)
        sys.exit(1)
    finally:
        store.close()
        client.close()
