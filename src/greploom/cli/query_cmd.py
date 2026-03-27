"""greploom query command — search the index and return graph-aware context."""

from __future__ import annotations

import json
import sys

import click

from greploom.config import GrepLoomConfig
from greploom.cpg_types import load_cpg
from greploom.index.embedder import EmbeddingClient
from greploom.index.store import IndexStore
from greploom.search.budget import assemble_context
from greploom.search.expand import expand_hits
from greploom.search.hybrid import hybrid_search


@click.command()
@click.argument("query_text")
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
def query(
    query_text: str,
    db_path: str | None,
    cpg_path: str | None,
    budget: int | None,
    top_k: int,
    output_format: str,
    embedding_model: str | None,
    embedding_url: str | None,
) -> None:
    """Search the index and return graph-aware context."""
    cfg = GrepLoomConfig.from_env()
    if db_path:
        cfg.db_path = db_path
    if budget is not None:
        cfg.token_budget = budget
    if embedding_model:
        cfg.embedding_model = embedding_model
    if embedding_url:
        cfg.embedding_url = embedding_url

    store = IndexStore(cfg.db_path)
    client = EmbeddingClient(cfg.embedding_url, cfg.embedding_model)
    try:
        query_embedding = client.embed_one(query_text)
        hits = hybrid_search(query_text, query_embedding, store, top_k=top_k)

        if cpg_path:
            cpg = load_cpg(cpg_path)
            expanded = expand_hits([h.node_id for h in hits], cpg)
            ctx = assemble_context(expanded, cfg.token_budget)

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
                click.echo(json.dumps(blocks, indent=2))
            else:
                click.echo(("\n\n").join(b.text for b in ctx.blocks))
        else:
            if output_format == "json":
                click.echo(
                    json.dumps(
                        [
                            {
                                "node_id": h.node_id,
                                "score": h.score,
                                "name": h.name,
                                "file": h.file,
                                "line": h.line,
                                "summary": h.summary,
                            }
                            for h in hits
                        ],
                        indent=2,
                    )
                )
            else:
                for h in hits:
                    loc = f"{h.file}:{h.line}" if h.file and h.line else (h.file or "")
                    click.echo(f"[{h.score:.4f}] {h.name}  {loc}\n  {h.summary}")
    except ConnectionError as exc:
        click.echo(f"Error: could not connect to embedding service — {exc}", err=True)
        sys.exit(1)
    finally:
        store.close()
        client.close()
