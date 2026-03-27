"""greploom MCP server — exposes search_code and index_code tools via FastMCP."""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from greploom.config import GrepLoomConfig
from greploom.cpg_types import load_cpg
from greploom.index import run_index
from greploom.index.embedder import EmbeddingClient
from greploom.index.store import IndexStore
from greploom.search.budget import assemble_context
from greploom.search.expand import expand_hits
from greploom.search.hybrid import hybrid_search


def create_server(config: GrepLoomConfig | None = None) -> FastMCP:
    """Factory function — creates and returns the FastMCP server instance.

    Accepts an optional pre-built config; otherwise tools build their own
    per-call configs from defaults and explicit parameters.
    """
    _default_config = config

    mcp = FastMCP(
        "greploom",
        instructions=(
            "Semantic code search over treeloom Code Property Graphs. "
            "Use search_code to find relevant code by natural language query. "
            "Use index_code to build or update the search index from a CPG JSON file."
        ),
    )

    @mcp.tool()
    def search_code(
        query: str,
        cpg_path: str,
        db_path: str = ".greploom/index.db",
        budget: int = 8192,
        top_k: int = 5,
    ) -> str:
        """Search code semantically and return graph-aware context."""
        cfg = _default_config or GrepLoomConfig(db_path=db_path, token_budget=budget)
        effective_db = cfg.db_path if _default_config else db_path
        effective_budget = cfg.token_budget if _default_config else budget

        try:
            with EmbeddingClient(cfg.embedding_url, cfg.embedding_model) as client:
                query_embedding = client.embed_one(query)
        except ConnectionError as exc:
            return f"Error: could not reach embedding service — {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Error: embedding failed — {exc}"

        try:
            with IndexStore(effective_db) as store:
                hits = hybrid_search(query, query_embedding, store, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            return f"Error: search index unavailable at {effective_db!r} — {exc}"

        if not hits:
            return "No results found."

        try:
            cpg = load_cpg(Path(cpg_path))
        except Exception as exc:  # noqa: BLE001
            return f"Error: could not load CPG from {cpg_path!r} — {exc}"

        hit_ids = [h.node_id for h in hits]
        expanded = expand_hits(hit_ids, cpg, depth=1)
        context = assemble_context(expanded, budget=effective_budget)

        if not context.blocks:
            return "No context could be assembled for the search results."

        return "\n\n".join(block.text for block in context.blocks)

    @mcp.tool()
    def index_code(
        cpg_path: str,
        db_path: str = ".greploom/index.db",
        tier: str = "enhanced",
    ) -> str:
        """Build or update the search index from a treeloom CPG JSON file."""
        cfg = _default_config or GrepLoomConfig(db_path=db_path, summary_tier=tier)
        effective_db = cfg.db_path if _default_config else db_path

        Path(effective_db).parent.mkdir(parents=True, exist_ok=True)

        try:
            result = run_index(Path(cpg_path), cfg)
        except Exception as exc:  # noqa: BLE001
            return f"Error: indexing failed — {exc}"

        return (
            f"Indexed {result.indexed} node(s), "
            f"skipped {result.skipped} unchanged, "
            f"{result.errors} error(s). "
            f"Total candidates: {result.total}."
        )

    return mcp
