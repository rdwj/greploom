"""Index pipeline: summarize CPG nodes → embed → store in SQLite."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from greploom.config import GrepLoomConfig
from greploom.cpg_types import load_cpg
from greploom.index.embedder import EmbeddingClient
from greploom.index.store import IndexStore
from greploom.index.summarizer import (
    INDEXABLE_KINDS,
    build_edges_from,
    build_node_lookup,
    summarize_node,
)
from greploom.version import __version__

log = logging.getLogger(__name__)


@dataclass
class IndexResult:
    total: int = 0
    indexed: int = 0
    skipped: int = 0
    errors: int = 0


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_index(
    cpg_path: Path,
    config: GrepLoomConfig,
    progress: Callable[[str], None] | None = None,
) -> IndexResult:
    """Index a treeloom CPG JSON file into the greploom search database.

    Flow:
    1. Load CPG, build lookup indexes.
    2. Summarize all indexable nodes and compute content hashes.
    3. Open store; filter out nodes whose hash matches what is already stored.
    4. Batch-embed all changed summaries.
    5. Upsert into store.
    """
    cpg = load_cpg(cpg_path)
    node_lookup = build_node_lookup(cpg)
    edges_from = build_edges_from(cpg)

    result = IndexResult()

    # Phase 1: collect summaries
    pending: list[tuple] = []  # (node, summary, content_hash)
    for node in cpg.nodes:
        if node.kind not in INDEXABLE_KINDS:
            continue
        result.total += 1
        summary = summarize_node(node, node_lookup, edges_from, config.summary_tier)
        if summary is None:
            continue
        pending.append((node, summary, _content_hash(summary)))

    # Phase 2: filter unchanged nodes using the store
    with IndexStore(config.db_path) as store:
        store.set_metadata("embedding_model", config.embedding_model)
        store.set_metadata("greploom_version", __version__)
        if not store.get_metadata("created_at"):
            store.set_metadata("created_at", datetime.now(timezone.utc).isoformat())
        store.set_metadata("indexed_at", datetime.now(timezone.utc).isoformat())

        to_embed: list[tuple] = []
        for node, summary, h in pending:
            if store.get_content_hash(node.id) == h:
                result.skipped += 1
                if progress:
                    progress(f"skip {node.name}")
            else:
                to_embed.append((node, summary, h))

        if not to_embed:
            return result

        # Phase 3: batch embed
        texts = [summary for _, summary, _ in to_embed]
        with EmbeddingClient(
            config.embedding_url, config.embedding_model, config.embedding_provider,
        ) as client:
            embeddings = client.embed(texts)

        # Phase 4: upsert
        if len(embeddings) != len(to_embed):
            raise RuntimeError(
                f"Embedding service returned {len(embeddings)} vectors "
                f"for {len(to_embed)} inputs"
            )
        for (node, summary, h), embedding in zip(to_embed, embeddings):
            try:
                store.upsert(
                    node_id=node.id,
                    kind=node.kind.value,
                    name=node.name,
                    file=node.location.file if node.location else None,
                    line=node.location.line if node.location else None,
                    summary=summary,
                    content_hash=h,
                    embedding=embedding,
                )
                result.indexed += 1
                if progress:
                    progress(f"index {node.name}")
            except Exception:
                log.exception("Failed to upsert node %s", node.id)
                result.errors += 1
                if progress:
                    progress(f"error {node.name}")

    return result
