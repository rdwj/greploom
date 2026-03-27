from __future__ import annotations

from dataclasses import dataclass

from greploom.index.store import IndexStore, SearchResult


@dataclass
class SearchHit:
    node_id: str
    score: float  # RRF fusion score, higher = better
    name: str
    file: str | None
    line: int | None
    summary: str


def hybrid_search(
    query: str,
    query_embedding: list[float],
    store: IndexStore,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """Hybrid BM25 + vector search with reciprocal rank fusion."""
    fetch = top_k * 2
    vec_results: list[SearchResult] = store.vector_search(query_embedding, limit=fetch)
    bm25_results: list[SearchResult] = store.bm25_search(query, limit=fetch)

    # node_id → (rrf_score, best_rank, result)
    # best_rank tracks which source gave the lower (better) rank for metadata selection
    fused: dict[str, tuple[float, int, SearchResult]] = {}

    for source in (vec_results, bm25_results):
        for rank, result in enumerate(source, start=1):
            contrib = 1.0 / (rrf_k + rank)
            nid = result.node_id
            if nid in fused:
                prev_score, prev_rank, prev_result = fused[nid]
                # Sum scores; keep metadata from the better-ranked appearance
                best_result = result if rank < prev_rank else prev_result
                fused[nid] = (prev_score + contrib, min(prev_rank, rank), best_result)
            else:
                fused[nid] = (contrib, rank, result)

    hits = [
        SearchHit(
            node_id=nid,
            score=score,
            name=result.name,
            file=result.file,
            line=result.line,
            summary=result.summary,
        )
        for nid, (score, _rank, result) in fused.items()
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
