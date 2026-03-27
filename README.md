# greploom

Semantic code search with graph-aware context retrieval, built on [treeloom](https://github.com/rdwj/treeloom).

greploom indexes a treeloom Code Property Graph into a hybrid search engine (vector embeddings + BM25) and assembles structurally-complete context neighborhoods for LLM consumption.

## Status

Pre-alpha. Design phase. See [loom-research](https://github.com/rdwj/loom-research) for the full design document.

## How It Works

```
Source code
    |
    v
treeloom build --> CPG (JSON)
    |
    v
greploom index --> vector store + BM25 index
    |
    v
greploom query "how is auth handled?" --> context bundle
    |
    v
LLM agent receives focused, graph-aware context
```

Vector search narrows to the right neighborhood. Graph traversal expands to the complete structural context. You need both.

## Planned Features

- **Hybrid search**: Vector similarity (sqlite-vec) + BM25 full-text (FTS5) with reciprocal rank fusion
- **Graph-aware context expansion**: For each search hit, walk the treeloom CPG to include callers, callees, imports, and data flow sources
- **Token budget management**: Return exactly as much context as the LLM can use, ranked by structural relevance
- **Three summary tiers**: fast (signatures), enhanced (signatures + docstrings + callees), llm (LLM-generated descriptions)
- **CLI and MCP server**: Usable from the command line or as an MCP tool for agents

## Planned CLI

```
greploom index CPG_JSON          # Build search index from treeloom CPG
greploom query "search terms"    # Hybrid semantic + symbol search
greploom callers SYMBOL          # Who calls this function?
greploom callees SYMBOL          # What does this function call?
greploom deps FILE               # What does this file depend on?
greploom impact SYMBOL           # What breaks if I change this?
greploom context FILE:LINE       # Full context neighborhood for a location
greploom serve                   # MCP server mode
greploom watch SOURCE_DIR        # Watch and re-index on changes
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| treeloom | CPG construction and graph queries |
| sqlite-vec | Vector storage and similarity search |
| click | CLI framework |
| tiktoken | Token counting for budget management |
| httpx | API calls to embedding services |
| ollama (optional) | Local embedding model inference |

## Installation

```bash
pip install greploom              # Core
pip install greploom[ollama]      # With local embedding support
```

## License

MIT
