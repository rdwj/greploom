# greploom -- Project Context

greploom is a semantic code search library with graph-aware context retrieval. It reads a treeloom Code Property Graph (JSON), indexes it for hybrid search (vector + BM25), and returns structurally-complete context neighborhoods suitable for LLM consumption.

See the [loom-research](https://github.com/rdwj/loom-research) repo for the full design document and architectural decisions.

## Tech Stack

- Python 3.10+, package name `greploom`
- Search: sqlite-vec (vector), FTS5 via APSW (BM25), reciprocal rank fusion
- Graph: reads treeloom CPG JSON format, walks edges for context expansion
- Embeddings: nomic-embed-text default model; any OpenAI-compatible endpoint
- CLI: click
- Token counting: tiktoken
- Build: Hatchling
- Testing: pytest, 80%+ coverage target
- Optional extras: `ollama` (ollama Python client), `mcp` (fastmcp server)

## Architecture

```
greploom/
├── src/greploom/
│   ├── cpg_types.py    # CPG JSON types, load_cpg()
│   ├── config.py       # GrepLoomConfig, env var loading
│   ├── version.py      # __version__
│   ├── __init__.py
│   ├── index/          # Indexing pipeline: summarizer, embedder, storage
│   │   ├── summarizer.py    # Generate text summaries from CPG nodes
│   │   ├── embedder.py      # Embed summaries via ollama or API
│   │   └── store.py         # SQLite storage (sqlite-vec + FTS5 + metadata)
│   ├── search/         # Search engine: hybrid search, ranking, context assembly
│   │   ├── hybrid.py        # BM25 + vector search with RRF
│   │   ├── expand.py        # Graph walk to assemble context neighborhoods
│   │   └── budget.py        # Token budget management
│   ├── cli/            # CLI commands
│   │   ├── index_cmd.py
│   │   ├── query_cmd.py
│   │   └── serve_cmd.py
│   └── mcp/            # MCP server (search_code, get_node_context, index_code)
│       └── server.py
├── tests/
│   ├── fixtures/
│   ├── test_index/
│   ├── test_search/
│   ├── test_cli/
│   └── test_mcp/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── llms.txt
└── llms-full.txt
```

## Design Principles

1. **treeloom is the graph, greploom is the search.** greploom reads treeloom's CPG JSON — it never parses source code or builds its own graph.
2. **SQLite-only storage.** sqlite-vec + FTS5. No servers, no Docker, one file. Portable and inspectable.
3. **Hybrid search.** Vector similarity for semantic queries ("where is authentication?") + BM25 for symbol queries ("find UserService"). Reciprocal rank fusion merges results.
4. **Graph expansion at query time.** Search finds candidate nodes. CPG walk adds callers, callees, imports, data flow sources. The agent gets code in structural context. Direct node lookup via `--node` / `get_node_context` bypasses search for known node IDs.
5. **Token budget management.** Returns exactly as much context as the LLM can use, ranked by structural relevance. Default: 8192 tokens.
6. **Incremental re-indexing.** Track content hash per node. On re-index, only re-embed changed nodes.
7. **Index metadata.** The SQLite store records the embedding model, greploom version, and timestamps. JSON output includes this metadata, and queries warn when the embedding model doesn't match the index.

## Relationship to treeloom

greploom depends on treeloom but does not import treeloom at runtime for core search operations. The dependency is via the CPG JSON format:

- `greploom index` reads a treeloom CPG JSON file and builds its search index
- `greploom query` reads both the search index (SQLite) and the CPG JSON (for graph expansion)
- The CPG JSON format is documented in treeloom's CLAUDE.md

This means greploom can work with any tool that produces treeloom-compatible CPG JSON, not just treeloom itself.

## Releasing

The version lives in two files that must stay in sync:
- `src/greploom/version.py` — `__version__ = "x.y.z"`
- `pyproject.toml` — `version = "x.y.z"`
