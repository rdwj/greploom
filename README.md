# greploom

Semantic code search with graph-aware context retrieval, built on [treeloom](https://github.com/rdwj/treeloom).

greploom reads a treeloom Code Property Graph (CPG JSON), indexes it for hybrid search (vector embeddings + BM25), and returns structurally-complete context neighborhoods for LLM consumption. Vector search finds the right neighborhood; graph traversal expands it to include callers, callees, imports, and data flow sources.

## Installation

```bash
pip install greploom           # Core — CLI and search engine
pip install greploom[mcp]      # Adds MCP server (requires fastmcp)
```

The default embedding model is `nomic-embed-text` via a local [Ollama](https://ollama.com) instance. Any OpenAI-compatible embedding endpoint works via `GREPLOOM_EMBEDDING_URL`.

## Quick Start

```bash
# 1. Build a CPG with treeloom
treeloom build src/ -o cpg.json

# 2. Index for search (creates .greploom/index.db)
greploom index cpg.json

# 3. Search
greploom query "where is authentication handled?"
```

## How It Works

```
Source code
    |
    v
treeloom build --> CPG (JSON)
    |
    v
greploom index --> vector store + BM25 index (SQLite)
    |
    v
greploom query "how is auth handled?" --> context bundle
    |
    v
LLM agent receives focused, graph-aware context
```

Storage is a single SQLite file using sqlite-vec for vectors and FTS5 for BM25. No server required, no Docker, portable and inspectable.

## CLI Reference

### `greploom index`

Build or update the search index from a treeloom CPG JSON file.

```
greploom index CPG_JSON [OPTIONS]

Arguments:
  CPG_JSON    Path to the treeloom CPG JSON file

Options:
  --db PATH              SQLite database path (default: .greploom/index.db)
  --tier [fast|enhanced] Summary tier (default: enhanced)
  --model TEXT           Embedding model name
  --ollama-url URL       Ollama server URL
  --force                Re-index all nodes, ignoring content hashes
```

Re-indexing is incremental by default — only nodes whose content has changed are re-embedded. Use `--force` to rebuild from scratch.

Summary tiers:
- `fast` — function signatures only; fastest to build
- `enhanced` — signatures, parameters, callees, decorators, and class methods; better recall

```bash
# Index with defaults
greploom index cpg.json

# Use a custom database path and force full re-index
greploom index cpg.json --db /tmp/myproject.db --force

# Point at a non-default Ollama instance
greploom index cpg.json --ollama-url http://gpu-box:11434
```

### `greploom query`

Search the index and return graph-aware context.

```
greploom query [QUERY_TEXT] [OPTIONS]

Arguments:
  QUERY_TEXT    Natural language or symbol query (mutually exclusive with --node)

Options:
  --db PATH               SQLite database path (default: .greploom/index.db)
  --cpg PATH              CPG JSON path for graph expansion
  --budget INT            Token budget (default: 8192)
  --top-k INT             Number of search results (default: 5)
  --format [context|json] Output format (default: context)
  --model TEXT            Embedding model name
  --ollama-url URL        Ollama server URL
  --node NODE_ID          CPG node ID for direct lookup (repeatable; bypasses search)
  --include-source        Include raw source text from the CPG when available
```

Without `--cpg`, the query returns ranked search hits with scores and summaries. With `--cpg`, hits are expanded through the graph and assembled into a context bundle trimmed to the token budget. All `--format json` output is wrapped in `{"metadata": {...}, "results": [...]}` including the embedding model, greploom version, and index timestamps. Each result object in CPG mode includes a `source` field (raw source text when `--include-source` is used, null otherwise) and a `structural_context` dict with `callers`, `callees`, `parameters`, `parent_class`, `data_sources`, and `imports` for programmatic consumers.

Use `--node` to retrieve context for known CPG node IDs directly, bypassing the search step entirely. Requires `--cpg`.

```bash
# Simple search — ranked hits with summaries
greploom query "user authentication"

# Full graph-expanded context, ready for an LLM
greploom query "where is authentication handled?" --cpg cpg.json

# JSON output with index metadata
greploom query "UserService" --cpg cpg.json --format json | jq '.metadata'

# Direct lookup by CPG node ID (bypasses search)
greploom query --node "function:src/auth.py:10:0:3" --cpg cpg.json

# Narrow token budget for smaller context windows
greploom query "error handling" --cpg cpg.json --budget 4096
```

### `greploom serve`

Start the MCP server.

```
greploom serve [OPTIONS]

Options:
  --db PATH                    SQLite database path
  --cpg PATH                   Default CPG JSON path
  --host TEXT                  Host to bind (default: 0.0.0.0)
  --port INT                   Port to listen on (default: 8901)
  --transport [streamable-http|stdio]  MCP transport (default: streamable-http)
```

```bash
# Start the MCP server on default port 8901
greploom serve --db .greploom/index.db --cpg cpg.json

# stdio transport for direct agent integration
greploom serve --transport stdio
```

## MCP Server

The MCP server exposes three tools:

**`search_code`** — Search code semantically and return graph-aware context.

Parameters: `query` (required), `cpg_path` (required), `db_path`, `budget`, `top_k`, `include_source`

**`get_node_context`** — Return graph-aware context for specific CPG node IDs, bypassing search.

Parameters: `node_ids` (required), `cpg_path` (required), `budget`, `include_source`

**`index_code`** — Build or update the search index from a CPG JSON file.

Parameters: `cpg_path` (required), `db_path`, `tier`

Example MCP server URL for agent configuration: `http://localhost:8901/mcp`

## Configuration

All settings can be provided via environment variables. CLI flags override environment variables for individual commands.

| Variable | Default | Description |
|---|---|---|
| `GREPLOOM_EMBEDDING_URL` | `http://localhost:11434` | Ollama or OpenAI-compatible endpoint |
| `GREPLOOM_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `GREPLOOM_DB_PATH` | `.greploom/index.db` | SQLite database path |
| `GREPLOOM_TOKEN_BUDGET` | `8192` | Default token budget for context assembly |
| `GREPLOOM_SUMMARY_TIER` | `enhanced` | Summary tier (`fast` or `enhanced`) |

To use an OpenAI-compatible embedding API instead of Ollama:

```bash
export GREPLOOM_EMBEDDING_URL=https://api.openai.com/v1
export GREPLOOM_EMBEDDING_MODEL=text-embedding-3-small
```

## Relationship to treeloom

greploom reads treeloom's CPG JSON format but does not import treeloom at runtime. `greploom index` reads the CPG JSON to build the search index; `greploom query` reads both the index and the CPG JSON for graph expansion. Any tool that produces treeloom-compatible CPG JSON will work.

greploom's query output includes structural summaries and graph context (callers, callees, parameters). Raw source text can be included via `--include-source` when the CPG contains source spans (treeloom 0.6.0+ with `--include-source`).

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute, development setup, testing
- [SECURITY.md](SECURITY.md) — vulnerability reporting policy
- [llms.txt](llms.txt) / [llms-full.txt](llms-full.txt) — LLM-friendly project documentation ([llmstxt.org](https://llmstxt.org/))
- [CLAUDE.md](CLAUDE.md) — project context for AI coding assistants

## Changelog

### Version 0.5.0

- JSON query output in CPG mode now includes a `source` field with raw source text (when `--include-source` is used) and a `structural_context` dict with `callers`, `callees`, `parameters`, `parent_class`, `data_sources`, and `imports` as structured data.
- Programmatic consumers can access source code and graph relationships directly without parsing the formatted markdown `text` field.

### Version 0.4.0

- All query output modes (search-only, `--cpg` expansion, `--node` lookup) now use a consistent `{"metadata": {...}, "results": [...]}` JSON envelope with the embedding model, greploom version, and index timestamps.
- Human-readable output shows a `Model: ... | Indexed: ...` header line.
- **Breaking:** JSON key renamed from `"blocks"` to `"results"`; search-only and `--node` output changed from bare arrays to the metadata envelope.

### Version 0.3.1

- Fixed fenced code block corruption when source text contains triple backticks (e.g., Markdown in docstrings).
- Fixed Click 8.2+ compatibility for `CliRunner` in tests.
- Bumped minimum Click dependency to `>=8.2`.

### Version 0.3.0

- `--include-source` flag for `greploom query`: include raw source text from the CPG in context blocks when available. Off by default. Requires treeloom 0.6.0+ CPG built with `--include-source`.
- MCP tools `search_code` and `get_node_context` gained `include_source` parameter.
- Bumped treeloom dependency to `>=0.6.0`.

### Version 0.2.0

- `--node` mode for `greploom query`: retrieve graph context for known CPG node IDs without running a search query.
- Index metadata: embedding model, greploom version, and timestamps are stored in the index and surfaced in all JSON output via the `{"metadata": ..., "results": ...}` envelope.
- MCP server: added `get_node_context` tool (direct node ID lookup, parallel to `--node` in the CLI).

### Version 0.1.0

Initial release — full indexing pipeline (summarize, embed, store), hybrid search with RRF, graph expansion for context neighborhoods, token budget management, CLI (index/query/serve), and MCP server with search_code/index_code tools.

## License

MIT
