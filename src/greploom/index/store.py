from __future__ import annotations

import struct
from dataclasses import dataclass

import apsw
import sqlite_vec


def _pack_f32(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


@dataclass
class SearchResult:
    node_id: str
    score: float
    name: str
    file: str | None
    line: int | None
    summary: str


@dataclass
class NodeRecord:
    node_id: str
    kind: str
    name: str
    file: str | None
    line: int | None
    summary: str
    content_hash: str


class IndexStore:
    """SQLite-backed search index using sqlite-vec (vector) and FTS5 (BM25)."""

    def __init__(self, db_path: str = ":memory:", embedding_dim: int = 768) -> None:
        self._embedding_dim = embedding_dim
        self._conn = apsw.Connection(db_path)
        self._conn.enableloadextension(True)
        self._conn.loadextension(sqlite_vec.loadable_path())
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(
        self,
        node_id: str,
        kind: str,
        name: str,
        file: str | None,
        line: int | None,
        summary: str,
        content_hash: str,
        embedding: list[float],
    ) -> bool:
        """Insert or update a node. Returns True if written, False if skipped (same hash)."""
        existing = list(
            self._conn.execute(
                "SELECT rowid, content_hash FROM nodes WHERE node_id = ?", [node_id]
            )
        )

        if existing and existing[0][1] == content_hash:
            return False  # unchanged

        packed = _pack_f32(embedding)

        # Wrap multi-table writes in a transaction for atomicity.
        with self._conn:
            if existing:
                rowid = existing[0][0]
                self._conn.execute(
                    """UPDATE nodes SET kind=?, name=?, file=?, line=?, summary=?,
                       content_hash=? WHERE node_id=?""",
                    [kind, name, file, line, summary, content_hash, node_id],
                )
                self._conn.execute(
                    "UPDATE vec_index SET embedding=? WHERE rowid=?", [packed, rowid]
                )
                self._conn.execute("DELETE FROM fts_index WHERE rowid=?", [rowid])
                self._conn.execute(
                    "INSERT INTO fts_index(rowid, name, summary) VALUES (?, ?, ?)",
                    [rowid, name, summary],
                )
            else:
                self._conn.execute(
                    """INSERT INTO nodes(node_id, kind, name, file, line, summary,
                       content_hash) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [node_id, kind, name, file, line, summary, content_hash],
                )
                rowid = list(
                    self._conn.execute(
                        "SELECT rowid FROM nodes WHERE node_id=?", [node_id]
                    )
                )[0][0]
                self._conn.execute(
                    "INSERT INTO vec_index(rowid, embedding) VALUES (?, ?)",
                    [rowid, packed],
                )
                self._conn.execute(
                    "INSERT INTO fts_index(rowid, name, summary) VALUES (?, ?, ?)",
                    [rowid, name, summary],
                )

        return True

    def get_content_hash(self, node_id: str) -> str | None:
        """Return the stored content hash for node_id, or None if not found."""
        rows = list(
            self._conn.execute("SELECT content_hash FROM nodes WHERE node_id=?", [node_id])
        )
        return rows[0][0] if rows else None

    def vector_search(self, query_embedding: list[float], limit: int = 10) -> list[SearchResult]:
        """Return nodes ranked by vector similarity (cosine distance, lower = better)."""
        packed = _pack_f32(query_embedding)
        # vec0 requires LIMIT inside the virtual table query; use a CTE to then JOIN nodes.
        rows = list(
            self._conn.execute(
                """WITH knn AS (
                       SELECT rowid, distance
                       FROM vec_index
                       WHERE embedding MATCH ?
                       ORDER BY distance
                       LIMIT ?
                   )
                   SELECT n.node_id, knn.distance, n.name, n.file, n.line, n.summary
                   FROM knn
                   JOIN nodes n ON n.rowid = knn.rowid""",
                [packed, limit],
            )
        )
        return [
            SearchResult(
                node_id=r[0],
                score=-r[1],  # negate: lower distance → higher score
                name=r[2],
                file=r[3],
                line=r[4],
                summary=r[5],
            )
            for r in rows
        ]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Escape a raw query string for FTS5 MATCH syntax.

        Wraps each whitespace-delimited term in double quotes so FTS5 treats
        them as literal tokens rather than interpreting special characters
        like ``*``, ``?``, ``-``, ``(``, ``)`` as query operators.
        """
        terms = query.split()
        if not terms:
            return '""'
        return " ".join(f'"{t}"' for t in terms)

    def bm25_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Return nodes ranked by BM25 full-text score."""
        fts_query = self._sanitize_fts_query(query)
        rows = list(
            self._conn.execute(
                """SELECT n.node_id, f.rank, n.name, n.file, n.line, n.summary
                   FROM fts_index f
                   JOIN nodes n ON n.rowid = f.rowid
                   WHERE fts_index MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                [fts_query, limit],
            )
        )
        return [
            SearchResult(
                node_id=r[0],
                score=r[1],  # FTS5 rank: negative float, more-negative = better match
                name=r[2],
                file=r[3],
                line=r[4],
                summary=r[5],
            )
            for r in rows
        ]

    def set_metadata(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", [key, value]
        )

    def get_metadata(self, key: str) -> str | None:
        rows = list(self._conn.execute("SELECT value FROM metadata WHERE key=?", [key]))
        return rows[0][0] if rows else None

    def get_all_metadata(self) -> dict[str, str]:
        rows = list(self._conn.execute("SELECT key, value FROM metadata"))
        return {r[0]: r[1] for r in rows}

    def get_node(self, node_id: str) -> NodeRecord | None:
        """Return the full NodeRecord for node_id, or None if not found."""
        rows = list(
            self._conn.execute(
                "SELECT node_id, kind, name, file, line, summary, content_hash "
                "FROM nodes WHERE node_id=?",
                [node_id],
            )
        )
        if not rows:
            return None
        r = rows[0]
        return NodeRecord(
            node_id=r[0], kind=r[1], name=r[2], file=r[3], line=r[4],
            summary=r[5], content_hash=r[6],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IndexStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        dim = self._embedding_dim
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                file TEXT,
                line INTEGER,
                summary TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(embedding float[{dim}])"
        )
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(name, summary)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
