"""SQLite-backed store for parent sections and child chunks.

The docstore is the single source of truth shared by both pipelines:
ingestion writes parents + children here (children additionally get their
``search_text`` embedded into Chroma), and retrieval reads children to build
the BM25 index and maps reranked children back to their parent sections.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    id         TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    section    TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS children (
    id          TEXT PRIMARY KEY,
    parent_id   TEXT NOT NULL REFERENCES parents(id),
    content     TEXT NOT NULL,
    search_text TEXT NOT NULL,
    section     TEXT NOT NULL,
    page_start  INTEGER NOT NULL,
    page_end    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_children_parent ON children(parent_id);
"""


@dataclass(frozen=True)
class ParentRecord:
    id: str
    content: str
    section: str  # heading path, e.g. "Ch. 4 Fungal diseases > Powdery mildew"
    page_start: int
    page_end: int


@dataclass(frozen=True)
class ChildRecord:
    id: str
    parent_id: str
    content: str
    search_text: str  # heading prefix + optional contextual blurb + content
    section: str
    page_start: int
    page_end: int


class DocStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM children")
            self._conn.execute("DELETE FROM parents")

    def add_parents(self, parents: list[ParentRecord]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO parents VALUES (?, ?, ?, ?, ?)",
                [(p.id, p.content, p.section, p.page_start, p.page_end) for p in parents],
            )

    def add_children(self, children: list[ChildRecord]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO children VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.id, c.parent_id, c.content, c.search_text, c.section,
                     c.page_start, c.page_end)
                    for c in children
                ],
            )

    def all_children(self) -> list[ChildRecord]:
        rows = self._conn.execute(
            "SELECT id, parent_id, content, search_text, section, page_start, page_end"
            " FROM children ORDER BY rowid"
        ).fetchall()
        return [ChildRecord(*row) for row in rows]

    def get_parents(self, ids: list[str]) -> dict[str, ParentRecord]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, content, section, page_start, page_end FROM parents"
            f" WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {row[0]: ParentRecord(*row) for row in rows}

    def counts(self) -> tuple[int, int]:
        (n_parents,) = self._conn.execute("SELECT COUNT(*) FROM parents").fetchone()
        (n_children,) = self._conn.execute("SELECT COUNT(*) FROM children").fetchone()
        return n_parents, n_children
