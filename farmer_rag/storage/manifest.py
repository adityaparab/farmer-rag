"""Index manifest: records what the persisted index was built with.

Retrieval refuses to run against an index whose embedding model or chunking
parameters no longer match the current settings — silently querying a stale
index is the classic hard-to-notice RAG failure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from farmer_rag.config import Settings

SCHEMA_VERSION = 1


class IndexError_(Exception):
    """Raised when the index is missing or incompatible with current settings."""


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    pdf_name: str
    pdf_sha256: str
    pdf_pages: int
    embedding_provider: str
    embedding_model: str
    child_chunk_tokens: int
    parent_chunk_tokens: int
    contextualized: bool
    n_parents: int
    n_children: int
    created_at: str

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        pdf_name: str,
        pdf_sha256: str,
        pdf_pages: int,
        n_parents: int,
        n_children: int,
    ) -> IndexManifest:
        return cls(
            schema_version=SCHEMA_VERSION,
            pdf_name=pdf_name,
            pdf_sha256=pdf_sha256,
            pdf_pages=pdf_pages,
            embedding_provider=settings.embedding_provider.value,
            embedding_model=settings.embedding_model,
            child_chunk_tokens=settings.child_chunk_tokens,
            parent_chunk_tokens=settings.parent_chunk_tokens,
            contextualized=settings.contextualize_chunks,
            n_parents=n_parents,
            n_children=n_children,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> IndexManifest:
        if not path.exists():
            raise IndexError_(
                "No index found. Run `farmer-rag ingest <pdf>` first (ingestion is CLI-only)."
            )
        try:
            manifest = cls(**json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError) as exc:  # corrupt JSON or schema drift
            raise IndexError_(
                "Index manifest is corrupt or unreadable — re-ingest with"
                " `farmer-rag ingest <pdf> --force`."
            ) from exc
        if manifest.schema_version != SCHEMA_VERSION:
            raise IndexError_(
                f"Index manifest schema v{manifest.schema_version} does not match the"
                f" app (v{SCHEMA_VERSION}) — re-ingest with `farmer-rag ingest <pdf> --force`."
            )
        return manifest

    def check_compatible(self, settings: Settings) -> None:
        mismatches: list[str] = []
        if self.embedding_provider != settings.embedding_provider.value:
            mismatches.append(
                f"embedding provider: index={self.embedding_provider},"
                f" settings={settings.embedding_provider.value}"
            )
        if self.embedding_model != settings.embedding_model:
            mismatches.append(
                f"embedding model: index={self.embedding_model},"
                f" settings={settings.embedding_model}"
            )
        if self.child_chunk_tokens != settings.child_chunk_tokens:
            mismatches.append(
                f"child chunk tokens: index={self.child_chunk_tokens},"
                f" settings={settings.child_chunk_tokens}"
            )
        if self.parent_chunk_tokens != settings.parent_chunk_tokens:
            mismatches.append(
                f"parent chunk tokens: index={self.parent_chunk_tokens},"
                f" settings={settings.parent_chunk_tokens}"
            )
        if self.contextualized != settings.contextualize_chunks:
            mismatches.append(
                f"contextualized chunks: index={self.contextualized},"
                f" settings={settings.contextualize_chunks}"
            )
        if mismatches:
            raise IndexError_(
                "Index is incompatible with current settings — re-ingest with"
                " `farmer-rag ingest <pdf> --force`.\n  - " + "\n  - ".join(mismatches)
            )
