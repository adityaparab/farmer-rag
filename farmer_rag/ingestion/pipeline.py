"""Ingestion orchestration. Manual CLI trigger only — never imported by the web UI."""

from __future__ import annotations

import logging
from pathlib import Path

from farmer_rag.config import Settings
from farmer_rag.ingestion.chunker import chunk_book
from farmer_rag.ingestion.contextualizer import contextualize_children
from farmer_rag.ingestion.indexer import index_chunks, wipe_index
from farmer_rag.ingestion.parser import parse_pdf, sha256_of
from farmer_rag.models import build_chat_model, build_embeddings
from farmer_rag.storage.manifest import IndexManifest

logger = logging.getLogger(__name__)


def ingest_pdf(settings: Settings, pdf_path: Path, *, force: bool = False) -> IndexManifest:
    """Parse, chunk, (optionally) contextualize, embed, and persist the book.

    Idempotent: if the same PDF (by content hash) is already indexed with the
    current settings, ingestion is skipped unless ``force`` is set.
    """
    if settings.manifest_path.exists() and not force:
        existing = IndexManifest.load(settings.manifest_path)
        if existing.pdf_sha256 == sha256_of(pdf_path):
            try:
                existing.check_compatible(settings)
            except Exception:
                logger.info("Existing index incompatible with settings; rebuilding")
            else:
                logger.info(
                    "'%s' already ingested (%d parents / %d children) — use --force to rebuild",
                    existing.pdf_name,
                    existing.n_parents,
                    existing.n_children,
                )
                return existing

    parsed = parse_pdf(pdf_path)
    result = chunk_book(
        parsed,
        child_tokens=settings.child_chunk_tokens,
        parent_tokens=settings.parent_chunk_tokens,
    )

    children = result.children
    if settings.contextualize_chunks:
        logger.info(
            "Contextualizing %d chunks via %s/%s (this makes one LLM call per chunk)",
            len(children),
            settings.llm_provider.value,
            settings.llm_model,
        )
        children = contextualize_children(build_chat_model(settings), result.parents, children)

    logger.info("Wiping any previous index")
    wipe_index(settings)
    index_chunks(settings, build_embeddings(settings), result.parents, children)

    manifest = IndexManifest.build(
        settings,
        pdf_name=parsed.name,
        pdf_sha256=parsed.sha256,
        pdf_pages=parsed.n_pages,
        n_parents=len(result.parents),
        n_children=len(children),
    )
    manifest.save(settings.manifest_path)
    logger.info(
        "Ingestion complete: %d parents / %d children indexed", len(result.parents), len(children)
    )
    return manifest
