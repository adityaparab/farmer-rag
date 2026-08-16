"""Persist chunks: children → Chroma (dense) + docstore (BM25 source + parents)."""

from __future__ import annotations

import logging
import shutil

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from farmer_rag.config import Settings
from farmer_rag.storage.docstore import ChildRecord, DocStore, ParentRecord

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64


def wipe_index(settings: Settings) -> None:
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.docstore_path.unlink(missing_ok=True)
    settings.manifest_path.unlink(missing_ok=True)


def index_chunks(
    settings: Settings,
    embeddings: Embeddings,
    parents: list[ParentRecord],
    children: list[ChildRecord],
) -> None:
    store = DocStore(settings.docstore_path)
    try:
        store.add_parents(parents)
        store.add_children(children)
    finally:
        store.close()
    logger.info("Docstore written: %d parents, %d children", len(parents), len(children))

    vectors = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    for start in range(0, len(children), _EMBED_BATCH):
        batch = children[start : start + _EMBED_BATCH]
        vectors.add_texts(
            texts=[c.search_text for c in batch],
            metadatas=[
                {
                    "child_id": c.id,
                    "parent_id": c.parent_id,
                    "section": c.section,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
                for c in batch
            ],
            ids=[c.id for c in batch],
        )
        done = min(start + _EMBED_BATCH, len(children))
        logger.info("Embedded %d/%d children", done, len(children))
