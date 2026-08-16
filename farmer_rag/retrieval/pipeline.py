"""Retrieval facade: hybrid search over children, rerank, map to parent sections.

This is the only entry point the generation layer uses. It owns index loading,
manifest compatibility checks, and the child→parent small-to-big expansion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from farmer_rag.config import Settings
from farmer_rag.models import build_embeddings
from farmer_rag.retrieval.bm25 import BM25Index
from farmer_rag.retrieval.dense import DenseIndex
from farmer_rag.retrieval.hybrid import rrf_fuse
from farmer_rag.retrieval.reranker import Reranker
from farmer_rag.storage.docstore import ChildRecord, DocStore, ParentRecord
from farmer_rag.storage.manifest import IndexError_, IndexManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedParent:
    record: ParentRecord
    score: float | None  # best child rerank score; None when reranking is off


class BookRetriever:
    """Loads the persisted index once; ``retrieve`` is then cheap to call."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.manifest = IndexManifest.load(settings.manifest_path)
        self.manifest.check_compatible(settings)
        # Snapshot marker: if the index is rebuilt while this retriever lives
        # (CLI re-ingest under a running web app), refuse to serve mixed data.
        self._manifest_mtime_ns = settings.manifest_path.stat().st_mtime_ns

        # One consistent in-memory snapshot of children AND parents — parent
        # ids are sequential per build, so mixing an old child match with a
        # freshly re-read docstore would silently return the wrong section.
        docstore = DocStore(settings.docstore_path)
        try:
            children = docstore.all_children()
            self._parents_by_id: dict[str, ParentRecord] = docstore.all_parents()
        finally:
            docstore.close()
        if not children:
            raise ValueError("Docstore is empty — re-run `farmer-rag ingest <pdf> --force`")
        self._children_by_id: dict[str, ChildRecord] = {c.id: c for c in children}
        self._bm25 = BM25Index(children)
        self._dense = DenseIndex(settings, build_embeddings(settings))
        self._reranker = Reranker(settings) if settings.reranker_enabled else None
        logger.info(
            "Retriever ready: %d children, reranker=%s",
            len(children),
            settings.reranker_model if self._reranker else "disabled",
        )

    def retrieve(self, queries: list[str]) -> list[RetrievedParent]:
        """Hybrid-retrieve for every query variant, fuse, rerank, expand to parents."""
        self._check_index_unchanged()
        settings = self._settings
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        if not queries:
            return []

        rankings: list[list[str]] = []
        for query in queries:
            rankings.append(self._dense.search(query, settings.dense_top_k))
            rankings.append(self._bm25.search(query, settings.sparse_top_k))
        fused_ids = rrf_fuse(rankings, k=settings.rrf_k)[: settings.rerank_candidates]
        candidates = [self._children_by_id[cid] for cid in fused_ids if cid in self._children_by_id]
        if not candidates:
            return []

        scored: list[tuple[ChildRecord, float | None]]
        if self._reranker is not None:
            # Rerank against the primary (first) query — variants only widen recall.
            scores = self._reranker.rerank(queries[0], [c.search_text for c in candidates])
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            scored = [(candidates[i], scores[i]) for i in order]
        else:
            scored = [(c, None) for c in candidates]

        # Children → unique parents, keeping the best child's rank and score.
        best_by_parent: dict[str, float | None] = {}
        parent_order: list[str] = []
        for child, score in scored:
            if child.parent_id not in best_by_parent:
                best_by_parent[child.parent_id] = score
                parent_order.append(child.parent_id)
        parent_order = parent_order[: settings.top_parents]

        results = [
            RetrievedParent(record=self._parents_by_id[pid], score=best_by_parent[pid])
            for pid in parent_order
            if pid in self._parents_by_id
        ]
        logger.debug(
            "Retrieved %d parents for %r (variants=%d)", len(results), queries[0], len(queries)
        )
        return results

    def _check_index_unchanged(self) -> None:
        try:
            current_mtime_ns = self._settings.manifest_path.stat().st_mtime_ns
        except FileNotFoundError as exc:
            raise IndexError_(
                "The index was removed while the app was running — re-ingest and restart."
            ) from exc
        if current_mtime_ns != self._manifest_mtime_ns:
            raise IndexError_(
                "The index was rebuilt while the app was running — restart the app to load it."
            )
