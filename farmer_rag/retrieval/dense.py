"""Dense retrieval over the persisted Chroma collection of child chunks."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from farmer_rag.config import Settings


class DenseIndex:
    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    def search(self, query: str, k: int) -> list[str]:
        """Return child ids in descending similarity order."""
        results = self._store.similarity_search(query, k=k)
        return [str(doc.metadata["child_id"]) for doc in results if "child_id" in doc.metadata]
