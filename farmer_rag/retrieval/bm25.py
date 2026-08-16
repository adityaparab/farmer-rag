"""In-memory BM25 over the docstore's child chunks.

Built at retriever startup from the persisted docstore (a few thousand chunks
for one book — sub-second). Unicode word tokenization keeps Latin remedy names
and Devanagari text searchable alike.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from farmer_rag.storage.docstore import ChildRecord

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, children: list[ChildRecord]) -> None:
        if not children:
            raise ValueError("BM25 index requires at least one chunk")
        self._children = children
        self._bm25 = BM25Okapi([tokenize(c.search_text) for c in children])

    def search(self, query: str, k: int) -> list[str]:
        """Return child ids in descending BM25 relevance order (positive scores only)."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        if scores.max() > 0.0:
            order = np.argsort(scores)[::-1][:k]
            return [self._children[i].id for i in order if scores[i] > 0.0]
        # Tiny corpora (1-2 chunks): BM25Okapi's idf floor goes non-positive for
        # every term, zeroing all scores. Fall back to plain token overlap so
        # exact-term matching still works.
        overlap = [
            sum(1 for t in set(tokens) if t in doc_freqs) for doc_freqs in self._bm25.doc_freqs
        ]
        order = sorted(range(len(overlap)), key=lambda i: overlap[i], reverse=True)[:k]
        return [self._children[i].id for i in order if overlap[i] > 0]
