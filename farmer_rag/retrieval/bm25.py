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
        order = np.argsort(scores)[::-1][:k]
        return [self._children[i].id for i in order if scores[i] > 0.0]
