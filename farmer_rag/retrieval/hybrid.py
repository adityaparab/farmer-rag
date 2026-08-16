"""Reciprocal-rank fusion of dense and sparse rankings across query variants."""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    """Fuse ranked id lists with RRF: score(id) = Σ 1 / (k + rank).

    Ids appearing high in several rankings (dense + BM25, multiple query
    variants) win. Returns ids in descending fused-score order.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
