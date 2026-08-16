"""In-process cross-encoder reranker.

Ollama cannot serve rerankers (no rerank API; community uploads only expose the
embedding layer, producing wrong scores), so the cross-encoder runs in-process
via sentence-transformers on MPS/CUDA/CPU. The model downloads from Hugging
Face on first use and is cached locally afterwards.
"""

from __future__ import annotations

import logging

from farmer_rag.config import RerankerDevice, Settings

logger = logging.getLogger(__name__)


def _resolve_device(device: RerankerDevice) -> str:
    if device is not RerankerDevice.AUTO:
        return device.value
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Reranker:
    def __init__(self, settings: Settings) -> None:
        # Deferred import: torch is heavy and only needed when reranking is on.
        from sentence_transformers import CrossEncoder

        device = _resolve_device(settings.reranker_device)
        logger.info("Loading reranker %s on %s", settings.reranker_model, device)
        self._model = CrossEncoder(settings.reranker_model, device=device)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query (higher = more relevant)."""
        if not texts:
            return []
        scores = self._model.predict([(query, text) for text in texts])
        return [float(s) for s in scores]
