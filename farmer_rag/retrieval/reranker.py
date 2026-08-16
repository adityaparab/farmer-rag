"""Cross-encoder reranking via an HTTP rerank endpoint.

Targets llama.cpp's ``llama-server`` started with ``--rerank`` (a Jina-style
API also offered by other rerank servers): POST ``{base_url}/rerank`` with
``{model, query, top_n, documents}`` returns ``{"results": [{"index": i,
"relevance_score": s}, ...]}``. Keeping the reranker behind HTTP means the app
itself carries no torch/model weights.

Failures raise ``RerankError`` with an actionable hint rather than silently
degrading to un-reranked results — disabling reranking is an explicit config
choice (``RERANKER_ENABLED=false``), never an accident.
"""

from __future__ import annotations

import logging

import httpx

from farmer_rag.config import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """The rerank server is unreachable or returned an unusable response."""


class Reranker:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        base_url = settings.reranker_base_url.rstrip("/")
        self._url = f"{base_url}/rerank"
        self._model = settings.reranker_model
        self._hint = (
            f"Is your reranker server running ({base_url})? Start it with e.g."
            " `llama-server -m <reranker>.gguf --rerank --port 8082`, or set"
            " RERANKER_ENABLED=false to run without reranking."
        )
        headers = {"Authorization": f"Bearer {settings.reranker_api_key}"}
        self._client = client or httpx.Client(headers=headers, timeout=settings.llm_timeout)
        logger.info("Reranker: %s (model %s)", self._url, self._model)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query (higher = more relevant)."""
        if not texts:
            return []
        try:
            response = self._client.post(
                self._url,
                json={
                    "model": self._model,
                    "query": query,
                    "top_n": len(texts),
                    "documents": texts,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise RerankError(f"Rerank request failed. {self._hint}") from exc

        scores = [0.0] * len(texts)
        try:
            for item in payload["results"]:
                scores[int(item["index"])] = float(item["relevance_score"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            preview = repr(payload)[:200]
            raise RerankError(
                f"Unexpected rerank response from {self._url}: {preview}"
            ) from exc
        return scores
