import json

import httpx
import pytest

from farmer_rag.config import Settings
from farmer_rag.retrieval.reranker import Reranker, RerankError


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_scores_map_back_to_input_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/rerank")
        body = json.loads(request.content)
        assert body["query"] == "powdery mildew"
        assert body["documents"] == ["a", "b", "c"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                    {"index": 1, "relevance_score": 0.1},
                ]
            },
        )

    reranker = Reranker(_settings(), client=_client(handler))
    assert reranker.rerank("powdery mildew", ["a", "b", "c"]) == [0.4, 0.1, 0.9]


def test_empty_input_makes_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert Reranker(_settings(), client=_client(handler)).rerank("q", []) == []


def test_server_down_raises_actionable_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    reranker = Reranker(_settings(), client=_client(handler))
    with pytest.raises(RerankError, match="RERANKER_ENABLED=false"):
        reranker.rerank("q", ["a"])


def test_documents_and_query_are_truncated_for_the_server():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    reranker = Reranker(_settings(), client=_client(handler))
    reranker.rerank("q" * 1000, ["d" * 5000])
    assert len(seen["query"]) == 300
    assert len(seen["documents"][0]) == 600  # default rerank_max_doc_chars


def test_server_error_surfaces_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "input (703 tokens) is too large"})

    reranker = Reranker(_settings(), client=_client(handler))
    with pytest.raises(RerankError, match="703 tokens") as excinfo:
        reranker.rerank("q", ["a"])
    assert "HTTP 500" in str(excinfo.value)
    assert "RERANK_MAX_DOC_CHARS" in str(excinfo.value)


def test_malformed_response_raises_rerank_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    reranker = Reranker(_settings(), client=_client(handler))
    with pytest.raises(RerankError, match="Unexpected rerank response"):
        reranker.rerank("q", ["a"])
