"""Application settings, loaded from environment variables / `.env`.

Every knob has a sane local-dev default; switching LLM provider is a single
`LLM_PROVIDER` change. Groq exposes no embeddings endpoint, so embeddings only
support `local` and `openai` (enforced by the `EmbeddingProvider` enum).
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(enum.StrEnum):
    LOCAL = "local"  # any OpenAI-compatible server: llama.cpp llama-server, vLLM, LM Studio
    OPENAI = "openai"
    GROQ = "groq"


class EmbeddingProvider(enum.StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Chat LLM ---
    # `local` targets an OpenAI-compatible server; for llama.cpp run e.g.
    #   llama-server -m <chat-model>.gguf --port 8080
    llm_provider: LLMProvider = LLMProvider.LOCAL
    llm_model: str = "qwen3-8b"
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = "local"
    llm_temperature: float = 0.1
    # Local reasoning models think before answering; a big prompt on a cold
    # single-slot server can take minutes. A short timeout with retries makes
    # it worse: the aborted request keeps occupying the server while the retry
    # queues behind it.
    llm_timeout: float = Field(default=300.0, description="Per-request timeout in seconds")
    llm_max_retries: int = Field(default=1, description="HTTP retries per LLM request")
    # Streaming needs a gateway whose SSE reliably terminates. If answers hang
    # while the model server sits idle (connections stuck in CLOSE_WAIT), the
    # proxy is dropping final stream chunks — set this to false.
    llm_streaming: bool = Field(
        default=True, description="Stream answer tokens; disable for flaky SSE gateways"
    )

    # --- Embeddings ---
    # llama.cpp: llama-server -m <embedding-model>.gguf --embeddings --port 8081
    # The model name is informational to llama.cpp but is recorded in the index
    # manifest — keep it matching the GGUF you actually serve.
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "qwen3-embedding-0.6b"
    embedding_base_url: str = "http://localhost:8081/v1"
    embedding_api_key: str = "local"

    # --- Reranker (served over HTTP, e.g. llama.cpp /v1/rerank) ---
    # llama.cpp: llama-server -m <reranker-model>.gguf --rerank --port 8082
    reranker_enabled: bool = True
    reranker_model: str = "bge-reranker-v2-m3"
    reranker_base_url: str = "http://localhost:8082/v1"
    reranker_api_key: str = "local"
    # Rerank servers hard-limit tokens per query+document pair (llama.cpp: the
    # physical batch size, often 512). Token-dense chunks (tables, Latin names)
    # can hit ~1.7 chars/token, so cap conservatively; rerankers weigh early
    # content most, so truncation barely affects scoring.
    rerank_max_doc_chars: int = 600

    # --- Ingestion ---
    contextualize_chunks: bool = False
    child_chunk_tokens: int = 250
    parent_chunk_tokens: int = 1000

    # --- Retrieval ---
    dense_top_k: int = 12
    sparse_top_k: int = 12
    rrf_k: int = 60
    rerank_candidates: int = 30
    top_parents: int = 6
    max_query_variants: int = 3

    # --- Corrective loop ---
    grading_enabled: bool = True
    max_retrieval_retries: int = 1

    # --- Storage / misc ---
    data_dir: Path = Path("data")
    log_level: str = "INFO"

    # Not env-configurable: a changed collection name would silently point
    # dense retrieval at an empty Chroma collection (BM25-only degradation).
    collection_name: ClassVar[str] = "farmer_rag_children"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def docstore_path(self) -> Path:
        return self.index_dir / "docstore.db"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @model_validator(mode="after")
    def _check_cloud_keys(self) -> Settings:
        placeholder_keys = {"", "local", "ollama", "changeme", "your-api-key"}
        if self.llm_provider is not LLMProvider.LOCAL and self.llm_api_key in placeholder_keys:
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider.value} requires a real LLM_API_KEY"
            )
        if (
            self.embedding_provider is not EmbeddingProvider.LOCAL
            and self.embedding_api_key in placeholder_keys
        ):
            raise ValueError(
                f"EMBEDDING_PROVIDER={self.embedding_provider.value} requires a real"
                " EMBEDDING_API_KEY"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
