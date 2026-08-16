"""Application settings, loaded from environment variables / `.env`.

Every knob has a sane local-dev default; switching LLM provider is a single
`LLM_PROVIDER` change. Groq exposes no embeddings endpoint, so embeddings only
support `local` and `openai` (enforced by the `EmbeddingProvider` enum).
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(enum.StrEnum):
    LOCAL = "local"  # any OpenAI-compatible server: Ollama /v1, LM Studio, vLLM
    OPENAI = "openai"
    GROQ = "groq"


class EmbeddingProvider(enum.StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class RerankerDevice(enum.StrEnum):
    AUTO = "auto"
    MPS = "mps"
    CPU = "cpu"
    CUDA = "cuda"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Chat LLM ---
    llm_provider: LLMProvider = LLMProvider.LOCAL
    llm_model: str = "qwen3:8b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.1
    llm_timeout: float = Field(default=120.0, description="Per-request timeout in seconds")

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: str = "ollama"

    # --- Reranker (in-process cross-encoder) ---
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: RerankerDevice = RerankerDevice.AUTO

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
    collection_name: str = "farmer_rag_children"
    log_level: str = "INFO"

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
        placeholder_keys = {"", "ollama", "changeme", "your-api-key"}
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
