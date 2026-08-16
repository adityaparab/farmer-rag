import pytest
from pydantic import ValidationError

from farmer_rag.config import EmbeddingProvider, LLMProvider, Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_local_defaults_need_no_keys():
    settings = _settings()
    assert settings.llm_provider is LLMProvider.LOCAL
    assert settings.embedding_provider is EmbeddingProvider.LOCAL
    assert settings.chroma_dir == settings.data_dir / "index" / "chroma"


def test_cloud_llm_requires_real_key():
    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        _settings(llm_provider="groq")
    settings = _settings(llm_provider="groq", llm_api_key="gsk_real")
    assert settings.llm_provider is LLMProvider.GROQ


def test_cloud_embeddings_require_real_key():
    with pytest.raises(ValidationError, match="EMBEDDING_API_KEY"):
        _settings(embedding_provider="openai")


def test_groq_is_not_a_valid_embedding_provider():
    with pytest.raises(ValidationError):
        _settings(embedding_provider="groq", embedding_api_key="gsk_x")
