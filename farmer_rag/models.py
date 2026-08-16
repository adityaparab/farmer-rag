"""Factories for chat models and embeddings, switched by provider config.

Local providers are any OpenAI-compatible server (llama.cpp ``llama-server``,
vLLM, LM Studio). Two quirks are handled here so the rest of the codebase never thinks
about providers:

- ``use_responses_api=False``: ChatOpenAI may infer the Responses API from the
  model name; local servers only implement ``/chat/completions``.
- ``check_embedding_ctx_length=False``: the default tokenizes with tiktoken and
  sends token arrays, which non-OpenAI servers reject; ``False`` sends strings.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from farmer_rag.config import EmbeddingProvider, LLMProvider, Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    match settings.llm_provider:
        case LLMProvider.LOCAL:
            return ChatOpenAI(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,  # type: ignore[arg-type]  # coerced to SecretStr
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout,
                use_responses_api=False,
                max_retries=2,
            )
        case LLMProvider.OPENAI:
            return ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,  # type: ignore[arg-type]
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout,
                max_retries=2,
            )
        case LLMProvider.GROQ:
            return ChatGroq(
                model=settings.llm_model,
                api_key=settings.llm_api_key,  # type: ignore[arg-type]
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout,
                max_retries=2,
            )


def build_embeddings(settings: Settings) -> Embeddings:
    match settings.embedding_provider:
        case EmbeddingProvider.LOCAL:
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,  # type: ignore[arg-type]
                check_embedding_ctx_length=False,
                chunk_size=32,  # local servers choke on OpenAI's default 1000-doc batches
            )
        case EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.embedding_api_key,  # type: ignore[arg-type]
            )
