# farmer-rag 🌱

Production-grade RAG application that answers questions **strictly from the contents** of a
book on applying homeopathy to plants and farming. Hybrid retrieval (dense + BM25) with local
cross-encoder reranking, a corrective LangGraph flow, page-cited answers, and a hard rule:
if the book doesn't cover it, the app says so instead of guessing.

See [PROJECT.md](PROJECT.md) for the architecture rationale and implementation plan.

## Requirements

- Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)
- For local development: an OpenAI-compatible model server, e.g. [Ollama](https://ollama.com):

```bash
ollama pull qwen3:8b
```

```bash
ollama pull qwen3-embedding:0.6b
```

The reranker (`BAAI/bge-reranker-v2-m3`) is downloaded from Hugging Face on first use and
runs in-process (MPS/CUDA/CPU) — no server needed.

## Setup

```bash
uv sync
```

```bash
cp .env.example .env
```

The defaults in `.env.example` target local Ollama. To use a cloud provider, set
`LLM_PROVIDER=groq` (or `openai`) plus `LLM_MODEL` and `LLM_API_KEY` — no code changes.
Embeddings support `local` and `openai` only (Groq has no embeddings API). **Changing the
embedding model requires re-ingestion.**

## Usage

Ingestion is a manual, CLI-only step (the web UI cannot ingest — by design):

```bash
uv run farmer-rag ingest path/to/book.pdf
```

Ask a single question:

```bash
uv run farmer-rag ask "What does the book recommend for powdery mildew?"
```

Interactive chat in the terminal:

```bash
uv run farmer-rag chat
```

Web UI (Streamlit chat, query-only):

```bash
uv run farmer-rag web
```

Index/config state:

```bash
uv run farmer-rag status
```

Retrieval quality eval against a golden question set (no LLM calls; see
`eval/golden.example.yaml`):

```bash
uv run farmer-rag eval eval/golden.yaml
```

## How answers stay grounded

1. Retrieval returns whole book *sections* (small-to-big), each labelled with its heading
   path and page range.
2. The answer prompt forbids outside knowledge, requires a `[n]` citation after every claim,
   and attributes recommendations to the book.
3. Citation indices are validated after generation; sources with page numbers are shown
   under every answer.
4. When retrieval finds nothing relevant (after one corrective query rewrite), the app
   abstains rather than falling back to model knowledge.

## Development

```bash
uv run pytest
```

```bash
uv run pyright
```

```bash
uv run ruff check .
```
