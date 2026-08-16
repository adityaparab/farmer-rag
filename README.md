# farmer-rag 🌱

Production-grade RAG application that answers questions **strictly from the contents** of a
book on applying homeopathy to plants and farming. Hybrid retrieval (dense + BM25) with local
cross-encoder reranking, a corrective LangGraph flow, page-cited answers, and a hard rule:
if the book doesn't cover it, the app says so instead of guessing.

See [PROJECT.md](PROJECT.md) for the architecture rationale and implementation plan.

## Requirements

- Python ≥ 3.12, [uv](https://docs.astral.sh/uv/)
- For local development: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`
  (OpenAI-compatible), one instance per role with your chosen GGUF models:

```bash
llama-server -m qwen3-8b-instruct.gguf --port 8080
```

```bash
llama-server -m qwen3-embedding-0.6b.gguf --embeddings --port 8081
```

```bash
llama-server -m bge-reranker-v2-m3.gguf --rerank --port 8082
```

The chat and embedding servers speak the OpenAI `/v1` API; the reranker uses llama.cpp's
Jina-style `/v1/rerank` endpoint. Any other OpenAI-compatible stack (vLLM, LM Studio) works
for chat/embeddings by pointing the base URLs at it; reranking needs a `/v1/rerank`-speaking
server (or set `RERANKER_ENABLED=false`).

## Setup

```bash
uv sync
```

```bash
cp .env.example .env
```

The defaults in `.env.example` target the three local llama.cpp servers above. To use a
cloud provider, set
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

## Troubleshooting slow or "hanging" answers

With local models the first query after a server (re)start is the worst case: the model
loads into memory and processes the full prompt before a single token appears. Reasoning
models additionally "think" before answering — the CLI and web UI show live pipeline status
("Searching the book…", "Writing the answer…"), and with a reasoning model expect a silent
gap before the first word of the answer.

If answers time out:

- Raise `LLM_TIMEOUT` (default 300s) rather than lowering it — and keep `LLM_MAX_RETRIES`
  low (default 1). On a single-slot server an aborted request keeps the server busy while
  the retry queues behind it, which compounds into a pile-up that looks like a hang.
- If answers hang **while the model server sits idle**, your gateway/proxy is dropping the
  final SSE stream chunks (symptom: the app's connections stuck in `CLOSE_WAIT`). Set
  `LLM_STREAMING=false` — answers then arrive in one piece over plain HTTP, which is
  immune to this; the live pipeline status still shows. Internal pipeline calls
  (query understanding, grading) never use streaming regardless.
- `GRADING_ENABLED=false` removes one LLM call per question if you need speed over the
  corrective loop.
- `farmer-rag eval` exercises retrieval without any LLM calls — use it to isolate whether
  a problem is retrieval or generation.

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
