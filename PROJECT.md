# farmer-rag — Implementation Plan

Production-grade RAG application answering questions **strictly from the contents** of a
300+ page book on the application of homeopathy in plants/farming. CLI + Streamlit web UI,
LangChain ecosystem, provider-switchable LLMs via `.env`.

---

## 1. Architecture decision

The corpus is a *single book* (~120–200K tokens). That constraint drove every choice below.
Strategies were researched against current (2025–26) evidence, including Anthropic's
contextual-retrieval benchmarks, the CRAG/Self-RAG papers, Microsoft GraphRAG, and chunking
evaluations (arXiv 2504.19754).

### Adopted

| Technique | Why |
|---|---|
| **Hybrid retrieval** (dense + BM25, RRF fusion) | Consensus production default. BM25 is critical here: remedy names (*Silicea*, *Arnica*), Latin plant/pest names, and potencies (6CH, 30CH, 200C) are exact-match tokens that dense embeddings handle poorly; dense catches paraphrased symptom descriptions. |
| **Cross-encoder reranking** (local, via llama.cpp) | Single highest-ROI post-retrieval step. Anthropic's measurements: hybrid + rerank cuts top-20 retrieval failure from 5.7% → 1.9%. Served by a llama.cpp `llama-server --rerank` instance over its Jina-style `/v1/rerank` HTTP endpoint (`bge-reranker-v2-m3` GGUF) — the app itself carries no torch/model weights. |
| **Parent/child (small-to-big) chunking** | Children (~250 tokens, heading-path prefixed) are embedded for precise matching; the *parent section* (~1000 tokens) is what the LLM sees. Farmer questions ("what do I spray for aphids?") need the full remedy section — dosage, preparation, cautions — not one sentence. Near-zero cost, 15–30% reported gains. |
| **Query understanding step** (1 cheap LLM call) | The dominant expected failure mode is vocabulary mismatch: farmers ask "white powder on tomato leaves", the book says "powdery mildew … Silicea". One call normalizes phrasing into book vocabulary, emits 2–3 retrieval variants, and condenses chat history into a standalone question. Variants are unioned pre-rerank. |
| **Corrective loop (CRAG-lite) in LangGraph** | The useful fragment of Agentic RAG for a closed corpus: grade reranked context → if weak, rewrite the query and retry once → if still weak, **abstain** (never fall back to model knowledge). Bounded retries; deterministic graph, not free agent tool-calling — more reliable with small local models. |
| **Structural grounding + citations** | Contexts are numbered with section titles + page ranges; the prompt requires per-claim `[n]` citations, attributes claims to *the book* ("the book recommends…" — not asserted as established agronomic fact), and mandates abstention when the book is silent. Cited indices are verified post-hoc and rendered as a sources list with page numbers. |
| **Contextual chunk blurbs** (optional, config flag) | Anthropic-style chunk contextualization: an LLM-written blurb situating each chunk in its chapter, prepended before embedding/BM25 indexing (35–49% retrieval-failure reduction). Off by default for local dev (slow on a local 8B); recommended when ingesting with a cloud provider. |

### Rejected (and why)

| Technique | Reason |
|---|---|
| **GraphRAG** | Worst cost/benefit here: LLM entity extraction over every chunk, a graph store to maintain, and its genuine advantage (global "what does the book say overall about X" questions) is better served later by a long-context escalation tier — the whole book fits in a modern context window. Revisit only if the corpus grows to many books. |
| **Full Self-RAG / 9-node agentic graphs** | 2–4× LLM calls and real debugging cost; headline CRAG gains lean on a web-search fallback that is *forbidden* here (book-only answers). We keep only grade → rewrite → retry → abstain. |
| **Sentence-window retrieval** | Empirically **underperforms** naive chunking (−23% in 2025 multi-dataset eval). |
| **HyDE** | Hurts well-formed queries (~−5%), adds a generation per query; mixed evidence. Multi-query rewriting wins. |
| **Fine-tuning** | Unnecessary, anti-grounding for a closed-book QA task. |

### Future enhancements (documented, not built now)

- **Long-context escalation tier**: for synthesis questions ("summarize the book's approach to
  soil health") or repeated low-confidence retrievals, send the *entire book* to a large-context
  cloud model with prompt caching. Replaces GraphRAG/RAPTOR entirely at this corpus size.
- **RAGAS-based answer evals** on top of the built-in retrieval eval harness.
- **Multilingual queries** (Hindi/Marathi): the chosen embedder (`qwen3-embedding` / `bge-m3`)
  and reranker are already multilingual; add query-language detection + answer-language control.

---

## 2. System design

Two strictly separate pipelines sharing only the storage layer and config.

### Ingestion (manual, CLI-only — never exposed in the web UI)

```
PDF ──► pymupdf4llm (markdown + page map)
    ──► structure-aware chunker (headings → parent sections ~1000 tok
                                  → child chunks ~250 tok, heading-path prefix)
    ──► [optional] contextual blurb per child (LLM, config flag)
    ──► indexer ──► Chroma (dense child vectors, persisted)
                ──► SQLite docstore (parents + children + metadata, feeds BM25)
                ──► manifest.json (pdf hash/pages, embedding provider/model,
                                   chunk params, contextualized flag, counts,
                                   timestamp)
```

The manifest makes index/config drift a **hard, clear error** at query time (e.g. switching
embedding models without re-ingesting).

### Retrieval + answer (LangGraph)

```
question + chat history
  └─► understand   : condense history → standalone question; emit ≤3 book-vocabulary variants
  └─► retrieve     : per variant → dense top-k (Chroma) + BM25 top-k (docstore)
                     → RRF fusion (k=60) → union → cross-encoder rerank
                       (llama.cpp /v1/rerank over HTTP)
                     → map children → parent sections, dedupe → top N parents
  └─► grade        : LLM judges context sufficiency (JSON, tolerant parsing)
        ├─ sufficient ──► generate
        └─ weak & retries left ──► rewrite query ──► retrieve (again)
        └─ weak & no retries ──► abstain
  └─► generate     : numbered contexts w/ section+pages; per-claim [n] citations;
                     book-attribution voice; abstain instruction; streaming
  └─► verify       : strip <think> blocks; validate citation indices; attach sources
```

### Grounding rules (enforced in prompt + post-processing)

1. Answer **only** from supplied excerpts; if the book doesn't cover it, say so plainly.
2. Every factual claim carries a `[n]` citation; invalid indices are stripped and logged.
3. Claims are attributed to the book, not asserted as established agronomic fact.
4. No web fallback, no model-knowledge fallback — abstention is a first-class outcome.

---

## 3. Configuration (.env-driven)

`pydantic-settings` loads `.env`; every knob has a sane default. Switching provider is one
variable. Groq offers **no embeddings endpoint** — embeddings support `local`/`openai` only
(validated at startup with a clear error).

| Variable | Values / default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `local` \| `openai` \| `groq` (default `local`) | `local` = any OpenAI-compatible server (llama.cpp `llama-server`, vLLM, LM Studio) |
| `LLM_MODEL` | default `qwen3-8b` | informational for llama.cpp; Groq example: `openai/gpt-oss-120b` |
| `LLM_BASE_URL` | default `http://localhost:8080/v1` | used when provider=`local` |
| `LLM_API_KEY` | default `local` | real key for `openai`/`groq` |
| `LLM_TEMPERATURE` | default `0.1` | |
| `LLM_TIMEOUT` | default `300` | per-request timeout, seconds — generous for slow local/reasoning models |
| `LLM_MAX_RETRIES` | default `1` | keep low for single-slot local servers (aborted requests still occupy them) |
| `LLM_STREAMING` | default `true` | set `false` for gateways with unreliable SSE (hangs with an idle server); internal pipeline calls never stream regardless |
| `EMBEDDING_PROVIDER` | `local` \| `openai` (default `local`) | |
| `EMBEDDING_MODEL` | default `qwen3-embedding-0.6b` | recorded in the manifest; OpenAI: `text-embedding-3-small` |
| `EMBEDDING_BASE_URL` | default `http://localhost:8081/v1` | |
| `EMBEDDING_API_KEY` | default `local` | |
| `RERANKER_ENABLED` | default `true` | |
| `RERANKER_MODEL` | default `bge-reranker-v2-m3` | informational for llama.cpp |
| `RERANKER_BASE_URL` | default `http://localhost:8082/v1` | Jina-style `/v1/rerank` endpoint |
| `RERANKER_API_KEY` | default `local` | |
| `RERANK_MAX_DOC_CHARS` | default `600` | truncation guard for the server's per-pair token cap |
| `CONTEXTUALIZE_CHUNKS` | default `false` | chunk blurbs at ingestion (slow locally) |
| `CHILD_CHUNK_TOKENS` / `PARENT_CHUNK_TOKENS` | default `250` / `1000` | re-ingestion required on change |
| `DATA_DIR` | default `./data` | indexes live under `data/index/` |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | default `12` / `12` | per query variant |
| `RRF_K` | default `60` | reciprocal-rank-fusion constant |
| `RERANK_CANDIDATES` / `TOP_PARENTS` | default `30` / `6` | |
| `MAX_QUERY_VARIANTS` | default `3` | |
| `MAX_RETRIEVAL_RETRIES` | default `1` | corrective-loop bound |
| `GRADING_ENABLED` | default `true` | disable to skip the corrective loop |
| `LOG_LEVEL` | default `INFO` | |

Local-dev model expectations — three llama.cpp `llama-server` instances (all OpenAI-style,
no Ollama):

```
llama-server -m <chat-model>.gguf --port 8080
llama-server -m <embedding-model>.gguf --embeddings --port 8081
llama-server -m <reranker-model>.gguf --rerank --port 8082
```

**Re-ingestion is required** whenever the embedding provider/model or chunk parameters change —
enforced via the manifest.

---

## 4. Project structure

Flat layout; repo root is the import root (pyright configured accordingly in `pyproject.toml`).

```
farmer_rag/
├── __init__.py
├── config.py              # pydantic-settings Settings + validation
├── log.py                 # logging setup
├── models.py              # LLM / embeddings factories (provider switch)
├── storage/
│   ├── docstore.py        # SQLite parent/child store
│   └── manifest.py        # index manifest read/write/compat-check
├── ingestion/             # CLI-only, manual trigger
│   ├── parser.py          # pymupdf4llm → markdown pages + page map
│   ├── chunker.py         # heading-aware parent/child chunking
│   ├── contextualizer.py  # optional chunk blurbs
│   └── pipeline.py        # ingest orchestration
├── retrieval/
│   ├── bm25.py            # BM25 over docstore children
│   ├── dense.py           # Chroma dense retrieval
│   ├── hybrid.py          # RRF fusion + variant union
│   ├── reranker.py        # HTTP client for llama.cpp /v1/rerank
│   └── pipeline.py        # retrieve() facade (children → parents)
├── generation/
│   ├── prompts.py
│   ├── parsing.py         # tolerant JSON parsing, <think> stripping (ThinkFilter)
│   ├── citations.py       # citation validation + sources assembly
│   └── graph.py           # LangGraph corrective flow + streaming
├── web/
│   └── app.py             # Streamlit chat (query-only; no ingestion controls)
└── cli.py                 # Typer app: ingest / ask / chat / status / eval / web
tests/                     # pytest unit tests (chunker, fusion, citations, config)
eval/golden.example.yaml   # golden-question template for retrieval eval
```

### Library choices (verified against LangChain 1.x, Aug 2026)

- `langchain` 1.3.x / `langchain-core` 1.5.x / `langgraph` 1.2.x / `langchain-text-splitters`
- `langchain-openai` (covers `local` OpenAI-compatible + `openai`; local quirks handled:
  `use_responses_api=False`, `check_embedding_ctx_length=False`), `langchain-groq`
- `langchain-chroma` + `chromadb` (in-process, persisted; first-party maintained)
- **Avoided**: `langchain-community` (sunset 2026-05, archived) — PDF loading uses `pymupdf4llm`
  directly; BM25 uses `rank-bm25` behind our own retriever; reranking is a plain `httpx` call
  to the llama.cpp `/v1/rerank` endpoint (no torch in the app). `langchain.retrievers` no
  longer exists in 1.x.
- `pymupdf4llm` (AGPL-3.0 — fine for this charity deployment; swap to MIT-licensed `docling`
  if the app is ever distributed commercially), `rank-bm25`, `httpx`,
  `pydantic-settings`, `typer`, `rich`, `streamlit`
- Dev: `pyright`, `ruff`, `pytest`

---

## 5. Step-by-step implementation plan

- [x] **Phase 0 — Research & plan**: verify ecosystem state, choose architecture, write this doc.
- [x] **Phase 1 — Scaffold & config**: uv deps; pyright + ruff config in `pyproject.toml`
      (root as import path); `Settings` with provider validation; `.env.example`; logging.
- [x] **Phase 2 — Storage layer**: SQLite docstore (parents/children), index manifest with
      compatibility checks.
- [x] **Phase 3 — Ingestion pipeline**: parser → chunker → (optional contextualizer) → indexer;
      `farmer-rag ingest <pdf>` with `--force`; idempotency via pdf hash.
- [x] **Phase 4 — Retrieval pipeline**: BM25 + dense retrievers, RRF fusion, reranker,
      parent mapping; `farmer-rag status` shows index/config state.
- [x] **Phase 5 — Generation**: prompts, LangGraph corrective flow, citation verification,
      streaming; `farmer-rag ask` / `farmer-rag chat`.
- [x] **Phase 6 — Web UI**: Streamlit chat with streaming, sources panel, abstention display;
      no ingestion affordances (read-only index status only).
- [x] **Phase 7 — Quality gates**: unit tests; pyright clean; ruff clean; retrieval eval
      harness (`farmer-rag eval` vs golden YAML); end-to-end smoke test with a sample PDF.
- [x] **Phase 8 — Docs & delivery**: README (setup, llama.cpp servers, usage), commit history in
      logical units, push.

## 6. Definition of done

- `farmer-rag ingest book.pdf` builds a persisted index; re-running is a no-op without `--force`.
- `farmer-rag ask "…"` and the Streamlit app answer with page-cited, book-attributed responses,
  and **abstain** on out-of-book questions.
- Ingestion is impossible from the web UI.
- Provider switch (`local` → `groq`/`openai`) is a `.env` edit, no code changes.
- `pyright` reports zero errors; tests pass; eval harness runs against a golden set.
