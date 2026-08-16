"""Streamlit chat UI. Query-only by design: this module never imports the
ingestion pipeline, and the UI offers no way to modify the index."""

from __future__ import annotations

import streamlit as st

from farmer_rag.config import Settings, get_settings
from farmer_rag.generation.graph import AnswerEngine, AnswerResult
from farmer_rag.log import setup_logging
from farmer_rag.storage.manifest import IndexError_, IndexManifest

st.set_page_config(page_title="Farmer RAG", page_icon="🌱", layout="centered")

_MAX_HISTORY_TURNS = 8


@st.cache_resource(show_spinner="Loading the book index…")
def _engine() -> AnswerEngine:
    settings = get_settings()
    setup_logging(settings.log_level)
    return AnswerEngine(settings)


def _load() -> tuple[Settings, AnswerEngine] | None:
    try:
        settings = get_settings()
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        return None
    try:
        return settings, _engine()
    except IndexError_ as exc:
        st.warning(str(exc))
        st.caption("Ingestion is deliberately CLI-only: `farmer-rag ingest <pdf>`")
        return None
    except Exception as exc:
        st.error(
            "Could not start the answer engine. Check that your model server is "
            f"running ({exc})"
        )
        return None


def _sidebar(settings: Settings, manifest: IndexManifest) -> None:
    with st.sidebar:
        st.header("📖 The book")
        st.markdown(
            f"**{manifest.pdf_name}**\n\n"
            f"- {manifest.pdf_pages} pages\n"
            f"- {manifest.n_parents} sections / {manifest.n_children} chunks\n"
            f"- indexed {manifest.created_at[:10]}"
        )
        st.divider()
        st.caption(
            f"LLM: `{settings.llm_provider.value} / {settings.llm_model}`  \n"
            f"Embeddings: `{manifest.embedding_model}`  \n"
            f"Reranker: `{settings.reranker_model if settings.reranker_enabled else 'off'}`"
        )
        st.caption(
            "Answers come **only** from the book above. "
            "Re-ingestion is available from the CLI only."
        )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**[{source.index}]** {source.section} — *{source.pages}*")


def main() -> None:
    loaded = _load()
    if loaded is None:
        st.stop()
        return
    settings, engine = loaded

    st.title("🌱 Farmer RAG")
    st.caption(
        "Ask about homeopathic plant care — answers are grounded in the book, with page citations."
    )
    _sidebar(settings, engine._retriever.manifest)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_sources(message.get("sources", []))

    question = st.chat_input("e.g. What does the book recommend for aphids on beans?")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history: list[tuple[str, str]] = [
        (m["role"], m["content"])
        for m in st.session_state.messages[:-1][-_MAX_HISTORY_TURNS:]
    ]

    with st.chat_message("assistant"):
        progress = st.status("Consulting the book…")
        placeholder = st.empty()
        streamed = ""
        result: AnswerResult | None = None
        try:
            for kind, payload in engine.stream(question, history):
                if kind == "status":
                    progress.update(label=str(payload))
                elif kind == "token":
                    if not streamed:
                        progress.update(label="Writing the answer…", state="complete")
                    streamed += str(payload)
                    placeholder.markdown(streamed + "▌")
                else:
                    assert isinstance(payload, AnswerResult)
                    result = payload
            progress.update(label="Done", state="complete")
        except Exception as exc:
            progress.update(label="Failed", state="error")
            placeholder.error(f"Answering failed: {exc}")
            st.session_state.messages.pop()
            return

        if result is None:
            placeholder.error("No answer was produced — please try again.")
            st.session_state.messages.pop()
            return

        placeholder.markdown(result.answer)
        _render_sources(result.sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


main()
