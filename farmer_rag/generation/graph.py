"""Corrective RAG flow (CRAG-lite) as a LangGraph state machine.

    understand ──► retrieve ──► [grade] ──► generate ──► END
                      ▲            │
                      └─ rewrite ◄─┘ (bounded by MAX_RETRIEVAL_RETRIES)

Design choices for a strictly-grounded, closed-corpus system:
- Deterministic graph, not free tool-calling — reliable with small local models.
- Grading fails *open* (unparseable grader output counts as sufficient) so a
  flaky local model cannot trap the loop.
- After retries are exhausted: with contexts we still generate (the answer
  prompt itself mandates abstention when excerpts don't cover the question);
  with no contexts at all we abstain outright. There is never a fallback to
  model knowledge or the web.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from farmer_rag.config import Settings
from farmer_rag.generation import prompts
from farmer_rag.generation.citations import Source, finalize_answer
from farmer_rag.generation.parsing import ThinkFilter, parse_json_object, strip_think
from farmer_rag.models import build_chat_model
from farmer_rag.retrieval.pipeline import BookRetriever, RetrievedParent

logger = logging.getLogger(__name__)

GENERATE_NODE = "generate"


class RAGState(TypedDict):
    question: str
    history: list[tuple[str, str]]
    standalone: str  # current retrieval query; _rewrite may replace it with a search query
    resolved_question: str  # self-contained question from _understand; never overwritten
    variants: list[str]
    contexts: list[RetrievedParent]
    retries: int
    sufficient: bool
    answer: str
    sources: list[Source]
    abstained: bool


def _initial_state(question: str, history: list[tuple[str, str]] | None) -> RAGState:
    return RAGState(
        question=question,
        history=history or [],
        standalone=question,
        resolved_question=question,
        variants=[question],
        contexts=[],
        retries=0,
        sufficient=True,
        answer="",
        sources=[],
        abstained=False,
    )


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[Source]
    abstained: bool
    contexts: list[RetrievedParent] = field(default_factory=list)
    standalone: str = ""


StreamEvent = tuple[Literal["token", "result"], str | AnswerResult]


class AnswerEngine:
    """Owns the LLM, the retriever, and the compiled graph. Build once, reuse."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = build_chat_model(settings)
        self._retriever = BookRetriever(settings)
        self._book_name = self._retriever.manifest.pdf_name
        self._graph = self._build_graph()

    # --- nodes ---

    def _understand(self, state: RAGState) -> dict[str, Any]:
        question = state["question"]
        standalone, variants = question, [question]
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(
                        prompts.UNDERSTAND_SYSTEM.format(
                            max_variants=self._settings.max_query_variants
                        )
                    ),
                    HumanMessage(
                        prompts.UNDERSTAND_USER.format(
                            history=prompts.render_history(state.get("history", [])),
                            question=question,
                        )
                    ),
                ]
            )
            data = parse_json_object(response.text)
            if data:
                if isinstance(data.get("standalone"), str) and data["standalone"].strip():
                    standalone = data["standalone"].strip()
                raw_variants = data.get("variants")
                if isinstance(raw_variants, list):
                    variants = [v.strip() for v in raw_variants if isinstance(v, str) and v.strip()]
        except Exception:
            logger.warning("Query understanding failed; using the raw question", exc_info=True)
        variants = variants[: self._settings.max_query_variants]
        logger.info("Standalone question: %r (%d variants)", standalone, len(variants))
        return {
            "standalone": standalone,
            "resolved_question": standalone,
            "variants": variants,
            "retries": 0,
        }

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        queries = [state["standalone"], *state.get("variants", [])]
        return {"contexts": self._retriever.retrieve(queries)}

    def _grade(self, state: RAGState) -> dict[str, Any]:
        sufficient = True  # fail open
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(prompts.GRADE_SYSTEM),
                    HumanMessage(
                        prompts.GRADE_USER.format(
                            question=state["standalone"],
                            excerpts=prompts.render_excerpts_brief(state.get("contexts", [])),
                        )
                    ),
                ]
            )
            data = parse_json_object(response.text)
            if data is not None and isinstance(data.get("sufficient"), bool):
                sufficient = data["sufficient"]
        except Exception:
            logger.warning("Grading failed; treating context as sufficient", exc_info=True)
        logger.info("Context graded %s", "sufficient" if sufficient else "weak")
        return {"sufficient": sufficient}

    def _rewrite(self, state: RAGState) -> dict[str, Any]:
        standalone = state["standalone"]
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(prompts.REWRITE_SYSTEM),
                    HumanMessage(f"Question: {state['question']}\nWeak query: {standalone}"),
                ]
            )
            rewritten = strip_think(response.text).strip().strip('"').splitlines()
            if rewritten and rewritten[0].strip():
                standalone = rewritten[0].strip()
        except Exception:
            logger.warning("Query rewrite failed; retrying with the same query", exc_info=True)
        logger.info("Retrying retrieval with rewritten query: %r", standalone)
        # Variants are dropped on retry: they contributed to the weak result.
        return {"standalone": standalone, "variants": [], "retries": state.get("retries", 0) + 1}

    def _generate(self, state: RAGState) -> dict[str, Any]:
        contexts = state.get("contexts", [])
        # Follow-ups arrive underspecified ("what potency should I use?"); give
        # the model the self-contained restatement alongside the raw question
        # (kept for language/tone fidelity — ANSWER_SYSTEM rule 5).
        resolved = state["resolved_question"]
        standalone_block = (
            f"\n(Restated as a self-contained question from the conversation: {resolved})"
            if resolved != state["question"]
            else ""
        )
        response = self._llm.invoke(
            [
                SystemMessage(prompts.ANSWER_SYSTEM.format(book_name=self._book_name)),
                HumanMessage(
                    prompts.ANSWER_USER.format(
                        contexts=prompts.render_contexts(contexts),
                        question=state["question"],
                        standalone_block=standalone_block,
                    )
                ),
            ]
        )
        answer, sources = finalize_answer(strip_think(response.text), contexts)
        return {"answer": answer, "sources": sources, "abstained": False}

    def _abstain(self, state: RAGState) -> dict[str, Any]:
        return {"answer": prompts.ABSTAIN_MESSAGE, "sources": [], "abstained": True}

    # --- routing ---

    def _route_after_retrieve(self, state: RAGState) -> str:
        if not state.get("contexts"):
            if state.get("retries", 0) < self._settings.max_retrieval_retries:
                return "rewrite"
            return "abstain"
        if self._settings.grading_enabled:
            return "grade"
        return GENERATE_NODE

    def _route_after_grade(self, state: RAGState) -> str:
        if state.get("sufficient", True):
            return GENERATE_NODE
        if state.get("retries", 0) < self._settings.max_retrieval_retries:
            return "rewrite"
        # Weak but non-empty context: generate anyway — the answer prompt
        # abstains itself if the excerpts truly don't cover the question.
        return GENERATE_NODE

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("understand", self._understand)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node(GENERATE_NODE, self._generate)
        graph.add_node("abstain", self._abstain)

        graph.add_edge(START, "understand")
        graph.add_edge("understand", "retrieve")
        graph.add_conditional_edges("retrieve", self._route_after_retrieve)
        graph.add_conditional_edges("grade", self._route_after_grade)
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge(GENERATE_NODE, END)
        graph.add_edge("abstain", END)
        return graph.compile()

    # --- public API ---

    def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> AnswerResult:
        final = cast(dict[str, Any], self._graph.invoke(_initial_state(question, history)))
        return self._result_from_state(final)

    def stream(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> Iterator[StreamEvent]:
        """Yield ("token", text) for the answer as it generates, then ("result", AnswerResult).

        The streamed tokens are raw model output (minus <think> blocks); the
        final result carries the citation-validated answer — replace the
        streamed text with it when rendering.
        """
        think_filter = ThinkFilter()
        final_state: dict[str, Any] | None = None
        for item in self._graph.stream(
            _initial_state(question, history),
            stream_mode=["messages", "values"],
        ):
            mode, payload = cast(tuple[str, Any], item)
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") == GENERATE_NODE and chunk.text:
                    text = think_filter.feed(chunk.text)
                    if text:
                        yield ("token", text)
            elif mode == "values":
                final_state = payload
        tail = think_filter.flush()
        if tail:
            yield ("token", tail)
        if final_state is None:  # defensive: stream ended without a values snapshot
            raise RuntimeError("Graph produced no final state")
        yield ("result", self._result_from_state(final_state))

    def _result_from_state(self, state: dict[str, Any]) -> AnswerResult:
        return AnswerResult(
            answer=state.get("answer", prompts.ABSTAIN_MESSAGE),
            sources=state.get("sources", []),
            abstained=state.get("abstained", False),
            contexts=state.get("contexts", []),
            standalone=state.get("standalone", ""),
        )
