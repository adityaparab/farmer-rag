"""Citation validation and source assembly for answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from farmer_rag.retrieval.pipeline import RetrievedParent

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Source:
    index: int
    section: str
    pages: str


def _pages_label(ctx: RetrievedParent) -> str:
    record = ctx.record
    if record.page_start == record.page_end:
        return f"p. {record.page_start}"
    return f"pp. {record.page_start}-{record.page_end}"


def finalize_answer(
    answer: str, contexts: list[RetrievedParent]
) -> tuple[str, list[Source]]:
    """Drop citations pointing at non-existent excerpts; list the sources used.

    If the model cited nothing, all supplied excerpts are listed as consulted.
    """
    n = len(contexts)
    valid: set[int] = set()

    def check(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= n:
            valid.add(idx)
            return match.group(0)
        return ""

    cleaned = _CITE_RE.sub(check, answer).strip()
    indices = sorted(valid) if valid else list(range(1, n + 1))
    sources = [
        Source(index=i, section=contexts[i - 1].record.section, pages=_pages_label(contexts[i - 1]))
        for i in indices
    ]
    return cleaned, sources
