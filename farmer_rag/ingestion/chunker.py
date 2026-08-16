"""Structure-aware parent/child chunking.

The markdown heading tree (detected by pymupdf4llm from font sizes) defines
*sections*. Each section becomes one or more *parents* (~PARENT_CHUNK_TOKENS,
what the LLM eventually reads), and each parent is split into small *children*
(~CHILD_CHUNK_TOKENS, what gets embedded and BM25-indexed). Children carry a
``[heading path]`` prefix in their ``search_text`` so isolated chunks stay
anchored to their chapter — the cheap version of contextual retrieval, upgraded
by the optional LLM contextualizer.

Books whose PDFs yield no headings degrade gracefully: everything lands in one
root section that is packed into page-labelled, size-bounded parents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from farmer_rag.ingestion.parser import ParsedPdf
from farmer_rag.storage.docstore import ChildRecord, ParentRecord

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_RULE_RE = re.compile(r"^[-=_*\s]{3,}$")  # horizontal rules / page-break artifacts
_CHARS_PER_TOKEN = 4


def _est_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _clean_title(raw: str) -> str:
    return raw.strip().strip("#").replace("**", "").replace("_", " ").strip()


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    lines: list[tuple[str, int]] = field(default_factory=list)  # (line, page)


@dataclass(frozen=True)
class _Paragraph:
    text: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class ChunkResult:
    parents: list[ParentRecord]
    children: list[ChildRecord]


def _split_sections(parsed: ParsedPdf) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[str] = []
    current = _Section(heading_path=())

    def flush(section: _Section) -> None:
        if any(line.strip() for line, _ in section.lines):
            sections.append(section)

    for page in parsed.pages:
        for raw in page.markdown.splitlines():
            line = raw.rstrip()
            if _RULE_RE.match(line):
                continue
            match = _HEADING_RE.match(line)
            if match:
                flush(current)
                level = len(match.group(1))
                title = _clean_title(match.group(2))
                stack = [*stack[: level - 1], title] if title else stack[: level - 1]
                current = _Section(heading_path=tuple(stack))
            else:
                current.lines.append((line, page.page))
    flush(current)
    return sections


def _paragraphs(section: _Section) -> list[_Paragraph]:
    paras: list[_Paragraph] = []
    buf: list[str] = []
    pages: list[int] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            paras.append(_Paragraph(text=text, page_start=min(pages), page_end=max(pages)))

    for line, page in section.lines:
        if line.strip():
            buf.append(line)
            pages.append(page)
        elif buf:
            flush()
            buf, pages = [], []
    if buf:
        flush()
    return paras


def chunk_book(parsed: ParsedPdf, *, child_tokens: int, parent_tokens: int) -> ChunkResult:
    parent_char_limit = parent_tokens * _CHARS_PER_TOKEN
    para_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_char_limit, chunk_overlap=0
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_tokens * _CHARS_PER_TOKEN,
        chunk_overlap=child_tokens * _CHARS_PER_TOKEN // 5,
    )

    parents: list[ParentRecord] = []
    children: list[ChildRecord] = []

    for section in _split_sections(parsed):
        paras = _paragraphs(section)
        # Oversized single paragraphs (e.g. heading-less books) get hard-split.
        normalized: list[_Paragraph] = []
        for para in paras:
            if _est_tokens(para.text) > parent_tokens:
                normalized.extend(
                    _Paragraph(piece, para.page_start, para.page_end)
                    for piece in para_splitter.split_text(para.text)
                )
            else:
                normalized.append(para)

        # Greedily pack paragraphs into parent-sized groups.
        groups: list[list[_Paragraph]] = []
        buf: list[_Paragraph] = []
        buf_tokens = 0
        for para in normalized:
            para_size = _est_tokens(para.text)
            if buf and buf_tokens + para_size > parent_tokens:
                groups.append(buf)
                buf, buf_tokens = [], 0
            buf.append(para)
            buf_tokens += para_size
        if buf:
            groups.append(buf)

        base_label = " > ".join(section.heading_path)
        for group_index, group in enumerate(groups):
            content = "\n\n".join(p.text for p in group)
            if len(content) < 40:  # page-number/footer noise
                continue
            page_start = min(p.page_start for p in group)
            page_end = max(p.page_end for p in group)
            label = base_label or f"Pages {page_start}-{page_end}"
            if len(groups) > 1 and base_label:
                label = f"{label} (part {group_index + 1})"
            parent_id = f"p{len(parents):05d}"
            parents.append(
                ParentRecord(
                    id=parent_id,
                    content=content,
                    section=label,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            pieces = child_splitter.split_text(content) or [content]
            for i, piece in enumerate(pieces):
                children.append(
                    ChildRecord(
                        id=f"{parent_id}-c{i:02d}",
                        parent_id=parent_id,
                        content=piece,
                        search_text=f"[{label}]\n{piece}",
                        section=label,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )

    logger.info("Chunked book into %d parents / %d children", len(parents), len(children))
    if not parents:
        raise ValueError("Chunking produced no content — check the PDF extraction quality.")
    return ChunkResult(parents=parents, children=children)
