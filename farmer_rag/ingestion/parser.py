"""PDF → per-page markdown via pymupdf4llm.

pymupdf4llm detects headings from font sizes and emits markdown ``#`` levels,
which the chunker uses to build the section tree. Page numbers are 1-indexed
throughout the app (matching what a reader sees in a PDF viewer).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pymupdf4llm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageMarkdown:
    page: int  # 1-indexed
    markdown: str


@dataclass(frozen=True)
class ParsedPdf:
    name: str
    sha256: str
    n_pages: int
    pages: list[PageMarkdown]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_pdf(path: Path) -> ParsedPdf:
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    logger.info("Parsing %s with pymupdf4llm (this can take a minute for a large book)", path.name)
    page_dicts = cast(
        list[dict[str, Any]],
        pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False),
    )
    pages = [
        PageMarkdown(page=i + 1, markdown=str(d.get("text") or ""))
        for i, d in enumerate(page_dicts)
    ]
    non_empty = sum(1 for p in pages if p.markdown.strip())
    logger.info("Parsed %d pages (%d with text)", len(pages), non_empty)
    if non_empty == 0:
        raise ValueError(
            f"{path.name} produced no extractable text — it may be a scanned PDF that"
            " needs OCR before ingestion."
        )
    return ParsedPdf(
        name=path.name, sha256=sha256_of(path), n_pages=len(pages), pages=pages
    )
