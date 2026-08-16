"""Optional Anthropic-style contextual retrieval: prepend an LLM-written blurb
situating each child chunk within its section before indexing.

Off by default for local development (hundreds of LLM calls through a local 8B
model is slow); recommended when ingesting with a cloud provider. Failures are
non-fatal — a chunk without a blurb still carries its heading-path prefix.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from farmer_rag.storage.docstore import ChildRecord, ParentRecord

logger = logging.getLogger(__name__)

_PROMPT = """\
<section title="{section}">
{parent_content}
</section>

<chunk>
{chunk}
</chunk>

The section above comes from a book about applying homeopathy to plants and \
farming. Write one or two short sentences situating the chunk within the \
section so the chunk can be found by search (name the topic, remedy, crop, or \
problem it concerns). Reply with the sentences only — no preamble, no quotes.\
"""

_MAX_PARENT_CHARS = 6000
_BATCH_SIZE = 8


def contextualize_children(
    llm: BaseChatModel,
    parents: list[ParentRecord],
    children: list[ChildRecord],
) -> list[ChildRecord]:
    parents_by_id = {p.id: p for p in parents}
    prompts = [
        [
            HumanMessage(
                _PROMPT.format(
                    section=child.section,
                    parent_content=parents_by_id[child.parent_id].content[:_MAX_PARENT_CHARS],
                    chunk=child.content,
                )
            )
        ]
        for child in children
    ]

    updated: list[ChildRecord] = []
    failures = 0
    for start in range(0, len(prompts), _BATCH_SIZE):
        batch_children = children[start : start + _BATCH_SIZE]
        try:
            responses = llm.batch(
                prompts[start : start + _BATCH_SIZE],  # type: ignore[arg-type]
                config={"max_concurrency": 4},
            )
        except Exception:
            logger.warning(
                "Contextualizer batch %d-%d failed; keeping heading-prefix only",
                start,
                start + len(batch_children),
                exc_info=True,
            )
            updated.extend(batch_children)
            failures += len(batch_children)
            continue
        for child, response in zip(batch_children, responses, strict=True):
            blurb = _clean(str(response.content))
            if blurb:
                updated.append(
                    replace(child, search_text=f"[{child.section}]\n{blurb}\n{child.content}")
                )
            else:
                updated.append(child)
        if (start // _BATCH_SIZE) % 5 == 0:
            done = min(start + _BATCH_SIZE, len(children))
            logger.info("Contextualized %d/%d chunks", done, len(children))

    if failures:
        logger.warning("%d/%d chunks kept without contextual blurbs", failures, len(children))
    return updated


def _clean(text: str) -> str:
    # Strip <think> blocks emitted by reasoning models (e.g. qwen3).
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return " ".join(text.split())[:600]
