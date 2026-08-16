"""Tolerant parsing of LLM output.

Local models are unreliable JSON emitters and reasoning models (qwen3) wrap
output in ``<think>`` blocks — every LLM-output consumer goes through here.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from LLM output; None if unparseable."""
    cleaned = strip_think(text)
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _held_prefix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of ``tag``."""
    for length in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:length]):
            return length
    return 0


class ThinkFilter:
    """Incrementally strip ``<think>…</think>`` from a token stream.

    Tags can be split across arbitrary chunk boundaries; a potential partial
    tag at the end of the buffer is held back until resolved.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_think:
                end = self._buf.find(self._CLOSE)
                if end == -1:
                    held = _held_prefix_len(self._buf, self._CLOSE)
                    self._buf = self._buf[len(self._buf) - held :] if held else ""
                    break
                self._buf = self._buf[end + len(self._CLOSE) :]
                self._in_think = False
            else:
                start = self._buf.find(self._OPEN)
                if start == -1:
                    held = _held_prefix_len(self._buf, self._OPEN)
                    emit_up_to = len(self._buf) - held
                    out.append(self._buf[:emit_up_to])
                    self._buf = self._buf[emit_up_to:]
                    break
                out.append(self._buf[:start])
                self._buf = self._buf[start + len(self._OPEN) :]
                self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        """Release held-back partial-tag text at end of stream.

        Text inside an unclosed <think> block stays dropped, matching
        ``strip_think``.
        """
        out = "" if self._in_think else self._buf
        self._buf = ""
        return out
