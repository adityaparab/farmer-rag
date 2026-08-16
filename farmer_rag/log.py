"""Logging setup shared by the CLI and the web app."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Chroma and httpx are chatty at INFO.
    for noisy in ("chromadb", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
