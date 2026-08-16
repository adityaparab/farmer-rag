"""Typer CLI. Ingestion is triggered here and only here — the web UI cannot ingest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from farmer_rag.config import Settings, get_settings
from farmer_rag.log import setup_logging
from farmer_rag.storage.manifest import IndexError_, IndexManifest

app = typer.Typer(
    name="farmer-rag",
    help="Question answering strictly grounded in a homeopathy-for-farming book.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True)


def _load_settings() -> Settings:
    try:
        settings = get_settings()
    except Exception as exc:  # pydantic validation errors are user config errors
        err_console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1) from exc
    setup_logging(settings.log_level)
    return settings


def _fail(message: str, exc: Exception | None = None) -> typer.Exit:
    err_console.print(f"[red]Error:[/red] {message}")
    if exc is not None and str(exc) not in message:
        err_console.print(f"[dim]{exc}[/dim]")
    return typer.Exit(1)


def _print_sources(sources: list) -> None:
    if not sources:
        return
    console.print("\n[bold]Sources[/bold] (from the book)")
    for source in sources:
        console.print(f"  [cyan]\\[{source.index}][/cyan] {source.section} — {source.pages}")


@app.command()
def ingest(
    pdf: Annotated[Path, typer.Argument(help="Path to the book PDF", exists=True, dir_okay=False)],
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even if already ingested")
    ] = False,
) -> None:
    """Parse, chunk, embed, and index the book (manual trigger, CLI-only)."""
    settings = _load_settings()
    from farmer_rag.ingestion.pipeline import ingest_pdf

    try:
        manifest = ingest_pdf(settings, pdf, force=force)
    except Exception as exc:
        raise _fail("Ingestion failed. Is your embedding server running "
                    f"({settings.embedding_provider.value}: {settings.embedding_base_url})?",
                    exc) from exc
    console.print(
        f"[green]Ingested[/green] '{manifest.pdf_name}': {manifest.pdf_pages} pages → "
        f"{manifest.n_parents} sections / {manifest.n_children} chunks"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question about the book")],
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Print retrieved sections before the answer")
    ] = False,
) -> None:
    """Ask a single question; the answer streams and cites book pages."""
    settings = _load_settings()
    engine = _build_engine(settings)

    from farmer_rag.generation.graph import AnswerResult

    result: AnswerResult | None = None
    try:
        with console.status("Consulting the book…"):
            stream = engine.stream(question)
            first_event = next(stream, None)
        for event in _chain(first_event, stream):
            kind, payload = event
            if kind == "token":
                print(payload, end="", flush=True)
            else:
                assert isinstance(payload, AnswerResult)
                result = payload
    except Exception as exc:
        raise _fail(f"Answering failed. Is your LLM server running "
                    f"({settings.llm_provider.value}: {settings.llm_model})?", exc) from exc
    print()
    if result is None:
        raise _fail("No answer produced")
    if show_context:
        console.print("\n[bold]Retrieved sections[/bold]")
        for i, ctx in enumerate(result.contexts, start=1):
            record = ctx.record
            console.print(f"  [{i}] {record.section} (pp. {record.page_start}-{record.page_end})")
    _print_sources(result.sources)


@app.command()
def chat() -> None:
    """Interactive multi-turn chat (type 'exit' to quit)."""
    settings = _load_settings()
    engine = _build_engine(settings)

    from farmer_rag.generation.graph import AnswerResult

    console.print("[bold green]farmer-rag chat[/bold green] — answers come only from the book. "
                  "Type 'exit' to quit.\n")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break
        result: AnswerResult | None = None
        try:
            for kind, payload in engine.stream(question, history):
                if kind == "token":
                    print(payload, end="", flush=True)
                else:
                    assert isinstance(payload, AnswerResult)
                    result = payload
        except Exception as exc:
            err_console.print(f"\n[red]Answering failed:[/red] {exc}")
            continue
        print()
        if result is not None:
            _print_sources(result.sources)
            history.extend([("user", question), ("assistant", result.answer)])
        print()
    console.print("bye")


@app.command()
def status() -> None:
    """Show current configuration and index state."""
    settings = _load_settings()
    table = Table(title="farmer-rag status", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("LLM", f"{settings.llm_provider.value} / {settings.llm_model}")
    table.add_row("Embeddings", f"{settings.embedding_provider.value} / {settings.embedding_model}")
    table.add_row(
        "Reranker",
        settings.reranker_model if settings.reranker_enabled else "disabled",
    )
    table.add_row("Data dir", str(settings.data_dir.resolve()))
    try:
        manifest = IndexManifest.load(settings.manifest_path)
        table.add_row(
            "Index",
            f"'{manifest.pdf_name}' — {manifest.pdf_pages} pages, "
            f"{manifest.n_parents} sections / {manifest.n_children} chunks "
            f"(embedded with {manifest.embedding_model}, created {manifest.created_at})",
        )
        try:
            manifest.check_compatible(settings)
            table.add_row("Compatibility", "[green]OK[/green]")
        except IndexError_ as exc:
            table.add_row("Compatibility", f"[red]{exc}[/red]")
    except IndexError_:
        table.add_row("Index", "[yellow]not built — run `farmer-rag ingest <pdf>`[/yellow]")
    console.print(table)


@app.command("eval")
def eval_cmd(
    golden: Annotated[
        Path, typer.Argument(help="YAML golden set (see eval/golden.example.yaml)", exists=True)
    ],
) -> None:
    """Measure retrieval hit-rate against a golden question set (no LLM calls)."""
    import yaml

    settings = _load_settings()
    from farmer_rag.retrieval.pipeline import BookRetriever

    try:
        retriever = BookRetriever(settings)
    except Exception as exc:
        raise _fail(str(exc), exc) from exc

    cases = yaml.safe_load(golden.read_text(encoding="utf-8")) or []
    if not isinstance(cases, list):
        raise _fail("Golden file must be a YAML list of cases")

    hits = 0
    evaluated = 0
    table = Table(title="Retrieval eval")
    table.add_column("Question", max_width=48)
    table.add_column("Hit", justify="center")
    table.add_column("Matched on")
    for case in cases:
        question = str(case.get("question", "")).strip()
        if not question or case.get("out_of_book"):
            continue
        expect_pages = {int(p) for p in case.get("expect_pages", [])}
        expect_keywords = [str(k).lower() for k in case.get("expect_keywords", [])]
        results = retriever.retrieve([question])
        matched: list[str] = []
        for ctx in results:
            pages = set(range(ctx.record.page_start, ctx.record.page_end + 1))
            if expect_pages & pages:
                matched.append(f"pages {sorted(expect_pages & pages)}")
            text = ctx.record.content.lower()
            matched.extend(f"'{kw}'" for kw in expect_keywords if kw in text)
        evaluated += 1
        hit = bool(matched)
        hits += hit
        table.add_row(question, "[green]✓[/green]" if hit else "[red]✗[/red]",
                      ", ".join(dict.fromkeys(matched)) or "—")
    console.print(table)
    if evaluated:
        console.print(f"\nHit rate: [bold]{hits}/{evaluated}[/bold] ({hits / evaluated:.0%})")
    else:
        console.print("No evaluable cases (out_of_book cases are skipped in retrieval eval).")


@app.command()
def web() -> None:
    """Launch the Streamlit web UI (query-only; ingestion stays in the CLI)."""
    _load_settings()
    app_path = Path(__file__).parent / "web" / "app.py"
    raise typer.Exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)], check=False
        ).returncode
    )


def _build_engine(settings: Settings):
    from farmer_rag.generation.graph import AnswerEngine

    try:
        return AnswerEngine(settings)
    except IndexError_ as exc:
        raise _fail(str(exc)) from exc
    except Exception as exc:
        raise _fail("Could not initialize the answer engine.", exc) from exc


def _chain(first, rest):
    if first is not None:
        yield first
    yield from rest


if __name__ == "__main__":
    app()
