"""End-to-end ingest→retrieve without any model server: deterministic fake
embeddings stand in for the embedding provider; the reranker is disabled."""

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import farmer_rag.retrieval.pipeline as retrieval_pipeline
from farmer_rag.config import Settings
from farmer_rag.ingestion.chunker import chunk_book
from farmer_rag.ingestion.indexer import index_chunks
from farmer_rag.ingestion.parser import PageMarkdown, ParsedPdf
from farmer_rag.retrieval.pipeline import BookRetriever
from farmer_rag.storage.manifest import IndexError_, IndexManifest

_BOOK = ParsedPdf(
    name="test-book.pdf",
    sha256="a" * 64,
    n_pages=3,
    pages=[
        PageMarkdown(
            1,
            "# Fungal diseases\n\n## Powdery mildew\n\nThe book recommends Silicea terra 6CH "
            "for powdery mildew on tomatoes and cucurbits. Spray in the evening, weekly.\n",
        ),
        PageMarkdown(
            2,
            "## Aphids\n\nFor aphid infestations on beans, Coccinella septempunctata 6CH is "
            "suggested as a spray made from the potentized ladybird preparation.\n",
        ),
        PageMarkdown(
            3,
            "# Soil health\n\nThe book discusses compost preparation and Calcarea carbonica "
            "for calcium-deficient soils in orchards and vegetable plots.\n",
        ),
    ],
)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    overrides = {
        "data_dir": tmp_path / "data",
        "reranker_enabled": False,
        "dense_top_k": 5,
        "sparse_top_k": 5,
        "top_parents": 3,
    }
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture()
def indexed(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    embeddings = DeterministicFakeEmbedding(size=64)
    result = chunk_book(
        _BOOK,
        child_tokens=settings.child_chunk_tokens,
        parent_tokens=settings.parent_chunk_tokens,
    )
    index_chunks(settings, embeddings, result.parents, result.children)
    IndexManifest.build(
        settings,
        pdf_name=_BOOK.name,
        pdf_sha256=_BOOK.sha256,
        pdf_pages=_BOOK.n_pages,
        n_parents=len(result.parents),
        n_children=len(result.children),
    ).save(settings.manifest_path)
    monkeypatch.setattr(retrieval_pipeline, "build_embeddings", lambda _: embeddings)
    return settings


def test_retrieves_relevant_parent_with_pages(indexed: Settings):
    retriever = BookRetriever(indexed)
    results = retriever.retrieve(["What helps against powdery mildew?"])
    assert results
    top = results[0].record
    assert "Silicea" in top.content
    assert top.section.endswith("Powdery mildew")
    assert (top.page_start, top.page_end) == (1, 1)


def test_variant_union_widens_recall(indexed: Settings):
    retriever = BookRetriever(indexed)
    results = retriever.retrieve(["aphids on beans", "ladybird preparation spray"])
    assert any("Coccinella" in r.record.content for r in results)


def test_missing_index_gives_actionable_error(settings: Settings):
    with pytest.raises(IndexError_, match="farmer-rag ingest"):
        BookRetriever(settings)


def test_incompatible_settings_are_rejected(indexed: Settings):
    changed = indexed.model_copy(update={"embedding_model": "some-other-model"})
    with pytest.raises(IndexError_, match="embedding model"):
        BookRetriever(changed)


def test_contextualize_flag_change_is_rejected(indexed: Settings):
    changed = indexed.model_copy(update={"contextualize_chunks": True})
    with pytest.raises(IndexError_, match="contextualized"):
        BookRetriever(changed)


def test_corrupt_manifest_gives_actionable_error(indexed: Settings):
    indexed.manifest_path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(IndexError_, match="--force"):
        BookRetriever(indexed)


def test_index_rebuilt_mid_run_is_refused(indexed: Settings):
    import os

    retriever = BookRetriever(indexed)
    assert retriever.retrieve(["powdery mildew"])
    # Simulate a CLI re-ingest while the app holds the old snapshot.
    stat = indexed.manifest_path.stat()
    os.utime(indexed.manifest_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(IndexError_, match="rebuilt"):
        retriever.retrieve(["powdery mildew"])
