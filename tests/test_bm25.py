from farmer_rag.retrieval.bm25 import BM25Index, tokenize
from farmer_rag.storage.docstore import ChildRecord


def _child(i: int, text: str) -> ChildRecord:
    return ChildRecord(
        id=f"p0000{i}-c00",
        parent_id=f"p0000{i}",
        content=text,
        search_text=text,
        section="S",
        page_start=1,
        page_end=1,
    )


def test_tokenize_is_unicode_aware():
    tokens = tokenize("Silicea 6CH, टमाटर!")
    assert "silicea" in tokens
    assert "6ch" in tokens
    assert any("ट" in t for t in tokens)  # Devanagari text stays searchable


def test_normal_corpus_ranks_matching_chunk_first():
    index = BM25Index(
        [
            _child(1, "Silicea terra for powdery mildew on tomatoes"),
            _child(2, "Compost preparation for soil health in orchards"),
            _child(3, "Arnica montana for transplant shock in seedlings"),
        ]
    )
    assert index.search("powdery mildew silicea", k=2)[0] == "p00001-c00"


def test_single_chunk_corpus_still_matches():
    # rank_bm25's idf floor zeroes every score when a term appears in most
    # documents; the token-overlap fallback must still find exact terms.
    index = BM25Index([_child(1, "Silicea terra for powdery mildew")])
    assert index.search("silicea powdery mildew", k=5) == ["p00001-c00"]
    assert index.search("unrelated words entirely", k=5) == []


def test_two_chunk_corpus_common_term():
    index = BM25Index(
        [
            _child(1, "Silicea for mildew"),
            _child(2, "Silicea dosage details"),
        ]
    )
    assert index.search("silicea", k=5) != []
