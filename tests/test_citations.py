from farmer_rag.generation.citations import finalize_answer
from farmer_rag.retrieval.pipeline import RetrievedParent
from farmer_rag.storage.docstore import ParentRecord


def _ctx(i: int, section: str = "Ch 1", pages: tuple[int, int] = (10, 12)) -> RetrievedParent:
    return RetrievedParent(
        record=ParentRecord(
            id=f"p{i}", content="text", section=section, page_start=pages[0], page_end=pages[1]
        ),
        score=None,
    )


def test_valid_citations_become_sources():
    answer, sources = finalize_answer("Use Silicea [1]. Spray weekly [2].", [_ctx(1), _ctx(2)])
    assert "[1]" in answer and "[2]" in answer
    assert [s.index for s in sources] == [1, 2]
    assert sources[0].pages == "pp. 10-12"


def test_invalid_citation_indices_are_stripped():
    answer, sources = finalize_answer("Claim [1]. Bogus [7].", [_ctx(1)])
    assert "[7]" not in answer
    assert "[1]" in answer
    assert [s.index for s in sources] == [1]


def test_uncited_answer_lists_all_contexts_as_sources():
    _, sources = finalize_answer("An answer with no citations.", [_ctx(1), _ctx(2)])
    assert [s.index for s in sources] == [1, 2]
