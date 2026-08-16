from farmer_rag.retrieval.hybrid import rrf_fuse


def test_items_in_multiple_rankings_win():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c", "d"}


def test_rank_position_matters():
    fused = rrf_fuse([["a", "b"], ["a", "b"]], k=60)
    assert fused == ["a", "b"]


def test_empty_rankings():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[], []], k=60) == []
