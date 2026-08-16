from farmer_rag.generation.parsing import ThinkFilter, parse_json_object, strip_think


def test_strip_think_removes_blocks_and_unclosed_tails():
    assert strip_think("<think>reasoning</think>answer") == "answer"
    assert strip_think("answer only") == "answer only"
    assert strip_think("<think>never closed") == ""


def test_parse_json_object_tolerates_noise():
    assert parse_json_object('{"sufficient": true}') == {"sufficient": True}
    assert parse_json_object('<think>hm</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"standalone": "q", "variants": ["a"]} done') == {
        "standalone": "q",
        "variants": ["a"],
    }
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None


def test_think_filter_handles_tags_split_across_chunks():
    f = ThinkFilter()
    out = "".join(
        f.feed(chunk) for chunk in ["<th", "ink>secret reasoning</th", "ink>The ", "answer"]
    )
    assert out == "The answer"


def test_think_filter_passes_plain_text_through():
    f = ThinkFilter()
    assert f.feed("hello ") + f.feed("world") == "hello world"


def test_think_filter_holds_back_partial_tag_then_releases():
    f = ThinkFilter()
    assert f.feed("answer <") == "answer "
    # "<" could open a think tag, so it is held; a non-tag continuation releases it.
    assert f.feed("3 apples") == "<3 apples"
