from farmer_rag.ingestion.chunker import chunk_book
from farmer_rag.ingestion.parser import PageMarkdown, ParsedPdf


def _pdf(pages: list[str]) -> ParsedPdf:
    return ParsedPdf(
        name="book.pdf",
        sha256="0" * 64,
        n_pages=len(pages),
        pages=[PageMarkdown(page=i + 1, markdown=md) for i, md in enumerate(pages)],
    )


def test_sections_follow_headings_and_pages():
    parsed = _pdf(
        [
            "# Chapter 1 Fungal diseases\n\nIntro text about fungi that is long enough to keep.\n",
            "## Powdery mildew\n\nSilicea terra 6CH is recommended for powdery mildew on "
            "tomatoes. Spray weekly in the evening for best results in the field.\n",
            "## Downy mildew\n\nA different remedy applies here, with dosage details that "
            "span this page and make the section meaningful.\n",
        ]
    )
    result = chunk_book(parsed, child_tokens=50, parent_tokens=200)

    sections = [p.section for p in result.parents]
    assert "Chapter 1 Fungal diseases" in sections
    assert "Chapter 1 Fungal diseases > Powdery mildew" in sections
    assert "Chapter 1 Fungal diseases > Downy mildew" in sections

    mildew = next(p for p in result.parents if p.section.endswith("> Powdery mildew"))
    assert mildew.page_start == 2 and mildew.page_end == 2
    assert "Silicea" in mildew.content


def test_children_carry_heading_prefix_and_map_to_parents():
    parsed = _pdf(["# Remedies\n\n" + "Arnica montana details. " * 40])
    result = chunk_book(parsed, child_tokens=40, parent_tokens=400)
    assert len(result.children) > 1
    parent_ids = {p.id for p in result.parents}
    for child in result.children:
        assert child.parent_id in parent_ids
        assert child.search_text.startswith("[Remedies")
        assert child.content in next(
            p for p in result.parents if p.id == child.parent_id
        ).content


def test_headingless_book_falls_back_to_page_labels():
    parsed = _pdf(["Plain text without any headings. " * 30, "More plain body text here. " * 30])
    result = chunk_book(parsed, child_tokens=50, parent_tokens=100)
    assert result.parents
    assert all(p.section.startswith("Pages ") for p in result.parents)


def test_oversized_sections_split_into_parts():
    parsed = _pdf(["# Big\n\n" + "word " * 1200])
    result = chunk_book(parsed, child_tokens=50, parent_tokens=150)
    big_parents = [p for p in result.parents if p.section.startswith("Big")]
    assert len(big_parents) > 1
    assert all("(part " in p.section for p in big_parents)
