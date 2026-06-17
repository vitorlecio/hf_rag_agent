import pytest

from hf_rag.ingestion.chunker import Chunk, Chunker, _has_atomic_block
from hf_rag.ingestion.fetcher import RawPage


@pytest.fixture(scope="module")
def chunker() -> Chunker:
    return Chunker()


def _page(content: str, title: str = "Test Page", path: str = "test.md") -> RawPage:
    return RawPage(path=path, title=title, content=content, source_url="https://x")


SHORT_PAGE = _page(
    """\
# Test Page

## Section A
This is some short content for section A.
"""
)

LONG_SECTION_PAGE = _page(
    """\
# Long Page

## Big Section
"""
    + ("This sentence repeats many times to exceed the token budget. " * 60),
    title="Long Page",
    path="long.md",
)

HIERARCHY_PAGE = _page(
    """\
# Hierarchy

## Section A
Content for A.

### Subsection A.1
Content for A.1.

### Subsection A.2
Content for A.2.

## Section B
Content for B.
""",
    title="Hierarchy",
    path="hierarchy.md",
)

NO_HEADINGS_PAGE = _page(
    """\
# Test Page

This page has only a title and prose. No second-level headings at all.
""",
    title="Test Page",
    path="no_headings.md",
)

CODE_BLOCK_PAGE = _page(
    """\
# Code Guide

## Example

Here is an example:

```python
"""
    + ("x = 1\n" * 50)
    + "```\n",
    title="Code Guide",
    path="code.md",
)

TABLE_PAGE = _page(
    """\
# Table Guide

## Comparison

| Model | Size | Accuracy |
|-------|------|----------|
"""
    + "| GPT-X | 7B | 0.91 |\n" * 30,
    title="Table Guide",
    path="table.md",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestChunkSchema:
    def test_required_fields_present(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.chunk_id
            assert c.page_path == SHORT_PAGE.path
            assert c.page_title == SHORT_PAGE.title
            assert c.content
            assert c.token_count > 0

    def test_chunk_id_format(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"test_{i:03d}"

    def test_page_metadata_propagated(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        for c in chunks:
            assert c.page_path == SHORT_PAGE.path
            assert c.page_title == SHORT_PAGE.title


# ---------------------------------------------------------------------------
# Title prepending
# ---------------------------------------------------------------------------


class TestTitlePrepending:
    def test_every_chunk_starts_with_title(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([HIERARCHY_PAGE])
        assert len(chunks) > 1
        for c in chunks:
            assert c.content.startswith(f"# {HIERARCHY_PAGE.title}")

    def test_title_not_duplicated(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        for c in chunks:
            assert c.content.count(f"# {SHORT_PAGE.title}") == 1


# ---------------------------------------------------------------------------
# Heading prepending
# ---------------------------------------------------------------------------


class TestHeadingPrepending:
    def test_chunk_with_heading_includes_it_in_content(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        assert all(c.heading == "Section A" for c in chunks)
        for c in chunks:
            assert "## Section A" in c.content

    def test_heading_not_duplicated(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        for c in chunks:
            assert c.content.count("## Section A") == 1

    def test_every_split_piece_carries_heading(self, chunker: Chunker) -> None:
        # A long section produces multiple chunks; only the first piece's raw markdown
        # contained the "## Big Section" line before splitting — all pieces must still
        # carry it via the explicit prefix.
        chunks = chunker.chunk_pages([LONG_SECTION_PAGE])
        assert len(chunks) > 1
        for c in chunks:
            assert "## Big Section" in c.content

    def test_no_heading_section_has_no_heading_line(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([NO_HEADINGS_PAGE])
        for c in chunks:
            assert c.heading == ""
            assert "##" not in c.content


# ---------------------------------------------------------------------------
# Size budget
# ---------------------------------------------------------------------------


class TestSizeBudget:
    def test_long_section_splits(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([LONG_SECTION_PAGE])
        assert len(chunks) > 1, "long prose section should produce multiple chunks"

    def test_prose_chunks_within_budget(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([LONG_SECTION_PAGE])
        for c in chunks:
            assert c.token_count <= 256, (
                f"{c.chunk_id} = {c.token_count} tokens (expected <= 256)"
            )


# ---------------------------------------------------------------------------
# Atomic blocks (Option A: never split tables or code blocks)
# ---------------------------------------------------------------------------


class TestAtomicBlocks:
    def test_code_block_not_split(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([CODE_BLOCK_PAGE])
        code_chunks = [c for c in chunks if "```" in c.content]
        assert len(code_chunks) == 1, (
            "code block must be kept in a single chunk, not split"
        )

    def test_table_not_split(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([TABLE_PAGE])
        table_chunks = [c for c in chunks if "| Model |" in c.content]
        assert len(table_chunks) == 1, "table must be kept in a single chunk, not split"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_chunk_ids_stable(self, chunker: Chunker) -> None:
        a = chunker.chunk_pages([HIERARCHY_PAGE])
        b = chunker.chunk_pages([HIERARCHY_PAGE])
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]

    def test_content_stable(self, chunker: Chunker) -> None:
        a = chunker.chunk_pages([HIERARCHY_PAGE])
        b = chunker.chunk_pages([HIERARCHY_PAGE])
        assert [c.content for c in a] == [c.content for c in b]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_headings_still_produces_chunk(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([NO_HEADINGS_PAGE])
        assert len(chunks) >= 1

    def test_no_headings_chunk_has_title_prefix(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([NO_HEADINGS_PAGE])
        for c in chunks:
            assert c.content.startswith(f"# {NO_HEADINGS_PAGE.title}")

    def test_heading_metadata_populated_for_h2(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([HIERARCHY_PAGE])
        headings = {c.heading for c in chunks}
        assert any(h for h in headings), "at least one chunk should carry a heading"


# ---------------------------------------------------------------------------
# Token count accuracy
# ---------------------------------------------------------------------------


class TestTokenCount:
    def test_token_count_matches_tokenizer(self, chunker: Chunker) -> None:
        chunks = chunker.chunk_pages([SHORT_PAGE])
        for c in chunks:
            expected = len(chunker._tokenizer.encode(c.content))
            assert c.token_count == expected


# ---------------------------------------------------------------------------
# _has_atomic_block (unit)
# ---------------------------------------------------------------------------


class TestHasAtomicBlock:
    def test_detects_fenced_code_block(self) -> None:
        assert _has_atomic_block("Some text.\n```python\nx = 1\n```\n")

    def test_detects_table(self) -> None:
        assert _has_atomic_block("| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n")

    def test_plain_prose_is_not_atomic(self) -> None:
        assert not _has_atomic_block("Just some plain text with no special blocks.")

    def test_inline_backtick_is_not_atomic(self) -> None:
        assert not _has_atomic_block("Use `my_fn()` to call the function.")
