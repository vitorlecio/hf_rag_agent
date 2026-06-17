import pytest

from hf_rag.eval.eval_set import EvalItem
from hf_rag.eval.runner import _is_code_item, _slice_ci
from hf_rag.ingestion.chunker import Chunk


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        page_path="training.md",
        page_title="Training",
        heading="",
        content=content,
        token_count=10,
    )


# ---------------------------------------------------------------------------
# _is_code_item
# ---------------------------------------------------------------------------


class TestIsCodeItem:
    def test_prose_chunk(self) -> None:
        chunks_by_id = {"c1": _make_chunk("c1", "# Training\n\nSome prose text.")}
        item = EvalItem(query="q", relevant_chunk_ids=["c1"])
        assert _is_code_item(item, chunks_by_id) is False

    def test_code_chunk(self) -> None:
        chunks_by_id = {"c1": _make_chunk("c1", "# Training\n\n```python\nx = 1\n```")}
        item = EvalItem(query="q", relevant_chunk_ids=["c1"])
        assert _is_code_item(item, chunks_by_id) is True

    def test_any_relevant_chunk_is_code(self) -> None:
        chunks_by_id = {
            "c1": _make_chunk("c1", "Prose only."),
            "c2": _make_chunk("c2", "```python\nx = 1\n```"),
        }
        item = EvalItem(query="q", relevant_chunk_ids=["c1", "c2"])
        assert _is_code_item(item, chunks_by_id) is True


# ---------------------------------------------------------------------------
# _slice_ci
# ---------------------------------------------------------------------------


class TestSliceCI:
    def test_filters_by_mask_value(self) -> None:
        scores = [1.0, 0.0, 1.0, 0.0]
        mask = [True, False, True, False]
        ci = _slice_ci(scores, mask, True)
        assert ci.mean == pytest.approx(1.0)

    def test_other_mask_value(self) -> None:
        scores = [1.0, 0.0, 1.0, 0.0]
        mask = [True, False, True, False]
        ci = _slice_ci(scores, mask, False)
        assert ci.mean == pytest.approx(0.0)
