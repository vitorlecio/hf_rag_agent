import json
from pathlib import Path

import pytest

from hf_rag.ingestion.chunker import Chunk
from hf_rag.ingestion.embedder import Embedder

EMBEDDING_DIM = 8  # small fake dimension — tests don't care about quality


def _fake_embedding_fn(texts: list[str]) -> list[list[float]]:
    """Returns deterministic fake embeddings; avoids loading any real model."""
    return [
        [float(i % EMBEDDING_DIM) / EMBEDDING_DIM] * EMBEDDING_DIM
        for i, _ in enumerate(texts)
    ]


def _chunk(idx: int, content: str = "Some content.") -> Chunk:
    return Chunk(
        chunk_id=f"training_{idx:03d}",
        page_path="training.md",
        page_title="Training",
        heading="Overview",
        content=content,
        token_count=len(content.split()),
    )


@pytest.fixture
def embedder(tmp_path: Path) -> Embedder:
    return Embedder(
        collection_name="test_collection",
        embedding_fn=_fake_embedding_fn,
        persist_dir=tmp_path / "chroma",
    )


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_stores_documents(self, embedder: Embedder) -> None:
        embedder.embed([_chunk(0), _chunk(1)])
        assert embedder._collection.count() == 2

    def test_is_idempotent(self, embedder: Embedder) -> None:
        chunks = [_chunk(0), _chunk(1)]
        embedder.embed(chunks)
        embedder.embed(chunks)
        assert embedder._collection.count() == 2

    def test_stores_metadata(self, embedder: Embedder) -> None:
        embedder.embed([_chunk(0, content="# Training\n\nHello.")])
        result = embedder._collection.get(ids=["training_000"], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["page_path"] == "training.md"
        assert meta["page_title"] == "Training"
        assert meta["heading"] == "Overview"
        assert "token_count" in meta


# ---------------------------------------------------------------------------
# Load (static)
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_roundtrip(self, tmp_path: Path) -> None:
        chunks = [_chunk(0), _chunk(1, content="# Training\n\nDifferent content.")]
        dest = tmp_path / "chunks.json"
        from dataclasses import asdict

        with open(dest, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f)

        loaded = Embedder.load(dest)
        assert len(loaded) == 2
        assert loaded[0].chunk_id == "training_000"
        assert loaded[1].content == "# Training\n\nDifferent content."

    def test_load_preserves_all_fields(self, tmp_path: Path) -> None:
        original = _chunk(5, content="# Training\n\nSpecial content.")
        dest = tmp_path / "chunks.json"
        from dataclasses import asdict

        with open(dest, "w", encoding="utf-8") as f:
            json.dump([asdict(original)], f)

        loaded = Embedder.load(dest)
        c = loaded[0]
        assert c.chunk_id == original.chunk_id
        assert c.page_path == original.page_path
        assert c.page_title == original.page_title
        assert c.heading == original.heading
        assert c.token_count == original.token_count
