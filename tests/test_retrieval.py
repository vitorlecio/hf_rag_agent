from pathlib import Path
from unittest.mock import MagicMock

import chromadb
import pytest

from hf_rag.retrieval.base import RetrievalResult, Retriever
from hf_rag.retrieval.dense import DenseRetriever
from hf_rag.retrieval.reranking import RerankingRetriever

DIM = 4  # small embedding dimension for tests


def _unit(index: int, dim: int = DIM) -> list[float]:
    """Return a unit vector with a 1 at position `index`."""
    v = [0.0] * dim
    v[index % dim] = 1.0
    return v


def _fake_embedding_fn(texts: list[str]) -> list[list[float]]:
    """Returns a deterministic fake embedding based on text content."""
    result = []
    for text in texts:
        # Map specific keywords to orthogonal unit vectors
        if "alpha" in text:
            result.append(_unit(0))
        elif "beta" in text:
            result.append(_unit(1))
        elif "gamma" in text:
            result.append(_unit(2))
        else:
            result.append(_unit(3))
    return result


@pytest.fixture
def collection(tmp_path: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = client.get_or_create_collection(
        name="test",
        metadata={"hnsw:space": "cosine"},
    )
    # Upsert three documents with orthogonal embeddings
    coll.upsert(
        ids=["alpha_000", "beta_000", "gamma_000"],
        documents=[
            "Content about alpha.",
            "Content about beta.",
            "Content about gamma.",
        ],
        embeddings=[_unit(0), _unit(1), _unit(2)],
        metadatas=[
            {
                "page_path": "alpha.md",
                "page_title": "Alpha",
                "heading": "Intro",
                "token_count": 5,
            },
            {
                "page_path": "beta.md",
                "page_title": "Beta",
                "heading": "Overview",
                "token_count": 5,
            },
            {
                "page_path": "gamma.md",
                "page_title": "Gamma",
                "heading": "Details",
                "token_count": 5,
            },
        ],
    )
    return coll


@pytest.fixture
def dense(collection: chromadb.Collection) -> DenseRetriever:
    return DenseRetriever(collection=collection, embedding_fn=_fake_embedding_fn)


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class TestDenseRetriever:
    def test_returns_k_results(self, dense: DenseRetriever) -> None:
        results = dense.retrieve("alpha", k=2)
        assert len(results) == 2

    def test_returns_fewer_than_k_when_collection_is_small(
        self, dense: DenseRetriever
    ) -> None:
        results = dense.retrieve("alpha", k=100)
        assert len(results) == 3  # collection only has 3 docs

    def test_empty_collection_returns_empty(self, tmp_path: Path) -> None:
        client = chromadb.PersistentClient(path=str(tmp_path / "empty"))
        coll = client.get_or_create_collection(
            "empty", metadata={"hnsw:space": "cosine"}
        )
        retriever = DenseRetriever(collection=coll, embedding_fn=_fake_embedding_fn)
        assert retriever.retrieve("anything", k=5) == []

    def test_top_result_is_most_similar(self, dense: DenseRetriever) -> None:
        results = dense.retrieve("alpha", k=3)
        assert results[0].chunk_id == "alpha_000"

    def test_result_fields_populated(self, dense: DenseRetriever) -> None:
        results = dense.retrieve("alpha", k=1)
        r = results[0]
        assert isinstance(r, RetrievalResult)
        assert r.chunk_id == "alpha_000"
        assert r.page_path == "alpha.md"
        assert r.page_title == "Alpha"
        assert r.heading == "Intro"
        assert r.content == "Content about alpha."

    def test_score_is_highest_for_top_result(self, dense: DenseRetriever) -> None:
        results = dense.retrieve("alpha", k=3)
        scores = [r.score for r in results]
        assert scores[0] == max(scores)

    def test_conforms_to_retriever_protocol(self, dense: DenseRetriever) -> None:
        assert isinstance(dense, Retriever)


# ---------------------------------------------------------------------------
# RerankingRetriever
# ---------------------------------------------------------------------------


class TestRerankingRetriever:
    def _make_reranker(
        self, dense: DenseRetriever, mock_scores: list[float]
    ) -> RerankingRetriever:
        reranker = RerankingRetriever.__new__(RerankingRetriever)
        reranker._dense = dense
        reranker._n_candidates = 3
        reranker._cross_encoder = MagicMock()
        reranker._cross_encoder.predict.return_value = mock_scores
        return reranker

    def test_reorders_by_cross_encoder_score(self, dense: DenseRetriever) -> None:
        # Dense would rank alpha first; cross-encoder says beta is best
        reranker = self._make_reranker(dense, mock_scores=[0.1, 0.9, 0.5])
        results = reranker.retrieve("alpha", k=3)
        # The cross-encoder is called with candidates in dense order (alpha, beta, gamma)
        # scores: alpha=0.1, beta=0.9, gamma=0.5 → reranked: beta, gamma, alpha
        assert results[0].chunk_id == "beta_000"
        assert results[1].chunk_id == "gamma_000"
        assert results[2].chunk_id == "alpha_000"

    def test_returns_at_most_k(self, dense: DenseRetriever) -> None:
        reranker = self._make_reranker(dense, mock_scores=[0.3, 0.1, 0.9])
        results = reranker.retrieve("alpha", k=2)
        assert len(results) == 2

    def test_scores_replaced_by_cross_encoder(self, dense: DenseRetriever) -> None:
        reranker = self._make_reranker(dense, mock_scores=[0.1, 0.9, 0.5])
        results = reranker.retrieve("alpha", k=3)
        # The top result should have the cross-encoder score, not the dense score
        assert results[0].score == pytest.approx(0.9)

    def test_conforms_to_retriever_protocol(self, dense: DenseRetriever) -> None:
        reranker = self._make_reranker(dense, mock_scores=[0.1, 0.9, 0.5])
        assert isinstance(reranker, Retriever)

    def test_empty_dense_returns_empty(self, tmp_path: Path) -> None:
        client = chromadb.PersistentClient(path=str(tmp_path / "empty"))
        coll = client.get_or_create_collection(
            "empty", metadata={"hnsw:space": "cosine"}
        )
        empty_dense = DenseRetriever(collection=coll, embedding_fn=_fake_embedding_fn)
        reranker = RerankingRetriever.__new__(RerankingRetriever)
        reranker._dense = empty_dense
        reranker._n_candidates = 5
        reranker._cross_encoder = MagicMock()
        assert reranker.retrieve("anything", k=3) == []
