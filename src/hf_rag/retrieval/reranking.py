from sentence_transformers import CrossEncoder

from hf_rag.config import DENSE_TOP_N, RERANK_TOP_K, RERANKER_MODEL
from hf_rag.retrieval.base import RetrievalResult
from hf_rag.retrieval.dense import DenseRetriever


class RerankingRetriever:
    """Wraps DenseRetriever and reranks candidates with a cross-encoder."""

    def __init__(
        self,
        dense: DenseRetriever,
        model: str = RERANKER_MODEL,
        n_candidates: int = DENSE_TOP_N,
    ) -> None:
        self._dense = dense
        self._n_candidates = n_candidates
        self._cross_encoder = CrossEncoder(model)

    def retrieve(self, query: str, k: int = RERANK_TOP_K) -> list[RetrievalResult]:
        candidates = self._dense.retrieve(query, k=self._n_candidates)
        if not candidates:
            return []
        scores = self._cross_encoder.predict([(query, c.content) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=float(s),
                page_path=r.page_path,
                page_title=r.page_title,
                heading=r.heading,
            )
            for r, s in ranked[:k]
        ]
