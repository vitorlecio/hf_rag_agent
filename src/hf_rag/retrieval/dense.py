import chromadb

from hf_rag.config import DENSE_TOP_K, EmbeddingFunction
from hf_rag.retrieval.base import RetrievalResult


class DenseRetriever:
    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_fn: EmbeddingFunction,
    ) -> None:
        self._collection = collection
        self._embedding_fn = embedding_fn

    def retrieve(self, query: str, k: int = DENSE_TOP_K) -> list[RetrievalResult]:
        n = min(k, self._collection.count())
        if n == 0:
            return []
        embedding = self._embedding_fn([query])[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return _parse_results(results)


def _parse_results(results: dict) -> list[RetrievalResult]:
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            content=doc,
            score=1.0 - dist,  # cosine distance -> cosine similarity
            page_path=meta.get("page_path", ""),
            page_title=meta.get("page_title", ""),
            heading=meta.get("heading", ""),
        )
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances)
    ]
