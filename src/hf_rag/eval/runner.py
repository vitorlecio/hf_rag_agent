import json
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from loguru import logger

from hf_rag.config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_OPENAI,
    DENSE_TOP_K,
    EVAL_SET_PATH,
    RERANK_TOP_K,
    make_openai_embedding_fn,
)
from hf_rag.eval.eval_set import EvalItem, generate, load
from hf_rag.eval.metrics import BootstrapCI, bootstrap_ci, hit_rate, mrr, precision_at_k
from hf_rag.ingestion.chunker import Chunk
from hf_rag.ingestion.embedder import Embedder
from hf_rag.retrieval.dense import DenseRetriever
from hf_rag.retrieval.reranking import RerankingRetriever

_RESULTS_PATH = Path("data/eval_results.json")
REQUEST_DELAY = (
    0.65  # seconds between embedding calls — stays under OpenAI's 100 RPM limit
)


def _score(
    retriever, items: list[EvalItem], k: int
) -> tuple[list[float], list[float], list[float]]:
    mrr_scores, hr_scores, p_scores = [], [], []
    for item in items:
        results = retriever.retrieve(item.query, k=k)
        ids = [r.chunk_id for r in results]
        relevant = set(item.relevant_chunk_ids)
        mrr_scores.append(mrr(ids, relevant))
        hr_scores.append(hit_rate(ids, relevant))
        p_scores.append(precision_at_k(ids, relevant))
        time.sleep(REQUEST_DELAY)
    return mrr_scores, hr_scores, p_scores


def _fmt(ci: BootstrapCI) -> str:
    return f"{ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]"


def _is_code_item(item: EvalItem, chunks_by_id: dict[str, Chunk]) -> bool:
    return any("```" in chunks_by_id[cid].content for cid in item.relevant_chunk_ids)


def _slice_ci(scores: list[float], mask: list[bool], value: bool) -> BootstrapCI:
    return bootstrap_ci([s for s, m in zip(scores, mask) if m == value])


def main() -> None:
    load_dotenv()

    eval_items = (
        load() if EVAL_SET_PATH.exists() else generate(Embedder.load(CHUNKS_PATH))
    )
    logger.info(f"Eval set: {len(eval_items)} items")

    embedding_fn = make_openai_embedding_fn()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_OPENAI)

    dense = DenseRetriever(collection=collection, embedding_fn=embedding_fn)
    reranking = RerankingRetriever(dense=dense)

    logger.info("Evaluating dense retrieval...")
    d_mrr, d_hr, d_p = _score(dense, eval_items, k=DENSE_TOP_K)

    logger.info("Evaluating reranking retrieval...")
    r_mrr, r_hr, r_p = _score(reranking, eval_items, k=RERANK_TOP_K)

    header = f"{'Metric':<20} {'Dense (k=' + str(DENSE_TOP_K) + ')':>22} {'Rerank (k=' + str(RERANK_TOP_K) + ')':>22}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for label, d_scores, r_scores in [
        ("MRR", d_mrr, r_mrr),
        ("Hit Rate", d_hr, r_hr),
        ("Precision@k", d_p, r_p),
    ]:
        d_ci = _fmt(bootstrap_ci(d_scores))
        r_ci = _fmt(bootstrap_ci(r_scores))
        print(f"{label:<20} {d_ci:>22} {r_ci:>22}")
    print(sep)

    chunks_by_id = {c.chunk_id: c for c in Embedder.load(CHUNKS_PATH)}
    is_code = [_is_code_item(item, chunks_by_id) for item in eval_items]
    n_code, n_prose = sum(is_code), len(is_code) - sum(is_code)
    logger.info(
        f"Eval items by chunk type: {n_code} code-containing, {n_prose} pure-prose"
    )

    stratified = {}
    for label, key, mask_value, n in [
        ("Code-containing", "code", True, n_code),
        ("Pure-prose", "prose", False, n_prose),
    ]:
        print(f"\n--- {label} chunks (n={n}) ---\n{sep}\n{header}\n{sep}")
        slice_metrics = {}
        for metric_label, metric_key, d_scores, r_scores in [
            ("MRR", "mrr", d_mrr, r_mrr),
            ("Hit Rate", "hit_rate", d_hr, r_hr),
            ("Precision@k", "precision_at_k", d_p, r_p),
        ]:
            d_ci = _slice_ci(d_scores, is_code, mask_value)
            r_ci = _slice_ci(r_scores, is_code, mask_value)
            print(f"{metric_label:<20} {_fmt(d_ci):>22} {_fmt(r_ci):>22}")
            slice_metrics[metric_key] = {"dense": d_ci.mean, "reranking": r_ci.mean}
        print(sep)
        stratified[key] = {"n": n, **slice_metrics}

    results = {
        "n_queries": len(eval_items),
        "dense": {
            "k": DENSE_TOP_K,
            "mrr": sum(d_mrr) / len(d_mrr),
            "hit_rate": sum(d_hr) / len(d_hr),
            "precision_at_k": sum(d_p) / len(d_p),
        },
        "reranking": {
            "k": RERANK_TOP_K,
            "n_candidates": dense._collection.count(),
            "mrr": sum(r_mrr) / len(r_mrr),
            "hit_rate": sum(r_hr) / len(r_hr),
            "precision_at_k": sum(r_p) / len(r_p),
        },
        "stratified": stratified,
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {_RESULTS_PATH}")
