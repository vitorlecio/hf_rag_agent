import random
from dataclasses import dataclass

import numpy as np


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    return float(any(cid in relevant_ids for cid in retrieved_ids))


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for cid in retrieved_ids if cid in relevant_ids) / len(retrieved_ids)


@dataclass
class BootstrapCI:
    mean: float
    lower: float
    upper: float


def bootstrap_ci(
    scores: list[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    rng = random.Random(42)
    n = len(scores)
    boot_means = [float(np.mean(rng.choices(scores, k=n))) for _ in range(n_iterations)]
    alpha = 1.0 - confidence
    return BootstrapCI(
        mean=float(np.mean(scores)),
        lower=float(np.percentile(boot_means, 100 * alpha / 2)),
        upper=float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )
