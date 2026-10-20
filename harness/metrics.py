import math

import numpy as np
from ranx import Qrels, Run, evaluate

from retrieval.ranking import top_k

def ndcg_at_k(scores: dict[str, float], qrel: dict[str, int], k: int = 10) -> float:
    """Hand-written nDCG@k (linear gain), used by the tests to cross-check ranx.
    The harness reports ranx's numbers (library_metrics)."""
    # 1. Rank with the shared top_k (score descending, ties by doc_id), keep the top k
    doc_ids = np.array(list(scores.keys()))
    doc_scores = np.array(list(scores.values()))
    top_k_doc_ids = [doc_id for doc_id, _ in top_k(doc_ids, doc_scores, k)]

    # 2. DCG: each doc's relevance, discounted by its rank
    dcg = 0.0
    for i, doc_id in enumerate(top_k_doc_ids):
        rank = i + 1
        relevance = qrel.get(doc_id, 0)                  # 0 if not labelled relevant
        dcg += relevance / math.log2(rank + 1)

    # 3. Ideal DCG: the best possible ranking, built from ALL labelled docs (from qrels!)
    ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        rank = i + 1
        idcg += relevance / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg

def library_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    metrics: list[str] | tuple[str, ...] = ("ndcg@10", "recall@100", "mrr@10"),
) -> dict[str, float]:
    """Mean of each metric over all queries, computed by ranx. Always returns {metric: value}."""
    metrics = list(metrics)
    result = evaluate(Qrels(qrels), Run(run), metrics)

    # ranx returns a bare number (not a dict) when asked for exactly one metric
    if len(metrics) == 1:
        return {metrics[0]: float(result)}
    return {name: float(value) for name, value in result.items()}