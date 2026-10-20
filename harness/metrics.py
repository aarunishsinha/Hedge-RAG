import math
from ranx import Qrels, Run, evaluate

def ndcg_at_k(scores: dict[str, float], qrel: dict[str, int], k: int = 10) -> float:
    # 1. Sort doc IDs by score, highest first, and keep the top k
    ranked_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
    top_k_doc_ids = ranked_doc_ids[:k]

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

def library_metrics(qrels, run, metrics=("ndcg@10", "recall@100", "mrr@10")):
    return evaluate(Qrels(qrels), Run(run), list(metrics))