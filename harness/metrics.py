import math
from ranx import Qrels, Run, evaluate

def ndcg_at_k(scores: dict[str, float], qrel: dict[str, int], k: int = 10) -> float:
    ranked = sorted(scores, key=scores.get, reverse=True)[:k]
    dcg = sum(qrel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked))
    ideal = sorted(qrel.values(), reverse=True)[:k]           # from qrels!
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def library_metrics(qrels, run, metrics=("ndcg@10", "recall@100", "mrr@10")):
    return evaluate(Qrels(qrels), Run(run), list(metrics))