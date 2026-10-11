import math, random
from ranx import Qrels, Run, evaluate
from data.beir_loader import load
from harness.metrics import ndcg_at_k

def test_drill():
    scores = {f"d{i}": 5 - i for i in range(5)}               # d0 ranked first
    qrel = {"d0": 3, "d1": 0, "d2": 2, "d3": 0, "d4": 1}
    assert math.isclose(ndcg_at_k(scores, qrel, k=5), 0.921, abs_tol=1e-3)

def test_matches_ranx():
    ds, rng = load("scifact"), random.Random(42)
    all_docs = list(ds.corpus)
    for n, qid in enumerate(list(ds.qrels)[:5]):
        relevant = list(ds.qrels[qid])
        if n == 0:
            relevant = relevant[1:]          # drop one relevant doc on purpose
        docs = relevant + rng.sample(all_docs, 20)
        scores = {d: rng.random() for d in docs}               # distinct scores
        mine = ndcg_at_k(scores, ds.qrels[qid])
        lib = evaluate(Qrels({qid: ds.qrels[qid]}), Run({qid: scores}), "ndcg@10")
        assert math.isclose(mine, lib, abs_tol=1e-6), (qid, mine, lib)