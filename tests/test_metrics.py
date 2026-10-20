import math, random
from ranx import Qrels, Run, evaluate
from data.beir_loader import load
from harness.metrics import ndcg_at_k

def test_drill():
    scores = {f"d{i}": 5 - i for i in range(5)}               # d0 ranked first
    qrel = {"d0": 3, "d1": 0, "d2": 2, "d3": 0, "d4": 1}
    assert math.isclose(ndcg_at_k(scores, qrel, k=5), 0.921, abs_tol=1e-3)

def test_matches_ranx():
    dataset = load("scifact")
    rng = random.Random(42)
    all_doc_ids = list(dataset.corpus.keys())

    first_five_query_ids = list(dataset.qrels.keys())[:5]

    for index, query_id in enumerate(first_five_query_ids):
        labels = dataset.qrels[query_id]                 # doc_id -> relevance
        relevant_doc_ids = list(labels.keys())

        if index == 0:
            relevant_doc_ids = relevant_doc_ids[1:]      # drop one relevant doc on purpose

        random_doc_ids = rng.sample(all_doc_ids, 20)
        doc_ids = relevant_doc_ids + random_doc_ids

        scores = {}                                      # doc_id -> score
        for doc_id in doc_ids:
            scores[doc_id] = rng.random()                # random floats, so no ties

        my_value = ndcg_at_k(scores, labels)

        qrels = Qrels({query_id: labels})
        run = Run({query_id: scores})
        ranx_value = evaluate(qrels, run, "ndcg@10")

        assert math.isclose(my_value, ranx_value, abs_tol=1e-6), (query_id, my_value, ranx_value)