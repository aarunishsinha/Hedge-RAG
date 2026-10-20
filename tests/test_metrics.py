import math, random
from ranx import Qrels, Run, evaluate
from data.beir_loader import load
from harness.metrics import library_metrics, ndcg_at_k

def test_drill():
    scores = {f"d{i}": 5 - i for i in range(5)}               # d0 ranked first
    qrel = {"d0": 3, "d1": 0, "d2": 2, "d3": 0, "d4": 1}
    assert math.isclose(ndcg_at_k(scores, qrel, k=5), 0.921, abs_tol=1e-3)

def test_matches_ranx():
    dataset = load("scifact")
    rng = random.Random(42)
    all_doc_ids = list(dataset.corpus.keys())

    first_five_query_ids = list(dataset.qrels.keys())[:5]

    drops = 0
    for query_id in first_five_query_ids:
        labels = dataset.qrels[query_id]                 # doc_id -> relevance
        relevant_doc_ids = list(labels.keys())

        # Drop one relevant doc on purpose, but only if another one stays in the
        # ranking. With a single relevant doc, dropping it makes both a correct
        # and a buggy ideal DCG return 0, and the test proves nothing.
        if len(relevant_doc_ids) >= 2:
            relevant_doc_ids = relevant_doc_ids[1:]
            drops += 1

        # Random filler docs come only from non-relevant docs, so the dropped
        # doc can't sneak back in and filler is never accidentally relevant
        non_relevant_doc_ids = []
        for doc_id in all_doc_ids:
            if doc_id not in labels:
                non_relevant_doc_ids.append(doc_id)
        random_doc_ids = rng.sample(non_relevant_doc_ids, 20)
        doc_ids = relevant_doc_ids + random_doc_ids

        scores = {}                                      # doc_id -> score
        for doc_id in doc_ids:
            scores[doc_id] = rng.random()                # random floats, so no ties

        my_value = ndcg_at_k(scores, labels)

        qrels = Qrels({query_id: labels})
        run = Run({query_id: scores})
        ranx_value = evaluate(qrels, run, "ndcg@10")

        assert math.isclose(my_value, ranx_value, abs_tol=1e-6), (query_id, my_value, ranx_value)

    assert drops > 0, "no query had 2+ relevant docs, so the missing-doc case wasn't tested"


def test_library_metrics_always_returns_a_dict():
    qrels = {"q1": {"a": 1}}
    run = {"q1": {"a": 2.0, "b": 1.0}}
    assert library_metrics(qrels, run, ["ndcg@10"]) == {"ndcg@10": 1.0}        # ranx alone gives a bare float here
    assert library_metrics(qrels, run, ["ndcg@10", "mrr@10"]) == {"ndcg@10": 1.0, "mrr@10": 1.0}
