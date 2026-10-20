import numpy as np
from retrieval.ranking import top_k


def test_sorts_by_score_descending():
    doc_ids = np.array(["a", "b", "c"])
    scores = np.array([0.1, 0.9, 0.5])
    assert top_k(doc_ids, scores, 3) == [("b", 0.9), ("c", 0.5), ("a", 0.1)]


def test_ties_break_by_doc_id_ascending():
    doc_ids = np.array(["d3", "d1", "d2"])
    scores = np.array([1.0, 1.0, 1.0])
    assert [doc for doc, _ in top_k(doc_ids, scores, 3)] == ["d1", "d2", "d3"]


def test_doc_ids_compare_as_strings():
    doc_ids = np.array(["9", "10"])
    scores = np.array([1.0, 1.0])
    assert [doc for doc, _ in top_k(doc_ids, scores, 2)] == ["10", "9"]


def test_tie_at_cutoff_keeps_smallest_doc_ids():
    # Four docs tie for the last two places; the tie-break decides, not array order.
    doc_ids = np.array(["z", "top", "y", "b", "a"])
    scores = np.array([0.5, 0.9, 0.5, 0.5, 0.5])
    assert [doc for doc, _ in top_k(doc_ids, scores, 3)] == ["top", "a", "b"]


def test_same_result_whatever_the_input_order():
    rng = np.random.default_rng(0)
    doc_ids = np.array([f"d{i}" for i in range(200)])
    scores = rng.integers(0, 5, size=200).astype(float)     # lots of ties
    expected = top_k(doc_ids, scores, 50)
    for _ in range(5):
        perm = rng.permutation(200)
        assert top_k(doc_ids[perm], scores[perm], 50) == expected


def test_keep_above_drops_non_matches_instead_of_padding():
    doc_ids = np.array(["a", "b", "c", "d"])
    scores = np.array([0.0, 2.0, 0.0, 1.0])
    assert top_k(doc_ids, scores, 3, keep_above=0.0) == [("b", 2.0), ("d", 1.0)]
    assert len(top_k(doc_ids, scores, 3)) == 3                 # without it, "a" pads the list


def test_keep_above_with_no_matches_returns_empty():
    doc_ids = np.array(["a", "b"])
    scores = np.array([0.0, 0.0])
    assert top_k(doc_ids, scores, 10, keep_above=0.0) == []


def test_k_larger_than_corpus_and_zero():
    doc_ids = np.array(["a", "b"])
    scores = np.array([0.2, 0.4])
    assert top_k(doc_ids, scores, 10) == [("b", 0.4), ("a", 0.2)]
    assert top_k(doc_ids, scores, 0) == []
