import pytest

from harness.config import BM25Config, RunConfig
from harness.run import new_run_dir, results_summary, retrieve, retriever_for
from retrieval.bm25 import BM25Retriever
from toy_corpus import CORPUS


def test_run_dirs_are_never_reused(tmp_path):
    first = new_run_dir(tmp_path, "20261020T120000Z_scifact_bm25")
    second = new_run_dir(tmp_path, "20261020T120000Z_scifact_bm25")       # same second
    third = new_run_dir(tmp_path, "20261020T120000Z_scifact_bm25")

    assert [first.name, second.name, third.name] == [
        "20261020T120000Z_scifact_bm25",
        "20261020T120000Z_scifact_bm25-2",
        "20261020T120000Z_scifact_bm25-3",
    ]
    assert first.is_dir() and second.is_dir() and third.is_dir()


def test_results_summary_counts_short_and_empty_lists():
    rankings = {
        "q1": [("a", 3.0), ("b", 2.0), ("c", 1.0)],    # full
        "q2": [("a", 1.0)],                            # short
        "q3": [],                                      # empty
    }
    assert results_summary(rankings, k=3) == {"queries": 3, "k": 3, "fewer_than_k": 2, "empty": 1}


def test_retriever_for_returns_matching_class_and_settings():
    cfg = RunConfig(dataset="scifact", retriever="bm25")
    retriever_cls, settings = retriever_for(cfg)
    assert retriever_cls.kind == "bm25"
    assert settings is cfg.bm25

    cfg = RunConfig(dataset="scifact", retriever="dense")
    retriever_cls, settings = retriever_for(cfg)
    assert retriever_cls.kind == "dense"
    assert settings is cfg.dense


def test_score_length_mismatch_is_an_error():
    retriever = BM25Retriever.build(CORPUS, BM25Config())
    retriever.doc_ids = retriever.doc_ids[:-1]                # one doc short
    with pytest.raises(ValueError, match="scores for"):
        retrieve(retriever, "cat", 3)
