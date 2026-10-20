import numpy as np
import pytest

from harness.config import BM25Config, DenseConfig
from retrieval.base import Retriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.index_store import load_or_build
from retrieval.ranking import top_k
from toy_corpus import CORPUS


# --- Base class contract ---

def test_retriever_missing_a_method_cannot_be_created():
    class Incomplete(Retriever):
        kind = "incomplete"
        no_match_score = None

        def score(self, query):
            return np.zeros(0)

    with pytest.raises(TypeError):
        Incomplete()


def test_retriever_missing_a_class_attribute_is_rejected():
    with pytest.raises(TypeError, match="no_match_score"):
        class NoThreshold(Retriever):
            kind = "no-threshold"


# --- BM25 ---

def test_stopword_only_query_scores_zero():
    retriever = BM25Retriever.build(CORPUS, BM25Config())
    assert not retriever.score("the").any()


def test_bm25_returns_only_matching_docs():
    retriever = BM25Retriever.build(CORPUS, BM25Config())
    scores = retriever.score("vaccines")                      # only d4 mentions vaccines
    ranked = top_k(retriever.doc_ids, scores, 3, keep_above=retriever.no_match_score)
    assert [doc_id for doc_id, _ in ranked] == ["d4"]


# --- Dense ---

def test_dense_build_records_truncated_docs(tmp_path):
    # With an 8-token limit, every toy doc (title + text + [CLS]/[SEP]) is too long
    _, info = load_or_build(DenseRetriever, "toy", CORPUS, DenseConfig(max_seq_length=8), root=tmp_path)
    truncation = info["build_stats"]["truncation"]
    assert truncation["docs_total"] == len(CORPUS)
    assert truncation["docs_truncated"] == len(CORPUS)
    assert truncation["longest_doc_tokens"] > 8

    _, info = load_or_build(DenseRetriever, "toy", CORPUS, DenseConfig(), root=tmp_path)
    assert info["build_stats"]["truncation"]["docs_truncated"] == 0     # all fit in 512
