import numpy as np
import pytest

from harness.config import BM25Config, DenseConfig
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.index_store import load_or_build
from toy_corpus import CORPUS, QUERIES

CASES = [(BM25Retriever, BM25Config()), (DenseRetriever, DenseConfig())]


@pytest.mark.parametrize("retriever_cls, cfg", CASES, ids=["bm25", "dense"])
def test_saved_index_gives_identical_scores(tmp_path, retriever_cls, cfg):
    built, info = load_or_build(retriever_cls, "toy", CORPUS, cfg, root=tmp_path)
    loaded, info2 = load_or_build(retriever_cls, "toy", CORPUS, cfg, root=tmp_path)

    assert info["action"] == "built"
    assert info2["action"] == "loaded"
    assert info2["index_id"] == info["index_id"]
    assert list(loaded.doc_ids) == list(built.doc_ids)
    for query in QUERIES:
        np.testing.assert_allclose(loaded.score(query), built.score(query), rtol=1e-6, atol=1e-6)


def test_rebuild_only_when_asked(tmp_path):
    cfg = BM25Config()
    _, first = load_or_build(BM25Retriever, "toy", CORPUS, cfg, root=tmp_path)
    _, second = load_or_build(BM25Retriever, "toy", CORPUS, cfg, root=tmp_path)
    _, third = load_or_build(BM25Retriever, "toy", CORPUS, cfg, rebuild=True, root=tmp_path)

    assert [first["action"], second["action"], third["action"]] == ["built", "loaded", "rebuilt"]
    assert not any(p.name.endswith(".tmp") for p in tmp_path.rglob("*"))   # no leftovers


def test_changed_settings_or_corpus_get_a_new_index(tmp_path):
    _, base = load_or_build(BM25Retriever, "toy", CORPUS, BM25Config(), root=tmp_path)
    _, new_k1 = load_or_build(BM25Retriever, "toy", CORPUS, BM25Config(k1=0.9), root=tmp_path)

    edited = {**CORPUS, "d5": {"title": "New", "text": "A new document."}}
    _, new_corpus = load_or_build(BM25Retriever, "toy", edited, BM25Config(), root=tmp_path)

    assert len({base["index_id"], new_k1["index_id"], new_corpus["index_id"]}) == 3
    assert new_k1["action"] == new_corpus["action"] == "built"


def test_query_only_settings_reuse_the_dense_index(tmp_path):
    _, base = load_or_build(DenseRetriever, "toy", CORPUS, DenseConfig(), root=tmp_path)
    _, other = load_or_build(DenseRetriever, "toy", CORPUS,
                             DenseConfig(query_prefix="", batch_size=8), root=tmp_path)
    assert other["index_id"] == base["index_id"]
    assert other["action"] == "loaded"
