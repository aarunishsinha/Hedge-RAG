import pytest
from data.beir_loader import load

EXPECTED = {"scifact": (5183, 300), "fiqa": (57638, 648)}

@pytest.mark.parametrize("name", EXPECTED)
def test_dataset(name):
    ds = load(name)
    n_docs, n_queries = EXPECTED[name]

    #1 Sizes match the publised number
    assert len(ds.corpus) == n_docs
    assert len(ds.queries) == n_queries

    #2 Every query in the answer has a question text and vice versa
    assert set(ds.queries) == set(ds.qrels)

    # 3. Every document the answer key mentions exists in the corpus
    missing = {d for rels in ds.qrels.values() for d in rels if d not in ds.corpus}
    assert not missing, f"{len(missing)} missing, e.g. {list(missing)[:5]}"

    # 4. All IDs are strings (the int-vs-str trap)
    assert all(isinstance(k, str) for k in ds.corpus)
    assert all(isinstance(d, str) for rels in ds.qrels.values() for d in rels)