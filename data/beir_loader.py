import csv, json, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
DATA_DIR = Path("/data")

@dataclass
class Dataset:
    corpus: dict[str, dict]            # doc_id -> {"title", "text"}
    queries: dict[str, str]            # query_id -> question
    qrels: dict[str, dict[str, int]]   # query_id -> {doc_id: relevance}

def _download(name: str) -> Path:
    root = DATA_DIR / name
    if not root.exists():
        zip_path = DATA_DIR / f"{name}.zip"
        urllib.request.urlretrieve(URL.format(name), zip_path)
        zipfile.ZipFile(zip_path).extractall(DATA_DIR)
    return root

def load(name: str, split: str = "test") -> Dataset:
    root = _download(name)

    # Relevance labels: one row per (query, doc, relevance)
    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)                                    # skip header
        for query_id, doc_id, relevance in rows:
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)

    # Queries: one JSON object per line
    queries: dict[str, str] = {}
    with open(root / "queries.jsonl") as f:
        for line in f:
            query = json.loads(line)
            query_id = str(query["_id"])
            if query_id in qrels:                     # keep this split only
                queries[query_id] = query["text"]

    # Corpus: one JSON object per line
    corpus: dict[str, dict] = {}
    with open(root / "corpus.jsonl") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc["_id"])
            corpus[doc_id] = {"title": doc.get("title", ""), "text": doc["text"]}

    return Dataset(corpus, queries, qrels)