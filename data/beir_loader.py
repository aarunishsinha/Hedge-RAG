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
    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)                                    # skip header
        for qid, did, rel in rows:
            qrels.setdefault(qid, {})[did] = int(rel)
    with open(root / "queries.jsonl") as f:
        queries = {str(q["_id"]): q["text"] for q in map(json.loads, f)
                   if str(q["_id"]) in qrels}         # keep this split only
    with open(root / "corpus.jsonl") as f:
        corpus = {str(d["_id"]): {"title": d.get("title", ""), "text": d["text"]}
                  for d in map(json.loads, f)}
    return Dataset(corpus, queries, qrels)