import json
from pathlib import Path
from typing import Self

import bm25s
import numpy as np
import Stemmer

from harness.config import BM25Config
from retrieval.base import Retriever
from retrieval.text import doc_text


class BM25Retriever(Retriever):
    kind = "bm25"
    no_match_score = 0.0           # 0 = shares no terms with the query

    def __init__(self, cfg: BM25Config, index: bm25s.BM25, doc_ids: np.ndarray):
        self.cfg = cfg
        self.index = index
        self.doc_ids = doc_ids
        self.stemmer = Stemmer.Stemmer(cfg.stemmer) if cfg.stemmer else None

    @staticmethod
    def index_params(cfg: BM25Config) -> dict:
        return cfg.model_dump()                  # every BM25 setting affects the index

    @classmethod
    def build(cls, corpus: dict[str, dict], cfg: BM25Config) -> Self:
        doc_ids = np.array(sorted(corpus))
        texts = [doc_text(corpus[doc_id]) for doc_id in doc_ids]

        retriever = cls(cfg, bm25s.BM25(k1=cfg.k1, b=cfg.b, method=cfg.method), doc_ids)
        tokens = retriever._tokenize(texts, return_ids=True, show_progress=True)
        retriever.index.index(tokens, show_progress=True)   # bm25s's build method is named index()
        return retriever

    def save(self, index_dir: Path) -> None:
        self.index.save(index_dir / "bm25", show_progress=False)
        with open(index_dir / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids.tolist(), f)

    @classmethod
    def load(cls, index_dir: Path, cfg: BM25Config) -> Self:
        index = bm25s.BM25.load(index_dir / "bm25", show_progress=False)
        with open(index_dir / "doc_ids.json") as f:
            doc_ids = np.array(json.load(f))
        return cls(cfg, index, doc_ids)

    def score(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize([query], return_ids=False, show_progress=False)[0]
        if not query_tokens:                     # e.g. a query made only of stopwords
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return self.index.get_scores(query_tokens)

    def _tokenize(self, texts: list[str], return_ids: bool, show_progress: bool):
        return bm25s.tokenize(
            texts,
            stopwords=self.cfg.stopwords,
            stemmer=self.stemmer,
            return_ids=return_ids,
            show_progress=show_progress,         # on for the corpus, off per query
        )
