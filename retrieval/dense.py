import json
import logging
from pathlib import Path
from typing import Self

import numpy as np
from sentence_transformers import SentenceTransformer

from harness.config import DenseConfig
from retrieval.base import Retriever
from retrieval.text import doc_text

log = logging.getLogger(__name__)

class DenseRetriever(Retriever):
    kind = "dense"
    no_match_score = None          # every doc gets a real similarity score

    def __init__(self, cfg: DenseConfig, embeddings: np.ndarray, doc_ids: np.ndarray,
                 model: SentenceTransformer | None = None):
        self.cfg = cfg
        self.embeddings = embeddings             # (n_docs, dim), L2-normalised
        self.doc_ids = doc_ids
        self.model = model if model is not None else _load_model(cfg)
        self._build_stats: dict = {}             # filled in by build(); empty after load()

    @staticmethod
    def index_params(cfg: DenseConfig) -> dict:
        # query_prefix only affects queries and batch_size only affects speed,
        # so neither is part of the index's identity.
        return cfg.model_dump(exclude={"query_prefix", "batch_size"})

    @classmethod
    def build(cls, corpus: dict[str, dict], cfg: DenseConfig) -> Self:
        doc_ids = np.array(sorted(corpus))
        texts = [doc_text(corpus[doc_id]) for doc_id in doc_ids]

        model = _load_model(cfg)
        truncation = _count_truncated_docs(model, texts, cfg.max_seq_length)
        log.info("%d of %d docs are longer than %d tokens and will be truncated (longest: %d tokens)",
                 truncation["docs_truncated"], truncation["docs_total"],
                 cfg.max_seq_length, truncation["longest_doc_tokens"])

        embeddings = model.encode(
            texts,
            batch_size=cfg.batch_size,
            normalize_embeddings=True,           # cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        retriever = cls(cfg, embeddings, doc_ids, model=model)
        retriever._build_stats = {"truncation": truncation}
        return retriever

    def build_stats(self) -> dict:
        return self._build_stats

    def save(self, index_dir: Path) -> None:
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with open(index_dir / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids.tolist(), f)

    @classmethod
    def load(cls, index_dir: Path, cfg: DenseConfig) -> Self:
        embeddings = np.load(index_dir / "embeddings.npy", mmap_mode="r")
        with open(index_dir / "doc_ids.json") as f:
            doc_ids = np.array(json.load(f))
        return cls(cfg, embeddings, doc_ids)

    def score(self, query: str) -> np.ndarray:
        query_vector = self.model.encode(
            query,
            prompt=self.cfg.query_prefix,        # BGE instruction, queries only
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self.embeddings @ query_vector


def _count_truncated_docs(model: SentenceTransformer, texts: list[str], max_seq_length: int) -> dict:
    """How many docs the model will cut off at max_seq_length tokens (special tokens included)."""
    encoded = model.tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
    lengths = [len(token_ids) for token_ids in encoded["input_ids"]]

    docs_truncated = 0
    for length in lengths:
        if length > max_seq_length:
            docs_truncated += 1

    return {
        "max_seq_length": max_seq_length,
        "docs_total": len(lengths),
        "docs_truncated": docs_truncated,
        "longest_doc_tokens": max(lengths, default=0),
    }


def _load_model(cfg: DenseConfig) -> SentenceTransformer:
    model = SentenceTransformer(cfg.model, revision=cfg.revision, device="cpu")
    model.max_seq_length = cfg.max_seq_length
    return model
