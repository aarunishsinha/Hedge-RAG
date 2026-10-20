from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

import numpy as np


class Retriever(ABC):
    """Base class for every retriever. Retrievers only score; ranking.top_k ranks.

    Subclasses must set the two class attributes below and implement every
    @abstractmethod; a subclass missing a method can't be instantiated.
    """

    kind: ClassVar[str]                      # "bm25" or "dense"; also the index folder name
    no_match_score: ClassVar[float | None]   # scores at or below this mean "not a match" (None: every score counts)

    doc_ids: np.ndarray                      # doc_ids[i] is the doc that scores[i] belongs to

    def __init_subclass__(cls, **kwargs):
        # Class attributes can't be @abstractmethod, so check them when the subclass is defined
        super().__init_subclass__(**kwargs)
        for name in ("kind", "no_match_score"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} must set class attribute {name!r}")

    @classmethod
    @abstractmethod
    def build(cls, corpus: dict[str, dict], cfg) -> Self: ...

    @classmethod
    @abstractmethod
    def load(cls, index_dir: Path, cfg) -> Self: ...

    @staticmethod
    @abstractmethod
    def index_params(cfg) -> dict:
        """Every setting that changes the stored index (used in its fingerprint)."""

    @abstractmethod
    def save(self, index_dir: Path) -> None: ...

    @abstractmethod
    def score(self, query: str) -> np.ndarray:
        """One score per doc, aligned with doc_ids. Higher is better."""

    def build_stats(self) -> dict:
        """Optional facts about the build, saved in the index manifest. Override to add some."""
        return {}
