from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

class BM25Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k1: float = 1.5
    b: float = 0.75
    method: Literal["lucene", "robertson", "atire", "bm25l", "bm25+"] = "lucene"
    stemmer: str | None = "english"              # PyStemmer language, or None to disable
    stopwords: str | None = "en"                 # bm25s stopword list, or None to disable

class DenseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "BAAI/bge-small-en-v1.5"
    revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"   # pinned HF commit
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    max_seq_length: int = 512
    batch_size: int = 64                         # only affects build speed, not results practically

class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")   # unknown keys are an error

    dataset: Literal["scifact", "fiqa"]
    split: str = "test"
    retriever: Literal["bm25", "dense"]
    bm25: BM25Config = BM25Config()
    dense: DenseConfig = DenseConfig()
    judge: str = "none"
    retrieve_k: int = 100
    rerank_k: int = 20
    metrics: list[str] = ["ndcg@10", "recall@100", "mrr@10"]
    seed: int = 42
    num_threads: int = 4                         # torch threads (dense only); fixed so latency is comparable
    warmup_queries: int = 5                      # run first, not timed

    @model_validator(mode="after")
    def metrics_fit_in_retrieve_k(self) -> "RunConfig":
        # A metric like "recall@100" needs at least 100 retrieved docs per query,
        # otherwise it silently reports a capped number.
        for metric in self.metrics:
            if "@" not in metric:
                continue                         # no cutoff: uses the whole run
            cutoff = int(metric.split("@")[1])
            if cutoff > self.retrieve_k:
                raise ValueError(
                    f"metric {metric!r} needs retrieve_k >= {cutoff}, got retrieve_k={self.retrieve_k}"
                )
        return self

def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:                        # closes the file properly
        return RunConfig(**yaml.safe_load(f))
