from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict

class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")   # unknown keys are an error

    dataset: Literal["scifact", "fiqa"]
    split: str = "test"
    judge: str = "none"
    retrieve_k: int = 100
    rerank_k: int = 20
    metrics: list[str] = ["ndcg@10", "recall@100", "mrr@10"]
    seed: int = 42

def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:                        # closes the file properly
        return RunConfig(**yaml.safe_load(f))