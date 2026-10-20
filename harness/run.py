"""Run one retrieval experiment from a config file.

    python -m harness.run configs/scifact_bm25.yaml            # loads the stored index
    python -m harness.run configs/scifact_bm25.yaml --rebuild  # rebuilds it on purpose
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.beir_loader import load
from harness.config import RunConfig, load_config
from harness.environment import environment
from harness.metrics import library_metrics
from retrieval.base import Retriever
from retrieval.index_store import load_or_build
from retrieval.ranking import top_k

log = logging.getLogger("harness.run")


def retriever_for(cfg: RunConfig) -> tuple[type[Retriever], object]:
    """The retriever class for this config, plus its settings section.

    The one place that maps cfg.retriever to code: add new retrievers here.
    Imports happen here, not at the top, so a BM25 run never loads torch /
    sentence-transformers (about a second of startup it doesn't need).
    """
    if cfg.retriever == "bm25":
        from retrieval.bm25 import BM25Retriever
        return BM25Retriever, cfg.bm25

    import torch
    torch.set_num_threads(cfg.num_threads)       # fix threads before the model is loaded
    from retrieval.dense import DenseRetriever
    return DenseRetriever, cfg.dense


def new_run_dir(runs_dir: Path, base_name: str) -> Path:
    """Create a fresh folder for this run; never reuse one from another run."""
    run_dir = runs_dir / base_name
    counter = 2
    while run_dir.exists():                      # e.g. two runs finishing in the same second
        run_dir = runs_dir / f"{base_name}-{counter}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def results_summary(rankings: dict[str, list[tuple[str, float]]], k: int) -> dict:
    """How many queries got fewer than k docs. BM25 isn't padded, so this affects recall@k."""
    fewer_than_k = 0
    empty = 0
    for ranked in rankings.values():
        if len(ranked) < k:
            fewer_than_k += 1
        if len(ranked) == 0:
            empty += 1
    return {"queries": len(rankings), "k": k, "fewer_than_k": fewer_than_k, "empty": empty}


def retrieve(retriever, query: str, k: int) -> list[tuple[str, float]]:
    """Raw query text -> ranked top-k. This whole step is what latency measures."""
    scores = retriever.score(query)
    # A mismatch would make top_k pair scores with the wrong docs, silently
    if len(scores) != len(retriever.doc_ids):
        raise ValueError(f"{retriever.kind}: got {len(scores)} scores for {len(retriever.doc_ids)} docs")
    return top_k(retriever.doc_ids, scores, k, keep_above=retriever.no_match_score)


def latency_summary(latencies_ms: list[float]) -> dict:
    values = np.array(latencies_ms)
    return {
        "count": len(values),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def run(cfg: RunConfig, rebuild: bool = False, runs_dir: Path = Path("runs")) -> dict:
    dataset = load(cfg.dataset, cfg.split)

    # 1. Index: load the stored one, or build it if missing / rebuild asked for
    retriever_cls, settings = retriever_for(cfg)
    retriever, index_info = load_or_build(
        retriever_cls, cfg.dataset, dataset.corpus, settings, rebuild=rebuild
    )

    query_ids = sorted(dataset.queries)

    # 2. Warm-up: the first calls pay one-off costs (lazy init, caches), so don't time them
    for query_id in query_ids[:cfg.warmup_queries]:
        retrieve(retriever, dataset.queries[query_id], cfg.retrieve_k)

    # 3. Retrieve every query on its own and time it: raw text -> ranked top-k
    rankings: dict[str, list[tuple[str, float]]] = {}
    latency_rows = []
    # The bar updates between iterations, outside the timed section, so it doesn't affect latency
    for query_id in tqdm(query_ids, desc=f"Retrieving ({cfg.retriever})", unit="query"):
        start = time.perf_counter()
        ranked = retrieve(retriever, dataset.queries[query_id], cfg.retrieve_k)
        elapsed_ms = (time.perf_counter() - start) * 1000

        rankings[query_id] = ranked
        latency_rows.append({"query_id": query_id, "latency_ms": elapsed_ms})

    # 4. Evaluate. ranx re-sorts by score and would break ties its own way,
    #    so give it rank-based scores (k, k-1, ..., 1) that keep our exact order.
    ranx_run = {}
    for query_id, ranked in rankings.items():
        ranx_run[query_id] = {}
        for rank, (doc_id, _) in enumerate(ranked):
            ranx_run[query_id][doc_id] = float(len(ranked) - rank)
    metrics = library_metrics(dataset.qrels, ranx_run, cfg.metrics)

    # 5. Write everything for this run
    finished = datetime.now(timezone.utc)
    run_dir = new_run_dir(runs_dir, f"{finished:%Y%m%dT%H%M%SZ}_{cfg.dataset}_{cfg.retriever}")
    run_id = run_dir.name

    with open(run_dir / "results.jsonl", "w") as f:
        for query_id, ranked in rankings.items():
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(json.dumps({"query_id": query_id, "rank": rank,
                                    "doc_id": doc_id, "score": score}) + "\n")

    with open(run_dir / "latency.jsonl", "w") as f:
        for row in latency_rows:
            f.write(json.dumps(row) + "\n")

    index_summary = {key: index_info[key] for key in
                     ("index_id", "index_dir", "action", "num_docs", "build_seconds", "save_seconds")}
    if "load_seconds" in index_info:
        index_summary["load_seconds"] = index_info["load_seconds"]
    index_summary["build_stats"] = index_info.get("build_stats", {})   # missing in older manifests

    summary = {
        "run_id": run_id,
        "config": cfg.model_dump(),
        "metrics": metrics,
        "results": results_summary(rankings, cfg.retrieve_k),
        "latency": latency_summary([row["latency_ms"] for row in latency_rows]),
        "index": index_summary,
        "environment": environment(),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Run %s -> %s", run_id, run_dir)
    log.info("Index %s (%s): build %.2fs", index_summary["index_id"], index_summary["action"],
             index_summary["build_seconds"])
    log.info("Latency: mean %.2fms, p50 %.2fms, p95 %.2fms, p99 %.2fms",
             summary["latency"]["mean_ms"], summary["latency"]["p50_ms"],
             summary["latency"]["p95_ms"], summary["latency"]["p99_ms"])
    log.info("Results: %d of %d queries got fewer than %d docs (%d got none)",
             summary["results"]["fewer_than_k"], summary["results"]["queries"],
             cfg.retrieve_k, summary["results"]["empty"])
    log.info("Metrics: %s", ", ".join(f"{k}={v:.4f}" for k, v in summary["metrics"].items()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one retrieval experiment.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the index even if a stored one exists")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config(args.config), rebuild=args.rebuild, runs_dir=args.runs_dir)


if __name__ == "__main__":
    main()
