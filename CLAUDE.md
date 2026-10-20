# Hedge-RAG

Calibrated RAG: comparing cross-encoder, LLM and decision models on quality, cost, latency and calibration. Current focus is BEIR datasets (SciFact, FiQA).

## Working guidelines

- **Log learnings.** Record bugs (resolved or observed), hard decisions and major design choices in [learnings.md](learnings.md), newest first, tagged Bug / Decision / Design with status Open / Resolved.
- **Do only what was asked.** When asked to explain, plan or review, don't change code. When asked for one change, don't bundle in other fixes; mention them instead.
- **Readable over clever.** Write code that someone coming from Java can follow: descriptive names, explicit loops instead of dense comprehensions or `map` one-liners, one step per line, short comments on intent.
- **Single responsibility, no duplication.** Shared logic lives in one place. Retrievers only score documents; ranking is done by the one shared function, `retrieval/ranking.py:top_k`, which every retriever and metric uses.
- **Deterministic ranking.** Sort by score descending, then doc ID ascending (string order, so `"10"` < `"9"`). Never rely on a library's tie-breaking: when handing a ranking to ranx, pass rank-based scores (`k, k-1, …, 1`) so it can't re-order ties. Never pad a ranking with non-matching docs: each retriever sets `no_match_score` (BM25: `0.0`) and `top_k(..., keep_above=...)` drops them.
- **Reproducible models.** Pin every Hugging Face model to a commit SHA (`revision`), never a branch.
- **nDCG uses linear gain** (gain = relevance label), matching published BEIR numbers and ranx's `ndcg`.
- **Strict configs.** Config models use `extra="forbid"`, so unknown keys are errors. Add new settings as fields.
- **Measure cost.** Log index build time and per-query latency (single query, raw text → ranked top-k, after untimed warm-up queries).
- **Indexes are persistent; rebuilding is intentional.** Indexes live on the `ir-data` volume and are loaded if present. They are only rebuilt with `--rebuild`. A change to the corpus, index settings or library versions changes the fingerprint and builds a new index alongside the old one.

## Layout

```
data/beir_loader.py      download + load a BEIR dataset into /data
harness/config.py        RunConfig (YAML -> validated settings)
harness/run.py           CLI: load/build index -> retrieve -> evaluate -> write runs/<run_id>/
harness/metrics.py       hand-written nDCG@k + ranx wrapper
harness/environment.py   versions / hardware info logged with every run and index
retrieval/ranking.py     top_k: the one deterministic ranking function
retrieval/bm25.py        BM25 (bm25s + PyStemmer)
retrieval/dense.py       bge-small-en-v1.5 (sentence-transformers), exact dot-product search
retrieval/index_store.py fingerprint, load-or-build, atomic save
configs/*.yaml           one file per experiment
```

## Environment

Everything runs in Docker (Linux, CPU-only torch). Dependencies are managed with uv; the lockfile resolves for Linux only.

Volumes (declared in [compose.yaml](compose.yaml)):

| Volume | Mounted at | Holds |
|---|---|---|
| repo folder | `/app` | live code, `runs/` outputs |
| `ir-data` | `/data` | datasets, `/data/indexes/<dataset>/<retriever>/<index_id>/` |
| `hf-cache` | `/root/.cache/huggingface` | downloaded models |

## Commands

From the repo root on the host:

```bash
docker compose build                      # build the image (after any dependency change)
docker compose run --rm hedge-rag bash    # open a shell in a container with all volumes mounted
```

Inside the container (working dir `/app`):

```bash
python -m pytest -q                                          # all tests
python -m harness.run configs/scifact_bm25.yaml              # BM25: builds index first time, loads after
python -m harness.run configs/scifact_dense.yaml             # dense (bge-small)
python -m harness.run configs/scifact_bm25.yaml --rebuild    # rebuild the index on purpose
ls /data/indexes/scifact/*/                                  # stored indexes
```

Each run writes `runs/<run_id>/` with `summary.json` (metrics, latency stats, index info, environment), `results.jsonl` (ranked docs per query) and `latency.jsonl` (per-query latency).

Models load offline: compose sets `HF_HUB_OFFLINE=1`, so runs read only from `hf-cache` and never contact the Hub. To download a new model or revision once, override it for that run: `docker compose run --rm -e HF_HUB_OFFLINE=0 hedge-rag bash`. On a fresh machine, do this once before running the tests or a dense run: the dense tests and runs fail offline until bge-small is in `hf-cache`.

Adding a dependency: edit `pyproject.toml`, run `uv lock` on the host, then `docker compose build`.
