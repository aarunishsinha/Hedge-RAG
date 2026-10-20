# Learnings

A running log of bugs resolved or observed, hard decisions, and major design choices.
Newest entries go at the top.

Each entry is tagged **Bug**, **Decision** or **Design**, with a status of **Open** or **Resolved**.

---

## Runs don't record which version of our code produced them

**Bug** (observed, deferred) · **Open** · logged 2026-10-20 · [harness/environment.py](harness/environment.py)

`summary.json` and index manifests record the config, library versions and hardware, but not the git commit of this repo or whether there were uncommitted changes. Two runs with the same config can come from different harness or retriever code, and nothing in the output says so. Deferred for now. The fix is to record the commit SHA plus a "dirty" flag. The slim image has no `git`, so either read `.git/HEAD` directly or install `git` in the Dockerfile.

## What the environment block means inside Docker on a Mac

**Design** · **Resolved** · logged 2026-10-20 · [harness/environment.py](harness/environment.py)

- `platform` and `cpu_count` describe Docker's Linux VM (its kernel, and the CPUs it's allowed), not macOS.
- `cpu_model` is read from `/proc/cpuinfo` and is the host's real CPU (here an Intel i9-9880H). Compare latency across machines by CPU model, not by count.
- `torch_threads` is `null` for BM25 runs. The harness imports each retriever only when it's used, so BM25 runs never load torch or sentence-transformers (about a second of startup). `num_threads` only applies to dense.

## 8.8% of SciFact docs are truncated by bge-small

**Bug** (observed, accepted for now) · **Open** · logged 2026-10-20 · [retrieval/dense.py](retrieval/dense.py)

455 of 5,183 SciFact docs (title + abstract) are longer than bge-small's 512-token limit; the longest is 1,939 tokens. The model only sees the first 512 tokens, so evidence near the end of a long abstract is invisible to dense retrieval but not to BM25. That's standard for this model, but it may explain some queries where BM25 beats dense. Dense builds now count this and save it in the manifest under `build_stats.truncation`. Indexes built before this change don't have it. Possible follow-ups: split long docs into chunks and score a doc by its best chunk, or use a model with a longer context.

## BM25 tokeniser drops single characters and splits on punctuation

**Bug** (observed, accepted for now) · **Open** · logged 2026-10-20 · [retrieval/bm25.py](retrieval/bm25.py)

bm25s's default token pattern `(?u)\b\w\w+\b` keeps only words of 2+ characters and splits on anything that isn't a letter or digit. On scientific text that loses signal: "COVID-19" → `covid` + `19`, "T cell" loses `t`, "vitamin D" loses `d`, "IL-6" → `il` only (since `6` is one character). Kept because it's the library default and Lucene-style analysers also split on hyphens, so it stays close to standard BM25 baselines. Changing the pattern is its own experiment: it changes the index fingerprint and needs before/after metrics on SciFact and FiQA.

## BM25 results are not padded with zero-score docs

**Decision** · **Resolved** · logged 2026-10-20 · [retrieval/ranking.py](retrieval/ranking.py) `keep_above`

When a query matched fewer than `retrieve_k` docs, `top_k` filled the remaining slots with score-0 docs (no shared terms) in doc ID order. If one of those happened to be relevant, recall@100, nDCG@10 and MRR got credit **by ID luck**. Now each retriever declares a `no_match_score` (BM25: `0.0`, dense: `None`) and `top_k` only returns docs scoring strictly above it, so BM25 may return fewer than k docs.

- Metrics only change where a padded doc was relevant. An empty slot counts the same as an irrelevant doc, and recall's denominator comes from the qrels.
- Matches Lucene, Elasticsearch and Pyserini, which only return docs that match at least one term, so it is closer to published BEIR BM25 numbers.
- ranx scores a query with an empty result as 0 and keeps it in the average, but **raises an error if a query is missing from the run**. The harness always includes every query, even with an empty result.

## Retrievers: one shared, deterministic ranking function

**Design** · **Resolved** · logged 2026-10-20 · [retrieval/ranking.py](retrieval/ranking.py)

Retrievers (BM25, dense) only return one score per doc. Ranking is done by a single function, `top_k`, used by both retrievers and by `ndcg_at_k`. Order is score descending, then doc ID ascending. Doc IDs compare as strings (`"10"` < `"9"`): deterministic, though not numeric.

- `top_k` shortlists every doc scoring at least the k-th best score (`>=`) and then sorts the shortlist, so ties at the cut-off are decided by the doc ID rule, not by `argpartition` or a library's internal top-k.
- **Gotcha:** ranx re-sorts a run by score and breaks ties its own way, which would undo our tie-break. The harness passes ranx rank-based scores (`k, k-1, …, 1`) and stores the real scores separately in `results.jsonl`.

## Persistent, fingerprinted indexes; rebuilding is intentional

**Design** · **Resolved** · logged 2026-10-20 · [retrieval/index_store.py](retrieval/index_store.py)

Indexes are stored on the `ir-data` volume at `/data/indexes/<dataset>/<retriever>/<index_id>/` and loaded if present. `--rebuild` is the only way to overwrite one. `index_id` is a hash of the corpus content, every index-affecting setting, and the relevant library versions, so any of those changing builds a new index instead of silently reusing a stale one. Query-only settings (BGE query prefix) and speed-only settings (batch size) are excluded, so changing them reuses the index. Saves go to a `.tmp` folder that is renamed into place, so a crash can't leave a half-written index that looks complete.

## Retriever choices

**Decision** · logged 2026-10-20

- **BM25 library: `bm25s`** (with PyStemmer, English stopwords, Lucene variant, k1=1.5, b=0.75). Fast and has save/load built in. Rejected Pyserini: closer to published BEIR BM25 numbers but needs a Java JDK in the image. Expect small differences from BEIR's Elasticsearch BM25.
- **Dense: `BAAI/bge-small-en-v1.5` pinned to commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`** (2024-02-22), so a model update on Hugging Face can't change results.
- **BGE query prefix on queries only** (`"Represent this sentence for searching relevant passages: "`), matching how BGE was evaluated on BEIR. Documents get no prefix.
- **Exact search, no FAISS.** Normalised embeddings, dot product over the whole corpus. FiQA (57k × 384 float32) is about 88 MB, small enough, and exact search adds no approximation error.
- **`bm25s.get_scores` crashes on an empty token list** (a query that is all stopwords). The BM25 retriever returns all-zero scores in that case.

## nDCG vs ranx test can't catch the bug it was written for on SciFact

**Bug** · **Resolved** · logged 2026-10-20 · [tests/test_metrics.py](tests/test_metrics.py) `test_matches_ranx`

**Context.** `test_matches_ranx` compares the hand-written `ndcg_at_k` with ranx on the first 5 SciFact test queries. For the first query it removes one relevant doc from the ranking. That is meant to catch a common nDCG bug: building the ideal DCG from the relevant docs that appear in the ranking instead of from all of the qrels. With that bug, a ranking that misses relevant docs still looks perfect (nDCG = 1.0 when the docs that are present rank first). With the correct version, the missing doc counts as a miss.

**Problem.** If the query has only one relevant doc, removing it leaves none in the ranking. The correct version and the buggy version then both return 0, so the test passes either way and proves nothing. A function that always returned 0 would also pass this case.

**Why this is worse on SciFact.** Most SciFact test queries have exactly one relevant doc:

| Relevant docs per query | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Queries | 277 | 14 | 4 | 3 | 2 |

The first query (`1`) has a single relevant doc, so the drop as written never tests the missing-doc case.

**Takeaway.** A test that removes a relevant doc needs at least two relevant docs on that query, so one stays and the two versions give different nonzero scores. Check how the data is actually distributed before relying on "the first N items" in a test.

**Solution.** Drop a relevant doc only from queries with **at least 2** relevant docs, so one stays in the ranking. (A threshold of "more than 2" would have matched none of the first 5 queries, so the drop would never happen.) The test also:

- asserts that at least one drop happened, so it fails loudly instead of silently testing nothing if the data changes;
- samples filler docs only from non-relevant docs, so the dropped doc can't be re-added by chance and filler is never accidentally relevant.

Verified by running the test against a deliberately buggy nDCG that builds the ideal DCG from the ranked docs. The test now fails on query `36`: buggy 0.431 vs ranx 0.264.
