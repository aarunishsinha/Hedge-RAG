# Hedge-RAG
Calibrated RAG: comparing cross-encoder, LLM and decision models on quality, cost, latency and calibration

## Metrics

nDCG uses **linear gain**: a document's gain is its relevance label (`rel`), not the exponential `2^rel - 1`. The project currently focuses on BEIR datasets, and the published BEIR benchmarks use linear gain (via `pytrec_eval`), so this keeps our numbers directly comparable. ranx's `ndcg` metric, which the tests check against, is linear as well.

On binary labels, such as SciFact's, linear and exponential gain give identical results. They only diverge on graded labels (e.g. 0–3).

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for copyright information.
