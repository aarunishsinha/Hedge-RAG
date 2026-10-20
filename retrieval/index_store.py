import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.environment import environment, package_versions
from retrieval.base import Retriever

log = logging.getLogger(__name__)

INDEX_ROOT = Path("/data/indexes")   # on the ir-data volume, so it survives container restarts

# Library versions that can change what an index contains
INDEX_PACKAGES = {
    "bm25": ["numpy", "bm25s", "PyStemmer"],
    "dense": ["numpy", "sentence-transformers", "transformers", "torch"],
}


def corpus_sha256(corpus: dict[str, dict]) -> str:
    """Hash of the corpus content, independent of file order."""
    digest = hashlib.sha256()
    for doc_id in sorted(corpus):
        doc = corpus[doc_id]
        digest.update(f"{doc_id}\0{doc.get('title', '')}\0{doc.get('text', '')}\n".encode())
    return digest.hexdigest()


def fingerprint_inputs(dataset: str, corpus: dict[str, dict], retriever_cls: type[Retriever], cfg) -> dict:
    """Everything that decides an index's content. Change any of it -> new index."""
    return {
        "dataset": dataset,
        "corpus_sha256": corpus_sha256(corpus),
        "retriever": retriever_cls.kind,
        "params": retriever_cls.index_params(cfg),
        "packages": package_versions(INDEX_PACKAGES[retriever_cls.kind]),
    }


def fingerprint(inputs: dict) -> str:
    canonical = json.dumps(inputs, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_or_build(
    retriever_cls: type[Retriever],
    dataset: str,
    corpus: dict[str, dict],
    cfg,
    rebuild: bool = False,
    root: Path = INDEX_ROOT,
) -> tuple[Retriever, dict]:
    """Load the stored index if there is one. Build only if missing or rebuild=True.

    Returns the retriever and an info dict (the manifest plus what happened).
    """
    inputs = fingerprint_inputs(dataset, corpus, retriever_cls, cfg)
    index_id = fingerprint(inputs)
    index_dir = root / dataset / retriever_cls.kind / index_id
    manifest_path = index_dir / "manifest.json"

    # Case 1: an index exists and no rebuild was asked for -> load it
    if manifest_path.exists() and not rebuild:
        start = time.perf_counter()
        retriever = retriever_cls.load(index_dir, cfg)
        load_seconds = time.perf_counter() - start

        with open(manifest_path) as f:
            manifest = json.load(f)
        log.info("Loaded %s index %s in %.2fs", retriever_cls.kind, index_dir, load_seconds)
        return retriever, {**manifest, "action": "loaded", "load_seconds": load_seconds}

    # Case 2: build it
    action = "rebuilt" if manifest_path.exists() else "built"
    log.info("Building %s index for %s (%d docs)", retriever_cls.kind, dataset, len(corpus))

    start = time.perf_counter()
    retriever = retriever_cls.build(corpus, cfg)
    build_seconds = time.perf_counter() - start

    # Save into a temp folder first, then swap it in, so a crash mid-save
    # never leaves a half-written index that looks complete.
    tmp_dir = index_dir.with_name(index_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    start = time.perf_counter()
    retriever.save(tmp_dir)
    save_seconds = time.perf_counter() - start

    manifest = {
        "index_id": index_id,
        "index_dir": str(index_dir),
        "fingerprint_inputs": inputs,
        "num_docs": len(retriever.doc_ids),
        "build_seconds": build_seconds,
        "save_seconds": save_seconds,
        "build_stats": retriever.build_stats(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
    }
    with open(tmp_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if index_dir.exists():
        shutil.rmtree(index_dir)
    tmp_dir.rename(index_dir)

    log.info("%s %s index in %.2fs (+%.2fs save) -> %s",
             action.capitalize(), retriever_cls.kind, build_seconds, save_seconds, index_dir)
    return retriever, {**manifest, "action": action}
