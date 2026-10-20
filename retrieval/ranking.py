import numpy as np


def top_k(
    doc_ids: np.ndarray,
    scores: np.ndarray,
    k: int,
    keep_above: float | None = None,
) -> list[tuple[str, float]]:
    """Return the k best (doc_id, score) pairs, best first.

    Order is fully deterministic: score descending, then doc_id ascending to
    break ties. doc_ids are compared as strings, so "10" sorts before "9".
    Every retriever and metric uses this one function to rank documents.

    keep_above: if set, only docs scoring strictly above it can be returned,
    so the result may have fewer than k docs. BM25 uses 0 here: a score of 0
    means the doc shares no terms with the query, and padding the list with
    such docs (in doc_id order) would hand out credit by ID luck.
    """
    doc_ids = np.asarray(doc_ids)
    scores = np.asarray(scores)

    # 0. Drop docs that don't count as a match at all
    if keep_above is not None:
        is_match = scores > keep_above
        doc_ids = doc_ids[is_match]
        scores = scores[is_match]

    k = min(k, len(scores))
    if k <= 0:
        return []

    # 1. Shortlist: everything scoring at least the k-th best score.
    #    Using ">=" keeps every doc tied at the cut-off, so the tie-break
    #    below decides which of them make it in, not argpartition.
    kth_best_score = np.partition(scores, len(scores) - k)[len(scores) - k]
    candidates = np.flatnonzero(scores >= kth_best_score)

    # 2. Sort the shortlist. lexsort uses its LAST key as the primary key:
    #    primary = -score (descending), secondary = doc_id (ascending).
    order = np.lexsort((doc_ids[candidates], -scores[candidates]))
    best = candidates[order[:k]]

    return [(str(doc_ids[i]), float(scores[i])) for i in best]
