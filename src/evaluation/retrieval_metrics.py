"""
Retrieval metric functions.
"""

from __future__ import annotations

import json

from src.evaluation.utils import dcg, safe_div


def retrieval_relevance(doc: dict, q_data: dict) -> float:
    if q_data.get("category") == "out_of_scope":
        return 0.0

    text = str(doc.get("text", ""))
    meta = doc.get("metadata", {})
    haystack = (text + "\n" + json.dumps(meta, ensure_ascii=False)).lower()

    score = 0.0
    for org in q_data.get("expected_orgs", []):
        if org.lower() in haystack:
            score += 2.0

    kw_hits = sum(1 for kw in q_data.get("expected_keywords", []) if kw.lower() in haystack)
    if kw_hits:
        score += min(2.0, kw_hits * 0.5)

    return score


def compute_retrieval_metrics(retrieved_docs: list[dict], q_data: dict) -> dict:
    gains = [retrieval_relevance(d, q_data) for d in retrieved_docs]
    relevant_flags = [1 if g > 0 else 0 for g in gains]

    hit_at_1 = 1.0 if relevant_flags[:1] and relevant_flags[0] else 0.0
    hit_at_3 = 1.0 if any(relevant_flags[:3]) else 0.0
    hit_at_5 = 1.0 if any(relevant_flags[:5]) else 0.0

    mrr = 0.0
    for i, rel in enumerate(relevant_flags, start=1):
        if rel:
            mrr = 1.0 / i
            break

    k = min(5, len(gains)) if gains else 0
    dcg_val = dcg(gains, k) if k else 0.0
    idcg_val = dcg(sorted(gains, reverse=True), k) if k else 0.0
    ndcg_at_5 = safe_div(dcg_val, idcg_val)

    precision_at_5 = safe_div(sum(relevant_flags[:5]), min(5, len(relevant_flags)))
    expected_relevant_docs = max(1, int(q_data.get("expected_relevant_docs", 1)))
    recall_proxy = min(1.0, safe_div(sum(relevant_flags), expected_relevant_docs))

    return {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_5": hit_at_5,
        "mrr": mrr,
        "ndcg_at_5": ndcg_at_5,
        "precision_at_5": precision_at_5,
        "recall_proxy": recall_proxy,
    }

