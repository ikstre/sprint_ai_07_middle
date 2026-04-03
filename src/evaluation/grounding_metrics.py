"""
Grounding and out-of-scope metrics.
"""

from __future__ import annotations

from src.evaluation.utils import safe_div, tokenize


def compute_grounding_metrics(answer: str, retrieved_docs: list[dict]) -> dict:
    answer_tokens = set(tokenize(answer))
    context = "\n".join(str(d.get("text", ""))[:2000] for d in retrieved_docs)
    context_tokens = set(tokenize(context))

    if not answer_tokens:
        return {"grounded_token_ratio": 0.0, "hallucination_risk_proxy": 1.0}

    overlap = len(answer_tokens.intersection(context_tokens))
    grounded_ratio = safe_div(overlap, len(answer_tokens))
    return {
        "grounded_token_ratio": grounded_ratio,
        "hallucination_risk_proxy": 1.0 - grounded_ratio,
    }


def compute_decline_accuracy(q_data: dict, answer: str):
    if q_data.get("expected_behavior") != "should_decline":
        return None

    decline_keywords = [
        "찾을 수 없",
        "없습니다",
        "확인되지",
        "문서에서 확인",
        "제공된 문서",
    ]
    return any(k in answer for k in decline_keywords)

