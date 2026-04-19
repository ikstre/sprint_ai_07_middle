"""
Generation metric suite.
"""

from __future__ import annotations

from typing import Optional

from src.evaluation.utils import safe_div, tokenize


class GenerationMetricSuite:
    def __init__(self):
        self._rouge_scorer = None
        self._meteor_fn = None
        self._bertscore_fn = None
        self._init_optional_metrics()

    def _init_optional_metrics(self) -> None:
        try:
            from rouge_score import rouge_scorer

            self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        except Exception:
            self._rouge_scorer = None

        try:
            from nltk.translate.meteor_score import meteor_score

            self._meteor_fn = meteor_score
        except Exception:
            self._meteor_fn = None

        try:
            from bert_score import score as bert_score_fn

            self._bertscore_fn = bert_score_fn
        except Exception:
            self._bertscore_fn = None

    def keyword_recall(self, answer: str, expected_keywords: list[str]) -> Optional[float]:
        if not expected_keywords:
            return None
        answer_lower = answer.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return safe_div(hits, len(expected_keywords))

    def field_coverage(self, answer: str, expected_fields: dict | None) -> Optional[float]:
        if not expected_fields:
            return None

        values = []
        for arr in expected_fields.values():
            values.extend(arr)

        if not values:
            return None

        answer_lower = answer.lower()
        hits = sum(1 for v in values if str(v).lower() in answer_lower)
        return safe_div(hits, len(values))

    def rouge_l(self, answer: str, reference: str) -> Optional[float]:
        if not self._rouge_scorer or not reference:
            return None
        try:
            return float(self._rouge_scorer.score(reference, answer)["rougeL"].fmeasure)
        except Exception:
            return None

    def meteor(self, answer: str, reference: str) -> Optional[float]:
        if not self._meteor_fn or not reference:
            return None
        try:
            return float(self._meteor_fn([tokenize(reference)], tokenize(answer)))
        except Exception:
            return None

    def bertscore(self, answer: str, reference: str, enabled: bool) -> Optional[float]:
        if not enabled or not self._bertscore_fn or not reference:
            return None
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, f1 = self._bertscore_fn(
                    [answer],
                    [reference],
                    lang="ko",
                    verbose=False,
                    rescale_with_baseline=False,  # ko 기준선 파일 미존재 → 비활성화
                )
            return float(f1[0])
        except Exception:
            return None

