"""
Evaluation utility functions.
"""

from __future__ import annotations

import math
import re


_KO_STOPWORDS = {
    # 조사
    "이", "가", "을", "를", "은", "는", "의", "에", "에서", "으로", "로", "와", "과",
    "도", "만", "까지", "부터", "에게", "한테", "께", "이나", "나", "이며", "며",
    # 어미·접속
    "있다", "있습니다", "없다", "없습니다", "합니다", "됩니다", "입니다", "습니다",
    "했습니다", "됩니다", "하여", "하고", "하며", "하는", "된", "되는", "되어",
    # 메타 표현 (LLM 생성 상투어)
    "문서에서", "확인되지", "않습니다", "명시된", "명시되어", "기반으로", "바탕으로",
    "따르면", "제시된", "해당", "관련", "내용", "사항", "요구사항", "정보",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _KO_STOPWORDS and len(t) > 1]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def dcg(gains: list[float], k: int) -> float:
    score = 0.0
    for i, g in enumerate(gains[:k]):
        score += (2**g - 1) / math.log2(i + 2)
    return score

