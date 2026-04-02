"""
Evaluation utility functions.
"""

from __future__ import annotations

import math
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣a-zA-Z0-9]+", text.lower())


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def dcg(gains: list[float], k: int) -> float:
    score = 0.0
    for i, g in enumerate(gains[:k]):
        score += (2**g - 1) / math.log2(i + 2)
    return score

