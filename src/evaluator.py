"""
Backward-compatible exports for evaluator.

Prefer importing from `src.evaluation`.
"""

from src.evaluation import EVALUATION_QUESTIONS, RAGEvaluator

__all__ = ["RAGEvaluator", "EVALUATION_QUESTIONS"]

