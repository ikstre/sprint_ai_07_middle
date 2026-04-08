"""
Main evaluator orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from configs.config import Config
from src.evaluation.dataset import EVALUATION_QUESTIONS, LLM_JUDGE_PROMPT
from src.evaluation.generation_metrics import GenerationMetricSuite
from src.evaluation.grounding_metrics import compute_decline_accuracy, compute_grounding_metrics
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.runtime_metrics import usage_to_tokens


class RAGEvaluator:
    """RAG system evaluator with retrieval/generation/runtime metrics."""

    def __init__(self, config: Config, generator=None):
        self.config = config
        self.generator = generator
        self.results: list[dict] = []
        self.generation_metrics = GenerationMetricSuite()

    def evaluate_with_llm_judge(self, question: str, answer: str, context: str) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)
        prompt = LLM_JUDGE_PROMPT.format(
            question=question,
            context=context[:3000],
            answer=answer,
        )

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=500,
        )

        raw = response.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "relevance": 3,
                "accuracy": 3,
                "faithfulness": 3,
                "completeness": 3,
                "conciseness": 3,
                "reasoning": str(raw)[:200],
            }

    def _evaluate_one(self, q_data: dict, use_llm_judge: bool, use_bertscore: bool, keep_memory: bool = False) -> dict:
        if self.generator is None:
            raise ValueError("generator is not set")

        if not keep_memory:
            self.generator.reset_memory()

        result = self.generator.generate(q_data["question"])

        answer = result["answer"]
        retrieved_docs = result["retrieved_docs"]
        reference = q_data.get("reference_answer", "")

        entry = {
            "id": q_data["id"],
            "question": q_data["question"],
            "category": q_data.get("category", "single_doc"),
            "answer": answer,
            "elapsed_time": float(result["elapsed_time"]),
            "num_retrieved": len(retrieved_docs),
        }

        entry.update(compute_retrieval_metrics(retrieved_docs, q_data))
        entry["keyword_recall"] = self.generation_metrics.keyword_recall(answer, q_data.get("expected_keywords", []))
        entry["field_coverage"] = self.generation_metrics.field_coverage(answer, q_data.get("expected_fields"))

        entry["rougeL_f1"] = self.generation_metrics.rouge_l(answer, reference)
        entry["meteor"] = self.generation_metrics.meteor(answer, reference)
        entry["bertscore_f1"] = self.generation_metrics.bertscore(answer, reference, enabled=use_bertscore)

        entry.update(compute_grounding_metrics(answer, retrieved_docs))

        decline_ok = compute_decline_accuracy(q_data, answer)
        if decline_ok is not None:
            entry["correctly_declined"] = bool(decline_ok)

        entry.update(usage_to_tokens(result))

        if use_llm_judge:
            context = "\n".join(str(d.get("text", ""))[:500] for d in retrieved_docs)
            judge_scores = self.evaluate_with_llm_judge(q_data["question"], answer, context)
            entry.update({f"llm_{k}": v for k, v in judge_scores.items()})

        return entry

    def run_evaluation_suite(
        self,
        questions: Optional[list[dict]] = None,
        use_llm_judge: bool = True,
        use_bertscore: bool = False,
    ) -> pd.DataFrame:
        if questions is None:
            questions = EVALUATION_QUESTIONS

        all_results = []

        for q_data in questions:
            print("\n" + "=" * 72)
            print(f"[{q_data['id']}] {q_data['question']}")
            print(f"category: {q_data.get('category', 'single_doc')}")

            entry = self._evaluate_one(
                q_data=q_data,
                use_llm_judge=use_llm_judge,
                use_bertscore=use_bertscore,
                keep_memory=False,
            )
            print(f"  elapsed: {entry['elapsed_time']:.2f}s | hit@5: {entry['hit_at_5']:.2f} | ndcg@5: {entry['ndcg_at_5']:.2f}")
            all_results.append(entry)

            if "follow_up" in q_data:
                fu = dict(q_data["follow_up"])
                fu["id"] = f"{q_data['id']}_followup"
                fu.setdefault("category", "follow_up")

                print(f"  [follow-up] {fu['question']}")
                fu_entry = self._evaluate_one(
                    q_data=fu,
                    use_llm_judge=use_llm_judge,
                    use_bertscore=use_bertscore,
                    keep_memory=True,
                )
                print(f"    elapsed: {fu_entry['elapsed_time']:.2f}s | hit@5: {fu_entry['hit_at_5']:.2f}")
                all_results.append(fu_entry)

        self.results = all_results
        return pd.DataFrame(all_results)

    def summary_report(self, df: Optional[pd.DataFrame] = None) -> dict:
        if df is None:
            df = pd.DataFrame(self.results)

        if df.empty:
            return {"total_questions": 0}

        report: dict = {
            "total_questions": int(len(df)),
            "avg_elapsed_time": float(df["elapsed_time"].mean()),
            "p50_elapsed_time": float(df["elapsed_time"].quantile(0.50)),
            "p95_elapsed_time": float(df["elapsed_time"].quantile(0.95)),
            "max_elapsed_time": float(df["elapsed_time"].max()),
            "avg_num_retrieved": float(df["num_retrieved"].mean()),
        }

        metric_cols = [
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
            "ndcg_at_5",
            "precision_at_5",
            "recall_proxy",
            "keyword_recall",
            "field_coverage",
            "rougeL_f1",
            "meteor",
            "bertscore_f1",
            "grounded_token_ratio",
            "hallucination_risk_proxy",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ]

        for col in metric_cols:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().any():
                    report[f"avg_{col}"] = float(numeric.mean())

        llm_cols = [c for c in df.columns if c.startswith("llm_") and c != "llm_reasoning"]
        for col in llm_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                report[f"avg_{col}"] = float(numeric.mean())

        if "correctly_declined" in df.columns:
            mask = df["correctly_declined"].notna()
            if mask.any():
                report["decline_accuracy"] = float(df.loc[mask, "correctly_declined"].astype(float).mean())

        report["by_category"] = {}
        for cat in df["category"].dropna().unique():
            cat_df = df[df["category"] == cat]
            cat_report = {
                "count": int(len(cat_df)),
                "avg_elapsed_time": float(cat_df["elapsed_time"].mean()),
            }
            for col in ["hit_at_5", "ndcg_at_5", "keyword_recall", "field_coverage", "grounded_token_ratio"]:
                if col in cat_df.columns:
                    n = pd.to_numeric(cat_df[col], errors="coerce")
                    if n.notna().any():
                        cat_report[f"avg_{col}"] = float(n.mean())
            report["by_category"][str(cat)] = cat_report

        return report

    def save_results(self, output_path: str = "evaluation/eval_results.json") -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"saved: {path}")

