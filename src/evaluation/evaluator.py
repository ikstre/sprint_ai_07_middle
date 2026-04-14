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


# judge 전용 단순 프롬프트 (gpt-5 계열 빈 응답 방지)
_JUDGE_PROMPT = """다음 질문과 답변을 평가하고 JSON만 반환하세요.

질문: {question}

컨텍스트: {context}

답변: {answer}

아래 JSON 형식으로만 답하세요. 다른 텍스트는 절대 포함하지 마세요.
{{"relevance": 점수, "accuracy": 점수, "faithfulness": 점수, "completeness": 점수, "conciseness": 점수}}

각 점수는 1~5 사이 정수입니다."""


class RAGEvaluator:
    """RAG system evaluator with retrieval/generation/runtime metrics."""

    def __init__(self, config: Config, generator=None):
        self.config = config
        self.generator = generator
        self.results: list[dict] = []
        self.generation_metrics = GenerationMetricSuite()

    # ── 단일 모델 judge (하위 호환용) ──────────────────────────────
    def evaluate_with_llm_judge(
        self, question: str, answer: str, context: str, model: str = "gpt-5-mini"
    ) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)
        prompt = _JUDGE_PROMPT.format(
            question=question,
            context=context[:3000],
            answer=answer,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=3000 if model == "gpt-5-nano" else 2000,  # ← 수정
            reasoning_effort="low",
        )

        raw = self._extract_content(response)
        try:
            return json.loads(self._clean_json(raw))
        except json.JSONDecodeError:
            return {
                "relevance": 3, "accuracy": 3, "faithfulness": 3,
                "completeness": 3, "conciseness": 3,
                "reasoning": str(raw)[:200],
            }

    # ── content 추출 (gpt-5 계열 대응) ────────────────────────────
    def _extract_content(self, response) -> str:
        message = response.choices[0].message
        raw = message.content

        if isinstance(raw, list):
            raw = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )

        if not raw:
            raw = (
                getattr(message, "output_text", None)
                or getattr(message, "refusal", None)
                or ""
            )

        return raw or ""

    # ── JSON 문자열 정리 ───────────────────────────────────────────
    @staticmethod
    def _clean_json(raw: str) -> str:
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    return part
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return raw[start:end]
        return raw

    # ── 멀티 모델 judge ────────────────────────────────────────────
    def evaluate_with_llm_judge_multi(
        self, question: str, answer: str, context: str
    ) -> list[dict]:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)
        results = []

        for model in self.config.eval_models:
            prompt = _JUDGE_PROMPT.format(
                question=question,
                context=context[:3000],
                answer=answer,
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=3000 if model == "gpt-5-nano" else 2000,  # ← 수정
                    reasoning_effort="low",
                )

                raw = self._extract_content(response)
                cleaned = self._clean_json(raw)

                print(f"  [judge] {model} 응답: {cleaned[:120]}")

                parsed = json.loads(cleaned)
                parsed["_model"] = model
                results.append(parsed)

            except Exception as e:
                print(f"  [judge] {model} 실패: {e}")
                continue

        return results

    def _evaluate_one(
        self,
        q_data: dict,
        use_llm_judge: bool,
        use_bertscore: bool,
        keep_memory: bool = False,
    ) -> dict:
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
        entry["keyword_recall"] = self.generation_metrics.keyword_recall(
            answer, q_data.get("expected_keywords", [])
        )
        entry["field_coverage"] = self.generation_metrics.field_coverage(
            answer, q_data.get("expected_fields")
        )
        entry["rougeL_f1"] = self.generation_metrics.rouge_l(answer, reference)
        entry["meteor"] = self.generation_metrics.meteor(answer, reference)
        entry["bertscore_f1"] = self.generation_metrics.bertscore(
            answer, reference, enabled=use_bertscore
        )
        entry.update(compute_grounding_metrics(answer, retrieved_docs))

        decline_ok = compute_decline_accuracy(q_data, answer)
        if decline_ok is not None:
            entry["correctly_declined"] = bool(decline_ok)

        entry.update(usage_to_tokens(result))

        if use_llm_judge:
            context = "\n".join(
                str(d.get("text", ""))[:500] for d in retrieved_docs
            )
            judge_results = self.evaluate_with_llm_judge_multi(
                q_data["question"], answer, context
            )

            if judge_results:
                score_keys = [
                    "relevance", "accuracy", "faithfulness",
                    "completeness", "conciseness",
                ]

                # 모델별 개별 점수 (예: llm_gpt_5_mini_relevance)
                for r in judge_results:
                    model_tag = r.get("_model", "unknown").replace("-", "_")
                    for k in score_keys:
                        if k in r:
                            entry[f"llm_{model_tag}_{k}"] = float(r[k])

                # 전체 평균 점수 (예: llm_avg_relevance)
                for k in score_keys:
                    vals = [float(r[k]) for r in judge_results if k in r]
                    if vals:
                        entry[f"llm_avg_{k}"] = sum(vals) / len(vals)

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
            print(
                f"  elapsed: {entry['elapsed_time']:.2f}s"
                f" | hit@5: {entry['hit_at_5']:.2f}"
                f" | ndcg@5: {entry['ndcg_at_5']:.2f}"
            )
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
                print(
                    f"    elapsed: {fu_entry['elapsed_time']:.2f}s"
                    f" | hit@5: {fu_entry['hit_at_5']:.2f}"
                )
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
            "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5",
            "precision_at_5", "recall_proxy", "keyword_recall", "field_coverage",
            "rougeL_f1", "meteor", "bertscore_f1",
            "grounded_token_ratio", "hallucination_risk_proxy",
            "prompt_tokens", "completion_tokens", "total_tokens",
        ]

        for col in metric_cols:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().any():
                    report[f"avg_{col}"] = float(numeric.mean())

        llm_cols = [c for c in df.columns if c.startswith("llm_") and "reasoning" not in c]
        for col in llm_cols:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                report[f"avg_{col}"] = float(numeric.mean())

        if "correctly_declined" in df.columns:
            mask = df["correctly_declined"].notna()
            if mask.any():
                report["decline_accuracy"] = float(
                    df.loc[mask, "correctly_declined"].astype(float).mean()
                )

        report["by_category"] = {}
        for cat in df["category"].dropna().unique():
            cat_df = df[df["category"] == cat]
            cat_report = {
                "count": int(len(cat_df)),
                "avg_elapsed_time": float(cat_df["elapsed_time"].mean()),
            }
            for col in [
                "hit_at_5", "ndcg_at_5", "keyword_recall",
                "field_coverage", "grounded_token_ratio",
            ]:
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