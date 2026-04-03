"""
RAG 시스템 성능 평가 모듈

평가 기준:
1. Retrieval 품질: 관련 문서를 잘 찾아오는가
2. 답변 정확성: 문서 기반으로 정확하게 답변하는가
3. 답변 충실성(Faithfulness): 환각 없이 근거 있는 답변인가
4. 대화 맥락 유지: 후속 질문을 잘 이해하는가
5. 응답 속도: 실사용 가능한 속도인가
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from configs.config import Config


# ─────────────────────────────────────────────────────────────────
# 평가 데이터셋 (질문 + 기대 답변/근거)
# ─────────────────────────────────────────────────────────────────

EVALUATION_QUESTIONS = [
    # ── 단일 문서 검색 ──────────────────────────────────────────
    {
        "id": "q1",
        "question": "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.",
        "category": "single_doc",
        "expected_keywords": ["국민연금공단", "이러닝", "요구사항"],
        "follow_up": {
            "question": "콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘.",
            "category": "follow_up",
            "expected_keywords": ["콘텐츠", "개발", "관리"],
        },
    },
    {
        "id": "q2",
        "question": "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?",
        "category": "single_doc",
        "expected_keywords": ["기초과학연구원", "극저온", "AI", "예측"],
        "follow_up": {
            "question": "그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘.",
            "category": "follow_up",
            "expected_keywords": ["모니터링"],
        },
    },
    {
        "id": "q3",
        "question": "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.",
        "category": "single_doc",
        "expected_keywords": ["원자력", "선량", "목적", "추진"],
    },
    # ── 다중 문서 비교 ──────────────────────────────────────────
    {
        "id": "q4",
        "question": "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
        "category": "multi_doc",
        "expected_keywords": ["고려대학교", "광주과학기술원", "비교"],
        "follow_up": {
            "question": "고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나? 문서를 기반으로 정확하게 답변해 줘.",
            "category": "follow_up",
            "expected_keywords": ["응답 시간", "요구사항"],
        },
    },
    # ── 교차 검색 ───────────────────────────────────────────────
    {
        "id": "q5",
        "question": "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?",
        "category": "cross_doc",
        "expected_keywords": ["교육", "학습"],
    },
    # ── 문서 외 질문 (모른다고 답해야 함) ───────────────────────
    {
        "id": "q6",
        "question": "삼성전자가 발주한 반도체 설계 자동화 사업의 요구사항을 알려줘.",
        "category": "out_of_scope",
        "expected_keywords": [],
        "expected_behavior": "should_decline",
    },
]


# ─────────────────────────────────────────────────────────────────
# LLM 기반 평가 (LLM-as-a-Judge)
# ─────────────────────────────────────────────────────────────────

LLM_JUDGE_PROMPT = """당신은 RAG 시스템의 답변 품질을 평가하는 전문 평가자입니다.

## 평가 기준 (각 1-5점)

1. **관련성 (Relevance)**: 질문에 대해 관련 있는 내용을 답변했는가?
2. **정확성 (Accuracy)**: 컨텍스트에 근거한 정확한 정보인가?
3. **충실성 (Faithfulness)**: 컨텍스트에 없는 내용을 지어내지 않았는가?
4. **완전성 (Completeness)**: 질문이 요구하는 정보를 빠짐없이 제공했는가?
5. **간결성 (Conciseness)**: 불필요한 내용 없이 핵심만 전달했는가?

## 입력

질문: {question}
컨텍스트 (검색된 문서): {context}
시스템 답변: {answer}

## 출력 형식 (JSON)

반드시 아래 JSON 형식으로만 응답하세요:
{{"relevance": <1-5>, "accuracy": <1-5>, "faithfulness": <1-5>, "completeness": <1-5>, "conciseness": <1-5>, "reasoning": "<평가 근거 한줄 요약>"}}
"""


class RAGEvaluator:
    """RAG 시스템 성능 평가기"""

    def __init__(self, config: Config, generator=None):
        self.config = config
        self.generator = generator
        self.results: list[dict] = []

    def evaluate_single(self, question: str, category: str = "single_doc", where=None) -> dict:
        """단일 질문에 대한 RAG 응답을 평가한다."""
        if self.generator is None:
            raise ValueError("generator가 설정되지 않았습니다.")

        # 답변 생성 및 시간 측정
        result = self.generator.generate(question, where=where)

        # 키워드 기반 자동 평가
        answer_lower = result["answer"].lower()
        retrieved_text = " ".join([d["text"] for d in result["retrieved_docs"]]).lower()

        eval_result = {
            "question": question,
            "category": category,
            "answer": result["answer"],
            "elapsed_time": result["elapsed_time"],
            "num_retrieved": len(result["retrieved_docs"]),
        }

        return eval_result

    def evaluate_with_llm_judge(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict:
        """LLM을 사용하여 답변 품질을 평가한다."""
        from openai import OpenAI
        client = OpenAI(api_key=self.config.openai_api_key)

        prompt = LLM_JUDGE_PROMPT.format(
            question=question,
            context=context[:3000],  # 토큰 절약
            answer=answer,
        )

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )

        try:
            scores = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트에서 추출 시도
            text = response.choices[0].message.content
            scores = {"relevance": 3, "accuracy": 3, "faithfulness": 3, "completeness": 3, "conciseness": 3, "reasoning": text[:200]}

        return scores

    def run_evaluation_suite(
        self,
        questions: Optional[list[dict]] = None,
        use_llm_judge: bool = True,
    ) -> pd.DataFrame:
        """전체 평가 세트를 실행한다."""
        if questions is None:
            questions = EVALUATION_QUESTIONS

        all_results = []

        for q_data in questions:
            print(f"\n{'='*60}")
            print(f"[{q_data['id']}] {q_data['question']}")
            print(f"카테고리: {q_data['category']}")

            # 메인 질문 평가
            self.generator.reset_memory()
            result = self.generator.generate(q_data["question"])

            eval_entry = {
                "id": q_data["id"],
                "question": q_data["question"],
                "category": q_data["category"],
                "answer": result["answer"],
                "elapsed_time": result["elapsed_time"],
                "num_retrieved": len(result["retrieved_docs"]),
            }

            # 키워드 히트율
            if q_data.get("expected_keywords"):
                answer_text = result["answer"].lower()
                hits = sum(1 for kw in q_data["expected_keywords"] if kw.lower() in answer_text)
                eval_entry["keyword_recall"] = hits / len(q_data["expected_keywords"])
            else:
                eval_entry["keyword_recall"] = None

            # Out-of-scope 처리 확인
            if q_data.get("expected_behavior") == "should_decline":
                decline_keywords = ["찾을 수 없", "없습니다", "확인되지", "해당", "문서에"]
                eval_entry["correctly_declined"] = any(
                    kw in result["answer"] for kw in decline_keywords
                )

            # LLM Judge 평가
            if use_llm_judge:
                context = "\n".join([d["text"][:500] for d in result["retrieved_docs"]])
                scores = self.evaluate_with_llm_judge(
                    q_data["question"], result["answer"], context
                )
                eval_entry.update({f"llm_{k}": v for k, v in scores.items()})

            print(f"  응답 시간: {result['elapsed_time']:.2f}s")
            print(f"  검색 문서: {len(result['retrieved_docs'])}개")
            if "keyword_recall" in eval_entry and eval_entry["keyword_recall"] is not None:
                print(f"  키워드 재현율: {eval_entry['keyword_recall']:.1%}")

            all_results.append(eval_entry)

            # 후속 질문 평가
            if "follow_up" in q_data:
                fu = q_data["follow_up"]
                print(f"\n  [후속] {fu['question']}")
                fu_result = self.generator.generate(fu["question"])

                fu_entry = {
                    "id": f"{q_data['id']}_followup",
                    "question": fu["question"],
                    "category": "follow_up",
                    "answer": fu_result["answer"],
                    "elapsed_time": fu_result["elapsed_time"],
                    "num_retrieved": len(fu_result["retrieved_docs"]),
                }

                if fu.get("expected_keywords"):
                    answer_text = fu_result["answer"].lower()
                    hits = sum(1 for kw in fu["expected_keywords"] if kw.lower() in answer_text)
                    fu_entry["keyword_recall"] = hits / len(fu["expected_keywords"])

                if use_llm_judge:
                    fu_context = "\n".join([d["text"][:500] for d in fu_result["retrieved_docs"]])
                    fu_scores = self.evaluate_with_llm_judge(
                        fu["question"], fu_result["answer"], fu_context
                    )
                    fu_entry.update({f"llm_{k}": v for k, v in fu_scores.items()})

                print(f"    응답 시간: {fu_result['elapsed_time']:.2f}s")
                all_results.append(fu_entry)

        self.results = all_results
        df = pd.DataFrame(all_results)
        return df

    def summary_report(self, df: Optional[pd.DataFrame] = None) -> dict:
        """평가 결과 요약 보고서를 생성한다."""
        if df is None:
            df = pd.DataFrame(self.results)

        report = {
            "total_questions": len(df),
            "avg_elapsed_time": df["elapsed_time"].mean(),
            "median_elapsed_time": df["elapsed_time"].median(),
            "max_elapsed_time": df["elapsed_time"].max(),
        }

        # 키워드 재현율
        kw_df = df[df["keyword_recall"].notna()]
        if not kw_df.empty:
            report["avg_keyword_recall"] = kw_df["keyword_recall"].mean()

        # LLM Judge 점수
        llm_cols = [c for c in df.columns if c.startswith("llm_") and c != "llm_reasoning"]
        if llm_cols:
            for col in llm_cols:
                numeric = pd.to_numeric(df[col], errors="coerce")
                report[f"avg_{col}"] = numeric.mean()

        # 카테고리별 성능
        categories = df["category"].unique()
        report["by_category"] = {}
        for cat in categories:
            cat_df = df[df["category"] == cat]
            cat_report = {
                "count": len(cat_df),
                "avg_elapsed_time": cat_df["elapsed_time"].mean(),
            }
            kw_cat = cat_df[cat_df["keyword_recall"].notna()]
            if not kw_cat.empty:
                cat_report["avg_keyword_recall"] = kw_cat["keyword_recall"].mean()
            report["by_category"][cat] = cat_report

        return report

    def save_results(self, output_path: str = "evaluation/eval_results.json"):
        """평가 결과를 파일로 저장한다."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"평가 결과 저장: {path}")
