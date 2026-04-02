"""
RAG 시스템 성능 평가 스크립트
여러 검색 설정을 비교하여 최적 파라미터를 찾는다.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from configs.config import Config
from src.rag_pipeline import RAGPipeline
from src.evaluator import RAGEvaluator, EVALUATION_QUESTIONS


def run_single_config(config: Config, label: str, use_llm_judge: bool = True):
    """단일 설정으로 평가를 실행한다."""
    print(f"\n{'='*50}")
    print(f"설정: {label}")
    print(f"{'='*50}")

    pipeline = RAGPipeline(config)
    pipeline.initialize_vectorstore()

    evaluator = RAGEvaluator(config, generator=pipeline.generator)
    df = evaluator.run_evaluation_suite(
        questions=EVALUATION_QUESTIONS,
        use_llm_judge=use_llm_judge,
    )

    # 결과 저장
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)
    evaluator.save_results(str(output_dir / f"eval_{label}.json"))
    df.to_csv(output_dir / f"eval_{label}.csv", index=False, encoding="utf-8-sig")

    summary = evaluator.summary_report(df)
    print(f"\n=== {label} 요약 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    return summary


def main():
    use_llm_judge = "--no-judge" not in sys.argv

    # 비교할 설정들
    configs = [
        {
            "label": "similarity_k5",
            "kwargs": {"retrieval_method": "similarity", "retrieval_top_k": 5},
        },
        {
            "label": "mmr_k5",
            "kwargs": {"retrieval_method": "mmr", "retrieval_top_k": 5},
        },
        {
            "label": "hybrid_k5",
            "kwargs": {"retrieval_method": "hybrid", "retrieval_top_k": 5},
        },
        {
            "label": "similarity_k10",
            "kwargs": {"retrieval_method": "similarity", "retrieval_top_k": 10},
        },
    ]

    all_summaries = {}
    for cfg in configs:
        config = Config(
            scenario="B",
            metadata_csv="data/data_list.csv",
            vectordb_dir="data/vectordb",
            **cfg["kwargs"],
        )
        summary = run_single_config(config, cfg["label"], use_llm_judge)
        all_summaries[cfg["label"]] = summary

    # 비교 테이블 출력
    print("\n" + "=" * 65)
    print("설정 비교")
    print("=" * 65)
    print(f"{'설정':<22} {'응답시간(평균)':>12} {'키워드재현율':>12} {'LLM평균점수':>12}")
    print("-" * 65)
    for label, s in all_summaries.items():
        t = f"{s.get('avg_elapsed_time', 0):.2f}s"
        kr = f"{s.get('avg_keyword_recall', 0):.1%}" if s.get('avg_keyword_recall') else "-"
        llm_scores = [v for k, v in s.items() if k.startswith("avg_llm_") and k != "avg_llm_reasoning" and isinstance(v, float)]
        llm_avg = f"{sum(llm_scores)/len(llm_scores):.2f}" if llm_scores else "-"
        print(f"{label:<22} {t:>12} {kr:>12} {llm_avg:>12}")


if __name__ == "__main__":
    main()
