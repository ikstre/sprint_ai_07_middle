"""
Gemma4 별도 실행 결과를 메인 AutoRAG trial에 합치는 스크립트.

Gemma4는 transformers 5.x가 필요해 별도 project dir에서 실행되므로
generator 노드의 parquet + summary.csv를 메인 trial에 병합합니다.
prompt_maker 노드는 두 실행 모두 동일한 평가용 LLM(EXAONE-4.0-1.2B)을
사용하므로 병합 대상에서 제외됩니다.

사용법:
  # run_pipeline.py 워크플로 (기본)
    python scripts/merge_gemma4_results.py \
        --main-dir evaluation/autorag_benchmark_csv \
        --gemma4-dir evaluation/autorag_benchmark_csv_gemma

  # 수동 실행 워크플로 (bash scripts/run_gemma4_optimization.sh)
    python scripts/merge_gemma4_results.py \
        --main-dir evaluation/autorag_benchmark_csv \
        --gemma4-dir evaluation/autorag_benchmark_gemma4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def merge_generator_results(main_trial: Path, gemma4_trial: Path) -> None:
    main_gen = main_trial / "post_retrieve_node_line" / "generator"
    gemma4_gen = gemma4_trial / "post_retrieve_node_line" / "generator"

    if not main_gen.exists():
        raise FileNotFoundError(f"메인 trial generator 디렉토리 없음: {main_gen}")
    if not gemma4_gen.exists():
        raise FileNotFoundError(f"Gemma4 trial generator 디렉토리 없음: {gemma4_gen}")

    # 기존 모듈 parquet 파일 수 파악 (0.parquet, 1.parquet, ...)
    existing = sorted(
        [p for p in main_gen.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    next_idx = len(existing)

    # Gemma4 trial의 parquet 파일들
    gemma4_parquets = sorted(
        [p for p in gemma4_gen.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    if not gemma4_parquets:
        raise FileNotFoundError(f"Gemma4 generator parquet 파일 없음: {gemma4_gen}")

    print(f"메인 trial 기존 모듈 수: {next_idx}")
    print(f"Gemma4 trial 모듈 수: {len(gemma4_parquets)}")

    # Gemma4 parquet 파일 복사 (새 인덱스로)
    idx_map: dict[str, str] = {}
    for i, src in enumerate(gemma4_parquets):
        new_name = f"{next_idx + i}.parquet"
        dst = main_gen / new_name
        shutil.copy2(src, dst)
        idx_map[src.name] = new_name
        print(f"  복사: {src.name} → {new_name}")

    # summary.csv 병합
    main_summary_path = main_gen / "summary.csv"
    gemma4_summary_path = gemma4_gen / "summary.csv"

    main_df = pd.read_csv(main_summary_path)
    gemma4_df = pd.read_csv(gemma4_summary_path)

    # Gemma4 summary의 filename을 새 인덱스로 교체
    gemma4_df["filename"] = gemma4_df["filename"].map(
        lambda f: idx_map.get(f, f)
    )
    # 병합된 행은 is_best=False로 초기화 (아래에서 재계산)
    gemma4_df["is_best"] = False

    merged_df = pd.concat([main_df, gemma4_df], ignore_index=True)

    # metric 컬럼 기준 is_best 재계산 (mean strategy)
    metric_cols = [
        c for c in merged_df.columns
        if c not in ("filename", "module_name", "module_params",
                     "execution_time", "average_output_token", "is_best")
    ]
    if metric_cols:
        merged_df["is_best"] = False
        best_idx = merged_df[metric_cols].mean(axis=1).idxmax()
        merged_df.loc[best_idx, "is_best"] = True
        print(f"\nis_best 재계산 완료 → {merged_df.loc[best_idx, 'module_name']} "
              f"({merged_df.loc[best_idx, 'filename']})")

    merged_df.to_csv(main_summary_path, index=False)
    print(f"\nsummary.csv 업데이트 완료: {main_summary_path}")
    print(merged_df[["filename", "module_name", "is_best"] + metric_cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma4 결과를 메인 trial에 병합")
    parser.add_argument(
        "--main-dir",
        type=str,
        default="evaluation/autorag_benchmark_csv",
        help="메인 AutoRAG project 디렉토리 (기본: evaluation/autorag_benchmark_csv)",
    )
    parser.add_argument(
        "--gemma4-dir",
        type=str,
        default="evaluation/autorag_benchmark_csv_gemma",
        help="Gemma4 전용 AutoRAG project 디렉토리 (기본: evaluation/autorag_benchmark_csv_gemma)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="trial 번호 (기본값: 0)",
    )
    args = parser.parse_args()

    main_trial = Path(args.main_dir) / str(args.trial)
    gemma4_trial = Path(args.gemma4_dir) / str(args.trial)

    if not main_trial.exists():
        raise FileNotFoundError(f"메인 trial 디렉토리 없음: {main_trial}")
    if not gemma4_trial.exists():
        raise FileNotFoundError(f"Gemma4 trial 디렉토리 없음: {gemma4_trial}")

    print(f"메인 trial: {main_trial}")
    print(f"Gemma4 trial: {gemma4_trial}\n")
    merge_generator_results(main_trial, gemma4_trial)


if __name__ == "__main__":
    main()
