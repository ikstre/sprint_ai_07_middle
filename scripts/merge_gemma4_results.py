"""
Gemma4 AutoRAG 결과를 메인 trial에 병합한다.

retrieval 노드는 모델 무관(동일 결과)이므로 건너뛰고,
post_retrieve_node_line(generator, prompt_maker)만 병합한다.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _merge_node(main_node_dir: Path, gemma_node_dir: Path) -> None:
    main_csv = main_node_dir / "summary.csv"
    gemma_csv = gemma_node_dir / "summary.csv"

    if not main_csv.exists() or not gemma_csv.exists():
        print(f"  [SKIP] summary.csv 없음: {main_node_dir.name}")
        return

    main_df = pd.read_csv(main_csv)
    gemma_df = pd.read_csv(gemma_csv)

    # 기존 parquet 파일 수 = 오프셋
    offset = len(main_df)

    # gemma parquet 파일을 main 디렉토리에 오프셋 적용해 복사
    copied = []
    for _, row in gemma_df.iterrows():
        src = gemma_node_dir / row["filename"]
        new_name = f"{int(row['filename'].replace('.parquet', '')) + offset}.parquet"
        dst = main_node_dir / new_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(new_name)

    # gemma summary에서 filename 오프셋 적용 후 concat
    gemma_df = gemma_df.copy()
    gemma_df["filename"] = gemma_df["filename"].apply(
        lambda f: f"{int(f.replace('.parquet', '')) + offset}.parquet"
    )

    merged = pd.concat([main_df, gemma_df], ignore_index=True)

    # is_best: 숫자 지표 컬럼 중 첫 번째를 기준으로 재계산
    metric_cols = [c for c in merged.columns if c not in ("filename", "module_name", "module_params", "execution_time", "is_best")]
    if metric_cols:
        primary = metric_cols[0]
        best_idx = merged[primary].idxmax()
        merged["is_best"] = False
        merged.loc[best_idx, "is_best"] = True

    merged.to_csv(main_csv, index=False)
    print(f"  ✓ {main_node_dir.name}: {len(main_df)}행 + {len(gemma_df)}행 → {len(merged)}행 (복사: {len(copied)}개)")

    # gemma best parquet도 복사
    for f in gemma_node_dir.glob("best_*.parquet"):
        dst = main_node_dir / f"best_gemma_{f.name}"
        if not dst.exists():
            shutil.copy2(f, dst)


def merge(main_dir: Path, gemma4_dir: Path) -> None:
    trial_main = main_dir / "0"
    trial_gemma = gemma4_dir / "0"

    if not trial_main.exists():
        raise SystemExit(f"[ERROR] 메인 trial 없음: {trial_main}")
    if not trial_gemma.exists():
        raise SystemExit(f"[ERROR] Gemma4 trial 없음: {trial_gemma}")

    post_nodes = ["generator", "prompt_maker"]
    for node in post_nodes:
        main_node = trial_main / "post_retrieve_node_line" / node
        gemma_node = trial_gemma / "post_retrieve_node_line" / node
        _merge_node(main_node, gemma_node)

    print("\n병합 완료.")
    print(f"  dashboard: autorag dashboard --trial_dir {trial_main}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma4 AutoRAG 결과를 메인 trial에 병합")
    parser.add_argument("--main-dir", required=True, type=Path)
    parser.add_argument("--gemma4-dir", required=True, type=Path)
    args = parser.parse_args()
    merge(args.main_dir, args.gemma4_dir)
