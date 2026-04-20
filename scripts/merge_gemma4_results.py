"""
Gemma4 AutoRAG 결과를 메인 trial에 병합한다.

retrieval 노드는 모델 무관(동일 결과)이므로 건너뛰고,
post_retrieve_node_line(generator, prompt_maker)만 병합한다.
중복 실행 방지: Gemma4 파일이 이미 복사돼 있으면 스킵한다.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

# 노드별 best 기준 지표 (높을수록 좋음)
_BEST_METRIC = {
    "generator": "meteor",
    "prompt_maker": "prompt_maker_meteor",
}


def _merge_node(main_node_dir: Path, gemma_node_dir: Path, node: str) -> None:
    main_csv = main_node_dir / "summary.csv"
    gemma_csv = gemma_node_dir / "summary.csv"

    if not main_csv.exists() or not gemma_csv.exists():
        print(f"  [SKIP] summary.csv 없음: {main_node_dir.name}")
        return

    main_df = pd.read_csv(main_csv)
    gemma_df = pd.read_csv(gemma_csv)

    offset = len(main_df)

    # 이미 병합된 경우 스킵 (gemma 첫 번째 파일이 오프셋 위치에 존재하는지 확인)
    first_gemma_dst = main_node_dir / f"{offset}.parquet"
    if first_gemma_dst.exists():
        print(f"  [SKIP] 이미 병합됨: {main_node_dir.name} (offset={offset})")
        return

    # gemma parquet 파일을 main 디렉토리에 오프셋 적용해 복사
    copied = []
    for _, row in gemma_df.iterrows():
        src = gemma_node_dir / row["filename"]
        new_name = f"{int(row['filename'].replace('.parquet', '')) + offset}.parquet"
        dst = main_node_dir / new_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(new_name)

    # gemma summary filename 오프셋 적용 후 concat
    gemma_df = gemma_df.copy()
    gemma_df["filename"] = gemma_df["filename"].apply(
        lambda f: f"{int(f.replace('.parquet', '')) + offset}.parquet"
    )

    merged = pd.concat([main_df, gemma_df], ignore_index=True)

    # is_best: 노드별 기준 지표로 재계산
    primary = _BEST_METRIC.get(node)
    if primary and primary in merged.columns:
        best_idx = merged[primary].idxmax()
    else:
        # 폴백: 첫 번째 숫자 지표
        metric_cols = [c for c in merged.columns
                       if c not in ("filename", "module_name", "module_params", "execution_time", "is_best")]
        best_idx = merged[metric_cols[0]].idxmax() if metric_cols else 0

    merged["is_best"] = False
    merged.loc[best_idx, "is_best"] = True
    merged.to_csv(main_csv, index=False)

    # best parquet 갱신
    for old in main_node_dir.glob("best_*.parquet"):
        old.unlink()
    best_src = main_node_dir / merged.loc[best_idx, "filename"]
    best_dst = main_node_dir / f"best_{best_idx}.parquet"
    shutil.copy2(best_src, best_dst)

    print(f"  ✓ {main_node_dir.name}: {len(main_df)}행 + {len(gemma_df)}행 → {len(merged)}행"
          f" | best: {merged.loc[best_idx, 'filename']} ({primary}={merged.loc[best_idx, primary]:.4f})"
          f" | 복사: {len(copied)}개")


def merge(main_dir: Path, gemma4_dir: Path) -> None:
    trial_main = main_dir / "0"
    trial_gemma = gemma4_dir / "0"

    if not trial_main.exists():
        raise SystemExit(f"[ERROR] 메인 trial 없음: {trial_main}")
    if not trial_gemma.exists():
        raise SystemExit(f"[ERROR] Gemma4 trial 없음: {trial_gemma}")

    for node in ["generator", "prompt_maker"]:
        main_node = trial_main / "post_retrieve_node_line" / node
        gemma_node = trial_gemma / "post_retrieve_node_line" / node
        _merge_node(main_node, gemma_node, node)

    print("\n병합 완료.")
    print(f"  dashboard: autorag dashboard --trial_dir {trial_main}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma4 AutoRAG 결과를 메인 trial에 병합")
    parser.add_argument("--main-dir", required=True, type=Path)
    parser.add_argument("--gemma4-dir", required=True, type=Path)
    args = parser.parse_args()
    merge(args.main_dir, args.gemma4_dir)
