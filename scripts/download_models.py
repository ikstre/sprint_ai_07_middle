"""
모델 다운로드 스크립트 (생성 모델 + 임베딩 모델)

생성 모델:
  - google/gemma-4-E4B-it  (~15GB BF16)
    Gemma 4 4B dense 모델, multimodal (vision+audio+text)
    → /srv/shared_data/models/gemma/Gemma4-E4B

임베딩 모델 (신규 3종):
  - intfloat/multilingual-e5-large         (~2.2GB)
  - BM-K/KoSimCSE-roberta-multitask        (~500MB)
  - upskyy/kf-deberta-multitask            (~900MB)

사용법:
  # 전체 (생성 모델 + 임베딩 모델)
  python scripts/download_models.py

  # 임베딩 모델만
  python scripts/download_models.py --embed-only

  # 생성 모델만
  python scripts/download_models.py --gen-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

GEMMA_DIR = Path("/srv/shared_data/models/gemma")
EMBED_DIR = Path("/srv/shared_data/models/embeddings")

GEN_MODELS = [
    {
        "repo_id": "google/gemma-4-E4B-it",
        "local_dir": GEMMA_DIR / "Gemma4-E4B",
        "desc": "Gemma4-E4B (~15GB BF16) — dense, multimodal, transformers 5.x 필요",
    },
]

EMBED_MODELS = [
    {
        "repo_id": "intfloat/multilingual-e5-large",
        "local_dir": EMBED_DIR / "multilingual-e5-large",
        "desc": "다국어 E5-large (~2.2GB) — 한국어 retrieval 강함",
    },
    {
        "repo_id": "BM-K/KoSimCSE-roberta-multitask",
        "local_dir": EMBED_DIR / "KoSimCSE-roberta-multitask",
        "desc": "한국어 SimCSE (~500MB) — 의미 유사도 특화",
    },
    {
        "repo_id": "upskyy/kf-deberta-multitask",
        "local_dir": EMBED_DIR / "kf-deberta-multitask",
        "desc": "한국어 DeBERTa (~900MB) — 문맥 이해 우수",
    },
]


def download_model(repo_id: str, local_dir: Path, desc: str,
                   ignore_patterns: list[str] | None = None) -> None:
    # snapshot_download은 누락 파일만 받으므로 항상 호출 (재개 가능)
    default_ignore = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*",
                      "pytorch_model*.bin", "onnx/*", "openvino/*"]
    if local_dir.exists():
        existing_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
        print(f"[RESUME/CHECK] {local_dir.name} ({existing_size / 1e9:.2f} GB 존재, 누락 파일 확인 중...)")
    else:
        print(f"[DOWNLOAD] {repo_id}")
        print(f"  {desc}")
        print(f"  → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=ignore_patterns or default_ignore,
    )
    size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
    print(f"  완료: {size / 1e9:.2f} GB\n")


def print_summary() -> None:
    print("=" * 60)
    print("현재 모델 목록:")
    for base in [GEMMA_DIR, EMBED_DIR]:
        if base.exists():
            print(f"\n{base}:")
            for p in sorted(base.iterdir()):
                if p.is_dir():
                    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    print(f"  {p.name:45s} {size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 다운로드")
    parser.add_argument("--embed-only", action="store_true", help="임베딩 모델만 다운로드")
    parser.add_argument("--gen-only", action="store_true", help="생성 모델만 다운로드")
    args = parser.parse_args()

    do_gen = not args.embed_only
    do_embed = not args.gen_only

    print("=" * 60)
    if do_gen and do_embed:
        print("전체 모델 다운로드 (생성 + 임베딩)")
    elif do_gen:
        print("생성 모델 다운로드")
    else:
        print("임베딩 모델 다운로드")
    print("=" * 60)
    print()

    if do_gen:
        print("── 생성 모델 ──────────────────────────────────────────")
        for m in GEN_MODELS:
            download_model(m["repo_id"], m["local_dir"], m["desc"],
                           m.get("ignore_patterns"))

    if do_embed:
        print("── 임베딩 모델 ────────────────────────────────────────")
        for m in EMBED_MODELS:
            download_model(m["repo_id"], m["local_dir"], m["desc"])

    print_summary()
    print()
    print("다음 단계:")
    print("  # 메인 실행 (Gemma4-E4B 제외 모든 모델)")
    print("  PYTHONNOUSERSITE=1 python scripts/run_autorag_optimization.py \\")
    print("    --qa-path data/autorag/qa.parquet \\")
    print("    --corpus-path data/autorag/corpus.parquet \\")
    print("    --config-path configs/autorag/local.yaml \\")
    print("    --project-dir evaluation/autorag_benchmark_local")
    print()
    print("  # Gemma4-E4B 별도 실행")
    print("  bash scripts/run_gemma4_optimization.sh")


if __name__ == "__main__":
    main()
