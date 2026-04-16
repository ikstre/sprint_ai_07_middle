"""
모델 다운로드 스크립트 (생성 모델 + 임베딩 모델)

README 기준 전체 모델 목록:

임베딩 모델 (5종):
  dragonkue/BGE-m3-ko                → embeddings/BGE-m3-ko              (2.2G)
  jhgan/ko-sroberta-multitask        → embeddings/ko-sroberta-multitask   (0.8G)
  intfloat/multilingual-e5-large     → embeddings/multilingual-e5-large   (2.2G)
  BM-K/KoSimCSE-roberta-multitask    → embeddings/KoSimCSE-roberta-multitask (0.4G)
  upskyy/kf-deberta-multitask        → embeddings/kf-deberta-multitask    (0.7G)

생성 모델 (7종):
  LGAI-EXAONE/EXAONE-4.0-1.2B       → exaone/EXAONE-4.0-1.2B            (2.4G)
  LGAI-EXAONE/EXAONE-Deep-2.4B      → exaone/EXAONE-Deep-2.4B           (4.5G)
  LGAI-EXAONE/EXAONE-Deep-7.8B      → exaone/EXAONE-Deep-7.8B           (15G)
  google/gemma-3-4b-it               → gemma/Gemma3-4B                   (8.1G)
  google/gemma-4-E4B-it              → gemma/Gemma4-E4B                  (15G)  ← transformers 5.x 필요
  kakaocorp/kanana-1.5-2.1b          → kanana/kanana-1.5-2.1b            (4.4G)
  skt/Midm-2.0-Mini-Instruct         → midm/Midm-2.0-Mini                (4.4G)

경로 기본값: MODEL_DIR 환경변수 → .env SRV_DATA_DIR/models → shared_data/models (프로젝트 상대경로)

사용법:
  # 전체 (임베딩 + 생성 모델)
  python scripts/download_models.py

  # 임베딩 모델만
  python scripts/download_models.py --embed-only

  # 생성 모델만
  python scripts/download_models.py --gen-only

  # 경량 생성 모델만 (7B 미만)
  python scripts/download_models.py --gen-only --small-only

  # 특정 모델만 (쉼표 구분)
  python scripts/download_models.py --models exaone,kanana,bge

  # 저장 경로 직접 지정
  python scripts/download_models.py --model-dir /path/to/models

  # HuggingFace 토큰 사용 (비공개 모델)
  HF_TOKEN=hf_xxx python scripts/download_models.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# .env 로드 후 configs/paths.py 임포트 (순서 중요)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # python-dotenv 미설치 환경에서는 환경변수 직접 설정 필요

try:
    from configs import paths as _paths
    _DEFAULT_MODEL_DIR = _paths.MODEL_DIR
except Exception:
    # configs 로드 실패 시 프로젝트 상대경로 사용
    _DEFAULT_MODEL_DIR = str(_PROJECT_ROOT / "shared_data" / "models")

from huggingface_hub import snapshot_download

# ── 임베딩 모델 정의 ────────────────────────────────────────────────────────
EMBED_MODELS: list[dict] = [
    {
        "key": "bge",
        "repo_id": "dragonkue/BGE-m3-ko",
        "subdir": "embeddings/BGE-m3-ko",
        "desc": "한국어 특화 BGE-m3 (~2.2GB) — 다국어, 1024-dim",
    },
    {
        "key": "sroberta",
        "repo_id": "jhgan/ko-sroberta-multitask",
        "subdir": "embeddings/ko-sroberta-multitask",
        "desc": "한국어 sRoBERTa (~0.8GB) — STS/NLI 멀티태스크, 768-dim",
    },
    {
        "key": "e5",
        "repo_id": "intfloat/multilingual-e5-large",
        "subdir": "embeddings/multilingual-e5-large",
        "desc": "다국어 E5-large (~2.2GB) — 한국어 retrieval 강함",
    },
    {
        "key": "kosimcse",
        "repo_id": "BM-K/KoSimCSE-roberta-multitask",
        "subdir": "embeddings/KoSimCSE-roberta-multitask",
        "desc": "한국어 SimCSE (~0.4GB) — 의미 유사도 특화",
    },
    {
        "key": "deberta",
        "repo_id": "upskyy/kf-deberta-multitask",
        "subdir": "embeddings/kf-deberta-multitask",
        "desc": "한국어 DeBERTa (~0.7GB) — 문맥 이해 우수",
    },
]

# ── 생성 모델 정의 ────────────────────────────────────────────────────────────
GEN_MODELS: list[dict] = [
    {
        "key": "exaone",
        "repo_id": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "subdir": "exaone/EXAONE-4.0-1.2B",
        "desc": "EXAONE-4.0-1.2B (~2.4GB) — 가장 빠름, trust_remote_code 필요",
        "size_gb": 2.4,
    },
    {
        "key": "exaone-deep-2.4b",
        "repo_id": "LGAI-EXAONE/EXAONE-Deep-2.4B",
        "subdir": "exaone/EXAONE-Deep-2.4B",
        "desc": "EXAONE-Deep-2.4B (~4.5GB) — 속도/성능 균형, trust_remote_code 필요",
        "size_gb": 4.5,
    },
    {
        "key": "exaone-deep-7.8b",
        "repo_id": "LGAI-EXAONE/EXAONE-Deep-7.8B",
        "subdir": "exaone/EXAONE-Deep-7.8B",
        "desc": "EXAONE-Deep-7.8B (~15GB) — 한국어 추론 특화, QLoRA 권장",
        "size_gb": 15.0,
    },
    {
        "key": "gemma3",
        "repo_id": "google/gemma-3-4b-it",
        "subdir": "gemma/Gemma3-4B",
        "desc": "Gemma3-4B (~8.1GB) — 다국어, max_model_len: 16384 권장",
        "size_gb": 8.1,
    },
    {
        "key": "gemma4",
        "repo_id": "google/gemma-4-E4B-it",
        "subdir": "gemma/Gemma4-E4B",
        "desc": "Gemma4-E4B (~15GB) — 멀티모달, transformers 5.x + vLLM 필요",
        "size_gb": 15.0,
    },
    {
        "key": "kanana",
        "repo_id": "kakaocorp/kanana-1.5-2.1b-instruct-2505",
        "subdir": "kanana/kanana-1.5-2.1b",
        "desc": "kanana-1.5-2.1b-instruct (~4.4GB) — 한국어 특화 경량, llama 계열",
        "size_gb": 4.4,
    },
    {
        "key": "midm",
        "repo_id": "K-intelligence/Midm-2.0-Mini-Instruct",
        "subdir": "midm/Midm-2.0-Mini",
        "desc": "Midm-2.0-Mini (~4.4GB) — 한국어 특화 경량, llama 계열",
        "size_gb": 4.4,
    },
]

_DEFAULT_IGNORE = [
    "*.msgpack", "flax_model*", "tf_model*", "rust_model*",
    "onnx/*", "openvino/*",
    # pytorch_model*.bin 은 제외하지 않음 — safetensors 없이 bin만 있는 모델 존재 (예: ko-sroberta)
]


def download_model(repo_id: str, local_dir: Path, desc: str) -> bool:
    """단일 모델 다운로드 (누락 파일만 재개 가능). 성공 여부 반환."""
    token = os.getenv("HF_TOKEN") or None
    if local_dir.exists():
        existing_gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e9
        print(f"[RESUME] {local_dir.name}  ({existing_gb:.2f} GB 존재, 누락 파일 확인 중...)")
    else:
        print(f"[DOWNLOAD] {repo_id}")
        print(f"  {desc}")
        print(f"  → {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            ignore_patterns=_DEFAULT_IGNORE,
            token=token,
        )
    except Exception as e:
        print(f"  [SKIP] {repo_id} 다운로드 실패: {e}\n")
        return False

    size_gb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e9
    print(f"  완료: {size_gb:.2f} GB\n")
    return True


def print_summary(model_dir: Path) -> None:
    print("=" * 60)
    print(f"모델 저장 위치: {model_dir}")
    for subdir in ["embeddings", "exaone", "gemma", "kanana", "midm"]:
        d = model_dir / subdir
        if d.exists():
            print(f"\n{d}:")
            for p in sorted(d.iterdir()):
                if p.is_dir():
                    size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
                    print(f"  {p.name:50s} {size_gb:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HuggingFace 모델 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=_DEFAULT_MODEL_DIR,
        help=f"모델 저장 루트 디렉토리 (기본: {_DEFAULT_MODEL_DIR})",
    )
    parser.add_argument("--embed-only", action="store_true", help="임베딩 모델만 다운로드")
    parser.add_argument("--gen-only", action="store_true", help="생성 모델만 다운로드")
    parser.add_argument(
        "--small-only",
        action="store_true",
        help="생성 모델 중 7GB 미만만 다운로드 (--gen-only와 함께 사용)",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "다운로드할 모델 key 목록 (쉼표 구분). "
            f"임베딩: {', '.join(m['key'] for m in EMBED_MODELS)}  "
            f"생성: {', '.join(m['key'] for m in GEN_MODELS)}"
        ),
    )
    parser.add_argument("--list", action="store_true", help="다운로드 가능한 모델 목록만 출력")
    args = parser.parse_args()

    if args.list:
        print("── 임베딩 모델 (" + str(len(EMBED_MODELS)) + "종) ─────────────────────────────")
        for m in EMBED_MODELS:
            print(f"  {m['key']:20s}  {m['repo_id']:45s}  {m['desc']}")
        print()
        print("── 생성 모델 (" + str(len(GEN_MODELS)) + "종) ──────────────────────────────────")
        for m in GEN_MODELS:
            size_tag = f"({m['size_gb']:.1f}GB)"
            print(f"  {m['key']:20s}  {m['repo_id']:45s}  {size_tag}")
        return

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # 다운로드 대상 결정
    selected_keys = {k.strip() for k in args.models.split(",") if k.strip()}

    do_embed = not args.gen_only
    do_gen = not args.embed_only

    embed_targets = [
        m for m in EMBED_MODELS
        if not selected_keys or m["key"] in selected_keys
    ] if do_embed else []

    gen_targets = [
        m for m in GEN_MODELS
        if (not selected_keys or m["key"] in selected_keys)
        and (not args.small_only or m.get("size_gb", 0) < 7.0)
    ] if do_gen else []

    if not embed_targets and not gen_targets:
        print("다운로드할 모델이 없습니다. --list 로 사용 가능한 모델을 확인하세요.")
        return

    total_gb = sum(m.get("size_gb", 0) for m in gen_targets)
    print("=" * 60)
    print(f"모델 저장 경로: {model_dir}")
    print(f"임베딩 {len(embed_targets)}종 / 생성 {len(gen_targets)}종")
    if gen_targets:
        print(f"예상 생성 모델 용량: ~{total_gb:.1f} GB (기존 파일 제외)")
    print("=" * 60)
    print()

    failed: list[str] = []

    if embed_targets:
        print("── 임베딩 모델 ─────────────────────────────────────────")
        for m in embed_targets:
            ok = download_model(m["repo_id"], model_dir / m["subdir"], m["desc"])
            if not ok:
                failed.append(m["repo_id"])

    if gen_targets:
        print("── 생성 모델 ───────────────────────────────────────────")
        for m in gen_targets:
            ok = download_model(m["repo_id"], model_dir / m["subdir"], m["desc"])
            if not ok:
                failed.append(m["repo_id"])

    print_summary(model_dir)
    if failed:
        print()
        print(f"[WARNING] 다음 {len(failed)}개 모델 다운로드 실패:")
        for r in failed:
            print(f"  - {r}")
    print()
    print("다음 단계:")
    print("  # AutoRAG 최적화 실행 (Gemma4 제외)")
    print("  PYTHONNOUSERSITE=1 python scripts/run_autorag_optimization.py \\")
    print("    --qa-path data/autorag_csv/qa.parquet \\")
    print("    --corpus-path data/autorag_csv/corpus.parquet \\")
    print("    --config-path configs/autorag/local_csv.yaml \\")
    print("    --project-dir evaluation/autorag_benchmark_csv")
    print()
    print("  # Gemma4-E4B 별도 실행 (transformers 5.x 필요)")
    print("  bash scripts/run_gemma4_optimization.sh")


if __name__ == "__main__":
    main()
