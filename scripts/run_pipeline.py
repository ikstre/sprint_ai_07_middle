"""
통합 파이프라인 실행기.

단계:
  data      CSV → corpus.parquet + qa.parquet
  finetune  지정 모델 LoRA 파인튜닝 (모델별 순차 실행)
  autorag   AutoRAG 최적화 평가

파인튜닝이 포함되면 완료된 모델을 AutoRAG config에 자동으로 추가합니다.

사용 예:
  # 전체: 데이터 + 파인튜닝(2종) + AutoRAG
  python scripts/run_pipeline.py --steps all \\
    --finetune-models kanana-nano,exaone

  # 데이터 준비 + AutoRAG (파인튜닝 생략)
  python scripts/run_pipeline.py --steps data,autorag

  # 파인튜닝 + AutoRAG (데이터 이미 준비됨)
  python scripts/run_pipeline.py --steps finetune,autorag \\
    --finetune-models kanana-nano

  # AutoRAG만
  python scripts/run_pipeline.py --steps autorag

사용 가능한 --finetune-models 값 (finetune_capable=True):
  [QLoRA 불필요 — BF16 LoRA]
  kanana-nano, kanana-1.5              — 2.1B
  exaone                               — EXAONE-4.0-1.2B
  exaone-deep-2.4b                     — EXAONE-Deep-2.4B
  midm                                 — Midm-2.0-Mini

  [QLoRA 자동 적용 — 레지스트리 qlora=True]
  gemma3                               — Gemma3-4B (권장)
  gemma4                               — Gemma4-E4B (권장)
  exaone-3.5-7.8b, exaone-deep-7.8b   — 7.8B (필수)

평가 전용 모델 (AutoRAG config에만 추가, 파인튜닝 불가):
  gemma4-26b  — Gemma4-26B-NVFP4, 22GB에서 학습 불가 (추론 전용)
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ── 프로젝트 루트 ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

# ── 모델 레지스트리 ─────────────────────────────────────────────────
# short name → 학습/서빙 파라미터
MODEL_REGISTRY: dict[str, dict] = {
    # ── 1.2B ~ 2.1B: BF16 LoRA, QLoRA 불필요 ─────────────────────────
    "kanana-nano": {
        "model_path": "/srv/shared_data/models/kanana/kanana-nano-2.1b",
        "trust_remote_code": False,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95], "max_model_len": 8192},
    },
    "kanana-1.5": {
        "model_path": "/srv/shared_data/models/kanana/kanana-1.5-2.1b",
        "trust_remote_code": False,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95], "max_model_len": 8192},
    },
    "exaone": {
        "model_path": "/srv/shared_data/models/exaone/EXAONE-4.0-1.2B",
        "trust_remote_code": True,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95]},
    },
    "exaone-deep-2.4b": {
        "model_path": "/srv/shared_data/models/exaone/EXAONE-Deep-2.4B",
        "trust_remote_code": True,
        "qlora": False,
        "finetune": {"batch_size": 2, "grad_accum": 8, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95]},
    },
    "midm": {
        "model_path": "/srv/shared_data/models/midm/Midm-2.0-Mini",
        "trust_remote_code": False,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.7, 0.8], "top_k": 20, "top_p": [0.80, 0.90]},
    },
    # ── 4B: QLoRA 권장 (22GB GPU에서 BF16 LoRA 가능하나 여유 확보) ──────
    "gemma3": {
        "model_path": "/srv/shared_data/models/gemma/Gemma3-4B",
        "trust_remote_code": False,
        "qlora": True,
        "finetune_capable": True,
        "finetune": {"batch_size": 2, "grad_accum": 8, "lora_r": 16},
        "vllm": {
            "temperature": [0.9, 1.0],
            "top_k": 64,
            "top_p": [0.90, 0.95],
            "max_model_len": 16384,
        },
    },
    # Gemma4-E4B: 15GB BF16, QLoRA 권장
    # needs_user_site=True: transformers 5.x(user-local) 필요
    "gemma4": {
        "model_path": "/srv/shared_data/models/gemma/Gemma4-E4B",
        "trust_remote_code": False,
        "qlora": True,
        "finetune_capable": True,
        "needs_user_site": True,
        "finetune": {"batch_size": 1, "grad_accum": 16, "lora_r": 8},
        "vllm": {
            "temperature": [0.9, 1.0],
            "top_k": 64,
            "top_p": [0.90, 0.95],
            "max_model_len": 16384,
        },
    },
    # ── 7.8B: QLoRA 필수 (22GB GPU에서 BF16 LoRA 불가) ──────────────────
    "exaone-3.5-7.8b": {
        "model_path": "/srv/shared_data/models/exaone/EXAONE-3.5-7.8B",
        "trust_remote_code": True,
        "qlora": True,
        "finetune": {"batch_size": 1, "grad_accum": 16, "lora_r": 8},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95], "max_model_len": 8192},
    },
    "exaone-deep-7.8b": {
        "model_path": "/srv/shared_data/models/exaone/EXAONE-Deep-7.8B",
        "trust_remote_code": True,
        "qlora": True,
        "finetune": {"batch_size": 1, "grad_accum": 16, "lora_r": 8},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95], "max_model_len": 8192},
    },
    # ── 추론 전용: AutoRAG config에만 추가 ───────────────────────────────
    # Gemma4-26B-NVFP4: 22GB 초과, LoRA 학습 불가
    "gemma4-26b": {
        "model_path": "/srv/shared_data/models/gemma/Gemma4-26B-NVFP4",
        "trust_remote_code": False,
        "qlora": False,
        "finetune_capable": False,
        "needs_user_site": True,
        "finetune": {},
        "vllm": {
            "temperature": [0.9, 1.0],
            "top_k": 64,
            "top_p": [0.90, 0.95],
            "max_model_len": 8192,
        },
    },
}


# ── 유틸리티 ────────────────────────────────────────────────────────

def _run(cmd: list[str], env_extra: dict | None = None, use_user_site: bool = False) -> None:
    """subprocess 실행. 실패 시 즉시 종료.

    Args:
        use_user_site: True면 PYTHONNOUSERSITE를 설정하지 않음.
            Gemma4처럼 user-local transformers 5.x 가 필요한 모델에 사용.
            False(기본)면 PYTHONNOUSERSITE=1 로 user-local 패키지를 차단 —
            kanana/midm 등이 transformers 5.x strict 검증 오류를 피하기 위해 필요.
    """
    env = os.environ.copy()
    if not use_user_site:
        env["PYTHONNOUSERSITE"] = "1"
    elif "PYTHONNOUSERSITE" in env:
        del env["PYTHONNOUSERSITE"]   # 부모 환경에 설정돼 있어도 해제
    if env_extra:
        env.update(env_extra)
    print("\n$ " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] 명령 실패 (returncode={result.returncode}): {cmd[0]}")
        sys.exit(result.returncode)


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Step 1: 데이터 준비 ─────────────────────────────────────────────

def step_data(args: argparse.Namespace) -> None:
    _section("Step 1 / data — CSV → corpus + qa")

    corpus = ROOT / args.data_dir / "corpus.parquet"
    qa = ROOT / args.data_dir / "qa.parquet"

    if corpus.exists() and qa.exists() and not args.force_data:
        print(f"  이미 존재: {corpus}, {qa}")
        print("  (재생성하려면 --force-data 옵션 추가)")
        return

    _run([
        PYTHON, str(ROOT / "scripts/prepare_autorag_from_csv.py"),
        "--csv-path", args.csv_path,
        "--output-dir", str(ROOT / args.data_dir),
        "--chunk-size", str(args.chunk_size),
        "--chunk-overlap", str(args.chunk_overlap),
    ])


# ── Step 2: 파인튜닝 ────────────────────────────────────────────────

def step_finetune(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """
    지정한 모델들을 순차적으로 파인튜닝.
    Returns: [(model_name, output_path), ...]  완료된 모델 목록
    """
    _section("Step 2 / finetune — LoRA 파인튜닝")

    model_names = [m.strip() for m in args.finetune_models.split(",") if m.strip()]
    if not model_names:
        print("  --finetune-models 미지정. 파인튜닝 스킵.")
        return []

    unknown = [m for m in model_names if m not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERROR] 알 수 없는 모델: {unknown}")
        print(f"  사용 가능: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    completed: list[tuple[str, Path]] = []

    for name in model_names:
        reg = MODEL_REGISTRY[name]

        # finetune_capable=False 모델은 스킵 (AutoRAG config에는 별도 추가)
        if not reg.get("finetune_capable", True):
            print(f"\n[스킵] {name}: finetune_capable=False (추론 전용 모델)")
            print(f"  → AutoRAG config에는 추가됩니다.")
            continue

        ft_params = reg["finetune"]
        output_dir = ROOT / "models/finetuned" / name
        final_dir = output_dir / "final"

        print(f"\n[파인튜닝] {name} → {output_dir}")

        if final_dir.exists() and not args.force_finetune:
            print(f"  이미 학습 완료: {final_dir}")
            print("  (재학습하려면 --force-finetune 옵션 추가)")
            completed.append((name, final_dir))
            continue

        cmd = [
            PYTHON, str(ROOT / "scripts/finetune_local.py"),
            "--model-path", reg["model_path"],
            "--output-dir", str(output_dir),
            "--qa-path", str(ROOT / args.data_dir / "qa.parquet"),
            "--corpus-path", str(ROOT / args.data_dir / "corpus.parquet"),
            "--epochs", str(args.finetune_epochs),
            "--early-stop-patience", str(args.early_stop_patience),
            "--batch-size", str(ft_params["batch_size"]),
            "--grad-accum", str(ft_params["grad_accum"]),
            "--lora-r", str(ft_params["lora_r"]),
            "--lr", str(args.finetune_lr),
            "--max-seq-length", str(args.max_seq_length),
        ]
        if reg["trust_remote_code"]:
            cmd.append("--trust-remote-code")
        # 레지스트리 기준 QLoRA 적용 (전역 --qlora로 강제 override 가능)
        if reg.get("qlora", False) or args.qlora:
            cmd.append("--qlora")
            print(f"  QLoRA: {'레지스트리 설정' if reg.get('qlora') else '--qlora 강제 적용'}")

        # 파인튜닝은 trl/bitsandbytes가 user site에 설치되어 있으므로
        # 모든 모델에서 user site 허용 (PYTHONNOUSERSITE 미설정)
        _run(cmd, use_user_site=True)
        completed.append((name, final_dir))

    return completed


# ── Step 3: 파인튜닝 모델 → config 주입 ────────────────────────────

def _build_pipeline_config(
    base_config_path: Path,
    finetuned: list[tuple[str, Path]],
    eval_only: list[str],
    output_config_path: Path,
) -> None:
    """
    base config를 로드하고 두 종류의 모델을 generator modules에 추가:
      - finetuned: 파인튜닝 완료 모델 (학습된 경로 사용)
      - eval_only: finetune_capable=False 모델 (원본 경로 그대로 추가)
    """
    with open(base_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # generator node 위치 찾기
    generator_node = None
    for node_line in config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("node_type") == "generator":
                generator_node = node
                break

    if generator_node is None:
        raise ValueError(f"generator node_type을 {base_config_path}에서 찾을 수 없습니다.")

    modules: list[dict] = generator_node.setdefault("modules", [])

    # 파인튜닝 완료 모델 추가 (학습된 경로)
    for name, final_dir in finetuned:
        reg = MODEL_REGISTRY[name]
        entry: dict = {
            "module_type": "vllm",
            "llm": str(final_dir),
            "max_tokens": [256, 512, 1024],
            "gpu_memory_utilization": 0.70,
            "kv_cache_dtype": "auto",
        }
        entry.update(copy.deepcopy(reg["vllm"]))
        if reg["trust_remote_code"]:
            entry["trust_remote_code"] = True
        modules.append(entry)
        print(f"  + 파인튜닝 모델 추가: {name} → {final_dir}")

    # 평가 전용 모델 추가 (원본 경로, 학습 없이 바로 추론)
    for name in eval_only:
        reg = MODEL_REGISTRY[name]
        entry = {
            "module_type": "vllm",
            "llm": reg["model_path"],
            "max_tokens": [256, 512, 1024],
            "gpu_memory_utilization": 0.70,
            "kv_cache_dtype": "auto",
        }
        entry.update(copy.deepcopy(reg["vllm"]))
        if reg["trust_remote_code"]:
            entry["trust_remote_code"] = True
        modules.append(entry)
        print(f"  + 평가 전용 모델 추가: {name} → {reg['model_path']}")

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  파이프라인 config 저장: {output_config_path}")


# ── Step 4: AutoRAG 최적화 ──────────────────────────────────────────

def _autorag_run(
    base_config: Path,
    finetuned: list[tuple[str, Path]],
    eval_only: list[str],
    config_out: Path,
    project_dir: Path,
    data_dir: Path,
    use_user_site: bool,
    label: str,
) -> None:
    """config 생성 후 AutoRAG 최적화 실행 (단일 그룹)."""
    if finetuned or eval_only:
        print(f"\n[{label}] 파이프라인 config 생성 중...")
        _build_pipeline_config(base_config, finetuned, eval_only, config_out)
        active_config = config_out
    else:
        active_config = base_config

    tag = "user-local transformers 5.x 사용" if use_user_site else "PYTHONNOUSERSITE=1"
    print(f"[{label}] AutoRAG 실행 ({tag})")
    _run(
        [
            PYTHON, str(ROOT / "scripts/run_autorag_optimization.py"),
            "--qa-path", str(data_dir / "qa.parquet"),
            "--corpus-path", str(data_dir / "corpus.parquet"),
            "--config-path", str(active_config),
            "--project-dir", str(project_dir),
        ],
        use_user_site=use_user_site,
    )

    trial_dir = project_dir / "0"
    print(f"\n[{label}] 결과:")
    print(f"  cat {trial_dir}/retrieve_node_line/*/summary.csv")
    print(f"  cat {trial_dir}/post_retrieve_node_line/*/summary.csv")
    print(f"  autorag dashboard --trial_dir {trial_dir}")


def step_autorag(
    args: argparse.Namespace,
    finetuned: list[tuple[str, Path]],
    eval_only: list[str],
) -> None:
    _section("Step 3 / autorag — AutoRAG 최적화 평가")

    base_config = ROOT / args.config_path
    if not base_config.exists():
        print(f"[ERROR] config 파일 없음: {base_config}")
        sys.exit(1)

    data_dir = ROOT / args.data_dir

    # ── 모델을 두 그룹으로 분리 ─────────────────────────────────────
    # group A: PYTHONNOUSERSITE=1  (kanana / midm / exaone / gemma3 등)
    # group B: user-local site 허용 (gemma4 / gemma4-26b — transformers 5.x 필요)

    ft_normal  = [(n, p) for n, p in finetuned  if not MODEL_REGISTRY[n].get("needs_user_site")]
    ft_gemma   = [(n, p) for n, p in finetuned  if     MODEL_REGISTRY[n].get("needs_user_site")]
    eo_normal  = [m      for m     in eval_only  if not MODEL_REGISTRY[m].get("needs_user_site")]
    eo_gemma   = [m      for m     in eval_only  if     MODEL_REGISTRY[m].get("needs_user_site")]

    # ── Group A 실행 (일반 모델) ────────────────────────────────────
    # vLLM은 LlamaConfig.validate_architecture 를 트리거하지 않으므로
    # user-local transformers 5.x 를 허용해도 문제 없음.
    # (PYTHONNOUSERSITE=1 을 설정하면 TokenizersBackend 등 5.x 전용
    #  tokenizer_class 를 찾지 못해 tokenizer 로드 실패)
    if ft_normal or eo_normal or (not finetuned and not eval_only):
        _autorag_run(
            base_config=base_config,
            finetuned=ft_normal,
            eval_only=eo_normal,
            config_out=ROOT / "configs/autorag/local_csv_pipeline.yaml",
            project_dir=ROOT / args.project_dir,
            data_dir=data_dir,
            use_user_site=True,
            label="일반 모델",
        )

    # ── Group B 실행 (Gemma4) — 별도 project dir 유지 ───────────────
    if ft_gemma or eo_gemma:
        gemma_project = ROOT / (args.project_dir + "_gemma")
        print(f"\n[Gemma4] 별도 프로젝트 디렉토리: {gemma_project}")
        _autorag_run(
            base_config=base_config,
            finetuned=ft_gemma,
            eval_only=eo_gemma,
            config_out=ROOT / "configs/autorag/local_csv_pipeline_gemma.yaml",
            project_dir=gemma_project,
            data_dir=data_dir,
            use_user_site=True,
            label="Gemma4",
        )

        # ── Group A + B 결과 자동 병합 ──────────────────────────────
        # Group A 실행 결과가 있을 때만 병합 (autorag 단계에서 Group A도 실행된 경우)
        main_project = ROOT / args.project_dir
        if main_project.exists() and (main_project / "0").exists():
            _section("Gemma4 결과 병합")
            _run(
                [
                    PYTHON, str(ROOT / "scripts/merge_gemma4_results.py"),
                    "--main-dir", str(main_project),
                    "--gemma4-dir", str(gemma_project),
                ],
                use_user_site=True,
            )
        else:
            print(f"\n[병합 스킵] 메인 trial 없음: {main_project / '0'}")
            print("  일반 모델 AutoRAG 실행 후 수동으로 병합하세요:")
            print(f"  python scripts/merge_gemma4_results.py \\")
            print(f"    --main-dir {main_project} \\")
            print(f"    --gemma4-dir {gemma_project}")


# ── 메인 ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="통합 파이프라인: 데이터 준비 → 파인튜닝 → AutoRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 단계 선택
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "실행할 단계 (쉼표 구분 또는 'all'). "
            "선택: data, finetune, autorag. 기본: all"
        ),
    )

    # 데이터 옵션
    parser.add_argument("--csv-path", default="/srv/shared_data/datasets/data_list_cleaned.csv")
    parser.add_argument("--data-dir", default="data/autorag_csv",
                        help="corpus/qa parquet 저장 위치 (기본: data/autorag_csv)")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--force-data", action="store_true",
                        help="corpus/qa가 존재해도 재생성")

    # 파인튜닝 옵션
    parser.add_argument(
        "--finetune-models",
        default="",
        help="파인튜닝할 모델 short name (쉼표 구분). "
             f"선택: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument("--finetune-epochs", type=int, default=5,
                        help="최대 학습 epoch 수 (early stop 시 조기 종료, 기본: 5)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="eval_loss 개선 없이 허용할 epoch 수 (0=비활성화, 기본: 3)")
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="학습 최대 시퀀스 길이 (기본: 1024)")
    parser.add_argument("--qlora", action="store_true",
                        help="모든 모델에 QLoRA 강제 적용 (기본: 레지스트리 qlora 필드 자동 적용)")
    parser.add_argument("--force-finetune", action="store_true",
                        help="모델이 이미 학습된 경우에도 재학습")

    # AutoRAG 옵션
    parser.add_argument("--config-path", default="configs/autorag/local_csv.yaml",
                        help="기본 AutoRAG config (파인튜닝 시 자동 확장)")
    parser.add_argument("--project-dir", default="evaluation/autorag_benchmark_csv",
                        help="AutoRAG 결과 저장 디렉토리")

    args = parser.parse_args()

    # 단계 파싱
    if args.steps.strip().lower() == "all":
        steps = {"data", "finetune", "autorag"}
    else:
        steps = {s.strip().lower() for s in args.steps.split(",")}

    invalid = steps - {"data", "finetune", "autorag"}
    if invalid:
        print(f"[ERROR] 알 수 없는 단계: {invalid}. 선택: data, finetune, autorag, all")
        sys.exit(1)

    # 지정 모델 분류: 학습 가능 / 평가 전용
    requested_models = [m.strip() for m in args.finetune_models.split(",") if m.strip()]
    unknown = [m for m in requested_models if m not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERROR] 알 수 없는 모델: {unknown}")
        print(f"  사용 가능: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    eval_only_models = [
        m for m in requested_models
        if not MODEL_REGISTRY[m].get("finetune_capable", True)
    ]
    finetune_models = [
        m for m in requested_models
        if MODEL_REGISTRY[m].get("finetune_capable", True)
    ]

    print(f"\n실행 단계: {sorted(steps)}")
    if finetune_models:
        print(f"파인튜닝 모델 ({len(finetune_models)}종): {', '.join(finetune_models)}")
    if eval_only_models:
        print(f"평가 전용 모델 ({len(eval_only_models)}종): {', '.join(eval_only_models)}")

    finetuned: list[tuple[str, Path]] = []

    if "data" in steps:
        step_data(args)

    if "finetune" in steps:
        finetuned = step_finetune(args)

    if "autorag" in steps:
        step_autorag(args, finetuned, eval_only_models)

    _section("파이프라인 완료")
    if finetuned:
        print(f"학습된 모델:")
        for name, path in finetuned:
            print(f"  {name}: {path}")
    if eval_only_models:
        print(f"평가 전용 모델 (원본 경로 사용): {', '.join(eval_only_models)}")
    if "autorag" in steps:
        trial_dir = ROOT / args.project_dir / "0"
        print(f"\nAutoRAG 결과 (일반 모델): {ROOT / args.project_dir}")
        print(f"  대시보드: autorag dashboard --trial_dir {trial_dir}")

        has_gemma = any(MODEL_REGISTRY[m].get("needs_user_site") for m in eval_only_models)
        has_gemma = has_gemma or any(
            MODEL_REGISTRY[n].get("needs_user_site") for n, _ in finetuned
        )
        if has_gemma:
            gemma_trial = ROOT / (args.project_dir + "_gemma") / "0"
            print(f"\nAutoRAG 결과 (Gemma4): {ROOT / (args.project_dir + '_gemma')}")
            print(f"  대시보드: autorag dashboard --trial_dir {gemma_trial}")


if __name__ == "__main__":
    main()
