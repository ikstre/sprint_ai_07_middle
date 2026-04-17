"""
통합 파이프라인 실행기.

단계:
  data      CSV → corpus.parquet + qa.parquet
  finetune  지정 모델 LoRA 파인튜닝 (모델별 순차 실행)
  autorag   AutoRAG 최적화 평가

파인튜닝이 포함되면 완료된 모델을 AutoRAG config에 자동으로 추가합니다.

단계:
  data       CSV → corpus.parquet + qa.parquet
  index      corpus.parquet → ChromaDB 인덱싱 (post_eval용 컬렉션)
  finetune   지정 모델 LoRA 파인튜닝 (모델별 순차 실행)
  autorag    AutoRAG 최적화 평가
  post_eval  AutoRAG 최적 config → run_evaluation.py

사용 예:
  # 전체: 데이터 + 인덱싱 + 파인튜닝 + AutoRAG + 평가
  python scripts/run_pipeline.py --steps all \\
    --finetune-models kanana-1.5,exaone

  # 파인튜닝 없이 전체 실행 (권장)
  python scripts/run_pipeline.py --steps data,index,autorag,post_eval

  # 데이터 + 인덱싱 + AutoRAG (파인튜닝 생략)
  python scripts/run_pipeline.py --steps data,index,autorag

  # 파인튜닝 + AutoRAG (데이터/인덱싱 이미 완료)
  python scripts/run_pipeline.py --steps finetune,autorag \\
    --finetune-models kanana-1.5

  # AutoRAG만
  python scripts/run_pipeline.py --steps autorag

사용 가능한 --finetune-models 값 (finetune_capable=True):
  [QLoRA 불필요 — BF16 LoRA]
  kanana-1.5                           — 2.1B
  exaone                               — EXAONE-4.0-1.2B
  exaone-deep-2.4b                     — EXAONE-Deep-2.4B
  midm                                 — Midm-2.0-Mini

  [QLoRA 자동 적용 — 레지스트리 qlora=True]
  gemma3                               — Gemma3-4B (권장)
  gemma4                               — Gemma4-E4B (권장)
  exaone-3.5-7.8b, exaone-deep-7.8b   — 7.8B (필수)

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

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()
from configs import paths

# ── 모델 레지스트리 ─────────────────────────────────────────────────
# short name → 학습/서빙 파라미터
# model_path는 paths.MODEL_DIR 기반 — .env의 MODEL_DIR 또는 SRV_DATA_DIR로 제어
def _mp(subpath: str) -> str:
    """paths.MODEL_DIR 아래 subpath 조합."""
    return f"{paths.MODEL_DIR}/{subpath}"


MODEL_REGISTRY: dict[str, dict] = {
    # ── 1.2B ~ 2.1B: BF16 LoRA, QLoRA 불필요 ─────────────────────────
    "kanana-1.5": {
        "model_path": _mp("kanana/kanana-1.5-2.1b"),
        "trust_remote_code": False,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95], "max_model_len": 8192},
    },
    "exaone": {
        "model_path": _mp("exaone/EXAONE-4.0-1.2B"),
        "trust_remote_code": True,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95]},
    },
    "exaone-deep-2.4b": {
        "model_path": _mp("exaone/EXAONE-Deep-2.4B"),
        "trust_remote_code": True,
        "qlora": False,
        "finetune": {"batch_size": 2, "grad_accum": 8, "lora_r": 16},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95]},
    },
    "midm": {
        "model_path": _mp("midm/Midm-2.0-Mini"),
        "trust_remote_code": False,
        "qlora": False,
        "finetune": {"batch_size": 4, "grad_accum": 4, "lora_r": 16},
        "vllm": {"temperature": [0.7, 0.8], "top_k": 20, "top_p": [0.80, 0.90]},
    },
    # ── 4B: QLoRA 권장 (22GB GPU에서 BF16 LoRA 가능하나 여유 확보) ──────
    "gemma3": {
        "model_path": _mp("gemma/Gemma3-4B"),
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
        "model_path": _mp("gemma/Gemma4-E4B"),
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
    # exaone-3.5-7.8b: 디스크 용량 제한으로 파인튜닝 제외 (추론 전용 base 모델만 유지)
    "exaone-deep-7.8b": {
        "model_path": _mp("exaone/EXAONE-Deep-7.8B"),
        "trust_remote_code": True,
        "qlora": True,
        "finetune": {"batch_size": 1, "grad_accum": 16, "lora_r": 8},
        "vllm": {"temperature": [0.1, 0.2], "top_p": [0.85, 0.95], "max_model_len": 8192, "gpu_memory_utilization": 0.90},
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


# ── Step 1b: ChromaDB 인덱싱 ────────────────────────────────────────

def step_index(args: argparse.Namespace) -> None:
    """corpus.parquet → ChromaDB 인덱싱 (post_eval / run_evaluation.py용)."""
    _section(f"Step 1b / index — corpus.parquet → ChromaDB ({args.eval_collection})")

    corpus = ROOT / args.data_dir / "corpus.parquet"
    if not corpus.exists():
        print(f"  [ERROR] corpus.parquet 없음: {corpus}")
        print("  data 단계를 먼저 실행하거나 --data-dir을 확인하세요.")
        sys.exit(1)

    cmd = [
        PYTHON, str(ROOT / "scripts/index_documents.py"),
        "--scenario", args.index_scenario,
        "--from-parquet", str(corpus),
        "--collection", args.eval_collection,
    ]
    if args.index_scenario == "A":
        cmd += ["--hf-embedding-model", args.hf_embedding_model]

    _run(cmd)


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

    trial_dirs = sorted([d for d in project_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name))
    trial_dir = trial_dirs[-1] if trial_dirs else project_dir / "0"
    print(f"\n[{label}] 결과 (trial {trial_dir.name}):")
    print(f"  cat {trial_dir}/retrieve_node_line/*/summary.csv")
    print(f"  cat {trial_dir}/post_retrieve_node_line/*/summary.csv")
    print(f"  autorag dashboard --trial_dir {trial_dir}")


def _resolve_yaml_env(config_path: Path) -> Path:
    """YAML 내 ${SRV_DATA_DIR} 등 우리 플레이스홀더를 실제 경로로 치환.

    - ${SRV_DATA_DIR}, ${MODEL_DIR} 등은 paths.py 기본값으로 치환
    - ${PROJECT_DIR}는 AutoRAG가 런타임에 직접 처리하므로 건드리지 않음
    치환이 필요하면 임시 파일에 저장 후 그 경로를 반환.
    """
    import re
    import tempfile

    # AutoRAG 자체 변수는 제외하고 우리 변수만 처리
    _AUTORAG_VARS = {"PROJECT_DIR"}

    # paths.py 기본값을 우선 사용 (.env 미설정 시에도 정상 동작)
    _OUR_VARS: dict[str, str] = {
        "SRV_DATA_DIR": paths.SRV_DATA_DIR,
        "MODEL_DIR":    paths.MODEL_DIR,
        "METADATA_CSV": paths.METADATA_CSV,
        "VECTORDB_DIR": paths.VECTORDB_DIR,
    }

    content = config_path.read_text(encoding="utf-8")

    def _sub(m: re.Match) -> str:
        key = m.group(1)
        if key in _AUTORAG_VARS:
            return m.group(0)          # AutoRAG 전용 변수는 그대로 유지
        if key in _OUR_VARS:
            return _OUR_VARS[key]
        # 나머지는 환경변수에서 탐색, 없으면 원본 유지
        val = os.getenv(key)
        if val is None:
            print(f"  [경고] 알 수 없는 플레이스홀더: ${{{key}}} — 치환 생략")
            return m.group(0)
        return val

    replaced = re.sub(r"\$\{([^}]+)\}", _sub, content)

    if replaced == content:
        return config_path  # 치환 없음 → 원본 사용

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        encoding="utf-8", prefix="autorag_resolved_",
    )
    tmp.write(replaced)
    tmp.close()
    print(f"  경로 치환 완료 ({paths.SRV_DATA_DIR}): {config_path.name} → {tmp.name}")
    return Path(tmp.name)


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

    # YAML 내 ${SRV_DATA_DIR} 등 환경변수 치환
    base_config = _resolve_yaml_env(base_config)

    data_dir = ROOT / args.data_dir

    # ── 모델을 두 그룹으로 분리 ─────────────────────────────────────
    # group A: PYTHONNOUSERSITE=1  (kanana / midm / exaone / gemma3 등)
    # group B: user-local site 허용 (gemma4 — transformers 5.x 필요)

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


# ── Step 4b: AutoRAG 최적 임베딩 모델 → ChromaDB 인덱싱 ────────────

# AutoRAG vectordb 이름 → index_documents.py --hf-embedding-model 키
_AUTORAG_VDB_TO_EMB: dict[str, str] = {
    "local_bge":        "bge",
    "local_sroberta":   "sroberta",
    "local_e5_large":   "e5",
    "local_kosimcse":   "kosimcse",
    "local_kf_deberta": "kf_deberta",
}


def _read_best_embedding(trial_dir: Path) -> str:
    """AutoRAG semantic_retrieval summary.csv에서 최고 성능 임베딩 모델 키를 반환.

    Returns:
        index_documents.py --hf-embedding-model 값 (기본: "bge")
    """
    import ast
    import pandas as pd

    best_emb = "bge"
    best_score = -1.0

    semantic_dir = trial_dir / "retrieve_node_line" / "semantic_retrieval"
    if not semantic_dir.exists():
        print(f"  [경고] semantic_retrieval 없음: {semantic_dir}")
        return best_emb

    for summary_csv in semantic_dir.rglob("summary.csv"):
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            continue

        score_col = next(
            (c for c in ["retrieval_f1", "retrieval_ndcg", "retrieval_map"] if c in df.columns),
            None,
        )
        if score_col is None:
            continue

        best_row = df.loc[df[score_col].idxmax()]
        score = float(best_row[score_col])
        if score <= best_score:
            continue

        try:
            params = best_row.get("module_params", "{}")
            if isinstance(params, str):
                params = ast.literal_eval(params)
            vdb_name = params.get("vectordb", "")
            emb_key = _AUTORAG_VDB_TO_EMB.get(vdb_name, "bge")
        except Exception:
            emb_key = "bge"

        best_score = score
        best_emb = emb_key

    print(f"  AutoRAG 최적 임베딩 모델: {best_emb} (score={best_score:.4f})")
    return best_emb


def step_best_index(args: argparse.Namespace, size: int) -> str:
    """AutoRAG 결과에서 최적 임베딩 모델을 읽어 ChromaDB 인덱싱 수행.

    Returns:
        생성된 컬렉션 이름
    """
    _section(f"Step 4b / best_index — AutoRAG 최적 임베딩으로 인덱싱 (chunk={size})")

    project_dir = ROOT / (f"evaluation/autorag_benchmark_csv_{size}" if args.chunk_sizes else args.project_dir)
    trial_dirs = sorted(
        [d for d in project_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    ) if project_dir.exists() else []

    if not trial_dirs:
        print(f"  [스킵] AutoRAG 결과 없음: {project_dir}")
        return ""

    trial_dir = trial_dirs[-1]
    best_emb = _read_best_embedding(trial_dir)
    collection = f"rfp_chunk{size}_{best_emb}"

    corpus = ROOT / f"data/autorag_csv_{size}/corpus.parquet"
    if not corpus.exists():
        print(f"  [ERROR] corpus.parquet 없음: {corpus}")
        return ""

    cmd = [
        PYTHON, str(ROOT / "scripts/index_documents.py"),
        "--scenario", "A",
        "--from-parquet", str(corpus),
        "--collection", collection,
        "--hf-embedding-model", best_emb,
    ]
    print(f"  컬렉션: {collection} | 임베딩: {best_emb}")
    _run(cmd)
    return collection


# ── Step 5: AutoRAG 결과 → run_evaluation.py 자동 실행 ──────────────

# AutoRAG 모듈명 → 우리 retrieval_method 매핑
_AUTORAG_METHOD_MAP: dict[str, str] = {
    "bm25":       "hybrid",      # BM25 단독 → 가장 가까운 hybrid
    "vectordb":   "similarity",  # 순수 벡터 → similarity
    "hybrid_rrf": "hybrid",
    "hybrid_cc":  "hybrid",
}


def _read_best_retrieval(trial_dir: Path) -> tuple[str, int]:
    """AutoRAG retrieve_node_line summary.csv에서 최고 성능 모듈과 top_k를 반환한다.

    Returns:
        (retrieval_method, top_k)  — 기본값 ("similarity", 5)
    """
    import ast
    import pandas as pd

    best_method = "similarity"
    best_top_k = 5
    best_score = -1.0

    retrieve_dir = trial_dir / "retrieve_node_line"
    if not retrieve_dir.exists():
        print(f"  [경고] retrieve_node_line 없음: {retrieve_dir}")
        return best_method, best_top_k

    for summary_csv in retrieve_dir.rglob("summary.csv"):
        try:
            df = pd.read_csv(summary_csv)
        except Exception:
            continue

        # 지표: retrieval_f1 → retrieval_ndcg → retrieval_map 순으로 시도
        score_col = next(
            (c for c in ["retrieval_f1", "retrieval_ndcg", "retrieval_map"] if c in df.columns),
            None,
        )
        if score_col is None:
            continue

        best_row = df.loc[df[score_col].idxmax()]
        score = float(best_row[score_col])

        if score <= best_score:
            continue

        # module_name 파싱 (patch로 model basename일 수도 있음)
        raw_name = str(best_row.get("module_name", "")).lower()
        method = next(
            (v for k, v in _AUTORAG_METHOD_MAP.items() if k in raw_name),
            "similarity",
        )

        # top_k 파싱 (module_params JSON에서 추출)
        top_k = 5
        try:
            params = best_row.get("module_params", "{}")
            if isinstance(params, str):
                params = ast.literal_eval(params)
            top_k = int(params.get("top_k", 5))
        except Exception:
            pass

        best_score = score
        best_method = method
        best_top_k = top_k

    print(f"  AutoRAG 최적 검색 방식: {best_method} | top_k={best_top_k} | score={best_score:.4f}")
    return best_method, best_top_k


def step_post_eval(args: argparse.Namespace) -> None:
    """AutoRAG 결과에서 최적 검색 설정을 읽어 run_evaluation.py를 자동 실행한다."""
    _section("Step 4 / post_eval — AutoRAG 최적 config로 RAG 평가")

    trial_dir = ROOT / args.project_dir / "0"
    if not trial_dir.exists():
        print(f"  [스킵] AutoRAG 결과 없음: {trial_dir}")
        print("  autorag 단계를 먼저 실행하세요.")
        return

    best_method, best_top_k = _read_best_retrieval(trial_dir)

    output_dir = ROOT / "evaluation" / "post_autorag"
    cmd = [
        PYTHON, str(ROOT / "scripts/run_evaluation.py"),
        "--mode", "core",
        "--collection", args.eval_collection,
        "--output-dir", str(output_dir),
    ]

    print(f"  retrieval_method={best_method} | top_k={best_top_k} | collection={args.eval_collection}")
    print(f"  출력 디렉토리: {output_dir}")

    # run_evaluation.py는 내부적으로 4가지 config를 돌리므로
    # best config만 추가로 출력 알림
    print(f"\n  ※ run_evaluation.py는 4가지 검색 config를 비교 실행합니다.")
    print(f"    AutoRAG 추천 best: {best_method} top_k={best_top_k}")

    _run(cmd, use_user_site=False)


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
            "선택: data, finetune, autorag, post_eval. 기본: all"
        ),
    )

    # 데이터 옵션
    parser.add_argument("--csv-path", default=paths.METADATA_CSV)
    parser.add_argument("--data-dir", default=paths.AUTORAG_DATA_DIR,
                        help=f"corpus/qa parquet 저장 위치 (기본: {paths.AUTORAG_DATA_DIR})")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-sizes", type=str, default="",
                        help="청크 크기 다중 실행 (쉼표 구분, 예: 600,800,1000,1200). 지정 시 각 크기별로 data+index+autorag 반복 실행")
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
    parser.add_argument("--finetune-epochs", type=int, default=10,
                        help="최대 학습 epoch 수 (early stop 시 조기 종료, 기본: 10)")
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
    parser.add_argument("--project-dir", default=paths.AUTORAG_PROJECT_DIR,
                        help=f"AutoRAG 결과 저장 디렉토리 (기본: {paths.AUTORAG_PROJECT_DIR})")

    # 인덱싱 옵션
    parser.add_argument(
        "--index-scenario",
        default="A",
        choices=["A", "B"],
        help="index 단계 시나리오 (기본: A — 로컬 HuggingFace 임베딩)",
    )
    parser.add_argument(
        "--hf-embedding-model",
        default="bge",
        choices=["bge", "sroberta"],
        help="index 단계 Scenario A 임베딩 모델 (기본: bge)",
    )

    # post_eval / index 공통 옵션
    parser.add_argument(
        "--eval-collection",
        default="",
        help="index/post_eval 단계에서 사용할 ChromaDB 컬렉션 (기본: rfp_chunk{chunk_size}_a 또는 rfp_chunk{chunk_size})",
    )

    args = parser.parse_args()

    # 단계 파싱
    _is_all = args.steps.strip().lower() == "all"
    if _is_all:
        steps = {"data", "index", "finetune", "autorag", "post_eval"}
    else:
        steps = {s.strip().lower() for s in args.steps.split(",")}

    valid_steps = {"data", "index", "finetune", "autorag", "best_index", "post_eval"}
    invalid = steps - valid_steps
    if invalid:
        print(f"[ERROR] 알 수 없는 단계: {invalid}. 선택: {', '.join(sorted(valid_steps))}, all")
        sys.exit(1)

    # --steps all 이면 best_index 포함
    if _is_all:
        steps.add("best_index")

    # --steps all 이고 --finetune-models 미지정이면 레지스트리 전체 모델 자동 포함
    if _is_all and not args.finetune_models.strip():
        args.finetune_models = ",".join(MODEL_REGISTRY.keys())
        print(f"[all 모드] 파인튜닝 모델 자동 선택: {args.finetune_models}")

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

    # ── 청크 크기 목록 결정 ─────────────────────────────────────────
    if args.chunk_sizes.strip():
        chunk_sizes = [int(s.strip()) for s in args.chunk_sizes.split(",") if s.strip()]
        multi_chunk = True
    else:
        chunk_sizes = [args.chunk_size]
        multi_chunk = False

    _orig_data_dir      = args.data_dir
    _orig_project_dir   = args.project_dir
    _orig_eval_col      = args.eval_collection  # "" = 자동 유도

    # ── 파인튜닝: 청크 크기 무관, 한 번만 실행 ─────────────────────
    # multi_chunk 시 첫 번째 크기 데이터를 먼저 준비한 뒤 학습
    finetuned: list[tuple[str, Path]] = []
    if "finetune" in steps:
        if multi_chunk:
            first_size = chunk_sizes[0]
            args.chunk_size = first_size
            args.data_dir = f"data/autorag_csv_{first_size}"
            if not _orig_eval_col:
                suffix = "_a" if args.index_scenario == "A" else ""
                args.eval_collection = f"rfp_chunk{first_size}{suffix}"
            if "data" in steps:
                step_data(args)
        finetuned = step_finetune(args)

    # finetune 단계를 건너뛴 경우, 디스크에서 완료된 모델을 자동 감지 (루프 전 1회)
    if "autorag" in steps and not finetuned and finetune_models:
        for name in finetune_models:
            final_dir = ROOT / "models/finetuned" / name / "final"
            if final_dir.exists():
                finetuned.append((name, final_dir))
                print(f"  [자동 감지] 기학습 모델: {name} → {final_dir}")

    # ── 청크 크기별 반복 실행 ────────────────────────────────────────
    for size in chunk_sizes:
        if multi_chunk:
            _section(f"청크 크기 {size}자 ({chunk_sizes.index(size)+1}/{len(chunk_sizes)})")
            args.chunk_size   = size
            args.data_dir     = f"data/autorag_csv_{size}"
            args.project_dir  = f"evaluation/autorag_benchmark_csv_{size}"

        # eval_collection: 명시적 지정 없으면 크기별 자동 유도
        if not _orig_eval_col:
            suffix = "_a" if args.index_scenario == "A" else ""
            args.eval_collection = f"rfp_chunk{size}{suffix}"
        else:
            args.eval_collection = _orig_eval_col

        if "data" in steps:
            step_data(args)

        if "index" in steps:
            step_index(args)

        if "autorag" in steps:
            step_autorag(args, finetuned, eval_only_models)

        if "best_index" in steps:
            step_best_index(args, size)

        if "post_eval" in steps:
            step_post_eval(args)

    # ── 최종 요약 ────────────────────────────────────────────────────
    _section("파이프라인 완료")
    if multi_chunk:
        print(f"처리된 청크 크기: {chunk_sizes}")
    if "index" in steps:
        if multi_chunk:
            for size in chunk_sizes:
                suffix = "_a" if args.index_scenario == "A" else ""
                col = _orig_eval_col or f"rfp_chunk{size}{suffix}"
                print(f"  인덱싱 컬렉션: {col} (scenario {args.index_scenario})")
        else:
            print(f"인덱싱 컬렉션: {args.eval_collection} (scenario {args.index_scenario})")
    if finetuned:
        print("학습된 모델:")
        for name, path in finetuned:
            print(f"  {name}: {path}")
    if eval_only_models:
        print(f"평가 전용 모델 (원본 경로 사용): {', '.join(eval_only_models)}")
    if "autorag" in steps:
        has_gemma = any(MODEL_REGISTRY[m].get("needs_user_site") for m in eval_only_models)
        has_gemma = has_gemma or any(
            MODEL_REGISTRY[n].get("needs_user_site") for n, _ in finetuned
        )
        for size in chunk_sizes:
            proj = ROOT / (f"evaluation/autorag_benchmark_csv_{size}" if multi_chunk else _orig_project_dir)
            trial_dirs = sorted([d for d in proj.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name)) if proj.exists() else []
            trial_dir = trial_dirs[-1] if trial_dirs else proj / "0"
            print(f"\nAutoRAG 결과 (일반 모델, chunk={size}): {proj}")
            print(f"  대시보드: autorag dashboard --trial_dir {trial_dir}")
            if has_gemma:
                gemma_proj = ROOT / (proj.name + "_gemma")
                gemma_trial = gemma_proj / "0"
                print(f"AutoRAG 결과 (Gemma4, chunk={size}): {gemma_proj}")
                print(f"  대시보드: autorag dashboard --trial_dir {gemma_trial}")
    if "post_eval" in steps:
        print(f"\npost_eval 결과: {ROOT / 'evaluation' / 'post_autorag'}")


if __name__ == "__main__":
    main()
