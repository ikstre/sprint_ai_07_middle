"""
기존 파인튜닝 모델 final/ 디렉토리 일괄 복구.

문제:
  save_pretrained()가 QLoRA 모델의 경우 merged full model(model.safetensors)을
  저장하지 않고 어댑터만 저장(adapter_model.safetensors + adapter_config.json).
  vLLM이 adapter_config.json을 보고 PEFT 모델로 인식하거나,
  adapter_model.safetensors의 base_model.* 키로 인해 로드 실패.

수정 방향:
  Case A. model.safetensors 있음 → adapter 파일만 제거 (kanana-nano 등)
  Case B. model.safetensors 없음 → 원본 기반 모델 + 어댑터 로드 후 merge, 저장

사용법:
  python scripts/repair_finetuned_models.py              # 전체 자동
  python scripts/repair_finetuned_models.py kanana-nano  # 개별 지정
  python scripts/repair_finetuned_models.py --dry-run    # 상태 확인만
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
FINETUNED_BASE = ROOT / "models" / "finetuned"


# ── 유틸리티 ────────────────────────────────────────────────────────

def _has_base_model_keys(safetensors_path: Path) -> bool:
    try:
        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            return any(k.startswith("base_model.") for k in f.keys())
    except Exception:
        return False


def _verify_model_safetensors(final_dir: Path) -> tuple[bool, str]:
    """model*.safetensors가 정상 merged 상태인지 확인."""
    shards = sorted(final_dir.glob("model*.safetensors"))
    if not shards:
        return False, "model*.safetensors 없음"

    try:
        from safetensors import safe_open
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            keys = list(f.keys())
        has_base = any(k.startswith("base_model.") for k in keys)
        has_lora = any("lora_" in k for k in keys)
        if has_base or has_lora:
            return False, f"비정상 키 (base_model={has_base}, lora={has_lora})"
        return True, f"정상 ({len(keys)}개 키)"
    except Exception as e:
        return False, f"읽기 실패: {e}"


def _remove_adapter_artifacts(final_dir: Path) -> list[str]:
    """adapter 파일 및 quantization_config 제거. 변경 항목 목록 반환."""
    changed = []

    # adapter_config.json 제거
    p = final_dir / "adapter_config.json"
    if p.exists():
        p.unlink()
        changed.append("adapter_config.json 제거")

    # adapter_model*.safetensors 제거
    for p in sorted(final_dir.glob("adapter_model*.safetensors")):
        p.unlink()
        changed.append(f"{p.name} 제거")

    # config.json quantization_config 제거
    cfg_path = final_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.pop("quantization_config", None) is not None:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            changed.append("config.json quantization_config 제거")

    return changed


# ── Case B: 원본 기반 모델 + 어댑터 → merge ──────────────────────────

def _merge_and_save(final_dir: Path, base_model_path: str) -> bool:
    """기반 모델 + 어댑터를 merge해 model.safetensors로 저장."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"    기반 모델 로드: {base_model_path}")
    trust = any(k in base_model_path.lower() for k in ("exaone",))
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=trust,
        )
    except Exception as e:
        print(f"    [오류] 기반 모델 로드 실패: {e}")
        return False

    print(f"    어댑터 로드: {final_dir}")
    try:
        peft_model = PeftModel.from_pretrained(base, str(final_dir))
    except Exception as e:
        print(f"    [오류] 어댑터 로드 실패: {e}")
        return False

    print(f"    LoRA merge_and_unload 중...")
    try:
        merged = peft_model.merge_and_unload()
    except Exception as e:
        print(f"    [오류] merge_and_unload 실패: {e}")
        return False

    # quantization_config 제거 후 저장
    if hasattr(merged, "config") and hasattr(merged.config, "quantization_config"):
        merged.config.quantization_config = None

    print(f"    merged model 저장 중...")
    try:
        merged.save_pretrained(str(final_dir))
    except Exception as e:
        print(f"    [오류] save_pretrained 실패: {e}")
        return False

    # 토크나이저 저장 (없는 경우)
    if not (final_dir / "tokenizer.json").exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust)
            tokenizer.save_pretrained(str(final_dir))
        except Exception:
            pass

    # GPU 메모리 해제
    try:
        import gc
        del merged, peft_model, base
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    return True


# ── 모델별 복구 ────────────────────────────────────────────────────

def repair_model(model_name: str, dry_run: bool = False) -> str:
    """복구 실행. 반환값: 'ok' | 'skipped' | 'failed'"""
    final_dir = FINETUNED_BASE / model_name / "final"

    print(f"\n{'─'*52}")
    print(f"  {model_name}")
    print(f"{'─'*52}")

    if not final_dir.exists():
        print(f"  [스킵] final/ 없음")
        return "skipped"

    ok, msg = _verify_model_safetensors(final_dir)
    print(f"  model.safetensors: {msg}")

    # ── Case A: model.safetensors 있음 → 어댑터 파일만 제거 ──────────
    if ok:
        if dry_run:
            has_adapter = (final_dir / "adapter_config.json").exists()
            print(f"  [dry-run] {'어댑터 파일 제거 필요' if has_adapter else '이미 정상'}")
            return "ok"
        changed = _remove_adapter_artifacts(final_dir)
        if changed:
            for c in changed:
                print(f"  ✓ {c}")
            print(f"  → 복구 완료 (Case A)")
        else:
            print(f"  이미 정상 — 변경 없음")
        return "ok"

    # ── Case B: model.safetensors 없음 → merge 필요 ──────────────────
    adapter_config_path = final_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"  [실패] adapter_config.json 없음 — 재학습 필요")
        return "failed"

    with open(adapter_config_path, encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_path = adapter_cfg.get("base_model_name_or_path", "")

    if not base_model_path:
        print(f"  [실패] base_model_name_or_path 없음 — 재학습 필요")
        return "failed"

    if not Path(base_model_path).exists():
        print(f"  [실패] 기반 모델 경로 없음: {base_model_path}")
        return "failed"

    print(f"  기반 모델: {base_model_path}")

    if dry_run:
        print(f"  [dry-run] merge 필요 — 'python scripts/repair_finetuned_models.py {model_name}' 로 실행")
        return "skipped"

    success = _merge_and_save(final_dir, base_model_path)
    if not success:
        return "failed"

    # merge 성공 후 어댑터 파일 정리
    changed = _remove_adapter_artifacts(final_dir)
    for c in changed:
        print(f"  ✓ {c}")

    # 최종 검증
    ok2, msg2 = _verify_model_safetensors(final_dir)
    if ok2:
        print(f"  ✓ 검증 완료: {msg2}")
        print(f"  → 복구 완료 (Case B)")
        return "ok"
    else:
        print(f"  [경고] 저장 후 검증 실패: {msg2}")
        return "failed"


# ── 메인 ───────────────────────────────────────────────────────────

def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    dry_run = "--dry-run" in sys.argv

    if not FINETUNED_BASE.exists():
        print(f"models/finetuned/ 없음: {FINETUNED_BASE}")
        sys.exit(1)

    if args:
        model_names = args
    else:
        model_names = sorted(
            d.name for d in FINETUNED_BASE.iterdir()
            if d.is_dir() and (d / "final").exists()
        )

    if dry_run:
        print(f"[dry-run 모드] 실제 변경 없음")

    print(f"복구 대상 ({len(model_names)}개): {model_names}\n")

    results: dict[str, list[str]] = {"ok": [], "skipped": [], "failed": []}
    for name in model_names:
        result = repair_model(name, dry_run=dry_run)
        results[result].append(name)

    print(f"\n{'='*52}")
    print(f"  완료: {len(results['ok'])}개 | 스킵: {len(results['skipped'])}개 | 실패: {len(results['failed'])}개")
    if results["ok"]:
        print(f"  ✓ OK    : {', '.join(results['ok'])}")
    if results["failed"]:
        print(f"  ✗ 실패  : {', '.join(results['failed'])}")
        print(f"    → 해당 모델은 run_pipeline.py --force-finetune으로 재학습 필요")


if __name__ == "__main__":
    main()
