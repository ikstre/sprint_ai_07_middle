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

import ctypes
import gc
import json
import sys
from pathlib import Path


def _malloc_trim() -> None:
    """libc malloc_trim으로 Python 힙 조각 메모리를 OS에 반환."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

ROOT = Path(__file__).parent.parent
FINETUNED_BASE = ROOT / "models" / "finetuned"

_SHARD_SIZE = 512 * 1024 ** 2  # 512 MB — RAM 부족 환경에서 힙 조각화 최소화


def _save_sharded(model, save_dir: Path) -> None:
    """GPU 메모리 제약 환경에서 샤드별 명시적 해제로 모델 저장.

    save_pretrained()는 내부 safetensors 버퍼를 즉시 해제하지 않아
    가용 RAM이 적을 때 OOM이 발생한다.
    본 함수는 GPU 텐서를 2 GB씩 CPU로 이동 → 저장 → 즉시 해제하며
    인덱스 파일(model.safetensors.index.json)도 생성한다.
    """
    from safetensors.torch import save_file

    # config / generation_config 저장
    if hasattr(model, "config"):
        model.config.save_pretrained(str(save_dir))
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.save_pretrained(str(save_dir))
        except Exception:
            pass

    # state_dict: GPU 텐서 참조 (CPU 복사 없음)
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    # 샤드 계획: 키 이름만 수집 (텐서 이동 없음)
    shards: list[list[str]] = []
    cur_keys: list[str] = []
    cur_size = 0
    for key in keys:
        tsize = state_dict[key].nelement() * state_dict[key].element_size()
        if cur_keys and cur_size + tsize > _SHARD_SIZE:
            shards.append(cur_keys)
            cur_keys, cur_size = [], 0
        cur_keys.append(key)
        cur_size += tsize
    if cur_keys:
        shards.append(cur_keys)

    n = len(shards)
    total_size = sum(t.nelement() * t.element_size() for t in state_dict.values())
    weight_map: dict[str, str] = {}

    print(f"    총 {n}개 샤드로 저장 (각 최대 512 MB)")
    for i, shard_keys in enumerate(shards):
        shard_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        print(f"    [{i+1}/{n}] {shard_name} 저장 중...")

        # 이 샤드만 CPU 이동 → 저장 → 즉시 해제 (힙 조각화 방지)
        cpu_shard = {k: state_dict[k].cpu() for k in shard_keys}
        save_file(cpu_shard, str(save_dir / shard_name))
        for k in shard_keys:
            weight_map[k] = shard_name
        del cpu_shard
        gc.collect()
        _malloc_trim()  # Python 힙 조각 메모리를 OS에 즉시 반환

    # 인덱스 파일 (단일 샤드면 생략 가능하지만 vLLM 호환을 위해 항상 생성)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(str(save_dir / "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"    model.safetensors.index.json 저장 완료")


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


# ── Case B-stream: mmap 기반 스트리밍 merge (RAM 부족 환경 전용) ─────────

def _stream_merge_and_save(final_dir: Path, base_model_path: str) -> bool:
    """기반 모델을 메모리맵(mmap)으로 읽고, LoRA 델타를 CPU에서 계산해 저장.

    transformers from_pretrained()를 전혀 사용하지 않으므로 CUDA 없이도 동작하며
    RAM 사용량이 최대 ~1GB에 불과하다.

    단일 model.safetensors 및 샤드(model-XXXXX-of-NNNNN.safetensors) 모두 지원.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_path = Path(base_model_path)
    adapter_sf = final_dir / "adapter_model.safetensors"
    adapter_cfg_path = final_dir / "adapter_config.json"

    # ── 기반 모델 파일 탐색 (단일 or 샤드) ──────────────────────────────
    single_sf = base_path / "model.safetensors"
    index_json = base_path / "model.safetensors.index.json"

    if single_sf.exists():
        # 단일 파일 모드
        base_shards: list[Path] = [single_sf]
        # key → shard_file 매핑 (단일 파일이면 모두 같은 파일)
        base_key_to_shard: dict[str, Path] = {}
        with safe_open(str(single_sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                base_key_to_shard[k] = single_sf
    elif index_json.exists():
        # 샤드 파일 모드
        with open(index_json, encoding="utf-8") as fj:
            idx = json.load(fj)
        weight_map_src: dict[str, str] = idx.get("weight_map", {})
        base_shards = sorted(set(base_path / v for v in weight_map_src.values()))
        base_key_to_shard = {k: base_path / v for k, v in weight_map_src.items()}
    else:
        print(f"    [stream] 기반 모델 파일 없음: {base_path}")
        return None  # type: ignore  # None → 폴백 신호

    if not adapter_sf.exists():
        print(f"    [stream] adapter_model.safetensors 없음: {adapter_sf}")
        return None  # type: ignore

    with open(adapter_cfg_path, encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    lora_alpha: float = float(adapter_cfg.get("lora_alpha", 16))
    lora_r: int = int(adapter_cfg.get("r", 16))
    scale: float = lora_alpha / lora_r

    print(f"    [stream] 어댑터 로딩 (lora_alpha={lora_alpha}, r={lora_r}, scale={scale:.4f})")

    # 어댑터 전체 로딩 (수십 MB 수준 → 문제없음)
    adapter_weights: dict[str, torch.Tensor] = {}
    with safe_open(str(adapter_sf), framework="pt", device="cpu") as f:
        for key in f.keys():
            adapter_weights[key] = f.get_tensor(key)
    print(f"    [stream] 어댑터 키 {len(adapter_weights)}개 로딩 완료")

    # key 변환: "base_model.model.X.lora_A.weight" → ("model.X", "lora_A")
    def _parse_adapter_key(key: str):
        if not key.startswith("base_model.model."):
            return None, None
        rest = key[len("base_model.model."):]
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if rest.endswith(suffix):
                param_key = "model." + rest[: -len(suffix)]
                return param_key, suffix.lstrip(".").split(".")[0]  # lora_A / lora_B
        return None, None

    lora_map: dict[str, dict[str, torch.Tensor]] = {}
    for akey, atensor in adapter_weights.items():
        base_key, lora_type = _parse_adapter_key(akey)
        if base_key is None:
            continue
        lora_map.setdefault(base_key, {})[lora_type] = atensor

    print(f"    [stream] LoRA 적용 대상 파라미터 {len(lora_map)}개")

    # ── 샤드 계획 (key 목록 및 크기를 mmap으로 수집) ─────────────────────
    shard_size = _SHARD_SIZE
    all_keys: list[str] = list(base_key_to_shard.keys())
    keys_with_size: list[tuple[str, int]] = []

    # 각 입력 샤드를 한 번씩 열어 크기 수집
    opened_shards: dict[Path, object] = {}
    for sf in base_shards:
        opened_shards[sf] = safe_open(str(sf), framework="pt", device="cpu").__enter__()

    try:
        for key in all_keys:
            sf_path = base_key_to_shard[key]
            meta = opened_shards[sf_path].get_slice(key)
            shape = meta.get_shape()
            nbytes = 2  # bf16 base
            for d in shape:
                nbytes *= d
            keys_with_size.append((key, nbytes))
    finally:
        for sf_path, f in opened_shards.items():
            try:
                f.__exit__(None, None, None)
            except Exception:
                pass

    out_shards: list[list[str]] = []
    cur_keys: list[str] = []
    cur_size = 0
    for key, nbytes in keys_with_size:
        if cur_keys and cur_size + nbytes > shard_size:
            out_shards.append(cur_keys)
            cur_keys, cur_size = [], 0
        cur_keys.append(key)
        cur_size += nbytes
    if cur_keys:
        out_shards.append(cur_keys)

    n = len(out_shards)
    total_size = sum(s for _, s in keys_with_size)
    weight_map: dict[str, str] = {}

    print(f"    [stream] 총 {n}개 샤드로 저장 (각 최대 {shard_size // 1024 ** 2} MB)"
          f" [기반 모델 샤드 {len(base_shards)}개]")
    final_dir.mkdir(parents=True, exist_ok=True)

    for i, shard_keys in enumerate(out_shards):
        shard_name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        print(f"    [{i+1}/{n}] {shard_name} 저장 중...", flush=True)

        # 이 출력 샤드에서 필요한 입력 파일만 열기
        needed_inputs = set(base_key_to_shard[k] for k in shard_keys)
        open_fhs: dict[Path, object] = {}
        for sf_path in needed_inputs:
            open_fhs[sf_path] = safe_open(str(sf_path), framework="pt", device="cpu").__enter__()

        cpu_shard: dict[str, torch.Tensor] = {}
        try:
            for key in shard_keys:
                sf_path = base_key_to_shard[key]
                tensor = open_fhs[sf_path].get_tensor(key).to(torch.bfloat16)

                if key in lora_map and "lora_A" in lora_map[key] and "lora_B" in lora_map[key]:
                    lora_A = lora_map[key]["lora_A"].to(torch.float32)
                    lora_B = lora_map[key]["lora_B"].to(torch.float32)
                    delta = (lora_B @ lora_A) * scale
                    tensor = (tensor.to(torch.float32) + delta).to(torch.bfloat16)

                cpu_shard[key] = tensor
                weight_map[key] = shard_name
        finally:
            for sf_path, f in open_fhs.items():
                try:
                    f.__exit__(None, None, None)
                except Exception:
                    pass

        save_file(cpu_shard, str(final_dir / shard_name))
        del cpu_shard
        gc.collect()
        _malloc_trim()

    # 인덱스 파일 생성
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(str(final_dir / "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"    [stream] model.safetensors.index.json 저장 완료")

    # config / tokenizer 복사 (기반 모델에서)
    for fname in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json"):
        src = Path(base_model_path) / fname
        dst = final_dir / fname
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    return True


# ── Case B: 원본 기반 모델 + 어댑터 → merge ──────────────────────────

def _merge_and_save(final_dir: Path, base_model_path: str) -> bool:
    """기반 모델 + 어댑터를 merge해 model.safetensors로 저장."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"    기반 모델 로드: {base_model_path}")
    trust = any(k in base_model_path.lower() for k in ("exaone",))

    # device_map="auto" 는 CPU로 일부 레이어를 오프로드할 수 있어
    # RAM이 부족한 환경에서 OOM 유발. CUDA가 있으면 GPU 전체 배치 우선 시도.
    if torch.cuda.is_available():
        _device_map: str | dict = {"": "cuda:0"}
    else:
        _device_map = "cpu"

    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=_device_map,
            low_cpu_mem_usage=True,   # meta 디바이스로 스켈레톤 생성 → CPU RAM 최소화
            trust_remote_code=trust,
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        # GPU VRAM 부족 시 CPU 전체 배치로 재시도
        print(f"    [경고] GPU 전체 배치 실패, CPU 배치로 재시도...")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=trust,
            )
        except Exception as e2:
            print(f"    [오류] 기반 모델 로드 실패: {e2}")
            return False
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
        # _save_sharded: 샤드마다 CPU 버퍼를 즉시 해제 → RAM 5GB 환경에서 OOM 방지
        _save_sharded(merged, final_dir)
    except Exception as e:
        print(f"    [오류] 저장 실패: {e}")
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

    # ── 스트리밍 merge 우선 시도 (RAM 최소화, mmap 기반) ─────────────────
    stream_result = _stream_merge_and_save(final_dir, base_model_path)
    if stream_result is True:
        success = True
    elif stream_result is None:
        # 폴백: base model이 sharded이거나 스트림 지원 불가 → 기존 방식
        print(f"  [stream 불가] _merge_and_save 로 폴백")
        success = _merge_and_save(final_dir, base_model_path)
    else:
        success = False

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
