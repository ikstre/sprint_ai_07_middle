"""
로컬 모델 LoRA/QLoRA 파인튜닝 스크립트

QA 데이터(qa.parquet + corpus.parquet)로 RAG 응답 품질에 특화된
instruction-following 모델을 학습합니다.

사전 설치:
    pip install peft trl bitsandbytes accelerate

실행 예시:
    # LoRA (GPU 메모리 충분할 때)
    python scripts/finetune_local.py \
        --model-path /srv/shared_data/models/kanana/kanana-1.5-2.1b \
        --output-dir models/finetuned/kanana-1.5 \
        --epochs 10

    # QLoRA (VRAM 제한 환경, 4B+ 모델)
    python scripts/finetune_local.py \
        --model-path /srv/shared_data/models/gemma/Gemma3-4B \
        --output-dir models/finetuned/gemma3 \
        --qlora \
        --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# 데이터셋 생성
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "당신은 RFP(제안요청서) 문서 전문가입니다. "
    "주어진 참고 문서를 바탕으로 질문에 정확하고 간결하게 한국어로 답변하세요."
)

PROMPT_TEMPLATE = (
    "다음 문서를 참고하여 질문에 한국어로 답변하세요.\n\n"
    "참고 문서:\n{context}\n\n"
    "질문: {query}\n\n"
    "답변:"
)


def build_dataset(
    qa_path: str,
    corpus_path: str,
    max_context_chars: int = 2000,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """qa.parquet + corpus.parquet → chat 형식 학습 데이터."""
    qa_df = pd.read_parquet(qa_path)
    corpus_df = pd.read_parquet(corpus_path)
    corpus_map: dict[str, str] = dict(zip(corpus_df["doc_id"], corpus_df["contents"]))

    examples: list[dict] = []
    for _, row in qa_df.iterrows():
        query: str = row["query"]
        generation_gt = row["generation_gt"]
        retrieval_gt = row["retrieval_gt"]

        # generation_gt 파싱
        if isinstance(generation_gt, list):
            answer = " ".join(str(g) for g in generation_gt)
        else:
            answer = str(generation_gt)

        # 참조 문서 취합 (retrieval_gt는 리스트 또는 numpy ndarray의 중첩 구조)
        doc_ids: list[str] = []
        if retrieval_gt is not None:
            outer = retrieval_gt.tolist() if isinstance(retrieval_gt, np.ndarray) else list(retrieval_gt)
            for group in outer:
                if isinstance(group, np.ndarray):
                    doc_ids.extend(group.tolist())
                elif isinstance(group, list):
                    doc_ids.extend(group)
                else:
                    doc_ids.append(str(group))

        context_parts: list[str] = []
        for doc_id in doc_ids:
            text = corpus_map.get(doc_id, "")
            if text:
                context_parts.append(text[:600])

        context = "\n---\n".join(context_parts)[:max_context_chars]
        if not context:
            continue

        user_content = PROMPT_TEMPLATE.format(context=context, query=query)
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ]
        })

    random.seed(seed)
    random.shuffle(examples)
    split = max(1, int(len(examples) * val_ratio))
    return examples[split:], examples[:split]


# ─────────────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────────────

def _cleanup_merged_model(model_dir: Path) -> None:
    """LoRA merge 후 저장된 모델에서 vLLM 로드 방해 요소 제거.

    save_pretrained()는 merged 모델(model.safetensors)을 올바르게 저장하지만
    동시에 PEFT 어댑터 파일도 함께 저장한다:
      - adapter_config.json   → vLLM/transformers가 PEFT 모델로 인식
      - adapter_model.safetensors  → base_model.* 키 포함 (LoRA 어댑터 원본)
      - quantization_config in config.json → bitsandbytes loader 선택

    이 세 가지를 제거하면 vLLM이 model.safetensors만 읽고 정상 로드된다.
    """
    import json

    # ── 1. adapter_config.json 제거 (PEFT 모델로 오인식 방지) ─────────
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        adapter_config.unlink()
        print(f"  adapter_config.json 제거")

    # ── 2. adapter_model.safetensors 제거 (LoRA 어댑터 원본 파일) ─────
    for adapter_shard in model_dir.glob("adapter_model*.safetensors"):
        adapter_shard.unlink()
        print(f"  {adapter_shard.name} 제거")

    # ── 3. config.json quantization_config 제거 ──────────────────────
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.pop("quantization_config", None) is not None:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"  config.json quantization_config 제거")


def _patch_transformers_validation() -> None:
    """transformers 5.x + huggingface_hub strict 검증 우회.

    kanana-nano 등 Llama 기반 한국어 모델은 hidden_size/num_heads 비율이
    표준 LLaMA 규격(정수배)을 벗어나지만 실제 GQA 연산에는 문제 없음.
    huggingface_hub @strict 데코레이터는 validate_* 메서드를
    __class_validators__ 리스트에 함수 참조로 저장하므로,
    메서드 교체가 아니라 리스트에서 직접 제거해야 패치가 적용됩니다.
    """
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
        if hasattr(LlamaConfig, "__class_validators__"):
            LlamaConfig.__class_validators__ = [
                v for v in LlamaConfig.__class_validators__
                if getattr(v, "__name__", "") != "validate_architecture"
            ]
    except Exception:
        pass


def _fix_missing_input_embeddings(model) -> None:  # type: ignore[type-arg]
    """get_input_embeddings가 구현되지 않은 모델 패치 (EXAONE-Deep 등).

    PEFT는 tied-weight 처리를 위해 model.get_input_embeddings()를 호출하는데,
    일부 커스텀 모델(trust_remote_code)은 이 메서드를 구현하지 않음.
    embed_tokens / wte 레이어를 탐색해 직접 연결한다.
    """
    try:
        model.get_input_embeddings()
        return  # 정상 동작 → 스킵
    except (NotImplementedError, AttributeError):
        pass

    emb = None
    for name, module in model.named_modules():
        if name.endswith(("embed_tokens", "wte")) and hasattr(module, "weight"):
            emb = module
            break

    if emb is None:
        return  # 임베딩 레이어 탐색 실패 → 그대로 진행

    model.get_input_embeddings = lambda: emb  # type: ignore[method-assign]
    # 내부 base model(model.model / model.transformer)도 패치
    for attr in ("model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None:
            try:
                sub.get_input_embeddings()
            except (NotImplementedError, AttributeError):
                sub.get_input_embeddings = lambda: emb  # type: ignore[method-assign]


def _get_lora_target_modules(model) -> list[str]:  # type: ignore[type-arg]
    """모델 아키텍처에 맞는 LoRA target_modules 반환.

    Gemma4는 선형 레이어를 Gemma4ClippableLinear로 래핑하므로
    PEFT가 지원하는 실제 Linear 레이어인 내부 .linear를 타겟해야 함.
    다른 모델(LLaMA 계열)은 q_proj 등을 직접 타겟.
    """
    base = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    for _, module in model.named_modules():
        if type(module).__name__ == "Gemma4ClippableLinear":
            print("  [Gemma4] LoRA target: q_proj.linear 등 내부 linear 레이어 사용")
            return [f"{m}.linear" for m in base]

    return base


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"필수 패키지 미설치: {e}")
        print("pip install peft trl bitsandbytes accelerate datasets")
        raise

    # kanana 등 Llama 기반 모델의 transformers 5.x strict 검증 우회
    _patch_transformers_validation()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 준비 ──────────────────────────────────────────────
    print("데이터셋 생성 중...")
    train_data, val_data = build_dataset(
        args.qa_path,
        args.corpus_path,
        max_context_chars=args.max_context_chars,
        val_ratio=args.val_ratio,
    )
    print(f"  train: {len(train_data)}개 / val: {len(val_data)}개")

    # JSONL 저장 (재사용 가능)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        with open(data_dir / f"{split_name}.jsonl", "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  데이터 저장: {data_dir}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    # ── 모델 로드 ─────────────────────────────────────────────────
    print(f"모델 로드: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_length  # trl 1.x: max_seq_length → tokenizer에 직접 설정

    load_kwargs: dict = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
    }

    if args.qlora:
        # 4-bit 양자화 (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = bnb_config
        print("  QLoRA (4-bit) 모드 활성화")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    _fix_missing_input_embeddings(model)
    model.enable_input_require_grads()

    # ── LoRA 설정 ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_get_lora_target_modules(model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 학습 설정 ─────────────────────────────────────────────────
    use_eval = val_dataset is not None
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch" if use_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,                          # 디스크 절약: 최대 2개 체크포인트만 유지
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    callbacks = []
    if use_eval and args.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience))
        print(f"  EarlyStopping 활성화 (patience={args.early_stop_patience} epochs)")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
    )

    # Gemma3/4: transformers 5.x에서 학습 시 token_type_ids 필수
    # SFTTrainer가 생성하지 않으므로 collator를 래핑해 all-zero로 주입
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if "gemma" in model_type.lower():
        _orig_collator = trainer.data_collator

        def _gemma_collator(features):
            batch = _orig_collator(features)
            if "token_type_ids" not in batch:
                batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

        trainer.data_collator = _gemma_collator
        print(f"  [{model_type}] token_type_ids collator 패치 적용")

    print("\n학습 시작...")
    trainer.train()

    # ── 저장: 어댑터 → final/ 저장 → 스트리밍 merge ──────────────────────
    # QLoRA/LoRA 학습 후 merge 전략:
    #   스트리밍 merge: 기반 모델을 mmap으로 읽고 LoRA 델타를 CPU에서 계산.
    #   GPU/RAM 상관없이 동작하며, 15B+ 대형 모델도 OOM 없이 처리.
    #   폴백: transformers 기반 GPU reload merge (소형 모델 또는 sharded 미지원 시).

    import gc
    import sys as _sys

    _scripts_dir = str(Path(__file__).parent.parent)
    if _scripts_dir not in _sys.path:
        _sys.path.insert(0, _scripts_dir)
    from scripts.repair_finetuned_models import (  # type: ignore
        _stream_merge_and_save, _remove_adapter_artifacts, _save_sharded,
    )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 어댑터를 final/ 에 저장 (merge 실패해도 repair_finetuned_models.py로 복구 가능)
    print(f"\n[1/2] LoRA 어댑터 저장 중: {final_dir}")
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Step 2: 학습 모델/트레이너 해제 후 스트리밍 merge
    print("\n[2/2] GPU 메모리 해제 후 스트리밍 merge 중...")
    _train_model = trainer.model
    del trainer, _train_model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 기존 model 샤드 삭제 (재학습 시 구버전 파일이 남아 디스크를 2배 소비하는 것 방지)
    for _old_shard in final_dir.glob("model-*.safetensors"):
        _old_shard.unlink()
    _old_index = final_dir / "model.safetensors.index.json"
    if _old_index.exists():
        _old_index.unlink()
    _old_single = final_dir / "model.safetensors"
    if _old_single.exists():
        _old_single.unlink()

    stream_result = _stream_merge_and_save(final_dir, args.model_path)

    if stream_result is None:
        # 폴백: GPU 재로드 merge (스트리밍 미지원 모델 구조)
        print(f"  [폴백] GPU 재로드 merge 시도...")
        from peft import PeftModel
        _dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _base = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=_dtype,
            device_map={"": "cuda:0"} if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=args.trust_remote_code,
        )
        _peft = PeftModel.from_pretrained(_base, str(final_dir))
        merged = _peft.merge_and_unload()
        if hasattr(merged, "config") and hasattr(merged.config, "quantization_config"):
            merged.config.quantization_config = None
        _save_sharded(merged, final_dir)
        tokenizer.save_pretrained(str(final_dir))
        del merged, _peft, _base
        gc.collect()
        torch.cuda.empty_cache()
    elif stream_result is False:
        print(f"  [경고] merge 실패 — 어댑터만 저장된 상태. repair_finetuned_models.py로 나중에 복구 가능.")

    # 저장 후처리: quantization_config 잔류 / adapter 파일 제거
    print("\n저장 후처리 (vLLM 호환성 보장)...")
    _cleanup_merged_model(final_dir)

    # save_pretrained()가 복사하지 않는 베이스 모델 설정 파일을 final/에 보완 복사.
    # - *.py: trust_remote_code 모델 아키텍처 코드 (exaone 등)
    # - preprocessor_config.json: 멀티모달 이미지 프로세서 (gemma3 등)
    # - *.model / sentencepiece 등 기타 프로세서 파일
    # 가중치(*.safetensors, *.bin)와 이미 존재하는 파일은 건너뜀.
    import shutil as _shutil
    _WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".ckpt"}
    _copied = []
    for _src in Path(args.model_path).iterdir():
        if _src.suffix in _WEIGHT_SUFFIXES or not _src.is_file():
            continue
        _dst = final_dir / _src.name
        if not _dst.exists():
            _shutil.copy2(_src, _dst)
            _copied.append(_src.name)
    if _copied:
        print(f"  베이스 모델 설정 파일 {len(_copied)}개 보완 복사: {_copied}")

    # Step 3: 체크포인트 삭제 (디스크 절약)
    _ckpt_deleted = 0
    for _ckpt in output_dir.glob("checkpoint-*"):
        if _ckpt.is_dir():
            import shutil
            shutil.rmtree(_ckpt)
            _ckpt_deleted += 1
    if _ckpt_deleted:
        print(f"\n체크포인트 {_ckpt_deleted}개 삭제 완료 (디스크 절약)")

    print(f"\n학습 완료. 저장 경로: {final_dir}")
    print("\nvLLM 서빙 방법:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {final_dir} --port 8001")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 모델 LoRA/QLoRA 파인튜닝")
    parser.add_argument("--model-path", required=True, help="모델 경로 또는 HuggingFace 모델 ID")
    parser.add_argument("--output-dir", required=True, help="학습 결과 저장 디렉토리")
    parser.add_argument("--qa-path", default="data/autorag_csv/qa.parquet",
                        help="QA parquet 경로 (기본: data/autorag_csv/qa.parquet)")
    parser.add_argument("--corpus-path", default="data/autorag_csv/corpus.parquet",
                        help="Corpus parquet 경로 (기본: data/autorag_csv/corpus.parquet)")
    parser.add_argument("--qlora", action="store_true", help="4-bit QLoRA 사용 (VRAM 절약)")
    parser.add_argument("--trust-remote-code", action="store_true", help="EXAONE 등 커스텀 코드 모델")
    parser.add_argument("--epochs", type=int, default=10,
                        help="최대 학습 epoch 수 (early stop 시 조기 종료, 기본: 10)")
    parser.add_argument("--early-stop-patience", type=int, default=3,
                        help="eval_loss 개선 없이 허용할 epoch 수 (0=비활성화, 기본: 3)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-context-chars", type=int, default=2000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
