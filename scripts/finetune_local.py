"""
로컬 모델 LoRA/QLoRA 파인튜닝 스크립트

QA 데이터(qa.parquet + corpus.parquet)로 RAG 응답 품질에 특화된
instruction-following 모델을 학습합니다.

사전 설치:
    pip install peft trl bitsandbytes accelerate

실행 예시:
    # LoRA (GPU 메모리 충분할 때)
    python scripts/finetune_local.py \
        --model-path /srv/shared_data/models/kanana/kanana-nano-2.1b \
        --output-dir models/finetuned/kanana-nano-rag \
        --epochs 3

    # QLoRA (8GB GPU 등 메모리 제한 환경)
    python scripts/finetune_local.py \
        --model-path /srv/shared_data/models/kanana/kanana-nano-2.1b \
        --output-dir models/finetuned/kanana-nano-rag \
        --qlora \
        --epochs 3

    # HuggingFace Hub 모델 (로컬 PC)
    python scripts/finetune_local.py \
        --model-path kakaocorp/kanana-nano-2.1b \
        --output-dir models/finetuned/kanana-nano-rag \
        --qlora
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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

        # 참조 문서 취합 (retrieval_gt는 리스트의 리스트)
        doc_ids: list[str] = []
        if isinstance(retrieval_gt, list):
            for group in retrieval_gt:
                if isinstance(group, list):
                    doc_ids.extend(group)
                else:
                    doc_ids.append(group)

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

def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"필수 패키지 미설치: {e}")
        print("pip install peft trl bitsandbytes accelerate datasets")
        raise

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

    load_kwargs: dict = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
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
    model.enable_input_require_grads()

    # ── LoRA 설정 ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 학습 설정 ─────────────────────────────────────────────────
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
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=val_dataset is not None,
        max_seq_length=args.max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\n학습 시작...")
    trainer.train()

    # ── 저장 ─────────────────────────────────────────────────────
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n학습 완료. 저장 경로: {output_dir / 'final'}")
    print("\nvLLM 서빙 방법:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {output_dir / 'final'} --port 8001")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 모델 LoRA/QLoRA 파인튜닝")
    parser.add_argument("--model-path", required=True, help="모델 경로 또는 HuggingFace 모델 ID")
    parser.add_argument("--output-dir", required=True, help="학습 결과 저장 디렉토리")
    parser.add_argument("--qa-path", default="data/autorag/qa.parquet")
    parser.add_argument("--corpus-path", default="data/autorag/corpus.parquet")
    parser.add_argument("--qlora", action="store_true", help="4-bit QLoRA 사용 (VRAM 절약)")
    parser.add_argument("--trust-remote-code", action="store_true", help="EXAONE 등 커스텀 코드 모델")
    parser.add_argument("--epochs", type=int, default=3)
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
