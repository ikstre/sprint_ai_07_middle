"""
OpenAI Fine-tuning 스크립트

QA 데이터(qa.parquet + corpus.parquet)로 OpenAI 모델을 파인튜닝합니다.
지원 모델: gpt-4o-mini-2024-07-18, gpt-4.1-mini, gpt-3.5-turbo-0125 등

사전 조건:
    OPENAI_API_KEY 환경변수 설정 (또는 .env 파일)

실행 단계:
    # 1. JSONL 데이터셋 생성 후 업로드하고 fine-tuning 작업 시작
    python scripts/finetune_openai.py start \
        --model gpt-4o-mini-2024-07-18 \
        --output-dir models/finetuned/openai

    # 2. 진행 상태 확인
    python scripts/finetune_openai.py status --job-id ftjob-xxxx

    # 3. 완료 후 tutorial.yaml의 llm을 fine-tuned 모델 ID로 교체
    #    예) llm: ft:gpt-4o-mini-2024-07-18:org:rag:xxxx
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
# 데이터셋 생성 (finetune_local.py 와 동일 포맷)
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


def build_jsonl_dataset(
    qa_path: str,
    corpus_path: str,
    output_path: Path,
    max_context_chars: int = 3000,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path]:
    """OpenAI fine-tuning 형식(JSONL)으로 데이터셋 생성."""
    qa_df = pd.read_parquet(qa_path)
    corpus_df = pd.read_parquet(corpus_path)
    corpus_map: dict[str, str] = dict(zip(corpus_df["doc_id"], corpus_df["contents"]))

    examples: list[dict] = []
    for _, row in qa_df.iterrows():
        query: str = row["query"]
        generation_gt = row["generation_gt"]
        retrieval_gt = row["retrieval_gt"]

        if isinstance(generation_gt, list):
            answer = " ".join(str(g) for g in generation_gt)
        else:
            answer = str(generation_gt)

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
                context_parts.append(text[:800])

        context = "\n---\n".join(context_parts)[:max_context_chars]
        if not context:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT_TEMPLATE.format(context=context, query=query)},
                {"role": "assistant", "content": answer},
            ]
        })

    random.seed(seed)
    random.shuffle(examples)
    split = max(1, int(len(examples) * val_ratio))
    train_examples = examples[split:]
    val_examples = examples[:split]

    output_path.mkdir(parents=True, exist_ok=True)
    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    for fpath, data in [(train_file, train_examples), (val_file, val_examples)]:
        with open(fpath, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"train: {len(train_examples)}개 → {train_file}")
    print(f"val:   {len(val_examples)}개 → {val_file}")
    return train_file, val_file


# ─────────────────────────────────────────────────────────────────
# OpenAI Fine-tuning API
# ─────────────────────────────────────────────────────────────────

def upload_file(client, file_path: Path) -> str:
    print(f"파일 업로드: {file_path} ({file_path.stat().st_size / 1024:.1f} KB)")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose="fine-tune")
    print(f"  파일 ID: {response.id}")
    return response.id


def start_finetuning(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI()
    output_dir = Path(args.output_dir)

    # 데이터셋 생성
    print("데이터셋 생성 중...")
    train_file, val_file = build_jsonl_dataset(
        args.qa_path,
        args.corpus_path,
        output_dir / "data",
        max_context_chars=args.max_context_chars,
        val_ratio=args.val_ratio,
    )

    # 파일 업로드
    train_id = upload_file(client, train_file)
    val_id = upload_file(client, val_file)

    # Fine-tuning 작업 생성
    hyperparams: dict = {}
    if args.epochs:
        hyperparams["n_epochs"] = args.epochs
    if args.batch_size:
        hyperparams["batch_size"] = args.batch_size
    if args.lr_multiplier:
        hyperparams["learning_rate_multiplier"] = args.lr_multiplier

    print(f"\nFine-tuning 작업 생성: {args.model}")
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=val_id,
        model=args.model,
        hyperparameters=hyperparams if hyperparams else None,
        suffix=args.suffix or "rag",
    )

    print(f"  작업 ID: {job.id}")
    print(f"  상태:    {job.status}")

    # 상태 정보 저장
    info = {
        "job_id": job.id,
        "model": args.model,
        "status": job.status,
        "train_file_id": train_id,
        "val_file_id": val_id,
    }
    info_path = output_dir / "job_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n작업 정보 저장: {info_path}")
    print(f"\n상태 확인 명령어:")
    print(f"  python scripts/finetune_openai.py status --job-id {job.id}")

    if args.wait:
        _wait_for_job(client, job.id, output_dir)


def _wait_for_job(client, job_id: str, output_dir: Path) -> None:
    print("\n완료 대기 중... (Ctrl+C로 중단 가능, 작업은 계속 실행됨)")
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  [{time.strftime('%H:%M:%S')}] 상태: {job.status}", end="")

        if job.status == "succeeded":
            print(f"\n\n파인튜닝 완료!")
            print(f"모델 ID: {job.fine_tuned_model}")
            print(f"\ntutorial.yaml 적용 방법:")
            print(f"  llm: {job.fine_tuned_model}")

            result = {"fine_tuned_model": job.fine_tuned_model, "job_id": job_id}
            with open(output_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)
            break
        elif job.status in ("failed", "cancelled"):
            print(f"\n\n실패: {job.status}")
            if job.error:
                print(f"오류: {job.error}")
            break
        else:
            print(" (60초 후 재확인...)", end="\r")
            time.sleep(60)


def check_status(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI()
    job = client.fine_tuning.jobs.retrieve(args.job_id)

    print(f"작업 ID:  {job.id}")
    print(f"모델:     {job.model}")
    print(f"상태:     {job.status}")
    print(f"생성 시각: {job.created_at}")

    if job.fine_tuned_model:
        print(f"\n완료 모델: {job.fine_tuned_model}")
        print(f"\ntutorial.yaml 적용 방법:")
        print(f"  llm: {job.fine_tuned_model}")

    # 최근 이벤트 출력
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=args.job_id, limit=10)
    print("\n최근 이벤트:")
    for event in reversed(list(events)):
        print(f"  [{event.created_at}] {event.message}")


def list_jobs(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI()
    jobs = client.fine_tuning.jobs.list(limit=args.limit)
    print(f"{'ID':<30} {'모델':<35} {'상태':<15} {'완료 모델'}")
    print("-" * 100)
    for job in jobs:
        ft_model = job.fine_tuned_model or "-"
        print(f"{job.id:<30} {job.model:<35} {job.status:<15} {ft_model}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI Fine-tuning")
    sub = parser.add_subparsers(dest="command", required=True)

    # start 서브커맨드
    p_start = sub.add_parser("start", help="파인튜닝 시작")
    p_start.add_argument("--model", default="gpt-4o-mini-2024-07-18",
                         help="베이스 모델 (기본: gpt-4o-mini-2024-07-18)")
    p_start.add_argument("--output-dir", default="models/finetuned/openai")
    p_start.add_argument("--qa-path", default="data/autorag/qa.parquet")
    p_start.add_argument("--corpus-path", default="data/autorag/corpus.parquet")
    p_start.add_argument("--epochs", type=int, default=None, help="학습 epoch 수 (미지정 시 OpenAI 자동)")
    p_start.add_argument("--batch-size", type=int, default=None)
    p_start.add_argument("--lr-multiplier", type=float, default=None)
    p_start.add_argument("--suffix", type=str, default="rag", help="모델 이름 접미사")
    p_start.add_argument("--max-context-chars", type=int, default=3000)
    p_start.add_argument("--val-ratio", type=float, default=0.1)
    p_start.add_argument("--wait", action="store_true", help="완료까지 대기")

    # status 서브커맨드
    p_status = sub.add_parser("status", help="작업 상태 확인")
    p_status.add_argument("--job-id", required=True)

    # list 서브커맨드
    p_list = sub.add_parser("list", help="파인튜닝 작업 목록")
    p_list.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if args.command == "start":
        start_finetuning(args)
    elif args.command == "status":
        check_status(args)
    elif args.command == "list":
        list_jobs(args)


if __name__ == "__main__":
    main()
