"""
CSV 기반 AutoRAG 데이터셋 생성 스크립트.

data_list_cleaned.csv → corpus.parquet + qa.parquet

기존 prepare_autorag_data.py의 핵심 문제점을 해결:
 1. retrieval_gt: 토큰 매칭 대신 공고번호/doc_id 직접 연결 → 100% 정확
 2. generation_gt: 키워드 목록 대신 실제 '사업 요약' 텍스트 사용 → METEOR/ROUGE 점수 정상화
 3. 청크 설계: chunk_0000에 사업 요약을 보장 → retrieval_gt가 항상 정답 청크를 가리킴

실행 예시:
  python scripts/prepare_autorag_from_csv.py \\
    --csv-path /srv/shared_data/datasets/data_list_cleaned.csv \\
    --output-dir data/autorag_csv \\
    --chunk-size 600 \\
    --chunk-overlap 100
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import naive_chunk
from src.document_loader import clean_text, apply_filter


# ─────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """null 바이트 및 제어 문자 제거."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\x00", "")
    return text.strip()


def _make_base_id(row: pd.Series, idx: int) -> str:
    """CSV 행에서 고유한 base_id를 생성한다."""
    공고번호 = str(row.get("공고 번호", "")).strip()
    if 공고번호 and 공고번호 not in ("nan", ""):
        return _sanitize(공고번호)
    발주기관 = _sanitize(str(row.get("발주 기관", "")))[:20]
    사업명 = _sanitize(str(row.get("사업명", "")))[:20]
    return f"{발주기관}_{사업명}_{idx:04d}"


# ─────────────────────────────────────────────────────────────────
# 다양한 질문 템플릿 (평가 다양성 확보)
# ─────────────────────────────────────────────────────────────────

# (템플릿, 타입, 답변소스)
# 답변소스: "summary"=사업 요약, "both"=사업 요약 + 핵심 키워드
QUESTION_TEMPLATES: list[tuple[str, str]] = [
    (
        "{발주기관}이 발주한 '{사업명}' 사업을 추진하는 목적과 배경은 무엇인가요?",
        "purpose",
    ),
    (
        "{발주기관}의 '{사업명}'에서 요구하는 주요 사항과 사업 범위를 설명하세요.",
        "scope",
    ),
    (
        "'{사업명}' 사업의 기대효과는 무엇인가요?",
        "effect",
    ),
]


# ─────────────────────────────────────────────────────────────────
# Corpus 생성
# ─────────────────────────────────────────────────────────────────

def build_corpus(df: pd.DataFrame, csv_path: str, chunk_size: int, chunk_overlap: int) -> pd.DataFrame:
    """
    CSV 행 → corpus.parquet rows.

    청크 전략:
      - chunk_0000: 사업 요약 전용 (항상 고정)
      - chunk_0001 이상: 상세 텍스트를 naive_chunk로 분할
    이 설계로 retrieval_gt = [base_id::chunk_0000] 이면 항상 사업 요약을 가리킨다.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []

    for idx, row in df.iterrows():
        base_id = _make_base_id(row, idx)
        발주기관 = _sanitize(str(row.get("발주 기관", "")))
        사업명 = _sanitize(str(row.get("사업명", "")))
        사업금액 = _sanitize(str(row.get("사업 금액", "")))
        요약 = _sanitize(str(row.get("사업 요약", "")))
        텍스트 = _sanitize(str(row.get("텍스트", "")))

        meta_base = {
            "last_modified_datetime": now,
            "filename": base_id,
            "발주기관": 발주기관,
            "사업명": 사업명,
            "사업금액": 사업금액,
        }

        # ── chunk_0000: 사업 요약 (필수 고정 청크) ──────────────────
        summary_contents = f"[발주기관] {발주기관}\n[사업명] {사업명}\n[사업 요약]\n{요약}"
        summary_contents = _sanitize(clean_text(apply_filter(summary_contents)))

        rows.append({
            "doc_id": f"{base_id}::chunk_0000",
            "contents": summary_contents,
            "metadata": {**meta_base, "chunk_index": 0, "chunk_type": "summary"},
            "path": csv_path,
            "start_end_idx": (0, len(summary_contents)),
        })

        # ── chunk_0001+: 상세 텍스트 분할 ───────────────────────────
        detail_text = _sanitize(clean_text(apply_filter(텍스트)))
        if len(detail_text) > chunk_size:
            sub_chunks = naive_chunk(detail_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for sc_idx, sc in enumerate(sub_chunks, start=1):
                sc_text = _sanitize(sc["text"])
                if not sc_text:
                    continue
                rows.append({
                    "doc_id": f"{base_id}::chunk_{sc_idx:04d}",
                    "contents": sc_text,
                    "metadata": {**meta_base, "chunk_index": sc_idx, "chunk_type": "detail"},
                    "path": csv_path,
                    "start_end_idx": (0, len(sc_text)),
                })
        elif detail_text and detail_text != summary_contents:
            # 짧은 텍스트는 chunk_0001 하나로
            rows.append({
                "doc_id": f"{base_id}::chunk_0001",
                "contents": detail_text,
                "metadata": {**meta_base, "chunk_index": 1, "chunk_type": "detail"},
                "path": csv_path,
                "start_end_idx": (0, len(detail_text)),
            })

    corpus_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["doc_id"])
        .reset_index(drop=True)
    )
    return corpus_df


# ─────────────────────────────────────────────────────────────────
# QA 생성
# ─────────────────────────────────────────────────────────────────

def build_qa(df: pd.DataFrame, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV 행 → qa.parquet rows.

    핵심 원칙:
      - generation_gt: 실제 '사업 요약' 텍스트 (METEOR/ROUGE 계산 가능)
      - retrieval_gt: base_id::chunk_0000 (사업 요약 청크를 정확히 가리킴)
    """
    rows: list[dict] = []

    for idx, row in df.iterrows():
        base_id = _make_base_id(row, idx)
        발주기관 = _sanitize(str(row.get("발주 기관", "")))
        사업명 = _sanitize(str(row.get("사업명", "")))
        요약 = _sanitize(str(row.get("사업 요약", "")))

        if not 요약 or 요약 == "nan":
            continue

        # 이 문서의 summary 청크가 corpus에 있는지 확인
        summary_doc_id = f"{base_id}::chunk_0000"
        if summary_doc_id not in corpus_df["doc_id"].values:
            continue

        # retrieval_gt: AutoRAG 형식 = [[...]] (list of list of doc_ids)
        # 사업 요약 청크 + 상세 첫 번째 청크도 포함 (더 나은 컨텍스트)
        detail_doc_id = f"{base_id}::chunk_0001"
        if detail_doc_id in corpus_df["doc_id"].values:
            retrieval_gt = [[summary_doc_id, detail_doc_id]]
        else:
            retrieval_gt = [[summary_doc_id]]

        # generation_gt: 실제 사업 요약 텍스트
        generation_gt = [요약]

        for template, q_type in QUESTION_TEMPLATES:
            question = template.format(발주기관=발주기관, 사업명=사업명)
            qid = f"{base_id}_{q_type}"

            rows.append({
                "qid": qid,
                "query": question,
                "retrieval_gt": retrieval_gt,
                "generation_gt": generation_gt,
            })

    qa_df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["qid"])
        .reset_index(drop=True)
    )
    return qa_df


# ─────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSV 기반 AutoRAG corpus/qa 파일 생성."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/srv/shared_data/datasets/data_list_cleaned.csv",
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/autorag_csv",
        help="출력 디렉토리 (corpus.parquet, qa.parquet 저장 위치)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="상세 텍스트 청크 크기 (문자 수). 기본 600.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="청크 간 중첩 크기. 기본 100.",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=50,
        help="이보다 짧은 텍스트 행은 스킵. 기본 50.",
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 및 정제 ──────────────────────────────────────
    print(f"[1/3] CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.dropna(subset=["텍스트"])
    df = df[df["텍스트"].astype(str).str.len() >= args.min_text_len]
    df = df.dropna(subset=["사업 요약"])
    df = df[df["사업 요약"].astype(str).str.len() >= 20]
    df = df.reset_index(drop=True)
    print(f"  전체 {before}행 → 유효 {len(df)}행 (텍스트/요약 존재)")

    # ── Corpus 생성 ───────────────────────────────────────────────
    print(f"\n[2/3] Corpus 생성 (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")
    corpus_df = build_corpus(df, csv_path, args.chunk_size, args.chunk_overlap)
    corpus_path = output_dir / "corpus.parquet"
    corpus_df.to_parquet(corpus_path, index=False)
    print(f"  corpus rows: {len(corpus_df)} → {corpus_path}")

    summary_chunks = corpus_df[corpus_df["doc_id"].str.endswith("::chunk_0000")]
    detail_chunks = corpus_df[~corpus_df["doc_id"].str.endswith("::chunk_0000")]
    print(f"  summary 청크(chunk_0000): {len(summary_chunks)}개")
    print(f"  detail 청크(chunk_0001+): {len(detail_chunks)}개")

    # ── QA 생성 ──────────────────────────────────────────────────
    print(f"\n[3/3] QA 생성")
    qa_df = build_qa(df, corpus_df)
    qa_path = output_dir / "qa.parquet"
    qa_df.to_parquet(qa_path, index=False)
    print(f"  qa rows: {len(qa_df)} → {qa_path}")
    print(f"  문서당 {len(QUESTION_TEMPLATES)}개 질문 × {len(df)}개 문서 = 최대 {len(df)*len(QUESTION_TEMPLATES)}개")

    # ── 검증 출력 ────────────────────────────────────────────────
    print("\n=== 샘플 QA 확인 ===")
    sample = qa_df.head(3)
    for _, r in sample.iterrows():
        print(f"  qid: {r['qid']}")
        print(f"  query: {r['query']}")
        print(f"  retrieval_gt: {r['retrieval_gt']}")
        print(f"  generation_gt[:80]: {str(r['generation_gt'])[:80]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
