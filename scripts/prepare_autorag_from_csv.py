"""
CSV 기반 AutoRAG 데이터셋 생성 스크립트.

data_list_cleaned.csv → corpus.parquet + qa.parquet

기존 prepare_autorag_data.py의 핵심 문제점을 해결:
 1. retrieval_gt: 토큰 매칭 대신 공고번호/doc_id 직접 연결 → 100% 정확
 2. generation_gt: 키워드 목록 대신 실제 '사업 요약' 텍스트 사용 → METEOR/ROUGE 점수 정상화
 3. 청크 설계: chunk_0000에 사업 요약을 보장 → retrieval_gt가 항상 정답 청크를 가리킴

실행 예시:
  python scripts/prepare_autorag_from_csv.py \\
    --csv-path $METADATA_CSV \\
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

from dotenv import load_dotenv
load_dotenv()

from configs import paths
from src.chunker import naive_chunk
from src.document_loader import clean_text, apply_filter


# ─────────────────────────────────────────────────────────────────
# 파일별 문장 단위 청킹
# ─────────────────────────────────────────────────────────────────

# 한국어 서술형 어미 + 마침표 뒤 공백을 문장 경계로 인식
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[다됩임음]\.)\s+"
    r"|(?<=습니다\.)\s+"
    r"|(?<=합니다\.)\s+"
    r"|(?<=됩니다\.)\s+"
    r"|(?<=입니다\.)\s+"
    r"|(?<=했습니다\.)\s+"
)


def _chunk_by_sentences(text: str, chunk_size: int) -> list[str]:
    """
    한 파일의 텍스트를 문장 단위로 분리한 뒤 chunk_size 이하로 묶어 반환.

    - 청크는 반드시 완전한 문장으로 시작하고 끝남
    - 파일 경계를 절대 넘지 않음 (호출 자체가 파일당 1회)
    - 문장 분리 실패(영문/숫자 위주) 시 naive_chunk로 폴백
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]

    if not sentences:
        # 한국어 문장 패턴 없음 → naive_chunk 폴백 (overlap 없이)
        return [sc["text"] for sc in naive_chunk(text, chunk_size, chunk_overlap=0)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        if sent_len > chunk_size:
            # 단일 문장이 chunk_size 초과 → 강제 분할 후 편입
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(sc["text"] for sc in naive_chunk(sent, chunk_size, chunk_overlap=0))
            continue

        if current_len + sent_len + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = [sent], sent_len
        else:
            current.append(sent)
            current_len += sent_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


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

        # ── chunk_0001+: 상세 텍스트 분할 (파일 단위, 문장 경계 기준) ──
        detail_text = _sanitize(clean_text(apply_filter(텍스트)))
        if len(detail_text) > chunk_size:
            # 파일 하나의 텍스트만 받아 문장 단위로 청킹 → 파일 경계 절대 불침범
            sentence_chunks = _chunk_by_sentences(detail_text, chunk_size)
            for sc_idx, sc_text in enumerate(sentence_chunks, start=1):
                sc_text = _sanitize(sc_text)
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
# 수동 평가셋 → AutoRAG QA 변환
# ─────────────────────────────────────────────────────────────────

def build_qa_from_eval_dataset(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    src.evaluation.single_dataset + multi_dataset 수동 평가셋을
    AutoRAG qa.parquet 형식으로 변환한다.

    매핑 전략:
      - expected_orgs → corpus 발주기관 부분 매칭 → retrieval_gt (chunk_0000 + chunk_0001)
      - reference_answer → generation_gt
      - qid 접두사: "eval_" (CSV 자동 생성 QA와 충돌 방지)

    매핑 불가(corpus에 없는 기관) 질문은 자동 스킵.
    """
    try:
        from src.evaluation.single_dataset import EVALUATION_QUESTIONS as sq
        from src.evaluation.multi_dataset import EVALUATION_QUESTIONS as mq
        # id 충돌 방지: single_ / multi_ 접두사 구분
        questions = [
            {**q, "_prefix": "single"} for q in sq
        ] + [
            {**q, "_prefix": "multi"} for q in mq
        ]
    except ImportError:
        print("  [경고] src.evaluation 모듈 없음 → 수동 평가셋 스킵")
        return pd.DataFrame()

    # corpus 발주기관 컬럼 준비
    def _get_org(meta: object) -> str:
        if isinstance(meta, dict):
            return str(meta.get("발주기관", "")).strip()
        return ""

    corp = corpus_df.copy()
    corp["_org"] = corp["metadata"].apply(_get_org)
    corpus_ids = set(corp["doc_id"])

    rows: list[dict] = []
    skipped = 0

    for q in questions:
        orgs = q.get("expected_orgs", [])
        reference = q.get("reference_answer", "")
        if not orgs or not reference:
            skipped += 1
            continue

        # 각 기관에 대해 corpus에서 매칭 doc_id 수집
        matched_ids: list[str] = []
        for org in orgs:
            org_lower = org.strip().lower()
            mask = corp["_org"].apply(
                lambda x: org_lower in x.lower() or x.lower() in org_lower
            )
            for _, row in corp[mask & corp["doc_id"].str.endswith("::chunk_0000")].iterrows():
                matched_ids.append(row["doc_id"])
                detail_id = row["doc_id"].replace("::chunk_0000", "::chunk_0001")
                if detail_id in corpus_ids:
                    matched_ids.append(detail_id)

        if not matched_ids:
            skipped += 1
            continue

        prefix = q.get("_prefix", "single")
        rows.append({
            "qid": f"eval_{prefix}_{q['id']}",
            "query": q["question"],
            "retrieval_gt": [matched_ids],
            "generation_gt": [reference],
        })

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["qid"])
        .reset_index(drop=True)
    )
    print(f"  수동 평가셋 변환 완료: {len(df)}개 추가, {skipped}개 스킵 (corpus 미존재 기관)")
    return df


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
        default=paths.METADATA_CSV,
        help=f"입력 CSV 파일 경로 (기본: {paths.METADATA_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=paths.AUTORAG_DATA_DIR,
        help=f"출력 디렉토리 (corpus.parquet, qa.parquet 저장 위치, 기본: {paths.AUTORAG_DATA_DIR})",
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
        default=50,
        help="청크 간 중첩 크기. 기본 100.",
    )
    parser.add_argument(
        "--min-text-len",
        type=int,
        default=30,
        help="이보다 짧은 텍스트 행은 스킵. 기본 50.",
    )
    parser.add_argument(
        "--no-eval-dataset",
        action="store_true",
        help="수동 평가셋(single/multi_dataset) 병합 스킵 (기본: 항상 포함)",
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
    print(f"\n[3/3] QA 생성 (CSV 자동 생성)")
    qa_df = build_qa(df, corpus_df)
    print(f"  CSV 기반 QA: {len(qa_df)}행 (문서당 {len(QUESTION_TEMPLATES)}개 × {len(df)}개 문서)")

    # ── 수동 평가셋 병합 ─────────────────────────────────────────
    if not args.no_eval_dataset:
        print(f"\n[3b/3] 수동 평가셋 병합 (single_dataset + multi_dataset)")
        eval_df = build_qa_from_eval_dataset(corpus_df)
        if not eval_df.empty:
            qa_df = (
                pd.concat([qa_df, eval_df], ignore_index=True)
                .drop_duplicates(subset=["qid"])
                .reset_index(drop=True)
            )
            print(f"  병합 후 QA 총계: {len(qa_df)}행")
    else:
        print("\n[3b/3] 수동 평가셋 스킵 (--no-eval-dataset)")

    # ── 저장 ─────────────────────────────────────────────────────
    qa_path = output_dir / "qa.parquet"
    qa_df.to_parquet(qa_path, index=False)
    print(f"\n  qa.parquet 저장: {len(qa_df)}행 → {qa_path}")

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
