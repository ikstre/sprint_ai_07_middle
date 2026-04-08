"""
Prepare AutoRAG dataset files (corpus.parquet, qa.parquet) from local RFP data.

Output format:
- corpus.parquet: doc_id, contents, metadata, path, start_end_idx
- qa.parquet: qid, query, retrieval_gt, generation_gt
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import chunk_documents
from src.document_loader import DocumentLoader
from src.embedder import _sanitize
from src.evaluator import EVALUATION_QUESTIONS


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", text.lower()))


def _build_corpus_rows(chunks: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    rows: list[dict] = []

    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        filename = str(meta.get("filename", "unknown"))
        file_path = str(meta.get("file_path", ""))
        chunk_index = int(meta.get("chunk_index", idx))

        doc_id = f"{filename}::chunk_{chunk_index:04d}"
        contents = _sanitize(str(chunk.get("text", "")))
        metadata = {
            "last_modified_datetime": now.isoformat(),
            "filename": _sanitize(filename),
            "발주기관": _sanitize(str(meta.get("발주 기관", meta.get("발주기관", "")))),
            "사업명": _sanitize(str(meta.get("사업명", ""))),
            "사업금액": _sanitize(str(meta.get("사업 금액", meta.get("사업금액", "")))),
            "chunk_index": chunk_index,
        }

        rows.append(
            {
                "doc_id": _sanitize(doc_id),
                "contents": contents,
                "metadata": metadata,           # AutoRAG는 dict 객체를 요구 (JSON 문자열 불가)
                "path": _sanitize(file_path),
                "start_end_idx": (0, len(contents)),
            }
        )

    return rows


def _pick_retrieval_gt(question: str, corpus_df: pd.DataFrame, top_k: int = 3) -> list[str]:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return [str(corpus_df.iloc[0]["doc_id"])]

    scored: list[tuple[int, str]] = []
    for _, row in corpus_df.iterrows():
        title_hint = f"{row.get('path', '')} {row.get('metadata', '')}"
        doc_tokens = _tokenize(title_hint)

        overlap = len(q_tokens.intersection(doc_tokens))
        if overlap == 0:
            preview = str(row.get("contents", ""))[:1200]
            overlap = len(q_tokens.intersection(_tokenize(preview)))

        scored.append((overlap, str(row["doc_id"])))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [doc_id for score, doc_id in scored if score > 0][:top_k]
    if not picked:
        picked = [str(corpus_df.iloc[0]["doc_id"])]
    return picked


def _build_qa_rows(corpus_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []

    for item in EVALUATION_QUESTIONS:
        qid = str(item["id"])
        query = str(item["question"])
        retrieval_gt = [_pick_retrieval_gt(query, corpus_df)]

        expected_keywords = item.get("expected_keywords", [])
        if expected_keywords:
            gt_text = f"핵심 키워드: {', '.join(expected_keywords)}"
        elif item.get("expected_behavior") == "should_decline":
            gt_text = "문서 근거가 없으면 모른다고 답변"
        else:
            gt_text = "문서 근거 기반 요약 답변"

        rows.append(
            {
                "qid": qid,
                "query": query,
                "retrieval_gt": retrieval_gt,
                "generation_gt": [gt_text],
            }
        )

        follow_up = item.get("follow_up")
        if follow_up:
            fu_qid = f"{qid}_followup"
            fu_query = str(follow_up["question"])
            fu_retrieval_gt = [_pick_retrieval_gt(fu_query, corpus_df)]

            fu_keywords = follow_up.get("expected_keywords", [])
            fu_gt_text = f"핵심 키워드: {', '.join(fu_keywords)}" if fu_keywords else "문서 근거 기반 후속 답변"

            rows.append(
                {
                    "qid": fu_qid,
                    "query": fu_query,
                    "retrieval_gt": fu_retrieval_gt,
                    "generation_gt": [fu_gt_text],
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare qa.parquet/corpus.parquet for AutoRAG.")
    parser.add_argument("--documents-dir", type=str, default="data")
    parser.add_argument("--metadata-csv", type=str, default="data/data_list.csv")
    parser.add_argument("--output-dir", type=str, default="data/autorag")
    parser.add_argument("--chunk-method", type=str, default="semantic", choices=["naive", "semantic"])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--csv-text-columns", type=str, default=None,
        help="CSV 파일에서 본문으로 사용할 컬럼명 (쉼표 구분). 미지정 시 자동 감지.",
    )
    parser.add_argument(
        "--csv-row-per-doc", action="store_true",
        help="CSV 각 행을 개별 문서로 처리. 미지정 시 CSV 전체를 하나의 문서로 처리.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_text_columns = args.csv_text_columns.split(",") if args.csv_text_columns else None
    print("[1/3] Load documents...")
    loader = DocumentLoader(
        documents_dir=args.documents_dir,
        metadata_csv=args.metadata_csv,
        csv_text_columns=csv_text_columns,
        csv_row_per_doc=args.csv_row_per_doc,
    )
    documents = loader.load_all()
    if not documents:
        raise RuntimeError("No documents loaded. Check --documents-dir.")

    print("[2/3] Chunk documents...")
    chunks = chunk_documents(
        documents,
        method=args.chunk_method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    corpus_rows = _build_corpus_rows(chunks)
    corpus_df = pd.DataFrame(corpus_rows).drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    corpus_path = output_dir / "corpus.parquet"
    corpus_df.to_parquet(corpus_path, index=False)
    print(f"  corpus rows: {len(corpus_df)} -> {corpus_path}")

    print("[3/3] Build QA dataset...")
    qa_rows = _build_qa_rows(corpus_df)
    qa_df = pd.DataFrame(qa_rows).drop_duplicates(subset=["qid"]).reset_index(drop=True)

    qa_path = output_dir / "qa.parquet"
    qa_df.to_parquet(qa_path, index=False)
    print(f"  qa rows: {len(qa_df)} -> {qa_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

