"""
중앙 경로 설정.

.env 의 환경변수를 읽어 모든 경로를 한 곳에서 관리합니다.
우선순위: 환경변수(.env) > 기본값

─────────────────────────────────────────────────────────────
환경변수 목록 (.env 에 추가하면 즉시 반영)
─────────────────────────────────────────────────────────────
  SRV_DATA_DIR          공용 서버 루트          (기본: /srv/shared_data)
  METADATA_CSV          메타데이터 CSV           (기본: {SRV_DATA_DIR}/datasets/data_list_cleaned.csv)
  PDF_DIR               원본 PDF 디렉토리        (기본: {SRV_DATA_DIR}/pdf)

  MODEL_DIR             모델 루트               (기본: {SRV_DATA_DIR}/models)
  EMBEDDING_MODEL_PATH  임베딩 모델              (기본: {MODEL_DIR}/embeddings/BGE-m3-ko)
  CHAT_MODEL_PATH       기본 채팅 모델           (기본: {MODEL_DIR}/exaone/EXAONE-4.0-1.2B)

  VECTORDB_DIR          로컬 ChromaDB 디렉토리   (기본: data/vectordb)
  PROCESSED_DIR         전처리 결과 디렉토리     (기본: data/processed)
  AUTORAG_DATA_DIR      AutoRAG parquet 디렉토리 (기본: data/autorag_csv)
  AUTORAG_PROJECT_DIR   AutoRAG 평가 결과        (기본: evaluation/autorag_benchmark_csv)
  AUTORAG_GEMMA4_DIR    Gemma4 전용 결과         (기본: {AUTORAG_PROJECT_DIR}_gemma)
  EVAL_OUTPUT_DIR       run_evaluation 결과      (기본: evaluation)
─────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── 서버 루트 ──────────────────────────────────────────────────────────────
SRV_DATA_DIR: str = os.getenv("SRV_DATA_DIR", "/srv/shared_data")

# ── 데이터셋 ───────────────────────────────────────────────────────────────
# 프로젝트 로컬 CSV (재파싱된 본문 포함)가 있으면 우선 사용하고,
# 없으면 공용 서버의 구버전 CSV로 fallback 한다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_METADATA_CSV = _PROJECT_ROOT / "data" / "data_list_cleaned.csv"
METADATA_CSV: str = os.getenv(
    "METADATA_CSV",
    str(_LOCAL_METADATA_CSV)
    if _LOCAL_METADATA_CSV.exists()
    else f"{SRV_DATA_DIR}/datasets/data_list_cleaned.csv",
)
PDF_DIR: str = os.getenv("PDF_DIR", f"{SRV_DATA_DIR}/pdf")

# ── 모델 경로 ──────────────────────────────────────────────────────────────
MODEL_DIR: str = os.getenv("MODEL_DIR", f"{SRV_DATA_DIR}/models")

EMBEDDING_MODEL_PATH: str = os.getenv(
    "EMBEDDING_MODEL_PATH",
    f"{MODEL_DIR}/embeddings/BGE-m3-ko",
)
CHAT_MODEL_PATH: str = os.getenv(
    "CHAT_MODEL_PATH",
    f"{MODEL_DIR}/exaone/EXAONE-4.0-1.2B",
)

# ── 로컬 데이터 경로 ───────────────────────────────────────────────────────
VECTORDB_DIR: str = os.getenv("VECTORDB_DIR", "data/vectordb")
PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "data/processed")
AUTORAG_DATA_DIR: str = os.getenv("AUTORAG_DATA_DIR", "data/autorag_csv")

# ── 평가 경로 ──────────────────────────────────────────────────────────────
AUTORAG_PROJECT_DIR: str = os.getenv(
    "AUTORAG_PROJECT_DIR", "evaluation/autorag_benchmark_csv"
)
AUTORAG_GEMMA4_DIR: str = os.getenv(
    "AUTORAG_GEMMA4_DIR", f"{AUTORAG_PROJECT_DIR}_gemma"
)
EVAL_OUTPUT_DIR: str = os.getenv("EVAL_OUTPUT_DIR", "evaluation")
