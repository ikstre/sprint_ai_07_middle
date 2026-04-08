"""
문서 인덱싱 스크립트 (2단계)

  1단계: 문서 로딩 + 청킹 → data/processed/{collection}_chunks.json 저장
  2단계: 청크 로딩 → 임베딩 → ChromaDB 저장

사용법:
  python scripts/index_documents.py                          # 전체 실행
  python scripts/index_documents.py --step chunk            # 1단계만
  python scripts/index_documents.py --step embed            # 2단계만 (청크 파일 재사용)
  python scripts/index_documents.py --collection rfp_chunk800 --chunk-size 800
"""
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from configs.config import Config
from src.document_loader import DocumentLoader
from src.chunker import chunk_documents
from src.rag_pipeline import RAGPipeline
from src.embedder import _sanitize


def _sanitize_chunks(chunks: list[dict]) -> list[dict]:
    """청크 텍스트와 메타데이터에서 서로게이트 문자를 제거한다."""
    result = []
    for chunk in chunks:
        result.append({
            "text": _sanitize(chunk["text"]),
            "metadata": {k: _sanitize(str(v)) for k, v in chunk.get("metadata", {}).items()},
        })
    return result


def step_chunk(args, config: Config) -> Path:
    """1단계: 문서 로딩 → 청킹 → JSON 저장"""
    print("\n[1/2] 문서 로딩 중...")
    csv_text_columns = args.csv_text_columns.split(",") if getattr(args, "csv_text_columns", None) else None
    loader = DocumentLoader(
        documents_dir=config.documents_dir,
        metadata_csv=config.metadata_csv,
        csv_text_columns=csv_text_columns,
        csv_row_per_doc=getattr(args, "csv_row_per_doc", False),
    )
    documents = loader.load_all()

    if not documents:
        print("로딩된 문서가 없습니다. documents_dir 경로를 확인하세요.")
        sys.exit(1)

    print(f"  로딩 완료: {len(documents)}개 문서")

    print(f"\n[2/2 준비] 청킹 중 (방법: {config.chunking_method}, 크기: {config.chunk_size})...")
    chunks = chunk_documents(
        documents,
        method=config.chunking_method,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    chunks = _sanitize_chunks(chunks)

    # 청크 저장
    out_dir = Path(config.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = out_dir / f"{args.collection}_chunks.json"

    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n청크 저장 완료: {chunk_file}")
    print(f"  총 문서: {len(documents)}개 | 총 청크: {len(chunks)}개")
    return chunk_file


def step_embed(args, config: Config, chunk_file: Path):
    """2단계: 청크 로딩 → 임베딩 → ChromaDB 저장"""
    if not chunk_file.exists():
        print(f"청크 파일을 찾을 수 없습니다: {chunk_file}")
        print("먼저 --step chunk 를 실행하세요.")
        sys.exit(1)

    print(f"\n청크 파일 로딩: {chunk_file}")
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  청크 수: {len(chunks)}개")
    print(f"\n임베딩 생성 및 벡터스토어 저장 중...")

    pipeline = RAGPipeline(config)
    pipeline.build_index(
        chunks,
        collection_name=args.collection,
        use_batch_api=getattr(args, "use_batch_api", False),
    )

    print(f"\n벡터스토어 저장 완료")
    print(f"  컬렉션: {args.collection}")
    print(f"  저장 위치: {config.vectordb_dir}")


def main():
    parser = argparse.ArgumentParser(description="RFP 문서 인덱싱 (2단계)")
    parser.add_argument(
        "--step", type=str, default="all", choices=["all", "chunk", "embed"],
        help="실행 단계: all(전체) / chunk(청킹만) / embed(임베딩만)",
    )
    parser.add_argument(
        "--scenario", type=str, default="B", choices=["A", "B"],
        help="A: HuggingFace 로컬, B: OpenAI API (기본값: B)",
    )
    # Scenario A 전용: 로컬 모델 경로 선택
    _LOCAL_EMB = {
        "bge": "/srv/shared_data/models/embeddings/BGE-m3-ko",
        "sroberta": "/srv/shared_data/models/embeddings/ko-sroberta-multitask",
    }
    parser.add_argument(
        "--hf-embedding-model",
        type=str,
        default="bge",
        choices=list(_LOCAL_EMB.keys()),
        help=(
            "Scenario A 임베딩 모델 선택 (기본값: bge)\n"
            "  bge       → BGE-m3-ko (1024-dim, 한국어 특화)\n"
            "  sroberta  → ko-sroberta-multitask (768-dim, 경량)"
        ),
    )
    parser.add_argument(
        "--method", type=str, default="naive", choices=["naive", "semantic"],
        help="청킹 전략 (기본값: naive)",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--collection", type=str, default="rfp_chunk1200",
        help="ChromaDB 컬렉션 이름",
    )
    parser.add_argument(
        "--documents-dir", type=str, default="data",
        help="문서 디렉토리 경로",
    )
    parser.add_argument(
        "--use-batch-api", action="store_true",
        help="OpenAI Batch API 사용 (500개 이상 청크 시 비용 50%% 절감, 처리 시간 증가)",
    )
    # CSV/TXT 전용 옵션
    parser.add_argument(
        "--csv-text-columns", type=str, default=None,
        help="CSV 파일에서 본문으로 사용할 컬럼명 (쉼표 구분, 예: '제목,내용,요구사항'). 미지정 시 자동 감지.",
    )
    parser.add_argument(
        "--csv-row-per-doc", action="store_true",
        help="CSV 각 행을 개별 문서로 처리. 미지정 시 CSV 전체를 하나의 문서로 처리.",
    )
    args = parser.parse_args()

    # Scenario A 임베딩 모델 경로 및 차원 결정
    hf_emb_path = _LOCAL_EMB.get(args.hf_embedding_model, args.hf_embedding_model)
    hf_emb_dim = 768 if args.hf_embedding_model == "sroberta" else 1024

    print("=" * 60)
    print("RFP 문서 인덱싱")
    print(f"  단계      : {args.step}")
    print(f"  시나리오  : {args.scenario}")
    print(f"  청킹 전략 : {args.method} | 크기: {args.chunk_size} | 중첩: {args.chunk_overlap}")
    print(f"  컬렉션    : {args.collection}")
    print(f"  Batch API : {'ON' if args.use_batch_api else 'OFF'}")
    if args.scenario == "A":
        print(f"  임베딩 모델: {hf_emb_path} (dim={hf_emb_dim})")
    print("=" * 60)

    config = Config(
        scenario=args.scenario,
        documents_dir=args.documents_dir,
        metadata_csv="data/data_list.csv",
        vectordb_dir="data/vectordb",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_method=args.method,
        hf_embedding_model=hf_emb_path,
        hf_embedding_dim=hf_emb_dim,
    )

    chunk_file = Path(config.processed_dir) / f"{args.collection}_chunks.json"

    if args.step == "chunk":
        step_chunk(args, config)
    elif args.step == "embed":
        step_embed(args, config, chunk_file)
    else:  # all
        chunk_file = step_chunk(args, config)
        step_embed(args, config, chunk_file)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
