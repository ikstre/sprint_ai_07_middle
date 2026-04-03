"""
문서 인덱싱 스크립트
: CSV 메타데이터 + 원본 파일 로딩 → 청킹 → 임베딩 → ChromaDB 저장
"""
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


def main():
    parser = argparse.ArgumentParser(description="RFP 문서 인덱싱")
    parser.add_argument(
        "--scenario", type=str, default="B", choices=["A", "B"],
        help="A: HuggingFace 로컬 모델, B: OpenAI API (기본값: B)",
    )
    parser.add_argument(
        "--method", type=str, default="naive", choices=["naive", "semantic"],
        help="청킹 전략 (기본값: naive)",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--collection", type=str, default="rfp_documents",
        help="ChromaDB 컬렉션 이름",
    )
    parser.add_argument(
        "--documents-dir", type=str, default="data",
        help="문서 디렉토리 경로 (기본값: data/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RFP 문서 인덱싱 시작")
    print(f"  시나리오: {args.scenario}")
    print(f"  청킹 전략: {args.method} | 크기: {args.chunk_size} | 중첩: {args.chunk_overlap}")
    print(f"  컬렉션: {args.collection}")
    print("=" * 60)

    # 설정
    config = Config(
        scenario=args.scenario,
        documents_dir=args.documents_dir,
        metadata_csv="data/data_list.csv",
        vectordb_dir="data/vectordb",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_method=args.method,
    )

    # 1. 문서 로딩
    print("\n[1/3] 문서 로딩 중...")
    loader = DocumentLoader(
        documents_dir=config.documents_dir,
        metadata_csv=config.metadata_csv,
    )
    documents = loader.load_all()

    if not documents:
        print("로딩된 문서가 없습니다. documents_dir 경로를 확인하세요.")
        sys.exit(1)

    # 2. 청킹
    print(f"\n[2/3] 문서 청킹 중 (방법: {config.chunking_method})...")
    chunks = chunk_documents(
        documents,
        method=config.chunking_method,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # 3. 벡터스토어 구축
    print("\n[3/3] 임베딩 생성 및 벡터스토어 저장 중...")
    pipeline = RAGPipeline(config)
    pipeline.build_index(chunks, collection_name=args.collection)

    print("\n" + "=" * 60)
    print("인덱싱 완료!")
    print(f"  총 문서: {len(documents)}개 | 총 청크: {len(chunks)}개")
    print(f"  저장 위치: {config.vectordb_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
