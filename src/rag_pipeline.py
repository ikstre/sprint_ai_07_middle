"""
RAG 파이프라인 - 질문 분석 → 메타데이터 필터 → 검색 → 답변 생성
configs/config.py, src/embedder.py, src/retriever.py, src/generator.py 를 조합하는 통합 레이어
"""
import re
import time

from configs.config import Config
from src.embedder import EmbeddingModel, VectorStore
from src.retriever import Retriever
from src.generator import RAGGenerator


class RAGPipeline:
    """
    인덱싱, 검색, 생성을 하나의 인터페이스로 묶는 파이프라인.
    Streamlit app 및 스크립트에서 사용한다.
    """

    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        self.vector_store = VectorStore(config, self.embedding_model)
        self.retriever = Retriever(config, self.vector_store, self.embedding_model)
        self.generator = RAGGenerator(config, self.retriever)
        self._initialized = False

    def initialize_vectorstore(self, collection_name: str = "rfp_documents"):
        """벡터스토어를 초기화(기존 컬렉션 로드)한다."""
        self.vector_store.initialize(collection_name)
        self._initialized = True

    def build_index(
        self,
        documents: list[dict],
        collection_name: str = "rfp_documents",
    ):
        """
        문서 리스트로 벡터 인덱스를 구축한다.

        Args:
            documents: chunk_documents()의 반환값 (각 항목: {"text": str, "metadata": dict})
        """
        self.vector_store.initialize(collection_name)
        self.vector_store.add_documents(documents)
        self._initialized = True

    def query(self, question: str, **kwargs) -> dict:
        """
        질문에 대한 RAG 답변을 생성한다.

        Returns:
            {
                "answer": str,
                "retrieved_docs": list[dict],
                "query_used": str,
                "elapsed_time": float,
                "sources": list[dict],  # 중복 제거된 출처 목록
            }
        """
        if not self._initialized:
            raise RuntimeError("initialize_vectorstore() 또는 build_index()를 먼저 호출하세요.")

        result = self.generator.generate(question, **kwargs)

        # 출처 정보 중복 제거
        seen = set()
        sources = []
        for doc in result["retrieved_docs"]:
            meta = doc.get("metadata", {})
            key = (meta.get("발주기관", ""), meta.get("사업명", ""))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "발주기관": meta.get("발주기관", meta.get("발주 기관", "N/A")),
                    "사업명": meta.get("사업명", "N/A"),
                    "사업금액": meta.get("사업금액", meta.get("사업 금액", "N/A")),
                })

        result["sources"] = sources
        return result

    def reset_conversation(self):
        """대화 히스토리를 초기화한다."""
        self.generator.reset_memory()

    def extract_metadata_filter(self, question: str) -> dict | None:
        """
        질문에서 발주기관명/사업명 키워드를 추출하여 메타데이터 필터를 만든다.
        Retriever.build_metadata_filter()를 래핑한다.
        """
        import pandas as pd
        metadata_path = self.config.metadata_csv
        try:
            df = pd.read_csv(metadata_path)
            return self.retriever.build_metadata_filter(question, df)
        except Exception:
            return None
