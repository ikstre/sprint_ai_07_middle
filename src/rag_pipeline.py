"""
RAG 파이프라인 - 질문 분석 -> 메타데이터 필터 -> 검색 -> 답변 생성
"""

from configs.config import Config
from src.embedder import EmbeddingModel, VectorStore
from src.generator import RAGGenerator
from src.retriever import Retriever


class RAGPipeline:
    """인덱싱, 검색, 생성을 하나의 인터페이스로 묶는 파이프라인"""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        self.vector_store = VectorStore(config, self.embedding_model)
        self.retriever = Retriever(config, self.vector_store, self.embedding_model)
        self.generator = RAGGenerator(config, self.retriever)
        self._initialized = False

    def initialize_vectorstore(self, collection_name: str = "rfp_documents"):
        """기존 컬렉션을 로드한다."""
        self.vector_store.initialize(collection_name=collection_name)
        self._initialized = True

    def build_index(
        self,
        documents: list[dict],
        collection_name: str = "rfp_documents",
        reset_collection: bool = True,
    ):
        """문서 리스트로 벡터 인덱스를 구축한다."""
        self.vector_store.initialize(
            collection_name=collection_name,
            reset_collection=reset_collection,
        )
        self.vector_store.add_documents(documents)
        self._initialized = True

    def query(self, question: str, **kwargs) -> dict:
        """질문에 대한 RAG 답변을 생성한다."""
        if not self._initialized:
            raise RuntimeError("initialize_vectorstore() 또는 build_index()를 먼저 호출하세요.")

        result = self.generator.generate(question, **kwargs)

        seen = set()
        sources = []
        for doc in result["retrieved_docs"]:
            meta = doc.get("metadata", {})
            org = meta.get("발주기관", meta.get("발주 기관", "N/A"))
            biz = meta.get("사업명", "N/A")
            amount = meta.get("사업금액", meta.get("사업 금액", "N/A"))

            key = (org, biz)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "발주기관": org,
                    "사업명": biz,
                    "사업금액": amount,
                }
            )

        result["sources"] = sources
        return result

    def reset_conversation(self):
        """대화 히스토리를 초기화한다."""
        self.generator.reset_memory()

    def extract_metadata_filter(self, question: str) -> dict | None:
        """질문에서 메타데이터 필터(기관명 등)를 추출한다."""
        import pandas as pd

        metadata_path = self.config.metadata_csv
        try:
            df = pd.read_csv(metadata_path)
            return self.retriever.build_metadata_filter(question, df)
        except Exception:
            return None
