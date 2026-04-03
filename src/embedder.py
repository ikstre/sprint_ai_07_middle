"""
임베딩 생성 및 Vector DB 관리
시나리오 A (HuggingFace) / 시나리오 B (OpenAI) 지원
"""

import uuid
from typing import Optional

import numpy as np

from configs.config import Config


def _sanitize(text: str) -> str:
    """서로게이트 등 잘못된 유니코드를 제거한다."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


class EmbeddingModel:
    """임베딩 모델 래퍼"""

    def __init__(self, config: Config):
        self.config = config
        self._model = None

    def _init_model(self):
        if self._model is not None:
            return

        if self.config.scenario == "B":
            if not self.config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.config.openai_api_key)
        else:
            if not self.config.hf_token:
                raise ValueError("HF_TOKEN is not set. .env 파일에 HF_TOKEN을 추가하세요.")
            from sentence_transformers import SentenceTransformer
            import huggingface_hub
            huggingface_hub.login(token=self.config.hf_token, add_to_git_credential=False)
            self._model = SentenceTransformer(
                self.config.hf_embedding_model,
                device=self.config.device,
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """텍스트 리스트를 임베딩 벡터로 변환한다."""
        self._init_model()

        if self.config.scenario == "B":
            return self._embed_openai(texts)
        return self._embed_hf(texts)

    def embed_query(self, query: str) -> list[float]:
        """단일 쿼리를 임베딩한다."""
        return self.embed_texts([query])[0]

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        """OpenAI API를 사용하여 임베딩을 생성한다."""
        texts = [_sanitize(t) if t.strip() else " " for t in texts]

        all_embeddings = []
        batch_size = 100  # OpenAI 300,000 토큰/요청 한도 대응 (청크당 ~200토큰 기준)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.config.openai_embedding_model,
                input=batch,
            )
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)

        return all_embeddings

    def _embed_hf(self, texts: list[str]) -> list[list[float]]:
        """HuggingFace 모델을 사용하여 임베딩을 생성한다."""
        embeddings = self._model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings.tolist()


class VectorStore:
    """ChromaDB 또는 FAISS 기반 벡터 스토어"""

    def __init__(self, config: Config, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self._store = None
        self._store_type = config.vectordb_type

    def _init_chroma(
        self,
        collection_name: str = "rfp_documents",
        reset_collection: bool = False,
    ):
        import chromadb

        self._chroma_client = chromadb.PersistentClient(path=self.config.vectordb_dir)
        if reset_collection:
            try:
                self._chroma_client.delete_collection(collection_name)
            except Exception:
                # Collection may not exist.
                pass

        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _add_chroma(self, chunks: list[dict], embeddings: list[list[float]]):
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [_sanitize(c["text"]) for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {k: _sanitize(str(v)) for k, v in c.get("metadata", {}).items() if v is not None}
            metadatas.append(meta)

        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            self._collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def _query_chroma(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for i in range(len(results["documents"][0])):
            output.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1 - results["distances"][0][i],
                }
            )
        return output

    def _init_faiss(self):
        import faiss

        self._faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
        self._faiss_docs = []
        self._faiss_metas = []

    def _add_faiss(self, chunks: list[dict], embeddings: list[list[float]]):
        import faiss as _faiss

        arr = np.array(embeddings, dtype="float32")
        _faiss.normalize_L2(arr)
        self._faiss_index.add(arr)
        self._faiss_docs.extend([c["text"] for c in chunks])
        self._faiss_metas.extend([c.get("metadata", {}) for c in chunks])

    def _query_faiss(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        import faiss as _faiss

        q = np.array([query_embedding], dtype="float32")
        _faiss.normalize_L2(q)
        scores, indices = self._faiss_index.search(q, top_k * 3 if where else top_k)

        output = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._faiss_metas[idx]

            if where:
                match = all(str(meta.get(k, "")) == str(v) for k, v in where.items())
                if not match:
                    continue

            output.append(
                {
                    "text": self._faiss_docs[idx],
                    "metadata": meta,
                    "score": float(score),
                }
            )
            if len(output) >= top_k:
                break

        return output

    def initialize(
        self,
        collection_name: str = "rfp_documents",
        reset_collection: bool = False,
    ):
        """벡터 스토어를 초기화한다."""
        if self._store_type == "chroma":
            self._init_chroma(collection_name, reset_collection=reset_collection)
        else:
            self._init_faiss()

    def add_documents(self, chunks: list[dict], show_progress: bool = True):
        """청크들을 임베딩하여 벡터 스토어에 추가한다."""
        if show_progress:
            print(f"임베딩 생성 중... ({len(chunks)}개 청크)")

        texts = [c["text"] for c in chunks]

        batch_size = 500
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = self.embedding_model.embed_texts(batch)
            all_embeddings.extend(embs)
            if show_progress:
                print(f"  {min(i + batch_size, len(texts))}/{len(texts)} 완료")

        if self._store_type == "chroma":
            self._add_chroma(chunks, all_embeddings)
        else:
            self._add_faiss(chunks, all_embeddings)

        if show_progress:
            print(f"벡터 스토어에 {len(chunks)}개 청크 저장 완료")

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """쿼리로 유사한 청크를 검색한다."""
        query_embedding = self.embedding_model.embed_query(query)

        if self._store_type == "chroma":
            return self._query_chroma(query_embedding, top_k, where)
        return self._query_faiss(query_embedding, top_k, where)

    def get_collection_count(self) -> int:
        """저장된 문서 수를 반환한다."""
        if self._store_type == "chroma":
            return self._collection.count()
        return self._faiss_index.ntotal
