"""
임베딩 생성 및 Vector DB 관리
시나리오 A (HuggingFace) / 시나리오 B (OpenAI) 지원
"""

import uuid
from typing import Optional, List, Dict, Any

import numpy as np

from configs.config import Config


def _sanitize(text: str) -> str:
    """서로게이트 등 잘못된 유니코드를 제거한다."""
    if not text:
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def _resolve_device() -> str:
    """cuda → mps → cpu 순서로 사용 가능한 디바이스를 감지한다."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class EmbeddingModel:
    """임베딩 모델 래퍼"""

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._client = None

    def _init_model(self):
        if self._model is not None or self._client is not None:
            return

        if self.config.scenario == "B":
            if not self.config.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.config.openai_api_key)
        else:
            from sentence_transformers import SentenceTransformer

            if self.config.hf_token:
                import huggingface_hub
                huggingface_hub.login(token=self.config.hf_token, add_to_git_credential=False)

            device = self.config.device
            if device == "auto":
                device = _resolve_device()

            print(f"[Scenario A] 임베딩 모델 로드: {self.config.hf_embedding_model} (device={device})")
            self._model = SentenceTransformer(
                self.config.hf_embedding_model,
                device=device,
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """텍스트 리스트를 임베딩 벡터로 변환한다."""
        self._init_model()

        if self.config.scenario == "B":
            return self._embed_openai(texts)
        return self._embed_hf(texts)

    def embed_query(self, query: str) -> List[float]:
        """단일 쿼리를 임베딩한다."""
        return self.embed_texts([query])[0]

    def _embed_openai(self, texts: List[str], use_batch_api: bool = False) -> List[List[float]]:
        """OpenAI API를 사용하여 임베딩을 생성한다."""
        texts = [_sanitize(t) if t.strip() else " " for t in texts]
        dim = getattr(self.config, "openai_embedding_dim", 1536) # 기본값 수정 가능

        if use_batch_api and len(texts) > 500:
            return self._embed_openai_batch(texts, dim)

        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(
                model=self.config.openai_embedding_model,
                input=batch,
                dimensions=dim,
            )
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)

        return all_embeddings

    def _embed_openai_batch(self, texts: List[str], dim: int) -> List[List[float]]:
        """OpenAI Batch API를 사용해 임베딩 생성 (비용 절감)"""
        import json
        import time
        import tempfile
        import os

        print(f"  [Batch API] {len(texts)}개 텍스트 배치 임베딩 시작...")

        requests = [
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.config.openai_embedding_model,
                    "input": text,
                    "dimensions": dim,
                },
            }
            for i, text in enumerate(texts)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                batch_file = self._client.files.create(file=f, purpose="batch")

            batch_job = self._client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )
            
            while True:
                job = self._client.batches.retrieve(batch_job.id)
                if job.status == "completed":
                    break
                if job.status in ("failed", "cancelled", "expired"):
                    raise RuntimeError(f"Batch API 실패: {job.status}")
                time.sleep(30)

            content = self._client.files.content(job.output_file_id).text
            results = [json.loads(line) for line in content.strip().split("\n")]
            results.sort(key=lambda x: int(x["custom_id"]))
            return [r["response"]["body"]["data"][0]["embedding"] for r in results]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _embed_hf(self, texts: List[str]) -> List[List[float]]:
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

    def _init_chroma(self, collection_name: str = "rfp_documents", reset_collection: bool = False):
        import chromadb
        self._chroma_client = chromadb.PersistentClient(path=self.config.vectordb_dir)
        if reset_collection:
            try:
                self._chroma_client.delete_collection(collection_name)
            except Exception:
                pass
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _add_chroma(self, chunks: List[Dict], embeddings: List[List[float]]):
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # [수정] 메타데이터의 공통 텍스트가 아닌, 상위 'text' 필드(진짜 본문)를 저장
        documents = [_sanitize(c.get("text", "")) for c in chunks]
        
        metadatas = []
        for c in chunks:
            # [수정] 메타데이터 내부에 불필요하게 포함된 중복 '텍스트' 필드 제거
            raw_meta = c.get("metadata", {})
            meta = {
                k: _sanitize(str(v)) 
                for k, v in raw_meta.items() 
                if v is not None and k != "텍스트" # 중복 필드 제외
            }
            metadatas.append(meta)

        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            self._collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def _query_chroma(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        output = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                output.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "score": 1 - results["distances"][0][i],
                })
        return output

    def _init_faiss(self):
        import faiss
        self._faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
        self._faiss_docs = []
        self._faiss_metas = []

    def _add_faiss(self, chunks: List[Dict], embeddings: List[List[float]]):
        import faiss as _faiss
        arr = np.array(embeddings, dtype="float32")
        _faiss.normalize_L2(arr)
        self._faiss_index.add(arr)
        
        # [수정] FAISS에서도 본문 텍스트를 상위 필드에서 가져옴
        self._faiss_docs.extend([c.get("text", "") for c in chunks])
        self._faiss_metas.extend([
            {k: v for k, v in c.get("metadata", {}).items() if k != "텍스트"} 
            for c in chunks
        ])

    def _query_faiss(self, query_embedding: List[float], top_k: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        import faiss as _faiss
        q = np.array([query_embedding], dtype="float32")
        _faiss.normalize_L2(q)
        scores, indices = self._faiss_index.search(q, top_k * 3 if where else top_k)

        output = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            meta = self._faiss_metas[idx]
            if where:
                match = all(str(meta.get(k, "")) == str(v) for k, v in where.items())
                if not match: continue

            output.append({
                "text": self._faiss_docs[idx],
                "metadata": meta,
                "score": float(score),
            })
            if len(output) >= top_k: break
        return output

    def initialize(self, collection_name: str = "rfp_documents", reset_collection: bool = False):
        if self._store_type == "chroma":
            self._init_chroma(collection_name, reset_collection=reset_collection)
        else:
            self._init_faiss()

    def add_documents(self, chunks: List[Dict], show_progress: bool = True, use_batch_api: bool = False):
        if show_progress:
            print(f"임베딩 생성 중... ({len(chunks)}개 청크)")

        # [핵심 수정] 메타데이터 필드가 아닌, 개별 본문(text) 리스트 생성
        texts = [c.get("text", "") for c in chunks]

        if use_batch_api and self.config.scenario == "B":
            self.embedding_model._init_model()
            all_embeddings = self.embedding_model._embed_openai(texts, use_batch_api=True)
        else:
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

    def search(self, query: str, top_k: int = 5, where: Optional[Dict] = None) -> List[Dict]:
        query_embedding = self.embedding_model.embed_query(query)
        if self._store_type == "chroma":
            return self._query_chroma(query_embedding, top_k, where)
        return self._query_faiss(query_embedding, top_k, where)

    def get_collection_count(self) -> int:
        if self._store_type == "chroma":
            return self._collection.count()
        return self._faiss_index.ntotal