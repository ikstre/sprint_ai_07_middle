"""
RFP RAG 시스템 설정
시나리오 A (HuggingFace/로컬) 와 시나리오 B (OpenAI API) 전환 가능
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # ── 시나리오 선택 ──────────────────────────────────────────────
    scenario: Literal["A", "B"] = "B"  # "A": HuggingFace 로컬, "B": OpenAI API

    # ── 데이터 경로 ────────────────────────────────────────────────
    documents_dir: str = "data/documents"
    processed_dir: str = "data/processed"
    metadata_csv: str = "data/data_list.csv"
    vectordb_dir: str = "data/vectordb"

    # ── 시나리오 B: OpenAI API 설정 ────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-5-mini"          # gpt-5-mini, gpt-5-nano, gpt-5
    openai_embedding_dim: int = 1536

    # ── 시나리오 A: HuggingFace 로컬 모델 설정 ─────────────────────
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    hf_embedding_model: str = "intfloat/multilingual-e5-large"
    hf_chat_model: str = "google/gemma-3-4b-it"
    hf_embedding_dim: int = 1024
    device: str = "cuda"  # GCP VM에서는 cuda 사용

    # ── 청킹 설정 ──────────────────────────────────────────────────
    chunk_size: int = 1200
    chunk_overlap: int = 200
    chunking_method: Literal["naive", "semantic"] = "naive"

    # ── Retrieval 설정 ─────────────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_method: Literal["similarity", "mmr", "hybrid"] = "similarity"
    mmr_lambda: float = 0.5           # MMR 다양성 파라미터
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_multi_query: bool = False

    # ── Generation 설정 ────────────────────────────────────────────
    # 시나리오 A (HuggingFace) 전용: temperature, top_p 지원
    # 시나리오 B (gpt-5 계열): temperature/top_p 미지원, max_completion_tokens만 사용
    temperature: float = 0.1          # Scenario A 전용
    top_p: float = 0.9                # Scenario A 전용
    max_tokens: int = 2048
    conversation_memory_k: int = 5    # 유지할 대화 턴 수

    # ── Vector DB 설정 ─────────────────────────────────────────────
    vectordb_type: Literal["chroma", "faiss"] = "chroma"

    @property
    def embedding_model(self) -> str:
        return self.openai_embedding_model if self.scenario == "B" else self.hf_embedding_model

    @property
    def embedding_dim(self) -> int:
        return self.openai_embedding_dim if self.scenario == "B" else self.hf_embedding_dim

    @property
    def chat_model(self) -> str:
        return self.openai_chat_model if self.scenario == "B" else self.hf_chat_model
