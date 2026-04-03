"""
Retrieval 모듈: 유사도 검색, MMR, Hybrid Search, Multi-Query, Reranking
"""

import re
from typing import Optional

import numpy as np

from configs.config import Config
from src.embedder import EmbeddingModel, VectorStore


class Retriever:
    """다양한 검색 전략을 지원하는 Retriever"""

    def __init__(
        self,
        config: Config,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self._reranker = None

    # ── 메타데이터 필터 구성 ────────────────────────────────────

    def build_metadata_filter(self, query: str, metadata_df=None) -> Optional[dict]:
        """쿼리에서 기관명/사업명을 추출하여 메타데이터 필터를 구성한다.

        사용자가 정확한 명칭을 입력하지 않을 수 있으므로
        fuzzy matching을 지원한다.
        """
        if metadata_df is None:
            return None

        # 기관명 매칭 시도
        org_col = None
        for col in metadata_df.columns:
            if "기관" in col or "발주" in col:
                org_col = col
                break

        if org_col:
            for _, row in metadata_df.iterrows():
                org_name = str(row[org_col])
                # 쿼리에 기관명이 포함되어 있는지 확인 (부분 매칭)
                if len(org_name) >= 2 and org_name in query:
                    return {org_col: org_name}
                # 역방향: 기관명의 핵심 키워드가 쿼리에 있는지
                keywords = re.findall(r"[가-힣]{2,}", org_name)
                for kw in keywords:
                    if len(kw) >= 3 and kw in query:
                        return {org_col: org_name}

        return None

    # ── 기본 유사도 검색 ───────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """코사인 유사도 기반 검색"""
        k = top_k or self.config.retrieval_top_k
        return self.vector_store.search(query, top_k=k, where=where)

    # ── MMR (Maximum Marginal Relevance) ───────────────────────

    def mmr_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: Optional[float] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """MMR 기반 검색: 관련성과 다양성의 균형을 맞춘다."""
        k = top_k or self.config.retrieval_top_k
        lam = lambda_mult if lambda_mult is not None else self.config.mmr_lambda

        # 후보군을 넉넉하게 가져옴
        candidates = self.vector_store.search(query, top_k=fetch_k, where=where)
        if len(candidates) <= k:
            return candidates

        # 쿼리 임베딩
        query_emb = np.array(self.embedding_model.embed_query(query))

        # 후보 임베딩
        candidate_texts = [c["text"] for c in candidates]
        candidate_embs = np.array(self.embedding_model.embed_texts(candidate_texts))

        # 정규화
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
        candidate_embs = candidate_embs / norms

        # 쿼리-후보 유사도
        query_sims = candidate_embs @ query_emb

        # MMR 선택
        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(k):
            if not remaining:
                break

            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                relevance = query_sims[idx]

                # 이미 선택된 문서와의 최대 유사도
                if selected_indices:
                    selected_embs = candidate_embs[selected_indices]
                    diversity_penalty = max(
                        candidate_embs[idx] @ se for se in selected_embs
                    )
                else:
                    diversity_penalty = 0

                mmr_score = lam * relevance - (1 - lam) * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        return [candidates[i] for i in selected_indices]

    # ── Multi-Query Retrieval ──────────────────────────────────

    def multi_query_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        llm_client=None,
    ) -> list[dict]:
        """쿼리를 여러 관점으로 변환하여 검색한 뒤 결과를 병합한다."""
        k = top_k or self.config.retrieval_top_k

        # LLM을 사용하여 다양한 쿼리 생성
        queries = self._generate_multi_queries(query, llm_client)
        queries.insert(0, query)  # 원본 쿼리 포함

        # 각 쿼리로 검색
        all_results = {}
        for q in queries:
            results = self.vector_store.search(q, top_k=k, where=where)
            for r in results:
                key = r["text"][:100]  # 중복 제거용 키
                if key not in all_results or r.get("score", 0) > all_results[key].get("score", 0):
                    all_results[key] = r

        # 점수순 정렬 후 top_k 반환
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        return sorted_results[:k]

    def _generate_multi_queries(self, query: str, llm_client=None) -> list[str]:
        """LLM을 사용하여 쿼리의 다양한 변형을 생성한다."""
        if llm_client is None:
            from openai import OpenAI
            llm_client = OpenAI(api_key=self.config.openai_api_key)

        prompt = f"""당신은 RFP(제안요청서) 검색 도우미입니다.
사용자의 질문을 3가지 다른 관점에서 재구성해 주세요.
각 질문은 한 줄에 하나씩, 번호 없이 작성해 주세요.

원본 질문: {query}

재구성된 질문들:"""

        response = llm_client.chat.completions.create(
            model=self.config.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )

        lines = response.choices[0].message.content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()][:3]

    # ── Reranking ──────────────────────────────────────────────

    def rerank(self, query: str, results: list[dict], top_k: Optional[int] = None) -> list[dict]:
        """Cross-encoder 기반 reranking"""
        k = top_k or self.config.retrieval_top_k

        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(
                    self.config.reranker_model, use_fp16=True
                )
            except ImportError:
                print("FlagEmbedding 미설치. reranking을 건너뜁니다.")
                return results[:k]

        pairs = [[query, r["text"]] for r in results]
        scores = self._reranker.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        for result, score in zip(results, scores):
            result["rerank_score"] = score

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:k]

    # ── Hybrid Search (키워드 + 벡터) ──────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        keyword_weight: float = 0.3,
    ) -> list[dict]:
        """벡터 검색 + 키워드 매칭을 결합한 하이브리드 검색"""
        k = top_k or self.config.retrieval_top_k

        # 벡터 검색 (넉넉하게)
        vector_results = self.vector_store.search(query, top_k=k * 3, where=where)

        # 키워드 점수 계산
        query_keywords = set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", query))

        for result in vector_results:
            text_lower = result["text"].lower()
            keyword_hits = sum(
                1 for kw in query_keywords if kw.lower() in text_lower
            )
            keyword_score = keyword_hits / max(len(query_keywords), 1)

            # 하이브리드 점수 = (1-w) * vector_score + w * keyword_score
            vector_score = result.get("score", 0)
            result["hybrid_score"] = (
                (1 - keyword_weight) * vector_score
                + keyword_weight * keyword_score
            )

        # 하이브리드 점수로 정렬
        sorted_results = sorted(
            vector_results,
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )
        return sorted_results[:k]

    # ── 통합 검색 인터페이스 ───────────────────────────────────

    def retrieve(
        self,
        query: str,
        method: Optional[str] = None,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        llm_client=None,
    ) -> list[dict]:
        """설정에 따라 적절한 검색 방법을 실행한다."""
        method = method or self.config.retrieval_method
        k = top_k or self.config.retrieval_top_k

        if method == "mmr":
            results = self.mmr_search(query, top_k=k, where=where)
        elif method == "hybrid":
            results = self.hybrid_search(query, top_k=k, where=where)
        else:
            results = self.similarity_search(query, top_k=k, where=where)

        # Multi-Query 적용
        if self.config.use_multi_query:
            results = self.multi_query_search(
                query, top_k=k, where=where, llm_client=llm_client
            )

        # Reranking 적용
        if self.config.use_reranker:
            results = self.rerank(query, results, top_k=k)

        return results
