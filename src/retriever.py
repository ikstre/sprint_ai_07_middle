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

    @staticmethod
    def _canonical_metadata_key(key: str) -> str:
        """CSV 컬럼명과 벡터스토어 메타데이터 키를 일치시키기 위한 정규화."""
        normalized = str(key).replace(" ", "")
        if normalized in {"발주기관", "발주처", "기관명"}:
            return "발주기관"
        if normalized in {"사업명", "사업명칭"}:
            return "사업명"
        return str(key)

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

    def build_metadata_filter(self, query: str, metadata_df=None) -> Optional[dict]:
        """쿼리에서 기관명을 추출해 메타데이터 필터를 구성한다."""
        if metadata_df is None:
            return None

        org_col = None
        for col in metadata_df.columns:
            if "기관" in col or "발주" in col:
                org_col = col
                break

        if org_col:
            for _, row in metadata_df.iterrows():
                org_name = str(row[org_col])
                if len(org_name) >= 2 and org_name in query:
                    return {self._canonical_metadata_key(org_col): org_name}
                keywords = re.findall(r"[가-힣]{2,}", org_name)
                for kw in keywords:
                    if len(kw) >= 3 and kw in query:
                        return {self._canonical_metadata_key(org_col): org_name}

        return None

    def similarity_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """코사인 유사도 기반 검색"""
        k = top_k or self.config.retrieval_top_k
        return self.vector_store.search(query, top_k=k, where=where)

    def mmr_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """MMR 기반 검색: 관련성과 다양성의 균형"""
        k = top_k or self.config.retrieval_top_k
        lam = lambda_mult if lambda_mult is not None else self.config.mmr_lambda
        fetch_k = fetch_k or max(k * getattr(self.config, "retrieval_fetch_k_multiplier", 2), k + 2)

        candidates = self.vector_store.search(query, top_k=fetch_k, where=where)
        if len(candidates) <= k:
            return candidates

        query_emb = np.array(self.embedding_model.embed_query(query))
        candidate_texts = [c["text"] for c in candidates]
        candidate_embs = np.array(self.embedding_model.embed_texts(candidate_texts))

        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
        candidate_embs = candidate_embs / norms

        query_sims = candidate_embs @ query_emb

        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(k):
            if not remaining:
                break

            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                relevance = query_sims[idx]
                if selected_indices:
                    selected_embs = candidate_embs[selected_indices]
                    diversity_penalty = max(candidate_embs[idx] @ se for se in selected_embs)
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

    def _retrieve_with_method(
        self,
        query: str,
        method: str,
        top_k: int,
        where: Optional[dict],
    ) -> list[dict]:
        if method == "mmr":
            return self.mmr_search(query, top_k=top_k, where=where)
        if method == "hybrid":
            return self.hybrid_search(query, top_k=top_k, where=where)
        return self.similarity_search(query, top_k=top_k, where=where)

    def multi_query_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        llm_client=None,
        base_method: str = "similarity",
    ) -> list[dict]:
        """쿼리를 여러 관점으로 변환해 검색 후 결과를 병합한다."""
        k = top_k or self.config.retrieval_top_k

        queries = self._generate_multi_queries(query, llm_client)
        queries.insert(0, query)

        all_results = {}
        for q in queries:
            results = self._retrieve_with_method(q, base_method, k, where)
            for r in results:
                key = r["text"][:100]
                if key not in all_results or r.get("score", 0) > all_results[key].get("score", 0):
                    all_results[key] = r

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        return sorted_results[:k]

    def _generate_multi_queries(self, query: str, llm_client=None) -> list[str]:
        """LLM을 사용해 쿼리 변형을 생성한다."""
        if llm_client is None:
            if not self.config.openai_api_key:
                return []
            from openai import OpenAI

            llm_client = OpenAI(api_key=self.config.openai_api_key)

        prompt = f"""당신은 RFP(제안요청서) 검색 도우미입니다.
사용자의 질문을 3가지 다른 관점에서 재구성해 주세요.
각 질문은 한 줄에 하나씩, 번호 없이 작성해 주세요.

원본 질문: {query}

재구성된 질문들:"""

        kwargs = dict(
            model=self.config.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
        )
        # gpt-5 계열은 temperature 미지원
        if not self.config.openai_chat_model.startswith("gpt-5"):
            kwargs["temperature"] = 0.7

        response = llm_client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()][:3]

    def rerank(self, query: str, results: list[dict], top_k: Optional[int] = None) -> list[dict]:
        """Cross-encoder 기반 reranking"""
        k = top_k or self.config.retrieval_top_k

        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker

                self._reranker = FlagReranker(self.config.reranker_model, use_fp16=True)
            except ImportError:
                print("FlagEmbedding 미설치: reranking을 건너뜁니다.")
                return results[:k]

        pairs = [[query, r["text"]] for r in results]
        scores = self._reranker.compute_score(pairs)

        if isinstance(scores, float):
            scores = [scores]

        for result, score in zip(results, scores):
            result["rerank_score"] = score

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:k]

    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        keyword_weight: float = 0.3,
    ) -> list[dict]:
        """벡터 검색 + 키워드 매칭 결합 하이브리드 검색"""
        k = top_k or self.config.retrieval_top_k

        vector_results = self.vector_store.search(query, top_k=k * 3, where=where)

        query_keywords = set(re.findall(r"[가-힣a-zA-Z0-9]{2,}", query))

        for result in vector_results:
            text_lower = result["text"].lower()
            keyword_hits = sum(1 for kw in query_keywords if kw.lower() in text_lower)
            keyword_score = keyword_hits / max(len(query_keywords), 1)

            vector_score = result.get("score", 0)
            result["hybrid_score"] = (1 - keyword_weight) * vector_score + keyword_weight * keyword_score

        sorted_results = sorted(vector_results, key=lambda x: x["hybrid_score"], reverse=True)
        return sorted_results[:k]

    @staticmethod
    def _source_key(doc: dict) -> str:
        """문서 출처 식별자 — 같은 RFP 문서 청크를 묶는 데 사용한다."""
        meta = doc.get("metadata", {})
        org = meta.get("발주기관", meta.get("발주 기관", ""))
        biz = meta.get("사업명", meta.get("filename", ""))
        return f"{org}||{biz}" if (org or biz) else "__unknown__"

    def _limit_per_source(self, results: list[dict], top_k: int) -> list[dict]:
        """동일 RFP 출처에서 최대 max_per_source개 청크만 허용한다.

        top_k=5 기준으로 한 출처가 3개 이상을 독점하면 다른 관련 문서가 밀려나므로
        출처당 최대 2개로 제한하되, 단일 출처일 때는 제한 완화(4개까지 허용).
        """
        max_per_source = getattr(self.config, "max_chunks_per_source", 2)
        if max_per_source <= 0:
            return results[:top_k]

        source_count: dict[str, int] = {}
        filtered = []
        for doc in results:
            key = self._source_key(doc)
            cnt = source_count.get(key, 0)
            if cnt < max_per_source:
                filtered.append(doc)
                source_count[key] = cnt + 1
            if len(filtered) >= top_k:
                break

        # 단일 출처만 남은 경우 제한 완화 (부족 시 원본에서 보충)
        if len(filtered) < top_k and len(source_count) == 1:
            existing_texts = {d["text"] for d in filtered}
            for doc in results:
                if doc["text"] not in existing_texts:
                    filtered.append(doc)
                    if len(filtered) >= top_k:
                        break

        return filtered

    def retrieve(
        self,
        query: str,
        method: Optional[str] = None,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        llm_client=None,
    ) -> list[dict]:
        """설정에 따라 적절한 검색 방식을 실행한다."""
        method = method or self.config.retrieval_method
        k = top_k or self.config.retrieval_top_k

        # per-source 제한 적용용 후보 확장은 하되, 과도한 재검색은 피한다.
        fetch_multiplier = max(1, getattr(self.config, "retrieval_fetch_k_multiplier", 2))
        fetch_k = max(k * fetch_multiplier, k + 2)

        if self.config.use_multi_query:
            results = self.multi_query_search(
                query,
                top_k=fetch_k,
                where=where,
                llm_client=llm_client,
                base_method=method,
            )
        else:
            results = self._retrieve_with_method(query, method, fetch_k, where)

        results = self._limit_per_source(results, k)

        if self.config.use_reranker:
            results = self.rerank(query, results, top_k=k)

        return results
