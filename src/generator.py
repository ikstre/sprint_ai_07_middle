"""
Generation 모듈: LLM을 사용한 답변 생성 + 대화 메모리
"""

import time
from typing import Optional

from configs.config import Config
from src.retriever import Retriever


# ─────────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 '입찰메이트'의 RFP(제안요청서) 분석 전문 AI 어시스턴트입니다.
컨설턴트들이 제안요청서의 핵심 정보를 빠르게 파악할 수 있도록 도와줍니다.

## 답변 원칙

1. **정확성**: 제공된 문서(컨텍스트)에 근거하여 답변하세요. 문서에 없는 내용은 추측하지 마세요.
2. **간결성**: 핵심 정보를 구조화하여 명확하게 전달하세요.
3. **출처 명시**: 답변에 사용한 문서의 사업명이나 발주기관을 언급하세요.
4. **모르면 모른다고**: 문서에서 관련 정보를 찾을 수 없으면 솔직하게 알려주세요.

## 답변 형식

- 요구사항 정리 시: 번호 매기기 또는 표 형태 활용
- 비교 요청 시: 항목별 비교표 사용
- 예산/일정 관련: 구체적인 수치를 정확히 인용
"""

RAG_PROMPT_TEMPLATE = """아래 제공된 RFP 문서 컨텍스트를 참고하여 질문에 답변해 주세요.

## 참고 문서 컨텍스트

{context}

## 질문

{question}

## 답변"""


# ─────────────────────────────────────────────────────────────────
# 대화 메모리
# ─────────────────────────────────────────────────────────────────

class ConversationMemory:
    """최근 k턴의 대화를 유지하는 슬라이딩 윈도우 메모리"""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]

    def add_user_message(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant_message(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def get_messages(self) -> list[dict]:
        """OpenAI messages 형태로 대화 이력을 반환한다."""
        return list(self.history)

    def clear(self):
        self.history = []

    def _trim(self):
        """최대 턴 수를 초과하면 오래된 대화를 제거한다."""
        # user+assistant 쌍 = 1턴
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_context_summary(self) -> str:
        """이전 대화 맥락을 요약 문자열로 반환한다."""
        if not self.history:
            return ""
        parts = []
        for msg in self.history[-4:]:  # 최근 2턴만 요약에 사용
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            parts.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────

class RAGGenerator:
    """RAG 기반 답변 생성기"""

    def __init__(self, config: Config, retriever: Retriever):
        self.config = config
        self.retriever = retriever
        self.memory = ConversationMemory(max_turns=config.conversation_memory_k)
        self._llm_client = None

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client

        if self.config.scenario == "B":
            from openai import OpenAI
            self._llm_client = OpenAI(api_key=self.config.openai_api_key)
        else:
            self._llm_client = self._init_hf_pipeline()

        return self._llm_client

    def _init_hf_pipeline(self):
        """HuggingFace 모델로 로컬 추론 파이프라인을 초기화한다."""
        from transformers import pipeline
        return pipeline(
            "text-generation",
            model=self.config.hf_chat_model,
            device_map="auto",
            max_new_tokens=self.config.max_tokens,
        )

    def _build_context(self, retrieved_docs: list[dict]) -> str:
        """검색된 문서들을 컨텍스트 문자열로 구성한다."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            source_info = []
            for key in ["사업명", "발주기관", "filename"]:
                if key in meta and meta[key]:
                    source_info.append(f"{key}: {meta[key]}")
            source_str = " | ".join(source_info) if source_info else f"문서 {i}"
            context_parts.append(f"[출처: {source_str}]\n{doc['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _enhance_query_with_context(self, query: str) -> str:
        """대화 맥락을 고려하여 쿼리를 보강한다."""
        context_summary = self.memory.get_context_summary()
        if not context_summary:
            return query

        # 대명사나 암시적 참조가 있으면 맥락으로 보강
        ambiguous_patterns = ["그", "이", "저", "해당", "그것", "위", "다른", "더"]
        needs_context = any(query.startswith(p) or f" {p} " in query for p in ambiguous_patterns)

        if needs_context or len(query) < 20:
            return f"[이전 대화 맥락]\n{context_summary}\n\n[현재 질문]\n{query}"
        return query

    def generate(
        self,
        query: str,
        retrieval_method: Optional[str] = None,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        stream: bool = False,
    ) -> dict:
        """사용자 질문에 대해 RAG 기반 답변을 생성한다.

        Returns:
            {
                "answer": str,
                "retrieved_docs": list[dict],
                "query_used": str,
                "elapsed_time": float,
            }
        """
        start_time = time.time()

        # 1. 대화 맥락을 고려한 쿼리 보강
        enhanced_query = self._enhance_query_with_context(query)

        # 2. Retrieval
        retrieved_docs = self.retriever.retrieve(
            enhanced_query,
            method=retrieval_method,
            top_k=top_k,
            where=where,
            llm_client=self._get_llm_client() if self.config.scenario == "B" else None,
        )

        # 3. 컨텍스트 구성
        context = self._build_context(retrieved_docs)

        # 4. 답변 생성
        rag_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context, question=query
        )

        answer = self._call_llm(rag_prompt, stream=stream)

        # 5. 대화 메모리 업데이트
        self.memory.add_user_message(query)
        self.memory.add_assistant_message(answer)

        elapsed = time.time() - start_time

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "query_used": enhanced_query,
            "elapsed_time": elapsed,
        }

    def _call_llm(self, user_prompt: str, stream: bool = False) -> str:
        """LLM을 호출하여 답변을 생성한다."""
        client = self._get_llm_client()

        if self.config.scenario == "B":
            return self._call_openai(client, user_prompt, stream)
        else:
            return self._call_hf(client, user_prompt)

    def _call_openai(self, client, user_prompt: str, stream: bool = False) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # 대화 이력 추가
        messages.extend(self.memory.get_messages())
        # 현재 RAG 프롬프트 추가
        messages.append({"role": "user", "content": user_prompt})

        if stream:
            response = client.chat.completions.create(
                model=self.config.openai_chat_model,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            chunks = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # 줄바꿈
            return "".join(chunks)
        else:
            response = client.chat.completions.create(
                model=self.config.openai_chat_model,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content

    def _call_hf(self, pipeline, user_prompt: str) -> str:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        result = pipeline(full_prompt)
        generated = result[0]["generated_text"]
        # 프롬프트 이후 생성된 부분만 추출
        if full_prompt in generated:
            generated = generated[len(full_prompt):].strip()
        return generated

    def reset_memory(self):
        """대화 메모리를 초기화한다."""
        self.memory.clear()

    def chat(self, query: str, **kwargs) -> str:
        """간편한 대화 인터페이스. 답변 텍스트만 반환한다."""
        result = self.generate(query, **kwargs)
        return result["answer"]
