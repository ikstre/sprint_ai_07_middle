"""
Generation module: RAG answer generation with optional conversation memory.
"""

from __future__ import annotations

import time
from typing import Optional

from configs.config import Config
from src.retriever import Retriever


SYSTEM_PROMPT = """당신은 '입찰메이트'의 RFP 분석 AI 어시스턴트입니다.
항상 검색된 문서 근거를 우선으로 답변하고, 문서에 없는 내용은 추측하지 마세요.
핵심 요구사항/기관/예산/제출방식/일정 중심으로 간결하게 답변하세요.
"""

RAG_PROMPT_TEMPLATE = """아래 컨텍스트를 근거로 질문에 답변하세요.

[컨텍스트]
{context}

[질문]
{question}

[답변 작성 규칙]
- 문서 근거 기반으로만 답변
- 비교 질문이면 항목별 표 또는 목록
- 근거가 없으면 "문서에서 확인되지 않음"이라고 명시
"""


class ConversationMemory:
    """Sliding window memory for conversation context."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def get_messages(self) -> list[dict]:
        return list(self.history)

    def clear(self) -> None:
        self.history = []

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_context_summary(self) -> str:
        if not self.history:
            return ""
        parts = []
        for msg in self.history[-4:]:
            role = "사용자" if msg["role"] == "user" else "어시스턴트"
            parts.append(f"{role}: {msg['content'][:200]}")
        return "\n".join(parts)


class RAGGenerator:
    """RAG-based answer generator."""

    def __init__(self, config: Config, retriever: Retriever):
        self.config = config
        self.retriever = retriever
        self.memory = ConversationMemory(max_turns=config.conversation_memory_k)
        self._llm_client = None
        self.last_usage: dict | None = None

    @staticmethod
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

    def _get_llm_client(self):
        if self._llm_client is not None:
            return self._llm_client

        if self.config.scenario == "B":
            from openai import OpenAI

            self._llm_client = OpenAI(api_key=self.config.openai_api_key)
        else:
            self._llm_client = self._init_hf_model()

        return self._llm_client

    def _init_hf_model(self):
        """AutoModelForCausalLM + AutoTokenizer로 로컬 또는 Hub 채팅 모델을 로드한다.

        /srv/shared_data/models/ 의 로컬 모델은 HF_TOKEN 없이 로드 가능.
        HuggingFace Hub 비공개 모델을 사용할 경우 .env에 HF_TOKEN을 설정한다.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # 로컬 경로 모델은 HF 토큰 불필요; Hub 비공개 모델일 때만 로그인
        if self.config.hf_token:
            import huggingface_hub
            huggingface_hub.login(token=self.config.hf_token, add_to_git_credential=False)

        device = self._resolve_device()
        print(f"[Scenario A] 사용 디바이스: {device}")
        print(f"[Scenario A] 채팅 모델 로드: {self.config.hf_chat_model}")

        tok_kwargs: dict = {}
        if self.config.hf_token:
            tok_kwargs["token"] = self.config.hf_token

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_chat_model,
            **tok_kwargs,
        )

        load_kwargs: dict = {"torch_dtype": torch.bfloat16}
        if self.config.hf_token:
            load_kwargs["token"] = self.config.hf_token
        if device == "cpu":
            load_kwargs["device_map"] = "cpu"
        else:
            load_kwargs["device_map"] = "auto"

        if getattr(self.config, "hf_load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs.pop("torch_dtype", None)  # 4-bit와 dtype 동시 지정 불가

        model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_chat_model,
            **load_kwargs,
        )
        model.eval()
        print(f"[Scenario A] 모델 로드 완료: {self.config.hf_chat_model}")
        return tokenizer, model

    def _build_context(self, retrieved_docs: list[dict]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            source_info = []
            for key in ["사업명", "발주기관", "발주 기관", "filename"]:
                if key in meta and meta[key]:
                    source_info.append(f"{key}: {meta[key]}")
            source_str = " | ".join(source_info) if source_info else f"문서 {i}"
            context_parts.append(f"[출처: {source_str}]\n{doc['text']}")

        return "\n\n---\n\n".join(context_parts)

    def _enhance_query_with_context(self, query: str) -> str:
        context_summary = self.memory.get_context_summary()
        if not context_summary:
            return query

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
        start_time = time.time()

        enhanced_query = self._enhance_query_with_context(query)

        retrieved_docs = self.retriever.retrieve(
            enhanced_query,
            method=retrieval_method,
            top_k=top_k,
            where=where,
            llm_client=self._get_llm_client() if self.config.scenario == "B" else None,
        )

        context = self._build_context(retrieved_docs)
        rag_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

        answer, usage = self._call_llm(rag_prompt, stream=stream, query=query)

        self.memory.add_user_message(query)
        self.memory.add_assistant_message(answer)

        elapsed = time.time() - start_time
        self.last_usage = usage

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "query_used": enhanced_query,
            "elapsed_time": elapsed,
            "usage": usage,
            "context_char_len": len(context),
            "answer_char_len": len(answer),
        }

    def _call_llm(self, user_prompt: str, stream: bool = False, query: str = "") -> tuple[str, dict | None]:
        client = self._get_llm_client()

        if self.config.scenario == "B":
            return self._call_openai(client, user_prompt, stream, query=query)
        return self._call_hf(client, user_prompt, query=query)

    def _route_model(self, query: str) -> str:
        """쿼리 복잡도에 따라 모델을 자동 선택한다."""
        if not self.config.auto_model_routing:
            return self.config.openai_chat_model
        if len(query) <= self.config.routing_complexity_threshold:
            return self.config.routing_simple_model
        return self.config.openai_chat_model

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """서로게이트 문자 제거 — OpenAI API JSON body 파싱 에러 방지."""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def _call_openai(self, client, user_prompt: str, stream: bool = False, query: str = "") -> tuple[str, dict | None]:
        user_prompt = self._sanitize_text(user_prompt)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.memory.get_messages())
        messages.append({"role": "user", "content": user_prompt})
        model = self._route_model(query or user_prompt)

        effort = getattr(self.config, "reasoning_effort", "medium")
        # gpt-5 계열(reasoning 모델)만 reasoning_effort 지원; 나머지는 생략
        is_reasoning_model = model.startswith(("gpt-5", "o1", "o3", "o4"))
        extra_kwargs: dict = {}
        if is_reasoning_model:
            extra_kwargs["reasoning_effort"] = effort

        if stream:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=self.config.max_tokens,
                stream=True,
                **extra_kwargs,
            )
            chunks = []
            for chunk in response:
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    chunks.append(delta.content)
            return "".join(chunks), None

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=self.config.max_tokens,
            **extra_kwargs,
        )

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        message = response.choices[0].message
        content = message.content

        # gpt-5 계열: content가 list[{"type":"text","text":"..."}] 형태로 올 수 있음
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        if not content:
            content = (
                getattr(message, "output_text", None)
                or getattr(message, "refusal", None)
                or ""
            )

        return content, usage

    def _call_hf(self, client, user_prompt: str, query: str = "") -> tuple[str, dict | None]:
        """Gemma-3 등 채팅 모델을 apply_chat_template으로 호출한다."""
        import torch

        tokenizer, model = client

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        # 대화 히스토리 추가 (system 이후, 현재 질문 이전)
        messages.extend(self.memory.get_messages())
        messages.append({"role": "user", "content": user_prompt})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=getattr(self.config, "hf_max_new_tokens", 1024),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 입력 토큰을 제외한 생성 토큰만 디코딩
        new_tokens = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response, None

    def reset_memory(self) -> None:
        self.memory.clear()

    def chat(self, query: str, **kwargs) -> str:
        result = self.generate(query, **kwargs)
        return result["answer"]
