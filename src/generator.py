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
답변에서 확인 가능한 기관명, 사업명, 금액, 일정은 문서 표현을 최대한 그대로 사용하세요.
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
- 확인 가능한 경우 답변 첫 부분에 발주기관과 사업명을 명시
- 금액, 일정, 제출방식, 요구사항은 문서에 나온 표현을 그대로 우선 사용
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

    @staticmethod
    def _collect_source_rows(retrieved_docs: list[dict]) -> list[dict]:
        rows = []
        seen = set()
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            row = {
                "발주기관": meta.get("발주기관") or meta.get("발주 기관"),
                "사업명": meta.get("사업명"),
                "사업금액": meta.get("사업금액") or meta.get("사업 금액"),
                "filename": meta.get("filename"),
            }
            key = tuple((k, row.get(k)) for k in ("발주기관", "사업명", "사업금액", "filename"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        return rows

    @staticmethod
    def _is_comparison_query(query: str) -> bool:
        comparison_keywords = ("비교", "차이", "각각", "서로", "vs", "대비")
        return any(keyword in query for keyword in comparison_keywords)

    def _build_source_summary(self, retrieved_docs: list[dict], query: str) -> str:
        rows = self._collect_source_rows(retrieved_docs)
        if not rows:
            return ""

        max_rows = 3 if self._is_comparison_query(query) else 1
        summary_lines = ["[문서 식별 정보]"]
        for row in rows[:max_rows]:
            parts = []
            if row.get("발주기관"):
                parts.append(f"발주기관: {row['발주기관']}")
            if row.get("사업명"):
                parts.append(f"사업명: {row['사업명']}")
            if row.get("사업금액"):
                parts.append(f"사업금액: {row['사업금액']}")
            if not parts and row.get("filename"):
                parts.append(f"파일명: {row['filename']}")
            if parts:
                summary_lines.append("- " + " | ".join(parts))

        return "\n".join(summary_lines) if len(summary_lines) > 1 else ""

    def _build_context(self, retrieved_docs: list[dict]) -> str:
        context_parts = []
        max_chars = max(0, getattr(self.config, "max_context_chars_per_doc", 0))
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            header_lines = [f"[문서 {i}]"]
            if meta.get("발주기관") or meta.get("발주 기관"):
                header_lines.append(f"발주기관: {meta.get('발주기관') or meta.get('발주 기관')}")
            if meta.get("사업명"):
                header_lines.append(f"사업명: {meta['사업명']}")
            if meta.get("사업금액") or meta.get("사업 금액"):
                header_lines.append(f"사업금액: {meta.get('사업금액') or meta.get('사업 금액')}")
            if meta.get("filename"):
                header_lines.append(f"파일명: {meta['filename']}")
            text = doc["text"]
            if max_chars and len(text) > max_chars:
                text = text[:max_chars].rstrip() + "\n...[후략]"
            context_parts.append("\n".join(header_lines) + f"\n[본문]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def _enhance_query_with_context(self, query: str) -> str:
        """이전 사용자 질문을 리트리버 쿼리에 접두어로 추가한다.

        LLM은 messages로 대화 이력을 받지만 리트리버는 쿼리 문자열만 본다.
        팔로우업 질문이 기관명·문서명을 생략해도 올바른 청크를 찾으려면
        이전 질문(기관명 포함)을 검색 쿼리에 명시해야 한다.
        어시스턴트 답변은 길고 노이즈가 많아 임베딩을 분산시키므로 제외한다.
        """
        prev_user_queries = [
            msg["content"][:150]
            for msg in self.memory.get_messages()
            if msg["role"] == "user"
        ]
        if not prev_user_queries:
            return query
        prev = " ".join(prev_user_queries[-2:])  # 최근 1~2개 질문만
        return f"{prev} {query}"

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
        self._last_retrieved_docs = retrieved_docs  # 평가기의 org 필터 추출용

        source_summary = self._build_source_summary(retrieved_docs, query)
        context = self._build_context(retrieved_docs)
        rag_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

        answer, usage = self._call_llm(rag_prompt, stream=stream, query=query)
        if source_summary and source_summary not in answer:
            answer = f"{source_summary}\n\n{answer}".strip()

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
