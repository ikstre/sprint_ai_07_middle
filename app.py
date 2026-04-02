"""
입찰메이트 RFP 분석 AI - Streamlit 챗봇
"""
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import Config
from src.rag_pipeline import RAGPipeline


# ── 페이지 설정 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="입찰메이트 RFP 분석 AI",
    page_icon="📋",
    layout="wide",
)

st.title("📋 입찰메이트 RFP 분석 AI")
st.caption("제안요청서(RFP)의 핵심 내용을 빠르게 파악하세요.")


# ── 사이드바 ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    model = st.selectbox(
        "LLM 모델",
        ["gpt-5-mini", "gpt-5-nano", "gpt-5"],
        index=0,
    )
    retrieval_method = st.selectbox(
        "검색 방식",
        ["similarity", "mmr", "hybrid"],
        index=0,
    )
    top_k = st.slider("Top-K (검색 결과 수)", 3, 15, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    use_reranker = st.checkbox("Re-ranking 사용", value=False)
    use_multi_query = st.checkbox("Multi-Query 사용", value=False)

    st.divider()
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        if "pipeline" in st.session_state:
            st.session_state.pipeline.reset_conversation()
        st.rerun()

    st.divider()
    st.markdown("### 💡 질문 예시")
    examples = [
        "국민연금공단이 발주한 이러닝시스템 사업 요구사항을 정리해 줘.",
        "한국원자력연구원 선량평가시스템 고도화 사업의 목적은?",
        "고려대학교와 광주과학기술원 학사 시스템 사업을 비교해 줘.",
        "기초과학연구원 극저온시스템 사업에서 AI 기반 예측 요구사항이 있나?",
        "교육 관련 사업을 발주한 기관들을 모두 알려줘.",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state.example_query = ex


# ── 파이프라인 초기화 ─────────────────────────────────────────────
@st.cache_resource
def init_pipeline(model_name, r_method, k, temp, reranker, multi_q):
    config = Config(
        scenario="B",
        openai_chat_model=model_name,
        metadata_csv="data/data_list.csv",
        vectordb_dir="data/vectordb",
        retrieval_method=r_method,
        retrieval_top_k=k,
        temperature=temp,
        use_reranker=reranker,
        use_multi_query=multi_q,
    )
    pipeline = RAGPipeline(config)
    pipeline.initialize_vectorstore()
    return pipeline


try:
    pipeline = init_pipeline(
        model, retrieval_method, top_k, temperature, use_reranker, use_multi_query
    )
    # 설정 변경 반영
    pipeline.config.retrieval_method = retrieval_method
    pipeline.config.retrieval_top_k = top_k
    pipeline.config.openai_chat_model = model
    pipeline.config.temperature = temperature
    pipeline.config.use_reranker = use_reranker
    pipeline.config.use_multi_query = use_multi_query
    st.session_state.pipeline = pipeline
except Exception as e:
    st.error(
        f"벡터스토어를 로딩할 수 없습니다. 먼저 인덱싱을 실행하세요.\n\n"
        f"```bash\npython scripts/index_documents.py\n```\n\n오류: {e}"
    )
    st.stop()


# ── 대화 UI ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 참고 문서"):
                for src in msg["sources"]:
                    amount = src.get("사업금액", "N/A")
                    try:
                        amount = f"{int(float(amount)):,}원" if amount != "N/A" else "N/A"
                    except (ValueError, TypeError):
                        pass
                    st.markdown(f"- **{src['발주기관']}** | {src['사업명']} | {amount}")
        if msg.get("timing"):
            st.caption(msg["timing"])

# 예시 버튼 처리
if "example_query" in st.session_state:
    user_input = st.session_state.pop("example_query")
else:
    user_input = st.chat_input("RFP에 대해 질문하세요...")

if user_input:
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 생성하는 중..."):
            # 메타데이터 필터 추출 시도
            where_filter = pipeline.extract_metadata_filter(user_input)
            result = pipeline.query(
                user_input,
                retrieval_method=retrieval_method,
                top_k=top_k,
                where=where_filter,
            )

        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("📎 참고 문서"):
                for src in result["sources"]:
                    amount = src.get("사업금액", "N/A")
                    try:
                        amount = f"{int(float(amount)):,}원" if amount != "N/A" else "N/A"
                    except (ValueError, TypeError):
                        pass
                    st.markdown(f"- **{src['발주기관']}** | {src['사업명']} | {amount}")

        elapsed = result.get("elapsed_time", 0)
        timing = f"⏱️ 총 응답 시간: {elapsed:.2f}s | 검색 문서: {len(result.get('retrieved_docs', []))}개"
        st.caption(timing)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "timing": timing,
    })
