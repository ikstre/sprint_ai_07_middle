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


st.set_page_config(
    page_title="입찰메이트 RFP 분석 AI",
    page_icon="📋",
    layout="wide",
)

st.title("📋 입찰메이트 RFP 분석 AI")
st.caption("제안요청서(RFP)의 핵심 내용을 빠르게 파악하세요.")


with st.sidebar:
    st.header("⚙️ 설정")

    scenario = st.radio(
        "실행 모드",
        ["B: OpenAI API", "A: 로컬 HuggingFace"],
        index=0,
        horizontal=True,
    )
    scenario_key = scenario.split(":")[0].strip()  # "A" or "B"

    st.divider()

    collection = st.selectbox(
        "컬렉션 (청크 크기)",
        ["rfp_chunk1200", "rfp_chunk800", "rfp_documents"],
        index=0,
        help="인덱싱 시 사용한 청크 크기별 컬렉션을 선택합니다.",
    )

    if scenario_key == "B":
        model = st.selectbox(
            "LLM 모델",
            ["gpt-5-mini", "gpt-5-nano", "gpt-5"],
            index=0,
        )
        temperature = 0.1   # gpt-5 미지원, 내부적으로만 유지
    else:
        model = st.selectbox(
            "LLM 모델",
            ["google/gemma-3-4b-it"],
            index=0,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    retrieval_method = st.selectbox(
        "검색 방식",
        ["similarity", "mmr", "hybrid"],
        index=0,
    )
    top_k = st.slider("Top-K (검색 결과 수)", 3, 15, 5)
    use_reranker = st.checkbox("Re-ranking 사용", value=False)
    use_multi_query = st.checkbox("Multi-Query 사용", value=False)

    if scenario_key == "B":
        st.divider()
        st.markdown("**OpenAI 고급 설정**")
        reasoning_effort = st.select_slider(
            "Reasoning Effort",
            options=["low", "medium", "high"],
            value="medium",
            help="low: 빠름/저렴, high: 정확/느림. 단순 질문은 low 권장.",
        )
        auto_model_routing = st.checkbox(
            "자동 모델 라우팅",
            value=True,
            help="짧은 질문은 gpt-5-nano, 복잡한 질문은 선택 모델 자동 적용.",
        )
    else:
        reasoning_effort = "medium"
        auto_model_routing = False

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


def _pipeline_signature() -> tuple:
    """세션 내에서 파이프라인 재생성이 필요한 설정만 signature로 사용한다."""
    return (
        scenario_key,
        collection,
        model,
        retrieval_method,
        top_k,
        temperature,
        use_reranker,
        use_multi_query,
        reasoning_effort,
        auto_model_routing,
    )


def _build_config() -> Config:
    base = dict(
        scenario=scenario_key,
        metadata_csv="data/data_list.csv",
        vectordb_dir="data/vectordb",
        retrieval_method=retrieval_method,
        retrieval_top_k=top_k,
        temperature=temperature,
        use_reranker=use_reranker,
        use_multi_query=use_multi_query,
        reasoning_effort=reasoning_effort,
        auto_model_routing=auto_model_routing,
    )
    if scenario_key == "B":
        base["openai_chat_model"] = model
    else:
        base["hf_chat_model"] = model
    return Config(**base)


if scenario_key == "B" and not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. `.env`를 확인해 주세요.")
    st.stop()
if scenario_key == "A" and not os.getenv("HF_TOKEN"):
    st.error("HF_TOKEN이 설정되지 않았습니다. `.env`를 확인해 주세요.")
    st.stop()


try:
    current_sig = _pipeline_signature()
    if st.session_state.get("pipeline_signature") != current_sig:
        pipeline = RAGPipeline(_build_config())
        pipeline.initialize_vectorstore(collection_name=collection)
        st.session_state.pipeline = pipeline
        st.session_state.pipeline_signature = current_sig
    pipeline = st.session_state.pipeline
except Exception as e:
    st.error(
        f"벡터스토어를 로딩할 수 없습니다. 먼저 인덱싱을 실행하세요.\n\n"
        f"```bash\npython scripts/index_documents.py\n```\n\n오류: {e}"
    )
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

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

if "example_query" in st.session_state:
    user_input = st.session_state.pop("example_query")
else:
    user_input = st.chat_input("RFP에 대해 질문하세요...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 생성하는 중..."):
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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "timing": timing,
        }
    )
