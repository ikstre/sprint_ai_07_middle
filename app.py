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
from configs import paths
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

    if scenario_key == "B":
        _collections = [
            "rfp_chunk600",      # CSV 기반 정제 데이터, 600자 (권장)
            "rfp_chunk1200",     # HWP 원문 추출, 1200자
        ]
    else:
        _collections = [
            "rfp_chunk1200_a",
            "rfp_chunk800_a",
            "rfp_chunk1200_a_sroberta",
        ]

    collection = st.selectbox(
        "컬렉션 (청크 크기)",
        _collections,
        index=0,
        help="인덱싱 시 사용한 청크 크기별 컬렉션을 선택합니다.",
    )

    if scenario_key == "B":
        model = st.selectbox(
            "LLM 모델",
            ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o"],
            index=0,
        )
        temperature = 0.1       # gpt-5 미지원, 내부적으로만 유지
        top_p = 0.9             # gpt-5 미지원, 내부적으로만 유지
        hf_max_new_tokens = 1024
        hf_load_in_4bit = False
        hf_embedding_model = "text-embedding-3-small"
        hf_embedding_dim = 512
    else:
        # paths.MODEL_DIR 기반 로컬 모델 목록 (.env의 MODEL_DIR 또는 SRV_DATA_DIR로 제어)
        _m = paths.MODEL_DIR
        _LOCAL_CHAT_MODELS = {
            "EXAONE-4.0-1.2B  (2.4G, 빠름)":          f"{_m}/exaone/EXAONE-4.0-1.2B",
            "EXAONE-Deep-2.4B (4.5G, 균형)":           f"{_m}/exaone/EXAONE-Deep-2.4B",
            "EXAONE-Deep-7.8B (15G,  고성능)":         f"{_m}/exaone/EXAONE-Deep-7.8B",
            "Gemma3-4B        (8.1G, 다국어)":          f"{_m}/gemma/Gemma3-4B",
            "Gemma4-E4B       (15G,  다국어)":          f"{_m}/gemma/Gemma4-E4B",
            "kanana-1.5-2.1b  (4.4G, 경량)":           f"{_m}/kanana/kanana-1.5-2.1b",
            "Midm-2.0-Mini    (4.4G, 경량)":           f"{_m}/midm/Midm-2.0-Mini",
        }
        model_label = st.selectbox(
            "LLM 모델 (로컬)",
            list(_LOCAL_CHAT_MODELS.keys()),
            index=0,
        )
        model = _LOCAL_CHAT_MODELS[model_label]
        _force_4bit = False

        _LOCAL_EMB_MODELS = {
            "BGE-m3-ko        (2.2G, 1024-dim, 한국어 특화)": f"{_m}/embeddings/BGE-m3-ko",
            "ko-sroberta-multitask (846M, 768-dim, 경량)": f"{_m}/embeddings/ko-sroberta-multitask",
        }
        emb_label = st.selectbox(
            "임베딩 모델 (로컬)",
            list(_LOCAL_EMB_MODELS.keys()),
            index=0,
        )
        hf_embedding_model = _LOCAL_EMB_MODELS[emb_label]
        hf_embedding_dim = 1024 if "BGE" in emb_label else 768

        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        top_p = st.slider("Top-P", 0.5, 1.0, 0.9, 0.05,
                          help="0.8~0.95: 일반 권장. 1.0: 비활성화(greedy 계열 권장)")
        hf_max_new_tokens = st.slider("최대 생성 토큰", 256, 4096, 1024, 256,
                                      help="생성할 최대 토큰 수. 높을수록 긴 답변 가능.")
        hf_load_in_4bit = st.checkbox(
                "4-bit 양자화 (bitsandbytes)",
                value=False,
                help="VRAM이 부족할 때 사용. bitsandbytes 패키지 및 CUDA 필요.",
            )

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
        hf_embedding_model,
        retrieval_method,
        top_k,
        temperature,
        top_p,
        hf_max_new_tokens,
        use_reranker,
        use_multi_query,
        reasoning_effort,
        auto_model_routing,
        hf_load_in_4bit,
    )


def _build_config() -> Config:
    base = dict(
        scenario=scenario_key,
        metadata_csv=paths.METADATA_CSV,
        vectordb_dir=paths.VECTORDB_DIR,
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
        base["hf_embedding_model"] = hf_embedding_model
        base["hf_embedding_dim"] = hf_embedding_dim
        base["hf_load_in_4bit"] = hf_load_in_4bit
        base["top_p"] = top_p
        base["hf_max_new_tokens"] = hf_max_new_tokens
    return Config(**base)


if scenario_key == "B" and not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. `.env`를 확인해 주세요.")
    st.stop()
# Scenario A: /srv/shared_data/models/ 로컬 모델은 HF_TOKEN 불필요 → 체크 생략


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
