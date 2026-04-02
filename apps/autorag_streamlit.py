"""
Streamlit app for AutoRAG optimized trial.

Usage:
  set AUTORAG_TRIAL_DIR=evaluation/autorag_benchmark/0
  streamlit run apps/autorag_streamlit.py
"""

from __future__ import annotations

import os

import streamlit as st

from src.autorag_runner import AutoRAGRuntime


@st.cache_resource
def get_runtime(trial_dir: str) -> AutoRAGRuntime:
    return AutoRAGRuntime(trial_dir=trial_dir)


st.set_page_config(page_title="AutoRAG Chat", page_icon=":robot_face:", layout="wide")
st.title("AutoRAG Optimized Chat")

trial_dir = st.sidebar.text_input(
    "Trial Dir",
    value=os.getenv("AUTORAG_TRIAL_DIR", "evaluation/autorag_benchmark/0"),
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your RFP corpus...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                runtime = get_runtime(trial_dir)
                answer = runtime.ask(question)
                answer_text = str(answer)
            except Exception as exc:
                answer_text = f"Error: {exc}"
        st.markdown(answer_text)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
