import os
import json
import pandas as pd

from configs.config import Config
from src.rag_pipeline import RAGPipeline


EVAL_CSV_PATH = "evaluation/eval_hybrid_k5.csv"
TARGET_QID = "q72"
COLLECTION_NAME = "rfp_chunk1200"
TOP_K = 5


def safe_preview(text, n=500):
    if not isinstance(text, str):
        return ""
    return text[:n].replace("\n", " ").strip()


def main():
    if not os.path.exists(EVAL_CSV_PATH):
        raise FileNotFoundError(f"평가 CSV를 찾을 수 없습니다: {EVAL_CSV_PATH}")

    df = pd.read_csv(EVAL_CSV_PATH)

    if "id" not in df.columns or "question" not in df.columns:
        raise ValueError("CSV에 'id' 또는 'question' 컬럼이 없습니다.")

    row = df[df["id"] == TARGET_QID]
    if row.empty:
        raise ValueError(f"{TARGET_QID} 행을 CSV에서 찾지 못했습니다.")

    question = row.iloc[0]["question"]
    answer = row.iloc[0]["answer"]

    print("=" * 100)
    print(f"[평가 ID] {TARGET_QID}")
    print(f"[질문] {question}")
    print(f"\n[기존 저장 답변]\n{answer}")
    print("=" * 100)

    # ✅ evaluation과 동일한 config 구조
    config = Config(
        scenario="B",
        metadata_csv="data/data_list.csv",
        vectordb_dir="data/vectordb",
        retrieval_method="hybrid",
        retrieval_top_k=TOP_K,
    )

    # ✅ pipeline 생성 및 vectorstore 연결
    pipeline = RAGPipeline(config)
    pipeline.initialize_vectorstore(collection_name=COLLECTION_NAME)
    retriever = pipeline.retriever

    # ✅ retriever 호출 (k= 제거 → 위치 인자 사용)
    if hasattr(retriever, "hybrid_search"):
        docs = retriever.hybrid_search(question, TOP_K)
    elif hasattr(retriever, "similarity_search"):
        docs = retriever.similarity_search(question, TOP_K)
    elif hasattr(retriever, "retrieve"):
        docs = retriever.retrieve(question, TOP_K)
    else:
        raise AttributeError("retriever에서 사용할 수 있는 검색 메서드를 찾지 못했습니다.")

    print(f"\n[재조회 문서 수] {len(docs)}")
    print("=" * 100)

    for i, doc in enumerate(docs, start=1):
        print(f"\n### Retrieved Doc {i}")
        print(f"[type] {type(doc)}")
        print(f"[raw repr] {repr(doc)}")
        print("-" * 100)


if __name__ == "__main__":
    main()