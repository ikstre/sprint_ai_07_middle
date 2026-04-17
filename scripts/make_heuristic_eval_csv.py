import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from tqdm import tqdm

from configs.config import Config
from src.rag_pipeline import RAGPipeline
from src.evaluation.single_dataset import EVALUATION_QUESTIONS


OUTPUT_DIR = "evaluation/heuristic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "heuristic_eval.csv")


def extract_doc_info(docs):
    filenames = []
    chunk_indices = []

    for doc in docs:
        # dict / Document 모두 대응
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            metadata = getattr(doc, "metadata", {})

        filename = metadata.get("filename", "")
        chunk_index = metadata.get("chunk_index", "")

        filenames.append(filename)
        chunk_indices.append(chunk_index)

    return filenames, chunk_indices


def check_doc_consistency(filenames):
    valid_filenames = [f for f in filenames if f]
    if valid_filenames and len(set(valid_filenames)) == 1:
        return "-"
    return "불일치"


def run_hybrid_k5_search(retriever, question):
    """
    hybrid_k5를 명시적으로 호출한다.
    retriever 구현에 따라 k 또는 top_k를 받을 수 있어 방어적으로 처리.
    반환값도 dict/list 모두 대응.
    """
    try:
        result = retriever.hybrid_search(question, k=5)
    except TypeError:
        result = retriever.hybrid_search(question, top_k=5)

    if isinstance(result, dict):
        docs = result.get("documents", [])
    else:
        docs = result

    return docs


def main(limit=None):
    config = Config()
    pipeline = RAGPipeline(config)

    # 기존 인덱싱 컬렉션 사용
    pipeline.initialize_vectorstore("rfp_chunk1200")
    retriever = pipeline.retriever

    results = []

    questions = EVALUATION_QUESTIONS
    if limit:
        questions = questions[:limit]

    for item in tqdm(questions):
        qid = item.get("id")
        question = item.get("question")
        category = item.get("category", "")

        # hybrid_k5 retrieval 수행
        docs = run_hybrid_k5_search(retriever, question)

        filenames, chunk_indices = extract_doc_info(docs)
        consistency = check_doc_consistency(filenames)

        # dataset에서 정답 filename 직접 매핑 불가 → top-1 기준
        answer_doc = filenames[0] if filenames else ""
        answer_chunk_pos = 1 if filenames else ""
        inclusion = "포함" if filenames else "미포함"

        # 추가 체크용 컬럼 계산
        top1_simple = ""
        if filenames and filenames[0]:
            top1_simple = filenames[0].split("_")[0]

        retrieval_type = "단일문서" if consistency == "-" else "혼합문서"
        priority = "높음" if consistency == "불일치" else "낮음"

        keyword_match = "불일치"
        if filenames and filenames[0]:
            question_tokens = question.split()
            if any(token and token in filenames[0] for token in question_tokens):
                keyword_match = "부분일치"

        row = {
            "id": qid,
            "question": question,
            "category": category,
            "answer": "",
            "retrieved_doc1_filename": filenames[0] if len(filenames) > 0 else "",
            "retrieved_doc1_chunk_index": chunk_indices[0] if len(chunk_indices) > 0 else "",
            "retrieved_doc2_filename": filenames[1] if len(filenames) > 1 else "",
            "retrieved_doc2_chunk_index": chunk_indices[1] if len(chunk_indices) > 1 else "",
            "retrieved_doc3_filename": filenames[2] if len(filenames) > 2 else "",
            "retrieved_doc3_chunk_index": chunk_indices[2] if len(chunk_indices) > 2 else "",
            "retrieved_doc4_filename": filenames[3] if len(filenames) > 3 else "",
            "retrieved_doc4_chunk_index": chunk_indices[3] if len(chunk_indices) > 3 else "",
            "retrieved_doc5_filename": filenames[4] if len(filenames) > 4 else "",
            "retrieved_doc5_chunk_index": chunk_indices[4] if len(chunk_indices) > 4 else "",
            "문서일치여부": consistency,
            "정답문서명": answer_doc,
            "정답문서_청크번호": answer_chunk_pos,
            "정답포함여부": inclusion,
            "top1_문서명_간단": top1_simple,
            "retrieval_판단": retrieval_type,
            "검증우선순위": priority,
            "top1_키워드매칭": keyword_match,
            "휴리스틱_메모": "",
        }

        results.append(row)

    if not results:
        print("저장할 결과가 없습니다.")
        return

    keys = results[0].keys()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ CSV 저장 완료: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(limit=args.limit)