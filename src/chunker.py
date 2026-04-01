"""
문서 청킹: Naive 분할과 의미 기반(Semantic) 분할
"""

import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────────
# Naive Chunking (RecursiveCharacterTextSplitter 기반)
# ─────────────────────────────────────────────────────────────────

def naive_chunk(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """고정 크기 기반 청킹. 단락/문장 경계를 우선 활용한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "chunk_index": i}
        for i, chunk in enumerate(chunks)
    ]


# ─────────────────────────────────────────────────────────────────
# Semantic Chunking (RFP 구조 기반)
# ─────────────────────────────────────────────────────────────────

# RFP에서 흔히 등장하는 섹션 제목 패턴
_SECTION_PATTERNS = [
    # 한글 번호 패턴: 제1장, 제2절, 1., 1-1., 가., 나. 등
    r"^제\s*\d+\s*[장절편관항]",
    r"^[가-힣]\.\s",
    r"^\d+\.\s+[가-힣]",
    r"^\d+-\d+[\.-]?\s",
    r"^[IVX]+\.\s",
    r"^[①②③④⑤⑥⑦⑧⑨⑩]",
    # 일반적인 섹션 키워드
    r"^(사업\s*개요|사업\s*목적|사업\s*범위|사업\s*내용|추진\s*배경|추진\s*목적)",
    r"^(제안\s*요청|요구\s*사항|기능\s*요구|성능\s*요구|보안\s*요구|품질\s*요구)",
    r"^(제안서\s*작성|제출\s*방법|평가\s*기준|계약\s*조건|입찰\s*참가)",
    r"^(기대\s*효과|일정|예산|과업\s*내용|산출물|납품)",
]

_COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in _SECTION_PATTERNS]


def _is_section_header(line: str) -> bool:
    """주어진 줄이 섹션 헤더인지 판별한다."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    return any(p.match(stripped) for p in _COMPILED_PATTERNS)


def semantic_chunk(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """RFP 문서 구조(섹션 제목)를 인식하여 의미 단위로 청킹한다.

    1차로 섹션 경계에서 분할하고,
    너무 큰 섹션은 naive_chunk로 재분할한다.
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current_header: Optional[str] = None
    current_lines: list[str] = []

    for line in lines:
        if _is_section_header(line):
            # 이전 섹션 저장
            if current_lines:
                sections.append({
                    "header": current_header,
                    "text": "\n".join(current_lines).strip(),
                })
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # 마지막 섹션 저장
    if current_lines:
        sections.append({
            "header": current_header,
            "text": "\n".join(current_lines).strip(),
        })

    # 각 섹션을 적절한 크기로 분할
    chunks: list[dict] = []
    chunk_idx = 0

    for section in sections:
        section_text = section["text"]
        if not section_text:
            continue

        if len(section_text) <= chunk_size:
            chunks.append({
                "text": section_text,
                "chunk_index": chunk_idx,
                "section_header": section["header"],
            })
            chunk_idx += 1
        else:
            # 큰 섹션은 재분할
            sub_chunks = naive_chunk(section_text, chunk_size, chunk_overlap)
            for sc in sub_chunks:
                sc["chunk_index"] = chunk_idx
                sc["section_header"] = section["header"]
                chunks.append(sc)
                chunk_idx += 1

    return chunks


# ─────────────────────────────────────────────────────────────────
# 통합 인터페이스
# ─────────────────────────────────────────────────────────────────

def chunk_document(
    document: dict,
    method: str = "naive",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """문서를 청킹하고 메타데이터를 각 청크에 부착한다.

    Args:
        document: {"text": str, "metadata": dict}
        method: "naive" 또는 "semantic"
        chunk_size: 청크 크기
        chunk_overlap: 청크 간 중첩 크기

    Returns:
        list of {"text": str, "metadata": dict, "chunk_index": int}
    """
    text = document["text"]
    metadata = document.get("metadata", {})

    if method == "semantic":
        raw_chunks = semantic_chunk(text, chunk_size, chunk_overlap)
    else:
        raw_chunks = naive_chunk(text, chunk_size, chunk_overlap)

    result = []
    for chunk in raw_chunks:
        chunk_meta = {**metadata}
        chunk_meta["chunk_index"] = chunk["chunk_index"]
        if "section_header" in chunk:
            chunk_meta["section_header"] = chunk["section_header"]

        result.append({
            "text": chunk["text"],
            "metadata": chunk_meta,
        })

    return result


def chunk_documents(
    documents: list[dict],
    method: str = "naive",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """여러 문서를 일괄 청킹한다."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, method, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        fname = doc.get("metadata", {}).get("filename", "unknown")
        print(f"  {fname}: {len(chunks)} chunks")

    print(f"\n총 {len(all_chunks)}개 청크 생성 완료")
    return all_chunks
