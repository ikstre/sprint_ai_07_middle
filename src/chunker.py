"""
문서 청킹: Naive 분할과 의미 기반(Semantic) 분할

개선 사항:
- RFP 섹션 패턴 확장 (조항, 표, 별표, 부록 등)
- 한국어 문장 경계 인식 (마침표·줄바꿈 조합)
- 최소 청크 크기 적용 (너무 짧은 청크는 앞 청크에 병합)
- 섹션 간 overlap 추가 (검색 경계 손실 방지)
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",        # 단락 경계 (최우선)
            "\n",          # 줄바꿈
            # ✅ 추가: RFP 문서 구조 마커
            "\n❍ ",        # ❍ 항목 (공공 RFP 특화)
            "\n- ",        # 하이픈 목록
            # ✅ 한국어 종결어미 (구체적인 것부터)
            "습니다. ",
            "됩니다. ",
            "합니다. ",
            "입니다. ",
            "한다. ",
            "된다. ",
            "있다. ",
            "없다. ",
            "다. ",        # 일반 서술형 (위 패턴 미매칭 시)
            ". ",          # 한/영 범용
            ".\n",
            "? ",
            "! ",
            "; ",
            ", ",
            # ❌ " " 와 "" 제거 → 어절/문자 단위 잘림 방지
        ],
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

_SECTION_PATTERNS = [
    # 장·절·편·관·항 단위
    r"^제\s*\d+\s*[장절편관항]",
    # 한글 번호 목록: 가., 나., 다.
    r"^[가-힣]\.\s",
    # 숫자 목록: 1., 2., 1-1., 1.1.
    r"^\d+\.\s+[가-힣A-Z]",
    r"^\d+-\d+[\.-]?\s",
    r"^\d+\.\d+[\.-]?\s",
    # 로마 숫자
    r"^[IVX]+\.\s",
    # 원문자
    r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]",
    # 조항 표기
    r"^제\s*\d+\s*조(\s*\(.*?\))?",
    # 별표·부록·첨부
    r"^(별표|별첨|부록|첨부)\s*\d*",
    # RFP 공통 대분류 키워드
    r"^(사업\s*개요|사업\s*목적|사업\s*범위|사업\s*내용|추진\s*배경|추진\s*목적|추진\s*경위)",
    r"^(제안\s*요청|요구\s*사항|기능\s*요구|성능\s*요구|보안\s*요구|품질\s*요구|인터페이스\s*요구)",
    r"^(제안서\s*작성|제출\s*방법|평가\s*기준|계약\s*조건|입찰\s*참가|참가\s*자격)",
    r"^(기대\s*효과|추진\s*일정|사업\s*예산|과업\s*내용|산출물|납품\s*기준|검수\s*기준)",
    r"^(현황\s*및\s*문제점|개선\s*방향|운영\s*방안|유지\s*보수|기술\s*지원)",
    r"^(보안\s*요건|개인정보|데이터\s*관리|시스템\s*구성|아키텍처)",
    # 표·그림 캡션 (짧은 헤더로 인식)
    r"^[<\[【]\s*(표|그림|[Tt]able|[Ff]igure)\s*\d",
]

_COMPILED_PATTERNS = [re.compile(p, re.MULTILINE) for p in _SECTION_PATTERNS]

# 최소 청크 길이: 이보다 짧으면 이전 청크에 병합
_MIN_CHUNK_SIZE = 80


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return any(p.match(stripped) for p in _COMPILED_PATTERNS)


def _merge_short_chunks(chunks: list[dict], min_size: int) -> list[dict]:
    """min_size보다 짧은 청크를 직전 청크에 병합한다."""
    if not chunks:
        return chunks

    merged: list[dict] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk["text"]) < min_size and merged:
            prev = merged[-1]
            prev["text"] = prev["text"].rstrip() + "\n" + chunk["text"]
        else:
            merged.append(chunk)

    # 인덱스 재부여
    for i, c in enumerate(merged):
        c["chunk_index"] = i
    return merged


def semantic_chunk(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[dict]:
    """RFP 문서 구조(섹션 제목)를 인식하여 의미 단위로 청킹한다.

    1단계: 섹션 경계에서 1차 분할
    2단계: 큰 섹션은 naive_chunk로 재분할
    3단계: 섹션 간 overlap 삽입 (검색 경계 손실 방지)
    4단계: 최소 크기 미달 청크 병합
    """
    lines = text.split("\n")
    sections: list[dict] = []
    current_header: Optional[str] = None
    current_lines: list[str] = []

    for line in lines:
        if _is_section_header(line):
            if current_lines:
                sections.append({
                    "header": current_header,
                    "text": "\n".join(current_lines).strip(),
                })
            current_header = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "header": current_header,
            "text": "\n".join(current_lines).strip(),
        })

    # 각 섹션을 적절한 크기로 분할
    chunks: list[dict] = []
    chunk_idx = 0
    prev_tail = ""  # 섹션 간 overlap용: 이전 섹션의 마지막 부분

    for section in sections:
        section_text = section["text"]
        if not section_text:
            continue

        # 섹션 간 overlap: 이전 섹션 끝부분을 현재 섹션 앞에 붙임
        if prev_tail and chunk_overlap > 0:
            section_text_with_overlap = prev_tail + "\n" + section_text
        else:
            section_text_with_overlap = section_text

        if len(section_text) <= chunk_size:
            chunks.append({
                "text": section_text_with_overlap,
                "chunk_index": chunk_idx,
                "section_header": section["header"],
            })
            chunk_idx += 1
        else:
            sub_chunks = naive_chunk(section_text_with_overlap, chunk_size, chunk_overlap)
            for sc in sub_chunks:
                sc["chunk_index"] = chunk_idx
                sc["section_header"] = section["header"]
                chunks.append(sc)
                chunk_idx += 1

        # 다음 섹션을 위한 tail 저장
        prev_tail = section_text[-chunk_overlap:] if len(section_text) > chunk_overlap else section_text

    # 최소 크기 미달 청크 병합
    chunks = _merge_short_chunks(chunks, _MIN_CHUNK_SIZE)
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
        chunk_size: 청크 크기 (문자 수)
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
