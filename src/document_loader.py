"""
문서 로더: PDF와 HWP 파일을 텍스트로 변환
"""

import os
import re
import struct
import zlib
from pathlib import Path
from typing import Optional

import pandas as pd


_META_KEY_ALIASES = {
    "발주기관": "발주기관",
    "발주 기관": "발주기관",
    "발주처": "발주기관",
    "기관명": "발주기관",
    "사업명": "사업명",
    "사업명칭": "사업명",
    "사업금액": "사업금액",
    "사업 금액": "사업금액",
    "공고번호": "공고번호",
    "공고 번호": "공고번호",
    "공고차수": "공고차수",
    "공고 차수": "공고차수",
    "공개일자": "공개일자",
    "공개 일자": "공개일자",
    "입찰참여시작일": "입찰참여시작일",
    "입찰 참여 시작일": "입찰참여시작일",
    "입찰참여마감일": "입찰참여마감일",
    "입찰 참여 마감일": "입찰참여마감일",
    "사업요약": "사업요약",
    "사업 요약": "사업요약",
}


def _normalize_meta_keys(meta: dict) -> dict:
    """CSV 컬럼명처럼 공백이 섞인 메타데이터 키를 canonical 형태로 통일한다."""
    normalized: dict = {}
    for key, value in meta.items():
        canonical = _META_KEY_ALIASES.get(str(key), str(key))
        normalized[canonical] = value
    return normalized


def _extract_text_from_hwp(file_path: str) -> str:
    """olefile을 사용하여 HWP 파일에서 텍스트를 추출한다."""
    try:
        import olefile
    except ImportError as exc:
        raise ImportError("olefile 패키지를 설치하세요: pip install olefile") from exc

    if not olefile.isOleFile(file_path):
        raise ValueError(f"유효한 HWP(OLE) 파일이 아닙니다: {file_path}")

    ole = olefile.OleFileIO(file_path)
    texts = []

    for stream_name in ole.listdir():
        joined = "/".join(stream_name)
        if not joined.startswith("BodyText/Section"):
            continue

        data = ole.openstream(stream_name).read()

        is_compressed = False
        if ole.exists("FileHeader"):
            header = ole.openstream("FileHeader").read()
            if len(header) > 36:
                flags = struct.unpack_from("<I", header, 36)[0]
                is_compressed = bool(flags & 1)

        if is_compressed:
            try:
                data = zlib.decompress(data, -15)
            except zlib.error:
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    continue

        section_text = _parse_hwp_body_text(data)
        if section_text:
            texts.append(section_text)

    ole.close()
    return "\n".join(texts)


def _parse_hwp_body_text(data: bytes) -> str:
    """HWP BodyText 바이너리 레코드에서 텍스트를 파싱한다."""
    texts = []
    i = 0
    while i < len(data) - 4:
        header = struct.unpack_from("<I", data, i)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4

        if size == 0xFFF:
            if i + 4 > len(data):
                break
            size = struct.unpack_from("<I", data, i)[0]
            i += 4

        if i + size > len(data):
            break

        if tag_id == 67:  # HWPTAG_PARA_TEXT
            record_data = data[i : i + size]
            text = _decode_hwp_para_text(record_data)
            if text.strip():
                texts.append(text.strip())

        i += size

    return "\n".join(texts)


def _decode_hwp_para_text(record_data: bytes) -> str:
    """HWPTAG_PARA_TEXT 레코드에서 유니코드 텍스트를 디코딩한다."""
    chars = []
    j = 0
    while j < len(record_data) - 1:
        code = struct.unpack_from("<H", record_data, j)[0]
        j += 2

        if code < 2:
            continue
        if code in (2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
            j += 12
            continue
        if code in (10, 13):
            chars.append("\n")
        elif code == 9:
            chars.append("\t")
        elif code == 24:
            chars.append("-")
        elif code in (30, 31):
            chars.append(" ")
        elif code > 31:
            chars.append(chr(code))

    return "".join(chars)


def _extract_text_from_pdf(file_path: str) -> str:
    """pdfplumber를 사용하여 PDF에서 텍스트를 추출한다."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber 패키지를 설치하세요: pip install pdfplumber") from exc

    texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n".join(texts)


def clean_text(text: str) -> str:
    """추출된 텍스트를 정리한다."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()

def apply_filter(text: str) -> str:
    """단계별 전처리 필터 적용"""
    text = filter_stage1(text)
    text = filter_stage2(text)
    text = filter_stage3(text)
    return text

def filter_stage1(text: str) -> str:
    """1차 필터: 기본 줄 정리 + 단순 노이즈 제거"""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 특수문자만 있는 줄 제거
        if re.fullmatch(r"[-=~_*#·.•|\\/:]+", line):
            continue

        # 너무 짧은 잡음 제거
        if len(line) <= 2:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def filter_stage2(text: str) -> str:
    """2차 필터: 보호 규칙 적용"""
    text = re.sub(r"\b([A-Za-z])/([A-Za-z])\b", r"\1__SLASH__\2", text)  # S/W, H/W
    text = re.sub(r"\be-mail\b", "e__HYPHEN__mail", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\.(?=\d)", "__DOT__", text)                   # 4.5
    text = re.sub(r"(?<=\d),(?=\d)", "__COMMA__", text)                  # 1,234
    text = re.sub(r"\b([A-Za-z]+)-([가-힣A-Za-z0-9]+)-(\d+)\b", r"\1__HYPHEN__\2__HYPHEN__\3", text)

    text = text.replace("__SLASH__", "/")
    text = text.replace("__HYPHEN__", "-")
    text = text.replace("__DOT__", ".")
    text = text.replace("__COMMA__", ",")
    text = text.replace("e__HYPHEN__mail", "e-mail")

    return text

def filter_stage3(text: str) -> str:
    """3차 필터: 문서 구조 노이즈 제거"""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 번호만 있는 줄 제거: 1. / 1.1 / 2.3.1 / (1) / ① / i
        if re.fullmatch(r"(\d+(\.\d+)*\.?)|(\(\d+\))|([①-⑳])|([ivxlcdmIVXLCDM]+)", line):
            continue

        # 목차성 점선 줄 제거: 4개 이상 연속된 점이 줄 끝(선택적 페이지 번호 포함)에 있을 때만
        # 본문 중 말줄임표(…/...)가 있는 줄이 통째로 버려지는 것을 방지한다.
        if re.search(r"\.{4,}\s*\d*\s*$", line):
            continue

        # 메타데이터성 줄 제거
        if re.match(r"^(문서번호|개정번호|버전|작성일|승인일)\s*[:：]?", line):
            continue

        # 불릿만 있는 줄 제거
        if re.fullmatch(r"[•·▪▫◦○●□■◆◇▶▷※]+", line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

class DocumentLoader:
    """RFP 문서(PDF/HWP) 및 메타데이터를 로드하는 클래스"""

    def __init__(
        self,
        documents_dir: str,
        metadata_csv: Optional[str] = None,
        csv_text_columns: Optional[list[str]] = None,
        csv_row_per_doc: bool = False,
    ):
        self.documents_dir = Path(documents_dir)
        self.csv_text_columns = csv_text_columns
        self.csv_row_per_doc = csv_row_per_doc
        self.metadata: Optional[pd.DataFrame] = None
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv)

    def load_single(self, file_path: str) -> dict:
        """단일 문서를 로드하여 텍스트와 메타데이터를 반환한다."""
        file_path = str(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            raw_text = _extract_text_from_pdf(file_path)
        elif ext == ".hwp":
            raw_text = _extract_text_from_hwp(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")

        text = clean_text(raw_text)
        text = apply_filter(text)
        filename = os.path.basename(file_path)

        meta = {"filename": filename, "file_path": file_path}
        if self.metadata is not None:
            matched = (
                self.metadata[self.metadata["파일명"] == filename]
                if "파일명" in self.metadata.columns
                else pd.DataFrame()
            )
            if matched.empty:
                for col in self.metadata.columns:
                    if "파일" in col or "file" in col.lower():
                        matched = self.metadata[
                            self.metadata[col]
                            .astype(str)
                            .str.contains(
                                re.escape(os.path.splitext(filename)[0]),
                                case=False,
                                na=False,
                            )
                        ]
                        if not matched.empty:
                            break
            if not matched.empty:
                row = matched.iloc[0].to_dict()
                meta.update(_normalize_meta_keys(row))

        return {"text": text, "metadata": meta}

    def load_from_csv(self) -> list[dict]:
        """CSV의 각 행을 개별 문서로 로드한다 (csv_row_per_doc=True 전용)."""
        if self.metadata is None:
            raise ValueError("csv_row_per_doc 사용 시 --metadata-csv 경로가 필요합니다.")

        text_cols = self.csv_text_columns or self._detect_text_columns()
        if not text_cols:
            raise ValueError(
                "CSV에서 텍스트 컬럼을 감지하지 못했습니다. "
                "--csv-text-columns 옵션으로 컬럼명을 지정하세요."
            )

        meta_cols = [c for c in self.metadata.columns if c not in text_cols]
        documents = []
        for idx, row in self.metadata.iterrows():
            text_parts = []
            for col in text_cols:
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip():
                    text_parts.append(str(val).strip())

            text = "\n\n".join(text_parts)
            if not text.strip():
                print(f"  ⚠ 행 {idx}: 텍스트 없음 (건너뜀)")
                continue

            text = clean_text(text)
            text = apply_filter(text)

            raw_meta = {c: row.get(c, "") for c in meta_cols}
            meta = _normalize_meta_keys(raw_meta)
            filename_hint = str(row.get("파일명", row.get("사업명", f"row_{idx}")))
            meta["filename"] = filename_hint
            meta["file_path"] = filename_hint

            documents.append({"text": text, "metadata": meta})
            print(f"  ✓ 로드 완료: {filename_hint[:60]} ({len(text)} chars)")

        print(f"\n총 {len(documents)}개 문서 로드 완료 (CSV 행)")
        return documents

    def _detect_text_columns(self) -> list[str]:
        """텍스트 컬럼을 휴리스틱으로 감지한다."""
        if self.metadata is None:
            return []
        candidates = ["텍스트", "text", "content", "본문", "내용"]
        found = [c for c in candidates if c in self.metadata.columns]
        if found:
            return found
        # 문자열 컬럼 중 평균 길이가 가장 긴 컬럼
        str_cols = self.metadata.select_dtypes(include="object").columns
        if len(str_cols) == 0:
            return []
        avg_lens = {c: self.metadata[c].dropna().astype(str).str.len().mean() for c in str_cols}
        best = max(avg_lens, key=avg_lens.get)
        return [best]

    def load_all(self) -> list[dict]:
        """문서를 로드한다. csv_row_per_doc=True면 CSV 행 기반, 아니면 파일 기반."""
        if self.csv_row_per_doc:
            return self.load_from_csv()

        documents = []
        supported_exts = {".pdf", ".hwp"}

        files = sorted(self.documents_dir.rglob("*"))
        for fp in files:
            if fp.suffix.lower() in supported_exts:
                try:
                    doc = self.load_single(str(fp))
                    documents.append(doc)
                    print(f"  ✓ 로드 완료: {fp.name} ({len(doc['text'])} chars)")
                except Exception as e:
                    print(f"  ✗ 로드 실패: {fp.name} - {e}")

        print(f"\n총 {len(documents)}개 문서 로드 완료")
        return documents

    def get_metadata_summary(self) -> Optional[pd.DataFrame]:
        """메타데이터 CSV 요약을 반환한다."""
        if self.metadata is not None:
            return self.metadata.describe(include="all")
        return None
