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


class DocumentLoader:
    """RFP 문서(PDF/HWP) 및 메타데이터를 로드하는 클래스"""

    def __init__(self, documents_dir: str, metadata_csv: Optional[str] = None):
        self.documents_dir = Path(documents_dir)
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
                meta.update(row)

        return {"text": text, "metadata": meta}

    def load_all(self) -> list[dict]:
        """문서 디렉토리의 모든 PDF/HWP 파일을 로드한다."""
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
