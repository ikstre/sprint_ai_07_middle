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


def _extract_text_from_txt(file_path: str) -> str:
    """TXT 파일에서 텍스트를 읽는다. UTF-8 → CP949 순으로 인코딩을 시도한다."""
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"TXT 파일 인코딩을 감지할 수 없습니다: {file_path}")


def _extract_text_from_csv(
    file_path: str,
    text_columns: Optional[list[str]] = None,
) -> tuple[str, list[dict]]:
    """CSV 파일을 읽어 (전체 텍스트, 행별 메타데이터 리스트)를 반환한다.

    Args:
        text_columns: 본문으로 사용할 컬럼명 리스트.
                      None이면 문자열 컬럼을 자동 감지한다.

    Returns:
        (joined_text, rows_as_dicts)
        - joined_text: 모든 행을 이어붙인 텍스트 (단일 문서로 청킹할 때 사용)
        - rows_as_dicts: 행별 dict 리스트 (행 단위 문서로 활용할 때 사용)
    """
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(file_path, encoding=encoding, dtype=str)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        raise ValueError(f"CSV 파일 인코딩을 감지할 수 없습니다: {file_path}")

    df = df.fillna("")

    if text_columns:
        use_cols = [c for c in text_columns if c in df.columns]
    else:
        # 문자열 컬럼 자동 감지 (숫자 전용 컬럼 제외)
        use_cols = [
            c for c in df.columns
            if not pd.to_numeric(df[c], errors="coerce").notna().all()
        ]

    if not use_cols:
        use_cols = list(df.columns)

    rows_as_dicts = df.to_dict(orient="records")

    parts = []
    for _, row in df.iterrows():
        row_text = "  ".join(
            f"{col}: {row[col]}" for col in use_cols if str(row[col]).strip()
        )
        if row_text.strip():
            parts.append(row_text)

    return "\n".join(parts), rows_as_dicts


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
    """RFP 문서(PDF/HWP/TXT/CSV) 및 메타데이터를 로드하는 클래스"""

    SUPPORTED_EXTS = {".pdf", ".hwp", ".txt", ".csv"}

    def __init__(
        self,
        documents_dir: str,
        metadata_csv: Optional[str] = None,
        csv_text_columns: Optional[list[str]] = None,
        csv_row_per_doc: bool = False,
    ):
        """
        Args:
            documents_dir: 문서 디렉토리 경로
            metadata_csv: 메타데이터 CSV 파일 경로 (data_list.csv)
            csv_text_columns: CSV 파일을 로드할 때 본문으로 사용할 컬럼명 리스트.
                              None이면 문자열 컬럼 자동 감지.
            csv_row_per_doc: True이면 CSV 각 행을 개별 문서로 처리.
                             False(기본값)이면 CSV 전체를 하나의 문서로 처리.
        """
        self.documents_dir = Path(documents_dir)
        self.metadata: Optional[pd.DataFrame] = None
        self.csv_text_columns = csv_text_columns
        self.csv_row_per_doc = csv_row_per_doc
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv)

    def _attach_metadata(self, meta: dict) -> dict:
        """파일명 기반으로 data_list.csv 메타데이터를 병합한다."""
        if self.metadata is None:
            return meta

        filename = meta["filename"]
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
            meta.update(matched.iloc[0].to_dict())
        return meta

    def load_single(self, file_path: str) -> list[dict]:
        """단일 문서를 로드하여 문서 dict 리스트를 반환한다.

        일반 포맷(PDF/HWP/TXT)은 항상 1개짜리 리스트를 반환한다.
        CSV는 csv_row_per_doc 설정에 따라 1개 또는 행 수만큼 반환한다.
        """
        file_path = str(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext == ".pdf":
            raw_text = _extract_text_from_pdf(file_path)
            docs = [{"text": clean_text(raw_text),
                     "metadata": self._attach_metadata({"filename": filename, "file_path": file_path})}]

        elif ext == ".hwp":
            raw_text = _extract_text_from_hwp(file_path)
            docs = [{"text": clean_text(raw_text),
                     "metadata": self._attach_metadata({"filename": filename, "file_path": file_path})}]

        elif ext == ".txt":
            raw_text = _extract_text_from_txt(file_path)
            docs = [{"text": clean_text(raw_text),
                     "metadata": self._attach_metadata({"filename": filename, "file_path": file_path})}]

        elif ext == ".csv":
            joined_text, rows = _extract_text_from_csv(file_path, self.csv_text_columns)
            base_meta = self._attach_metadata({"filename": filename, "file_path": file_path})

            if self.csv_row_per_doc:
                # 행 하나를 개별 문서로 처리
                docs = []
                for i, row in enumerate(rows):
                    row_text = "  ".join(
                        f"{k}: {v}" for k, v in row.items() if str(v).strip()
                    )
                    if not row_text.strip():
                        continue
                    row_meta = {**base_meta, **{k: str(v) for k, v in row.items()}, "row_index": i}
                    docs.append({"text": clean_text(row_text), "metadata": row_meta})
            else:
                # CSV 전체를 하나의 문서로 처리
                docs = [{"text": clean_text(joined_text), "metadata": base_meta}]

        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext} ({file_path})")

        return docs

    def load_all(self) -> list[dict]:
        """문서 디렉토리의 모든 지원 파일(PDF/HWP/TXT/CSV)을 로드한다."""
        documents = []

        files = sorted(self.documents_dir.rglob("*"))
        for fp in files:
            if fp.suffix.lower() not in self.SUPPORTED_EXTS:
                continue
            # data_list.csv 자체는 메타데이터 파일이므로 제외
            if fp.name == "data_list.csv":
                continue
            try:
                docs = self.load_single(str(fp))
                documents.extend(docs)
                label = f"{len(docs)}행" if fp.suffix.lower() == ".csv" and self.csv_row_per_doc else f"{len(docs[0]['text'])} chars"
                print(f"  ✓ 로드 완료: {fp.name} ({label})")
            except Exception as e:
                print(f"  ✗ 로드 실패: {fp.name} - {e}")

        print(f"\n총 {len(documents)}개 문서 로드 완료")
        return documents

    def get_metadata_summary(self) -> Optional[pd.DataFrame]:
        """메타데이터 CSV 요약을 반환한다."""
        if self.metadata is not None:
            return self.metadata.describe(include="all")
        return None
