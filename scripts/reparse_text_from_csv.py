"""
CSV 행을 기준으로 원본 PDF/HWP를 다시 파싱하여 '텍스트' 컬럼을 재생성한다.

기본 동작:
  - 입력 CSV의 행 순서를 유지
  - 파일명/파일형식으로 /srv/shared_data/pdf 에서 원본 문서를 다시 읽음
  - 파싱 텍스트를 '텍스트' 컬럼에 반영
  - 길이 변화 리포트를 별도 CSV로 저장

예시:
  python scripts/reparse_text_from_csv.py \
    --input-csv data/data_list_cleaned.csv \
    --output-csv data/data_list_reparsed.csv \
    --report-csv data/reparse_report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_cleaner import SmartOriginFrequencyMatcher


def sanitize_text(value: str) -> str:
    return str(value).encode("utf-8", errors="ignore").decode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV 기준 전수 재파싱")
    parser.add_argument(
        "--input-csv",
        default="data/data_list_cleaned.csv",
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--output-csv",
        default="data/data_list_reparsed.csv",
        help="재파싱 결과 CSV 경로",
    )
    parser.add_argument(
        "--report-csv",
        default="data/reparse_report.csv",
        help="길이 변화 리포트 CSV 경로",
    )
    parser.add_argument(
        "--raw-dir",
        default="/srv/shared_data/pdf",
        help="원본 PDF/HWP 디렉토리",
    )
    parser.add_argument(
        "--min-improvement",
        type=int,
        default=200,
        help="의미 있는 길이 증가로 볼 최소 증가량",
    )
    return parser.parse_args()


def parse_file(matcher: SmartOriginFrequencyMatcher, raw_dir: Path, filename: str, filetype: str) -> tuple[str, str]:
    file_path = raw_dir / filename
    if not file_path.exists():
        return "", "missing_source"

    filetype = str(filetype).strip().lower()
    try:
        if filetype == "hwp" or file_path.suffix.lower() == ".hwp":
            text = matcher.parse_hwp(str(file_path))
        elif filetype == "pdf" or file_path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(str(file_path))
            text = " ".join(d.page_content for d in loader.load())
        else:
            return "", f"unsupported:{filetype or file_path.suffix.lower()}"
    except Exception as exc:
        return "", f"parse_error:{type(exc).__name__}"

    text = matcher.clean_text_content(text)
    if not text:
        return "", "empty_parsed"
    return text, "reparsed"


def classify_row(old_len: int, new_len: int, status: str, min_improvement: int) -> str:
    if status != "reparsed":
        return status
    if old_len == 0 and new_len > 0:
        return "recovered_from_empty"
    if new_len - old_len >= min_improvement:
        return "expanded"
    if new_len == old_len:
        return "unchanged"
    if new_len > old_len:
        return "slightly_expanded"
    return "shorter_after_reparse"


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    report_csv = Path(args.report_csv)
    raw_dir = Path(args.raw_dir)

    if not input_csv.exists():
        raise SystemExit(f"입력 CSV를 찾을 수 없습니다: {input_csv}")
    if not raw_dir.exists():
        raise SystemExit(f"원본 디렉토리를 찾을 수 없습니다: {raw_dir}")

    df = pd.read_csv(input_csv).fillna("")
    if "텍스트" not in df.columns:
        df["텍스트"] = ""

    matcher = SmartOriginFrequencyMatcher(str(raw_dir), str(input_csv), "")
    reports: list[dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="reparse"):
        filename = str(row.get("파일명", "")).strip()
        filetype = str(row.get("파일형식", "")).strip()
        old_text = str(row.get("텍스트", ""))
        old_len = len(old_text)

        if not filename:
            reports.append(
                {
                    "row_index": idx,
                    "파일명": filename,
                    "파일형식": filetype,
                    "old_len": old_len,
                    "new_len": old_len,
                    "delta": 0,
                    "status": "missing_filename",
                }
            )
            continue

        new_text, status = parse_file(matcher, raw_dir, filename, filetype)
        if status == "reparsed":
            df.at[idx, "텍스트"] = new_text
            new_len = len(new_text)
        else:
            new_len = old_len

        reports.append(
            {
                "row_index": idx,
                "파일명": filename,
                "파일형식": filetype,
                "old_len": old_len,
                "new_len": new_len,
                "delta": new_len - old_len,
                "status": classify_row(old_len, new_len, status, args.min_improvement),
            }
        )

    df["텍스트"] = df["텍스트"].astype(str).map(sanitize_text)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report_df = pd.DataFrame(reports).sort_values(["status", "delta"], ascending=[True, False])
    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    print(f"입력 CSV   : {input_csv}")
    print(f"출력 CSV   : {output_csv}")
    print(f"리포트 CSV : {report_csv}")
    print("\n상태별 집계")
    print(report_df["status"].value_counts().to_string())
    print("\n길이 증가 상위 20건")
    print(report_df.sort_values("delta", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
