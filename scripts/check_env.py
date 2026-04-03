"""
로컬 환경 설정 진단 스크립트.

사용법:
  python scripts/check_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def mask(value: str) -> str:
    if not value:
        return "(없음)"
    return "*" * len(value)


def check_env_vars() -> bool:
    print("\n[1] 환경 변수")
    ok = True

    openai_key = os.getenv("OPENAI_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN", "")
    autorag_python = os.getenv("AUTORAG_PYTHON", "")

    print(f"  OPENAI_API_KEY : {mask(openai_key)}")
    print(f"  HF_TOKEN       : {mask(hf_token)}")
    print(f"  AUTORAG_PYTHON : {autorag_python or '(없음)'}")

    if not openai_key:
        print("  [WARN] OPENAI_API_KEY 미설정")
        ok = False
    if autorag_python and not Path(autorag_python).exists():
        print(f"  [WARN] AUTORAG_PYTHON 경로 없음: {autorag_python}")
        ok = False

    return ok


def check_openai() -> bool:
    print("\n[2] OpenAI API 연결")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        print("  [SKIP] OPENAI_API_KEY 없음")
        return False

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        models = client.models.list()
        model_ids = sorted(m.id for m in models)

        chat_models = [m for m in model_ids if "gpt" in m]
        embedding_models = [m for m in model_ids if "embedding" in m]

        print(f"  연결 성공 — 총 {len(model_ids)}개 모델 접근 가능")
        print(f"  Chat 모델    : {chat_models if chat_models else '없음'}")
        print(f"  Embedding 모델: {embedding_models if embedding_models else '없음'}")

        if not embedding_models:
            print("  [WARN] 임베딩 모델 없음 → 코드잇에 text-embedding-3-small 권한 요청 필요")
            return False
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_data_dirs() -> bool:
    print("\n[3] 데이터 경로")
    dirs = {
        "data/documents": "원본 문서",
        "data/processed": "전처리 결과",
        "data/vectordb": "벡터DB",
    }
    ok = True
    for path, label in dirs.items():
        p = Path(path)
        if p.exists():
            files = list(p.iterdir())
            print(f"  {path:<20} ✓  ({len(files)}개 항목) — {label}")
        else:
            print(f"  {path:<20} ✗  없음 — {label}")
            ok = False
    return ok


def check_packages() -> bool:
    print("\n[4] 핵심 패키지")
    packages = [
        ("openai", "OpenAI SDK"),
        ("chromadb", "ChromaDB"),
        ("langchain_text_splitters", "LangChain splitter"),
        ("streamlit", "Streamlit"),
        ("pypdf", "PDF 파서"),
    ]
    ok = True
    for pkg, label in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  {label:<22} ✓  {version}")
        except ImportError:
            print(f"  {label:<22} ✗  미설치")
            ok = False
    return ok


def main() -> None:
    print("=" * 50)
    print("환경 진단")
    print(f"Python: {sys.executable}")
    print("=" * 50)

    results = {
        "환경 변수": check_env_vars(),
        "OpenAI 연결": check_openai(),
        "데이터 경로": check_data_dirs(),
        "패키지": check_packages(),
    }

    print("\n" + "=" * 50)
    print("진단 결과 요약")
    print("=" * 50)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<12} {status}")
        if not passed:
            all_pass = False

    print("=" * 50)
    if all_pass:
        print("모든 항목 정상. 인덱싱을 시작하세요:")
        print("  python scripts/index_documents.py")
    else:
        print("위 FAIL 항목을 먼저 해결하세요.")


if __name__ == "__main__":
    main()
