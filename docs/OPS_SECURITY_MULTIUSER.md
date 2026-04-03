# Ops, Security, Multi-User Guide

## 멀티 컨설턴트 동시 사용 가정
- 현재 `app.py`는 Streamlit `session_state`를 사용하므로 사용자별 대화 이력이 분리됩니다.
- 파이프라인은 세션별로 생성되며, 벡터스토어는 공용 persisted DB(`data/vectordb`)를 조회합니다.
- 동시 요청이 많아지면 Streamlit 단일 프로세스보다 API 서버(FastAPI) + 워커 스케일아웃 구조를 권장합니다.

## 권장 운영 구조 (Scenario B 서버)
- RAG API를 별도 프로세스로 분리하고, Streamlit은 프론트로 사용.
- 서버 측에서 다음을 추가:
  - 요청 단위 timeout
  - rate limiting
  - request id 기반 구조화 로깅
  - 사용자별 대화 메모리 저장소 분리(세션/토큰 기준)

## 보안 체크리스트
- `.env` 파일은 절대 저장소에 커밋하지 않습니다.
- `OPENAI_API_KEY`는 서버 환경변수로 주입하고, 앱 로그에 출력하지 않습니다.
- 예외 메시지에 원문 문서 텍스트가 섞여 노출되지 않도록 주의합니다.
- 필요 시 문서 원문 대신 출처 메타데이터와 발췌 요약만 사용자 응답에 노출합니다.
- 운영 로그에서 개인정보/민감정보 마스킹 규칙을 둡니다.

## Windows 로컬 + GPU 권장안 (Scenario A)
- 벡터DB:
  - Windows 네이티브에서는 `faiss-gpu` 제약이 있어 `chroma` 기본 사용을 권장합니다.
  - GPU 가속은 임베딩/생성/리랭커 쪽에서 활용합니다 (`torch` CUDA, SentenceTransformer GPU).
- 옵션:
  - FAISS GPU가 꼭 필요하면 WSL2(Ubuntu) 환경에서 실행하는 방식을 권장합니다.
  - 네이티브 Windows는 `chroma + GPU 임베딩/리랭커` 조합이 안정적입니다.

## Python 버전 권장 매트릭스
- 기본 RAG 서비스/평가: Python `3.14` 가능 (`requirements.txt`)
- AutoRAG 실험: Python `3.11` 또는 `3.12` 권장 (`requirements.txt`에서 자동 설치 분기)

## 장애 대응 포인트
- 인덱스 누락/불일치: `scripts/index_documents.py` 재실행 후 컬렉션 카운트 확인.
- 응답 지연 급증: `core` 평가에서 `p95_elapsed_time` 추적.
- 환각 증가: `grounded_token_ratio` 하락 여부와 프롬프트/검색 top-k를 동시 점검.
