# Codebase Map

## 목적
- 모듈별 책임 경계를 명확히 하여 변경 영향 범위를 빠르게 파악합니다.
- 신규 엔지니어가 디버깅/개선 포인트를 빠르게 찾을 수 있게 합니다.

## 엔트리포인트
- `app.py`: 메인 Streamlit 앱. 세션 단위 파이프라인/대화 메모리 유지.
- `scripts/index_documents.py`: 문서 로딩, 청킹, 임베딩, 벡터스토어 적재.
- `scripts/run_evaluation.py`: `core`/`detailed` 평가 실행 및 리포트 저장.
- `scripts/check_release_gate.py`: `core` 평가 + 게이트 판정을 단일 커맨드로 실행.
- `scripts/prepare_autorag_data.py`: AutoRAG용 `qa.parquet`, `corpus.parquet` 생성.
- `scripts/run_autorag_optimization.py`: AutoRAG 탐색/최적화 실행.

## 핵심 모듈
- `configs/config.py`
  - 전체 런타임 설정 (Scenario A/B, Retrieval, Generation, VectorDB).
- `src/document_loader.py`
  - `pdf`, `hwp` 파일 로딩 + `data_list.csv` 메타데이터 병합.
- `src/chunker.py`
  - `naive`, `semantic` 청킹 로직.
- `src/embedder.py`
  - 임베딩 생성 (`OpenAI` 또는 `SentenceTransformer`) + VectorStore 래퍼.
- `src/retriever.py`
  - similarity/MMR/hybrid, multi-query, reranker.
- `src/generator.py`
  - RAG 프롬프트 생성, LLM 호출, 대화 메모리.
- `src/rag_pipeline.py`
  - 검색/생성/출처 정리를 묶은 오케스트레이션 레이어.

## 평가 모듈
- `src/evaluation/dataset.py`: 평가 질문셋(단일/다중/후속/out-of-scope).
- `src/evaluation/retrieval_metrics.py`: hit@k, mrr, ndcg 등.
- `src/evaluation/generation_metrics.py`: keyword recall, field coverage, rouge, meteor, bertscore.
- `src/evaluation/grounding_metrics.py`: grounded token ratio, decline accuracy.
- `src/evaluation/runtime_metrics.py`: latency, token usage 집계.
- `src/evaluation/evaluator.py`: 평가 실행 오케스트레이터.
- `src/evaluator.py`: 하위호환 re-export.

## API/앱 확장
- `apps/autorag_api.py`: AutoRAG 결과를 감싼 FastAPI 서버.
- `apps/autorag_streamlit.py`: AutoRAG 결과를 감싼 Streamlit 앱.
- `src/autorag_runner.py`: trial 폴더 로딩 및 질의 래퍼.

## 변경 가이드
- 검색 로직 변경: `src/retriever.py` + `scripts/run_evaluation.py`로 회귀 확인.
- 생성 프롬프트 변경: `src/generator.py` + `detailed` 평가로 품질 확인.
- 메타데이터 필터 변경: `src/retriever.py`, `src/rag_pipeline.py` 동시 점검.
- 평가 지표 추가: `src/evaluation/*` 모듈 추가 후 `summary_report`와 CLI 출력 반영.
