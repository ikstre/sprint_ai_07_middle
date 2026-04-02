# sprint_ai_07_middle

기업/공공 RFP 문서를 대상으로 한 RAG 시스템 프로젝트입니다.  
본 저장소는 아래 3가지를 모두 제공합니다.

1. 실서비스용 RAG 앱/파이프라인
2. AutoRAG 기반 자동 최적화 실험 파이프라인
3. 운영/튜닝을 분리한 평가 체계 (`core` vs `detailed`)

---

## 문서 허브

- 아키텍처/개발 흐름: `docs/ENGINEERING_GUIDE.md`
- 코드 책임 분리 맵: `docs/CODEBASE_MAP.md`
- 평가 실행/게이트 기준: `docs/EVALUATION_GUIDE.md`
- 멀티 사용자 운영/보안 가이드: `docs/OPS_SECURITY_MULTIUSER.md`
- AutoRAG 실험/배포 가이드: `docs/AUTORAG_GUIDE.md`
- 노트북 실행 가이드: `docs/NOTEBOOK_GUIDE.md`
- 보고서 템플릿: `docs/REPORT_TEMPLATE.md`

---

## 1) 최종 구현 범위

### A. RAG 서비스
- 문서 로딩: PDF/HWP
- 청킹: naive/semantic
- 검색: similarity/MMR/hybrid + (옵션) multi-query/rerank
- 생성: OpenAI 기반 답변 생성 + 대화 메모리
- UI: Streamlit 앱

핵심 엔트리:
- `app.py`
- `src/rag_pipeline.py`

### B. AutoRAG 실험/배포
- `qa.parquet`, `corpus.parquet` 생성 자동화
- AutoRAG 평가/탐색 실행
- AutoRAG API/Web 실행 스크립트
- FastAPI/Streamlit 래퍼 앱

핵심 엔트리:
- `scripts/prepare_autorag_data.py`
- `scripts/run_autorag_optimization.py`
- `scripts/run_autorag_api.py`
- `scripts/run_autorag_web.py`
- `apps/autorag_api.py`
- `apps/autorag_streamlit.py`

### C. 평가 프레임워크 (분리 설계)
- `core` 모드: 운영 핵심 지표 중심 (빠르고 비용 절감)
- `detailed` 모드: 모델/프롬프트/튜닝 분석 지표 포함

핵심 엔트리:
- `scripts/run_evaluation.py`
- `scripts/check_release_gate.py`
- `src/evaluation/*`

---

## 2) 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에서 통합 관리하는 주요 패키지:
- RAG: `openai`, `langchain-text-splitters`, `chromadb`, `faiss-cpu`
- 평가: `rouge-score`, `nltk`, `bert-score`
- 평가: `ragas` (Python 버전에 따라 자동 분기)
- 서비스: `streamlit`, `fastapi`, `uvicorn`
- AutoRAG: Python `<3.13`에서 자동 설치, Python `>=3.13`에서는 자동 skip
- LangChain splitters는 Python 버전에 따라 자동으로 호환 버전이 선택됩니다.

설치 이슈 메모:
- `pyhwp`, `python-hwp`는 Python 3.14 환경에서 설치 불가 케이스가 많아 필수 목록에서 제외했습니다.
- 본 프로젝트의 HWP 파싱은 `olefile` + 내부 파서(`src/document_loader.py`)로 동작합니다.
- AutoRAG 실행이 필요하면 Python `3.11/3.12` 환경을 권장합니다.

---

## 3) 빠른 실행 가이드

### 3-1. 인덱싱

```bash
python scripts/index_documents.py \
  --scenario B \
  --method semantic \
  --chunk-size 800 \
  --chunk-overlap 200
```

### 3-2. 서비스 앱 실행

```bash
streamlit run app.py
```

---

## 4) AutoRAG 사용 가이드

### 4-1. AutoRAG 데이터 생성

```bash
python scripts/prepare_autorag_data.py \
  --documents-dir data \
  --metadata-csv data/data_list.csv \
  --output-dir data/autorag \
  --chunk-method semantic \
  --chunk-size 800 \
  --chunk-overlap 200
```

### 4-2. AutoRAG 최적화

```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/tutorial.yaml \
  --project-dir evaluation/autorag_benchmark
```

### 4-3. AutoRAG 배포 실행

```bash
autorag run_api --trial_dir evaluation/autorag_benchmark/0 --host 0.0.0.0 --port 8000
autorag run_web --trial_path evaluation/autorag_benchmark/0
```

보조 스크립트:
```bash
python scripts/run_autorag_api.py --trial-dir evaluation/autorag_benchmark/0
python scripts/run_autorag_web.py --trial-path evaluation/autorag_benchmark/0
```

---

## 5) 평가 실행 가이드

## 5-1. core 모드 (운영 권장)
- 목적: 빠른 품질 모니터링 / 회귀 체크
- 기본: LLM Judge OFF, BERTScore OFF
- 기본: Gate ON (`--gate auto`)

```bash
python scripts/run_evaluation.py --mode core --output-dir evaluation
```

생성 파일 예시:
- `evaluation/eval_similarity_k5_core.csv` (핵심 컬럼만)
- `evaluation/summary_similarity_k5_core.json`
- `evaluation/gate_report_core.json`
- `evaluation/gate_report_core.md`

커스텀 Gate 기준 적용:
```bash
python scripts/run_evaluation.py \
  --mode core \
  --gate on \
  --gate-thresholds configs/evaluation/core_gate.default.json \
  --output-dir evaluation
```

릴리즈 게이트 원커맨드:
```bash
python scripts/check_release_gate.py
```

## 5-2. detailed 모드 (튜닝/실험)
- 목적: 모델/프롬프트/검색 전략 튜닝
- 기본: LLM Judge ON, BERTScore ON

```bash
python scripts/run_evaluation.py --mode detailed --output-dir evaluation
```

생성 파일 예시:
- `evaluation/eval_similarity_k5_detailed.csv` (상세 지표 포함)
- `evaluation/summary_similarity_k5_detailed.json`

## 5-3. 빠른 테스트

```bash
python scripts/run_evaluation.py --mode detailed --test-limit 2 --output-dir evaluation
```

---

## 6) 평가 지표 정의

### 6-1. Retrieval
- `hit_at_1/3/5`
- `mrr`
- `ndcg_at_5`
- `precision_at_5`
- `recall_proxy`

### 6-2. Generation
- `keyword_recall`
- `field_coverage`
- `rougeL_f1`
- `meteor`
- `bertscore_f1` (detailed 중심)

### 6-3. Grounding / 신뢰성
- `grounded_token_ratio`
- `hallucination_risk_proxy`
- `decline_accuracy` (out-of-scope 대응)

### 6-4. Runtime / 비용
- `avg/p50/p95 elapsed_time`
- `prompt/completion/total_tokens`

---

## 7) 보고서 작성 템플릿 (권장)

### 7-1. 목표/가정
- 대상 사용자: 입찰 컨설턴트
- 핵심 요구: 정확성 + 속도 + 비용 균형

### 7-2. 실험 설정
- 비교군: similarity/MMR/hybrid, top_k
- 모드: core / detailed
- 질문셋: 단일문서/다중문서/후속질문/out-of-scope

### 7-3. 결과 표 (예시)
- 운영 관점: `p95`, `hit@5`, `field_coverage`, `grounded_token_ratio`
- 튜닝 관점: `bertscore_f1`, `llm_*`, `rougeL_f1`, `meteor`

### 7-4. 의사결정
- 운영 기본값: core 기준 최적 config
- 튜닝/개선 트랙: detailed 기준 개선 후보

### 7-5. 한계/개선
- HWP 파싱 정확도 개선
- 메타데이터 필터 정교화
- 재랭킹 비용 최적화

---

## 8) 운영 체크리스트

- `.env`/키 노출 금지
- 인덱싱 재실행 시 컬렉션 중복 적재 여부 점검
- 멀티 사용자 환경에서 세션 메모리 분리 유지
- 배포 전 `--mode core --test-limit N`으로 회귀 테스트

---

## 9) 현재 구조 요약

- `src/evaluation/`: 평가 모듈 분리 구조
- `src/evaluator.py`: 하위호환 re-export
- `requirements.txt`: AutoRAG/평가/서빙 패키지 통합 관리

