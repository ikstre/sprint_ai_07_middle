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

인덱싱은 **청킹 → 임베딩** 2단계로 구성됩니다. 에러 발생 시 완료된 단계부터 재실행할 수 있습니다.

```bash
# 전체 실행 (기본, 1200자 청크)
python scripts/index_documents.py --collection rfp_chunk1200

# 비교용 800자 컬렉션 추가 인덱싱
python scripts/index_documents.py --chunk-size 800 --collection rfp_chunk800

# 단계별 실행 (디버깅 시)
python scripts/index_documents.py --step chunk --collection rfp_chunk1200   # 1단계: 청킹만
python scripts/index_documents.py --step embed --collection rfp_chunk1200   # 2단계: 임베딩만

# Batch API 사용 (500개 이상 청크 시 비용 50% 절감)
python scripts/index_documents.py --use-batch-api --collection rfp_chunk1200
```

**컬렉션 개념**: 청크 크기별로 별도 벡터DB 컬렉션을 생성해 성능을 비교할 수 있습니다.

| 컬렉션 | 청크 크기 | 특징 |
|--------|---------|------|
| `rfp_chunk1200` | 1200자 (기본값) | 문맥 풍부, 청크 수 적음 |
| `rfp_chunk800` | 800자 | 정밀도 높음, 청크 수 많음 |

### 3-2. 서비스 앱 실행

```bash
streamlit run app.py
```

#### 사이드바 주요 설정

| 설정 | 옵션 | 설명 |
|------|------|------|
| 실행 모드 | B: OpenAI API / A: 로컬 HuggingFace | Scenario 전환 |
| 컬렉션 | rfp_chunk1200 / rfp_chunk800 | 인덱싱된 컬렉션 선택 |
| LLM 모델 | gpt-5-mini / gpt-5-nano / gpt-5 (B안) | 생성 모델 |
| 검색 방식 | similarity / mmr / hybrid | 벡터 검색 전략 |
| Reasoning Effort | low / medium / high | gpt-5 추론 깊이 (B안 전용) |
| 자동 모델 라우팅 | 체크박스 | 단순 질문→nano, 복잡→mini 자동 전환 |

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

---

## 10) OpenAI 모델 최적화 설정

### 10-1. Reasoning Effort (응답 속도/비용 조절)

gpt-5 계열 reasoning 모델은 내부 추론 깊이를 조절할 수 있습니다.  
앱 사이드바 **Reasoning Effort** 슬라이더 또는 `config.py`에서 설정합니다.

| 값 | 용도 | 속도 | 비용 |
|----|------|------|------|
| `low` | 단순 사실 조회 | 빠름 | 저렴 |
| `medium` | 일반 질문 (기본값) | 보통 | 보통 |
| `high` | 복잡한 비교/분석 | 느림 | 비쌈 |

```python
# configs/config.py
reasoning_effort: str = "medium"
```

### 10-2. 자동 모델 라우팅

쿼리 길이 기준으로 단순 질문은 `gpt-5-nano`, 복잡한 질문은 설정 모델(`gpt-5-mini`)을 자동 선택합니다.  
앱 사이드바 **자동 모델 라우팅** 체크박스로 제어합니다.

```python
auto_model_routing: bool = True
routing_simple_model: str = "gpt-5-nano"   # 단순 질문
routing_complexity_threshold: int = 50     # 글자 수 기준
```

### 10-3. 임베딩 차원 축소

`text-embedding-3-small`은 출력 차원을 줄여도 품질 저하가 적습니다.  
기본값 512 (원본 1536 대비 저장 공간 67% 절감, 검색 속도 향상).

```python
openai_embedding_dim: int = 512  # 축소 (원본: 1536)
```

> **주의**: 차원을 변경한 경우 기존 벡터DB와 호환되지 않으므로 재인덱싱 필요.
> ```bash
> python scripts/index_documents.py --collection rfp_chunk1200
> ```

### 10-4. Batch API (대용량 인덱싱 비용 50% 절감)

500개 이상 청크 인덱싱 시 OpenAI Batch API를 사용하면 비용이 절반으로 줄어듭니다.  
처리 시간이 수 분~수 시간 소요될 수 있으므로 최초 인덱싱 또는 재인덱싱 시 사용을 권장합니다.

```bash
python scripts/index_documents.py --use-batch-api --collection rfp_chunk1200
```

### 10-5. Scenario A (HuggingFace) 모델 학습/파인튜닝

모델 자체를 학습시키거나 파인튜닝하는 작업은 **Scenario A (HuggingFace)** 기반입니다.  
GCP VM에서 GPU를 활용해 도메인 특화 학습을 진행합니다.

| 항목 | 내용 |
|------|------|
| 기반 모델 | `google/gemma-3-4b-it` |
| 임베딩 모델 | `intfloat/multilingual-e5-large` |
| 학습 방법 | LoRA / QLoRA (PEFT) 권장 |
| 학습 데이터 | `data/` 내 RFP 문서 + 정제된 QA 쌍 |
| 실행 환경 | GCP VM (CUDA), conda 환경 |

파인튜닝 후 `config.py`의 `hf_chat_model` 경로를 학습된 체크포인트로 변경하면 적용됩니다.

---

## 11) 트러블슈팅

### 11-1. UnicodeEncodeError: surrogates not allowed

**증상**
```
UnicodeEncodeError: 'utf-8' codec can't encode characters: surrogates not allowed
```

**원인**  
HWP 바이너리 파싱 중 일부 바이트가 유효하지 않은 유니코드 서로게이트(`\uD800`~`\uDFFF`)로 잘못 디코딩됨.  
OpenAI 임베딩 API 및 ChromaDB 저장 시 모두 오류 발생.

**해결**  
`src/embedder.py`에 모듈 레벨 `_sanitize()` 함수 추가. 청킹 직후(`scripts/index_documents.py`) 및 ChromaDB 저장(`src/embedder.py`) 시 텍스트와 메타데이터 전체에 정제 적용.

```python
def _sanitize(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")
```

---

### 11-2. openai.PermissionDeniedError: model not found (403)

**증상**
```
openai.PermissionDeniedError: Error code: 403
Project does not have access to model `text-embedding-3-small`
```

**원인**  
`.env`의 `OPENAI_API_KEY`가 해당 모델 접근 권한이 없는 프로젝트 소속 키임.

**해결**  
OpenAI 플랫폼 → 해당 프로젝트 → Settings → Model access 에서 `text-embedding-3-small` 활성화 확인.  
또는 `python scripts/check_env.py` 실행으로 현재 키로 접근 가능한 모델 목록 확인.

---

### 11-3. openai.BadRequestError: max_tokens not supported (400)

**증상**
```
BadRequestError: 'max_tokens' is not supported with this model.
Use 'max_completion_tokens' instead.
```

**원인**  
gpt-5 계열 모델은 `max_tokens` 파라미터를 지원하지 않음.

**해결**  
`src/generator.py` 및 `src/retriever.py`에서 `max_tokens` → `max_completion_tokens` 로 변경.

---

### 11-4. openai.BadRequestError: temperature not supported (400)

**증상**
```
BadRequestError: 'temperature' does not support 0.1 with this model.
Only the default (1) value is supported.
```

**원인**  
gpt-5 계열 reasoning 모델은 `temperature`, `top_p` 파라미터를 지원하지 않음.

**해결**  
`src/generator.py`의 `_call_openai()`에서 `temperature`, `top_p` 제거.  
`src/retriever.py`의 `_generate_multi_queries()`에서 모델명 기준으로 조건부 적용.  
해당 파라미터는 Scenario A (HuggingFace) 전용으로 유지.

---

### 11-5. gpt-5 응답 텍스트가 빈 문자열로 반환

**증상**  
앱 화면에 답변 텍스트가 나타나지 않음. usage 확인 시 `completion_tokens`가 `max_completion_tokens` 한도와 동일.

**원인**  
gpt-5-mini는 reasoning 모델로 내부 추론 토큰을 먼저 소비함. `max_completion_tokens=2048`이 너무 낮아 추론만 하다가 실제 응답 출력 전에 한도 도달.

**해결**  
`configs/config.py`에서 `max_tokens` 값을 충분히 크게 설정.

```python
max_tokens: int = 16000  # reasoning 모델은 내부 추론 토큰 포함으로 충분히 크게 설정
```

---

### 11-6. AutoRAG 의존성 충돌 (langchain-text-splitters 버전)

**증상**
```
ERROR: Cannot install langchain-community and langchain-text-splitters>=1.1.1
because these package versions have conflicting dependencies.
```

**원인**  
`AutoRAG==0.3.21` → `langchain-community==0.2.19` → `langchain-text-splitters<0.3.0` 요구,  
메인 프로젝트는 `langchain-text-splitters>=1.1.1` 요구. 동일 환경에서 공존 불가.

**해결**  
AutoRAG를 별도 conda 환경(`autorag`)으로 분리.  
`requirements-autorag.txt` 생성, `openai>=1.40.0,<2.0.0` 고정.  
각 AutoRAG 스크립트(`scripts/run_autorag_*.py`) 상단에 `.env`의 `AUTORAG_PYTHON` 경로로 자동 인터프리터 전환 로직 추가.

```bash
# autorag 환경 생성
conda create -n autorag python=3.11
conda activate autorag
pip install -r requirements-autorag.txt

# .env 설정
AUTORAG_PYTHON=/path/to/envs/autorag/bin/python
```

---

### 11-7. 환경 진단

실행 전 아래 스크립트로 전체 환경을 점검할 수 있습니다.

```bash
python scripts/check_env.py
```

확인 항목: API 키 로드 여부, OpenAI 모델 접근 권한, 데이터 폴더 존재 여부, 핵심 패키지 설치 상태.

