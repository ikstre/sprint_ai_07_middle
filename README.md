# sprint_ai_07_middle

기업/공공 RFP 문서를 대상으로 한 RAG 시스템 프로젝트입니다.  
본 저장소는 아래 3가지를 모두 제공합니다.

1. 실서비스용 RAG 앱/파이프라인 (Scenario A/B 모두 지원)
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

| 항목 | Scenario B (OpenAI API) | Scenario A (로컬 HuggingFace) |
|------|------------------------|------------------------------|
| 문서 로딩 | PDF / HWP / TXT / CSV | 동일 |
| 청킹 | naive / semantic | 동일 |
| 임베딩 | text-embedding-3-small (dim=512) | BGE-m3-ko (dim=1024) / ko-sroberta (dim=768) |
| 검색 | similarity / MMR / hybrid + multi-query / rerank | 동일 |
| 생성 | gpt-5-mini / gpt-5-nano / gpt-5 | EXAONE / Gemma3 / Gemma4 / kanana / Midm 등 로컬 모델 |
| 대화 메모리 | 슬라이딩 윈도우 (최근 5턴) | 동일 |
| UI | Streamlit 앱 (사이드바에서 A/B 전환) | 동일 |

핵심 엔트리:
- `app.py`
- `src/rag_pipeline.py`

### B. AutoRAG 실험/배포

| config 파일 | 시나리오 | Generator | 임베딩 |
|------------|---------|-----------|--------|
| `configs/autorag/tutorial.yaml` | B (OpenAI) | openai_llm — gpt-5-mini | text-embedding-3-small |
| `configs/autorag/local.yaml` | A (GPU 서버, 22GB) | vllm — 5종 모델 | BGE-m3-ko + ko-sroberta |
| `configs/autorag/local_pc.yaml` | A-PC (8GB GPU) | vllm — 4종 모델 | BAAI/bge-m3 + ko-sroberta (HF Hub) |

핵심 엔트리:
- `scripts/prepare_autorag_data.py`
- `scripts/run_autorag_optimization.py`
- `apps/autorag_api.py`, `apps/autorag_streamlit.py`

### C. 평가 프레임워크 (분리 설계)
- `core` 모드: 운영 핵심 지표 중심 (빠르고 비용 절감)
- `detailed` 모드: 모델/프롬프트/튜닝 분석 지표 포함

핵심 엔트리:
- `scripts/run_evaluation.py`
- `scripts/check_release_gate.py`
- `src/evaluation/*`

### D. 파인튜닝
- `scripts/finetune_local.py`: LoRA / QLoRA (로컬 오픈소스 모델)
- `scripts/finetune_openai.py`: OpenAI Fine-tuning API

---

## 2) 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에서 통합 관리하는 주요 패키지:
- RAG: `openai`, `langchain-text-splitters`, `chromadb`, `faiss-cpu`
- HuggingFace (Scenario A): `transformers`, `sentence-transformers`, `torch`, `accelerate`
- 평가: `rouge-score`, `nltk`, `bert-score`, `ragas`
- 서비스: `streamlit`, `fastapi`, `uvicorn`

파인튜닝 추가 패키지:
```bash
pip install peft trl bitsandbytes accelerate datasets
```

설치 이슈 메모:
- HWP 파싱은 `olefile` + 내부 파서(`src/document_loader.py`)로 동작합니다.
- Scenario A 로컬 모델은 `/srv/shared_data/models/`에서 직접 로드하며 HF_TOKEN이 불필요합니다.
- 로컬 PC에서는 HuggingFace Hub에서 자동 다운로드됩니다 (`HF_HOME` 환경변수로 캐시 경로 지정 가능).

---

## 3) 빠른 실행 가이드

### 3-1. 인덱싱

인덱싱은 **청킹 → 임베딩** 2단계로 구성됩니다.

#### Scenario B (OpenAI API)

```bash
# 기본 (chunk_size=1200)
python scripts/index_documents.py --scenario B --collection rfp_chunk1200

# 비교용 800자 컬렉션
python scripts/index_documents.py --scenario B --chunk-size 800 --collection rfp_chunk800

# Batch API 사용 (비용 50% 절감)
python scripts/index_documents.py --scenario B --use-batch-api --collection rfp_chunk1200

# 단계별 실행
python scripts/index_documents.py --scenario B --step chunk --collection rfp_chunk1200
python scripts/index_documents.py --scenario B --step embed --collection rfp_chunk1200
```

#### Scenario A (로컬 HuggingFace)

```bash
# 기본 (BGE-m3-ko 임베딩, chunk_size=1200)
python scripts/index_documents.py \
  --scenario A \
  --hf-embedding-model bge \
  --chunk-size 1200 \
  --collection rfp_chunk1200_a

# 비교용 — ko-sroberta 임베딩 (경량, dim=768)
python scripts/index_documents.py \
  --scenario A \
  --hf-embedding-model sroberta \
  --chunk-size 1200 \
  --collection rfp_chunk1200_a_sroberta

# B안 청킹 파일 재사용 → A안 임베딩만 실행
python scripts/index_documents.py \
  --scenario A --step embed \
  --hf-embedding-model bge \
  --collection rfp_chunk1200_a
```

#### 컬렉션 목록

| 컬렉션명 | 시나리오 | 임베딩 | chunk_size |
|---------|---------|--------|-----------|
| `rfp_chunk1200` | B (OpenAI) | text-embedding-3-small (512) | 1200 |
| `rfp_chunk800` | B (OpenAI) | text-embedding-3-small (512) | 800 |
| `rfp_chunk1200_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 1200 |
| `rfp_chunk800_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 800 |
| `rfp_chunk1200_a_sroberta` | A (ko-sroberta) | ko-sroberta (768) | 1200 |

> A안과 B안은 임베딩 차원이 달라 컬렉션을 반드시 분리해야 합니다.

---

### 3-2. 서비스 앱 실행

```bash
streamlit run app.py
```

#### 사이드바 주요 설정

| 설정 | Scenario B (OpenAI) | Scenario A (로컬 HF) |
|------|--------------------|--------------------|
| 실행 모드 | B: OpenAI API | A: 로컬 HuggingFace |
| 컬렉션 | rfp_chunk1200 / rfp_chunk800 | rfp_chunk1200_a / rfp_chunk1200_a_sroberta |
| LLM 모델 | gpt-5-mini / gpt-5-nano / gpt-5 | EXAONE / Gemma3 / Gemma4 등 9종 |
| 임베딩 모델 | text-embedding-3-small (고정) | BGE-m3-ko / ko-sroberta 선택 |
| 검색 방식 | similarity / mmr / hybrid | 동일 |
| Reasoning Effort | low / medium / high | 해당 없음 |
| Temperature | 미지원 (gpt-5 계열) | 0.0 ~ 1.0 슬라이더 |

> 실행 모드를 전환하면 컬렉션 목록이 자동으로 해당 시나리오 컬렉션으로 바뀝니다.

---

## 4) AutoRAG 사용 가이드

### 4-1. AutoRAG 데이터 생성

```bash
python scripts/prepare_autorag_data.py \
  --documents-dir data \
  --metadata-csv data/data_list.csv \
  --output-dir data/autorag \
  --chunk-method semantic \
  --chunk-size 600 \
  --chunk-overlap 150
```

산출물: `data/autorag/corpus.parquet`, `data/autorag/qa.parquet`

### 4-2. AutoRAG 최적화

**Scenario B (OpenAI)**
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/tutorial.yaml \
  --project-dir evaluation/autorag_benchmark
```

**Scenario A — GPU 서버 (22GB VRAM)** — 실행 전 `nvidia-smi`로 GPU 점유 확인 권장
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/local.yaml \
  --project-dir evaluation/autorag_benchmark_local
```

**Scenario A-PC — 로컬 PC (RTX 4070 / 3060Ti, 8GB VRAM)**
```bash
# 첫 실행 시 HuggingFace에서 모델 자동 다운로드 (수 GB)
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/local_pc.yaml \
  --project-dir evaluation/autorag_benchmark_pc
```

> **AutoRAG 0.3.22 노드명 변경**: 구버전의 `node_type: retrieval` 단일 노드는 지원 종료.  
> `lexical_retrieval` / `semantic_retrieval` / `hybrid_retrieval` 로 분리해서 사용해야 합니다.
>
> **내장 패치** (`run_autorag_optimization.py`):
> - ChromaDB `add_embedding` batch 초과 자동 분할 처리 (11567 > 5461 오류 방지)
> - ChromaDB `is_exist` SQLite 변수 초과 자동 분할 처리 (500개 단위)
> - 각 모델 평가 후 VRAM 자동 해제 (`gc.collect` + `cuda.empty_cache`)
> - 위 패치는 `tutorial.yaml`, `local.yaml`, `local_pc.yaml` 모두에 자동 적용됨

### 4-3. AutoRAG 결과 확인

```bash
# 요약 CSV
cat evaluation/autorag_benchmark_local/0/summary.csv

# 대시보드 (시각화)
autorag dashboard --trial_dir evaluation/autorag_benchmark_local/0

# 최적 config 추출
autorag extract_best_config \
  --trial_path evaluation/autorag_benchmark_local/0 \
  --output_path evaluation/autorag_benchmark_local/best_config.yaml

# API 서빙
autorag run_api --trial_dir evaluation/autorag_benchmark_local/0 --host 0.0.0.0 --port 8000
```

---

## 5) 파인튜닝

### 5-1. 로컬 LoRA/QLoRA

```bash
# GPU 서버 — LoRA
python scripts/finetune_local.py \
  --model-path /srv/shared_data/models/kanana/kanana-nano-2.1b \
  --output-dir models/finetuned/kanana-nano-rag \
  --epochs 3

# 로컬 PC (8GB GPU) — QLoRA (4-bit 양자화)
python scripts/finetune_local.py \
  --model-path kakaocorp/kanana-nano-2.1b \
  --output-dir models/finetuned/kanana-nano-rag \
  --qlora \
  --epochs 3
```

### 5-2. OpenAI Fine-tuning

```bash
# 시작 (데이터 자동 생성 + 업로드)
python scripts/finetune_openai.py start \
  --model gpt-4o-mini-2024-07-18 \
  --output-dir models/finetuned/openai

# 상태 확인
python scripts/finetune_openai.py status --job-id ftjob-xxxx

# 목록
python scripts/finetune_openai.py list
```

완료 후 `configs/autorag/tutorial.yaml`의 `llm`을 파인튜닝 모델 ID로 교체:
```yaml
llm: ft:gpt-4o-mini-2024-07-18:org:rag:xxxx
```

---

## 6) 평가 실행 가이드

### core 모드 (운영 권장)

```bash
python scripts/run_evaluation.py --mode core --output-dir evaluation
```

릴리즈 게이트 원커맨드:
```bash
python scripts/check_release_gate.py
```

### detailed 모드 (튜닝/실험)

```bash
python scripts/run_evaluation.py --mode detailed --output-dir evaluation
```

### 빠른 테스트

```bash
python scripts/run_evaluation.py --mode detailed --test-limit 2 --output-dir evaluation
```

---

## 7) 평가 지표 정의

### Retrieval
- `hit_at_1/3/5`, `mrr`, `ndcg_at_5`, `precision_at_5`, `recall_proxy`

### Generation
- `keyword_recall`, `field_coverage`, `rougeL_f1`, `meteor`, `bertscore_f1`

### Grounding / 신뢰성
- `grounded_token_ratio`, `hallucination_risk_proxy`, `decline_accuracy`

### Runtime / 비용
- `avg/p50/p95 elapsed_time`, `prompt/completion/total_tokens`

---

## 8) Scenario A 로컬 모델 목록

모델 저장 위치 (서버): `/srv/shared_data/models/`  
로컬 PC: HuggingFace Hub 자동 다운로드

### 임베딩 모델

| CLI 옵션 | 서버 경로 / HF Hub ID | 차원 | 특징 |
|---------|----------------------|------|------|
| `--hf-embedding-model bge` | `BGE-m3-ko` / `BAAI/bge-m3` | 1024 | 다국어, 고성능 |
| `--hf-embedding-model sroberta` | `ko-sroberta-multitask` / `jhgan/ko-sroberta-multitask` | 768 | 한국어 특화, 경량 |

### AutoRAG 평가 모델 (Scenario A — 서버, `local.yaml`)

| 모델명 | 크기 | 특이사항 |
|--------|------|---------|
| EXAONE-4.0-1.2B | 2.4G | trust_remote_code 필요 |
| kanana-nano-2.1b | 4.0G | llama 계열, 한국어 특화 |
| kanana-1.5-2.1b | 4.4G | llama 계열, 한국어 특화 |
| Midm-2.0-Mini | 4.4G | llama 계열, 한국어 특화 |
| Gemma3-4B | 8.1G | 순차 로드 (이전 모델 VRAM 해제 후) |

> Gemma4-E4B(15G)는 fp8 KV 캐시 적용 시 22GB GPU에서 로드 가능 — 필요 시 `local.yaml`에 추가 가능.  
> Gemma4-26B-A4B(49G)는 22GB VRAM 초과로 단일 GPU 로드 불가.

### AutoRAG 평가 모델 (Scenario A-PC — 8GB GPU)

| HF Hub ID | 크기 | 비고 |
|-----------|------|------|
| `LGAI-EXAONE/EXAONE-4.0-1.2B` | 2.4G | trust_remote_code 필요 |
| `kakaocorp/kanana-nano-2.1b` | 4.0G | |
| `kakaocorp/kanana-1.5-2.1b` | 4.4G | |
| `skt/Midm-2.0-Mini-Instruct` | 4.4G | |

### 채팅 모델 (앱 사이드바 선택, 서버)

| 모델명 | 경로 | 크기 | VRAM 적합성 | 특징 |
|--------|------|------|------------|------|
| EXAONE-4.0-1.2B | `exaone/EXAONE-4.0-1.2B` | 2.4G | ✅ | 가장 빠름, 테스트 용도 |
| EXAONE-Deep-2.4B | `exaone/EXAONE-Deep-2.4B` | 4.5G | ✅ | 속도/성능 균형 |
| EXAONE-3.5-7.8B | `exaone/EXAONE-3.5-7.8B` | 30G | ✅ (22GB 서버) | 한국어 고성능 |
| EXAONE-Deep-7.8B | `exaone/EXAONE-Deep-7.8B` | 15G | ✅ | 한국어 추론 특화 |
| Gemma3-4B | `gemma/Gemma3-4B` | 8.1G | ✅ | 다국어 |
| Gemma4-E4B | `gemma/Gemma4-E4B` | 15G | ⚠️ (fp8 필요) | 멀티모달, 다국어 |
| Gemma4-26B-A4B | `gemma/Gemma4-26B-A4B` | 49G | ❌ (22GB 초과) | MoE 26B/4B활성 |
| kanana-nano-2.1b | `kanana/kanana-nano-2.1b` | 4.0G | ✅ | 한국어 특화, 경량 |
| kanana-1.5-2.1b | `kanana/kanana-1.5-2.1b` | 4.4G | ✅ | 한국어 특화, 경량 |
| Midm-2.0-Mini | `midm/Midm-2.0-Mini` | 4.4G | ✅ | 한국어 특화, 경량 |

> 모두 `/srv/shared_data/models/` 하위 경로.

---

## 9) OpenAI 모델 최적화 설정 (Scenario B)

### Reasoning Effort

| 값 | 용도 | 속도 | 비용 |
|----|------|------|------|
| `low` | 단순 사실 조회 | 빠름 | 저렴 |
| `medium` | 일반 질문 (기본값) | 보통 | 보통 |
| `high` | 복잡한 비교/분석 | 느림 | 비쌈 |

### gpt-5 계열 파라미터 제한사항

| 파라미터 | gpt-4 계열 | gpt-5 계열 |
|---------|-----------|-----------|
| `temperature` | 지원 | **미지원** (Scenario A 전용) |
| `max_tokens` | 지원 | **미지원** (`max_completion_tokens` 사용) |
| `reasoning_effort` | 미지원 | low/medium/high |

---

## 10) 운영 체크리스트

- `.env` / API 키 노출 금지
- A안/B안 컬렉션 혼용 금지 (임베딩 차원 불일치로 검색 오류 발생)
- 인덱싱 재실행 시 컬렉션 중복 적재 여부 점검
- 배포 전 `--mode core --test-limit N`으로 회귀 테스트
- AutoRAG 실행 전 `nvidia-smi`로 다른 팀원 GPU 점유 확인

---

## 11) 트러블슈팅

### UnicodeEncodeError: surrogates not allowed
HWP 파싱 중 유효하지 않은 유니코드 서로게이트 발생.  
`_sanitize()` 함수가 청킹 및 ChromaDB 저장 시 자동 정제합니다.

### AutoRAG `KeyError: retrieval is not supported`
AutoRAG 0.3.22에서 `node_type: retrieval` 노드명 변경.  
`lexical_retrieval` / `semantic_retrieval` / `hybrid_retrieval` 로 분리해서 사용해야 합니다.

### openai.BadRequestError: max_tokens not supported
gpt-5 계열은 `max_tokens` 대신 `max_completion_tokens` 사용.  
`src/evaluation/evaluator.py`, `src/generator.py` 에서 이미 적용됨.

### ChromaDB batch size 초과 (11567 > 5461)
`run_autorag_optimization.py` 내장 패치로 자동 처리됨.

### ChromaDB is_exist SQLite 변수 초과
`run_autorag_optimization.py` 내장 패치로 자동 처리됨 (500개 단위 배치).

### vLLM max_model_len 초과
`max_model_len` 미지정 시 각 모델의 `config.json`에서 자동 참조됨.  
`local.yaml`, `local_pc.yaml` 모두 `max_model_len` 미지정으로 설정되어 있음.

### 환경 진단

```bash
python scripts/check_env.py
```
