# Engineering Guide

## Purpose
This project provides:
1. A production-facing RAG service for RFP analysis.
2. An AutoRAG-based optimization workflow.
3. A two-layer evaluation strategy (`core` for operations, `detailed` for tuning).
4. Fine-tuning pipeline (local LoRA/QLoRA + OpenAI Fine-tuning API).

## Recommended Reading Order
1. `docs/CODEBASE_MAP.md`
2. `docs/EVALUATION_GUIDE.md`
3. `docs/OPS_SECURITY_MULTIUSER.md`
4. `docs/AUTORAG_GUIDE.md`

## Directory Map
- `app.py`: Main Streamlit chat app (service UI).
- `src/`: Core RAG modules.
- `src/evaluation/`: Modular evaluation package.
- `scripts/`: CLI entrypoints for indexing, evaluation, AutoRAG, fine-tuning.
- `apps/`: Optional API/UI wrappers for AutoRAG runtime.
- `configs/config.py`: 런타임 설정 (시나리오, 모델, 검색 파라미터).
- `configs/paths.py`: **중앙 경로 레지스트리** — `.env` 환경변수 기반 경로 관리.
- `configs/autorag/`: AutoRAG search configs (B / A-server / A-PC).
- `notebooks/`: 보조 분석 노트북 (운영 워크플로는 `scripts/`를 사용).

## Operating Principle
- Production decision: optimize for stable `core` metrics and gate pass.
- Tuning decision: use `detailed` mode for metric decomposition and model/prompt iteration.
- Fine-tuning decision: compare AutoRAG results before/after fine-tuning to verify actual improvement.

## Design Decisions
- Separation of concerns:
  - pipeline/retrieval/generation/evaluation are split into independent modules.
  - `src/evaluator.py` keeps backward compatibility, while new logic lives under `src/evaluation/`.
- Operational safety:
  - core mode excludes expensive metrics by default.
  - gate report provides explicit pass/fail criteria for release checks.
- Experiment agility:
  - AutoRAG path is maintained in parallel, not replacing the hand-tuned path.

## Recommended Workflow
1. Index documents.
2. Run `core` evaluation (with gate).
3. If gate fails, run `detailed` evaluation.
4. Apply retrieval/generation changes or fine-tune.
5. Re-run `core` for regression safety.

---

## 경로 설정 (.env)

모든 경로는 `configs/paths.py`에서 중앙 관리합니다. `.env`에 필요한 항목만 추가하면 전체 스크립트에 즉시 반영됩니다.

```dotenv
# 서버 공용 데이터 루트 (기본: /srv/shared_data)
SRV_DATA_DIR=/srv/shared_data

# 이하는 SRV_DATA_DIR로부터 자동 파생되며, 필요 시 개별 오버라이드 가능
# METADATA_CSV=/srv/shared_data/datasets/data_list_cleaned.csv
# MODEL_DIR=/srv/shared_data/models
# EMBEDDING_MODEL_PATH=/srv/shared_data/models/embeddings/BGE-m3-ko
# CHAT_MODEL_PATH=/srv/shared_data/models/exaone/EXAONE-4.0-1.2B
# VECTORDB_DIR=data/vectordb
# AUTORAG_DATA_DIR=data/autorag_csv
# AUTORAG_PROJECT_DIR=evaluation/autorag_benchmark_csv
```

AutoRAG YAML 파일(`configs/autorag/*.yaml`)의 `${SRV_DATA_DIR}` 플레이스홀더는 `run_pipeline.py` 실행 시 자동 치환됩니다.

---

## Scenario A / B 선택 가이드

| 항목 | Scenario B (OpenAI API) | Scenario A — 서버 | Scenario A-PC — 로컬 PC |
|------|------------------------|--------------------|------------------------|
| 실행 환경 | 로컬/GCP 모두 가능 | GCP GPU 서버 L4 (22GB) | RTX 4070 / 3060Ti (8GB) |
| 채팅 모델 | gpt-5-mini / gpt-5-nano / gpt-5 | EXAONE, Gemma3, Gemma4-E4B, kanana, Midm | 동일 (HF Hub 다운로드) |
| 임베딩 모델 | text-embedding-3-small (dim=512) | BGE-m3-ko / ko-sroberta / E5-large / KoSimCSE / kf-DeBERTa | BAAI/bge-m3 / ko-sroberta (HF Hub) |
| 파인튜닝 | OpenAI Fine-tuning API | LoRA / QLoRA | QLoRA (4-bit, 8GB 대응) |
| 필수 환경변수 | `OPENAI_API_KEY` | 없음 | 없음 (HF 모델 자동 다운로드) |
| AutoRAG config | `tutorial.yaml` | `local.yaml` | `local_pc.yaml` |

> **전환 방법**: 앱 실행 시 사이드바 **실행 모드** 라디오 버튼으로 A ↔ B 실시간 전환 가능.
> 단, A안과 B안은 **임베딩 차원이 달라** 컬렉션을 별도로 인덱싱해야 합니다.

---

## 지원 문서 포맷

현재 일반 파일 로딩 경로에서 `DocumentLoader`가 직접 처리하는 포맷은 아래 두 가지입니다.

| 포맷 | 확장자 | 인코딩 자동 감지 | 특이사항 |
|------|--------|----------------|---------|
| HWP | `.hwp` | 바이너리 파싱 | OLE 구조 직접 파싱 |
| PDF | `.pdf` | — | pdfplumber 사용 |

> `data_list.csv`(메타데이터 파일)는 자동으로 인덱싱에서 제외됩니다.

### CSV 로드 옵션

| 옵션 | 설명 | 사용 상황 |
|------|------|----------|
| 기본 | 일반 파일 로딩 경로 사용 | PDF/HWP 인덱싱 |
| `--csv-row-per-doc` | 메타데이터 CSV 각 행을 개별 문서로 처리 | 행 하나가 RFP 한 건인 경우 |
| `--csv-text-columns` | 본문으로 사용할 컬럼 직접 지정 | 특정 컬럼만 검색 대상으로 쓸 때 |

---

## 인덱싱 2단계 구조

두 시나리오 모두 동일한 2단계 구조를 따릅니다.

```
[1단계: chunk] 문서 로딩 → 청킹 → data/processed/{collection}_chunks.json 저장
[2단계: embed] 청크 파일 로딩 → 임베딩 생성 → ChromaDB 저장
```

- 임베딩 중 오류 시 1단계 재실행 없이 `--step embed`만 재실행 가능
- 청크 크기별로 컬렉션을 여러 개 만들어 앱에서 실시간 비교 가능
- 청킹 결과(`.json`)는 A/B안이 **공유 가능** → B안 청킹 후 A안 임베딩만 재실행 가능

---

## Scenario B (OpenAI) 인덱싱 명령어

```bash
# 기본 (chunk_size=1200)
python scripts/index_documents.py \
  --scenario B \
  --chunk-size 1200 \
  --collection rfp_chunk1200

# 청크 크기 800 비교용
python scripts/index_documents.py \
  --scenario B \
  --chunk-size 800 \
  --collection rfp_chunk800

# OpenAI Batch API 활용 (500개 이상 청크 시 비용 50% 절감)
python scripts/index_documents.py \
  --scenario B \
  --collection rfp_chunk1200 \
  --use-batch-api

# Semantic 청킹 (RFP 섹션 구조 인식)
python scripts/index_documents.py \
  --scenario B \
  --method semantic \
  --collection rfp_semantic

# 1단계(청킹)만 실행 → 2단계(임베딩) 나중에 별도 실행
python scripts/index_documents.py --scenario B --step chunk --collection rfp_chunk1200
python scripts/index_documents.py --scenario B --step embed --collection rfp_chunk1200
```

---

## Scenario A (로컬 HuggingFace) 인덱싱 명령어

A안은 B안과 동일한 CLI 구조입니다. `--scenario A` + `--hf-embedding-model` 옵션만 추가됩니다.

```bash
# 기본 (BGE-m3-ko 임베딩, chunk_size=1200)
python scripts/index_documents.py \
  --scenario A \
  --hf-embedding-model bge \
  --chunk-size 1200 \
  --collection rfp_chunk1200_a

# 경량 임베딩 모델 비교 (ko-sroberta, dim=768)
python scripts/index_documents.py \
  --scenario A \
  --hf-embedding-model sroberta \
  --chunk-size 1200 \
  --collection rfp_chunk1200_a_sroberta

# B안 청킹 파일 재사용 → A안 임베딩만 실행
python scripts/index_documents.py \
  --scenario A \
  --step embed \
  --hf-embedding-model bge \
  --collection rfp_chunk1200_a
```

---

## AutoRAG 활성화

Gemma4 기반 AutoRAG는 메인 스택과 분리된 `requirements-gemma4.txt` 환경을 권장합니다.

### Step 1. 데이터 준비

```bash
# 권장 설정 (CSV 기반 ground truth)
# --csv-path와 --output-dir은 .env의 METADATA_CSV, AUTORAG_DATA_DIR이 기본값
python scripts/prepare_autorag_from_csv.py \
  --chunk-size 600 \
  --chunk-overlap 100
```

산출물: `data/autorag_csv/corpus.parquet`, `data/autorag_csv/qa.parquet`

### Step 2. 최적화 실행

AutoRAG config 파일은 시나리오별로 분리되어 있습니다.

| 파일 | 시나리오 | Generator | 임베딩 비교 |
|------|---------|-----------|-----------|
| `configs/autorag/tutorial.yaml` | B (OpenAI) | `openai_llm` — gpt-5-mini | text-embedding-3-small |
| `configs/autorag/local.yaml` | A (서버, 22GB) | `vllm` — 5종 모델 | 5종 (BGE / sroberta / E5 / SimCSE / DeBERTa) |
| `configs/autorag/local_pc.yaml` | A (PC, 8GB) | `vllm` — 4종 모델 | BAAI/bge-m3 vs ko-sroberta (HF Hub) |

**Scenario B (OpenAI)**
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag_csv/qa.parquet \
  --corpus-path data/autorag_csv/corpus.parquet \
  --config-path configs/autorag/tutorial.yaml \
  --project-dir evaluation/autorag_benchmark
```

**Scenario A — GPU 서버**
```bash
# 통합 파이프라인 권장
python scripts/run_pipeline.py --steps data,autorag

# 필요 시 파인튜닝 포함
python scripts/run_pipeline.py --steps finetune,autorag \
  --finetune-models kanana-1.5
```

**Scenario A-PC — 로컬 PC (8GB GPU)**
```bash
# 첫 실행 시 HuggingFace에서 모델 자동 다운로드
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag_csv/qa.parquet \
  --corpus-path data/autorag_csv/corpus.parquet \
  --config-path configs/autorag/local_pc.yaml \
  --project-dir evaluation/autorag_benchmark_pc
```

> **내장 패치** (`run_autorag_optimization.py`에 모두 포함, 세 config 공통 적용):
> 1. ChromaDB `add_embedding` — max_batch_size(5461) 초과 시 자동 분할
> 2. ChromaDB `is_exist` — SQLite 변수 제한(999) 초과 시 500개 단위 배치 분할
> 3. VRAM 순차 해제 — `BaseModule.run_evaluator` 후 `gc.collect` + `cuda.empty_cache`

### Step 3. 결과 확인

```bash
# 요약 CSV
cat evaluation/autorag_benchmark_csv/0/summary.csv

# 대시보드
autorag dashboard --trial_dir evaluation/autorag_benchmark_csv/0

# 최적 config 추출
autorag extract_best_config \
  --trial_path evaluation/autorag_benchmark_local/0 \
  --output_path evaluation/autorag_benchmark_local/best_config.yaml

# API 서빙
autorag run_api --trial_dir evaluation/autorag_benchmark_local/0 --host 0.0.0.0 --port 8000
```

---

## 파인튜닝

AutoRAG 평가 결과를 토대로 RAG 응답 품질에 특화된 모델을 파인튜닝할 수 있습니다.

### 로컬 LoRA/QLoRA

```bash
# 사전 설치
pip install peft trl bitsandbytes accelerate datasets

# GPU 서버 — LoRA ($MODEL_DIR은 .env의 MODEL_DIR 또는 SRV_DATA_DIR/models)
python scripts/finetune_local.py \
  --model-path $MODEL_DIR/kanana/kanana-1.5-2.1b \
  --output-dir models/finetuned/kanana-1.5 \
  --epochs 5 --lora-r 16

# QLoRA (4B 이상, 레지스트리 qlora=True 시 자동 적용)
python scripts/finetune_local.py \
  --model-path $MODEL_DIR/gemma/Gemma3-4B \
  --output-dir models/finetuned/gemma3 \
  --qlora --epochs 5

# EXAONE (trust_remote_code 필요)
python scripts/finetune_local.py \
  --model-path $MODEL_DIR/exaone/EXAONE-4.0-1.2B \
  --output-dir models/finetuned/exaone \
  --trust-remote-code --epochs 5
```

파인튜닝 완료 후 vLLM 서빙:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/finetuned/kanana-1.5/final \
  --port 8001
```

### OpenAI Fine-tuning

```bash
# 파인튜닝 시작
python scripts/finetune_openai.py start \
  --model gpt-4o-mini-2024-07-18 \
  --output-dir models/finetuned/openai

# 상태 확인
python scripts/finetune_openai.py status --job-id ftjob-xxxx

# 목록 조회
python scripts/finetune_openai.py list
```

완료 후 `configs/autorag/tutorial.yaml`의 `llm` 교체:
```yaml
llm: ft:gpt-4o-mini-2024-07-18:org:rag:xxxx
```

---

## 사용 가능한 로컬 모델 목록 (`$MODEL_DIR` — 기본: `/srv/shared_data/models/`)

### 임베딩 모델

#### 앱 / 인덱싱용 (`--hf-embedding-model`)

| 옵션 | 경로 | 크기 | 차원 | 특징 |
|------|------|------|------|------|
| `bge` (기본값) | `.../embeddings/BGE-m3-ko` | 2.2G | 1024 | 다국어, 고성능 |
| `sroberta` | `.../embeddings/ko-sroberta-multitask` | 846M | 768 | 한국어 특화, 경량 |

#### AutoRAG 비교 평가용 (vectordb 5종)

| 이름 | 경로 | 크기 | HF Hub ID | 특징 |
|------|------|------|-----------|------|
| `local_bge` | `.../embeddings/BGE-m3-ko` | 2.2G | `dragonkue/BGE-m3-ko` | 한국어 특화 BGE |
| `local_sroberta` | `.../embeddings/ko-sroberta-multitask` | 0.8G | `jhgan/ko-sroberta-multitask` | 한국어 sRoBERTa |
| `local_e5_large` | `.../embeddings/multilingual-e5-large` | 2.2G | `intfloat/multilingual-e5-large` | 다국어 E5-large |
| `local_kosimcse` | `.../embeddings/KoSimCSE-roberta-multitask` | 0.4G | `BM-K/KoSimCSE-roberta-multitask` | 한국어 SimCSE |
| `local_kf_deberta` | `.../embeddings/kf-deberta-multitask` | 0.7G | `upskyy/kf-deberta-multitask` | 한국어 DeBERTa |

### 채팅 모델 전체 목록 (서버 `/srv/shared_data/models/`)

| 모델명 | 경로 | 크기 | 22GB VRAM |
|--------|------|------|-----------|
| EXAONE-4.0-1.2B | `exaone/EXAONE-4.0-1.2B` | 2.4G | ✅ |
| EXAONE-Deep-2.4B | `exaone/EXAONE-Deep-2.4B` | 4.5G | ✅ |
| EXAONE-Deep-7.8B | `exaone/EXAONE-Deep-7.8B` | 15G | ✅ |
| Gemma3-4B | `gemma/Gemma3-4B` | 8.1G | ✅ |
| Gemma4-E4B | `gemma/Gemma4-E4B` | 15G | ✅ |
| kanana-1.5-2.1b | `kanana/kanana-1.5-2.1b` | 4.4G | ✅ |
| Midm-2.0-Mini | `midm/Midm-2.0-Mini` | 4.4G | ✅ |


### AutoRAG 평가 모델 (서버, `local.yaml` 기준)

| 모델명 | 크기 | gpu_memory_utilization | kv_cache_dtype |
|--------|------|----------------------|----------------|
| EXAONE-4.0-1.2B | 2.4G | 0.70 | auto |
| kanana-1.5-2.1b | 4.4G | 0.70 | auto |
| Midm-2.0-Mini | 4.4G | 0.70 | auto |
| Gemma3-4B | 8.1G | 0.70 | auto (max_model_len: 16384) |

### AutoRAG 평가 모델 (서버, `local.yaml` 내 Gemma4 그룹 기준)

| 모델명 | 크기 | gpu_memory_utilization | 특이사항 |
|--------|------|----------------------|---------|
| Gemma4-E4B | 15G | 0.85 | BF16, dense, max_model_len: 16384 |

### AutoRAG 평가 모델 (로컬 PC)

| HF Hub ID | 크기 | gpu_memory_utilization | kv_cache_dtype |
|-----------|------|----------------------|----------------|
| LGAI-EXAONE/EXAONE-4.0-1.2B | 2.4G | 0.80 | auto |
| kakaocorp/kanana-1.5-2.1b | 4.4G | 0.80 | auto |
| skt/Midm-2.0-Mini-Instruct | 4.4G | 0.80 | auto |

> `kv_cache_dtype: auto`: RTX 4070 (Ada sm_89) → fp8 / RTX 3060Ti (Ampere sm_86) → fp16 자동 선택

---

## 컬렉션 관리 (A/B안 분리 운영)

| 컬렉션명 | 시나리오 | 임베딩 | chunk_size | 비고 |
|---------|---------|--------|-----------|------|
| `rfp_chunk600` | B (OpenAI) | text-embedding-3-small (512) | 600 | 비교용 컬렉션 |
| `rfp_chunk800` | B (OpenAI) | text-embedding-3-small (512) | 800 | 비교용 — 최신 재평가에서 best_config는 유지됐지만 시간 gate는 미통과 |
| `rfp_chunk1000` | B (OpenAI) | text-embedding-3-small (512) | 1000 | 비교용 컬렉션 |
| `rfp_chunk1200` | B (OpenAI) | text-embedding-3-small (512) | 1200 | HWP 원문 추출 |
| `rfp_chunk600_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 600 | A안 비교용 컬렉션 |
| `rfp_chunk800_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 800 | **A안 기본값** |
| `rfp_chunk1000_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 1000 | A안 비교용 컬렉션 |
| `rfp_chunk1200_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 1200 | |

---

## gpt-5 계열 파라미터 제한사항

| 파라미터 | gpt-4 계열 | gpt-5 계열 |
|---------|-----------|-----------|
| `temperature` | 지원 | **미지원** (Scenario A 전용) |
| `top_p` | 지원 | **미지원** |
| `max_tokens` | 지원 | **미지원** → `max_completion_tokens` 사용 |
| `reasoning_effort` | 미지원 | low/medium/high |

> `src/generator.py`와 `src/evaluation/evaluator.py` 모두 `max_completion_tokens` 적용 완료.

## Complexity Note
- Remaining complexity hotspots:
  - retrieval strategy combination (`multi-query + rerank + hybrid`)
  - metadata filtering quality for near-match institution names
  - scenario A/B branching and local environment differences
- Mitigation:
  - keep `core` regression fixed and small,
  - add feature flags by config (not hard-coded branches),
  - document each extension in `docs/CODEBASE_MAP.md` first.
