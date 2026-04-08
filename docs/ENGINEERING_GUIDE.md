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
- `configs/autorag/`: AutoRAG search configs (B / A-server / A-PC).
- `notebooks/`: Reproducible notebook workflows.

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

## Scenario A / B 선택 가이드

| 항목 | Scenario B (OpenAI API) | Scenario A — 서버 | Scenario A-PC — 로컬 PC |
|------|------------------------|--------------------|------------------------|
| 실행 환경 | 로컬/GCP 모두 가능 | GCP GPU 서버 (22GB) | RTX 4070 / 3060Ti (8GB) |
| 채팅 모델 | gpt-5-mini / gpt-5-nano / gpt-5 | EXAONE, Gemma3, kanana, Midm | 동일 (HF Hub 다운로드) |
| 임베딩 모델 | text-embedding-3-small (dim=512) | BGE-m3-ko / ko-sroberta | BAAI/bge-m3 / ko-sroberta (HF Hub) |
| 파인튜닝 | OpenAI Fine-tuning API | LoRA / QLoRA | QLoRA (4-bit, 8GB 대응) |
| 필수 환경변수 | `OPENAI_API_KEY` | 없음 | 없음 (HF 모델 자동 다운로드) |
| AutoRAG config | `tutorial.yaml` | `local.yaml` | `local_pc.yaml` |

> **전환 방법**: 앱 실행 시 사이드바 **실행 모드** 라디오 버튼으로 A ↔ B 실시간 전환 가능.
> 단, A안과 B안은 **임베딩 차원이 달라** 컬렉션을 별도로 인덱싱해야 합니다.

---

## 지원 문서 포맷

`DocumentLoader`는 아래 4가지 포맷을 자동 감지합니다.

| 포맷 | 확장자 | 인코딩 자동 감지 | 특이사항 |
|------|--------|----------------|---------|
| HWP | `.hwp` | 바이너리 파싱 | OLE 구조 직접 파싱 |
| PDF | `.pdf` | — | pdfplumber 사용 |
| 텍스트 | `.txt` | UTF-8 → CP949 → EUC-KR 순 시도 | 일반 텍스트 |
| CSV | `.csv` | UTF-8 → CP949 → EUC-KR 순 시도 | 아래 CSV 옵션 참고 |

> `data_list.csv`(메타데이터 파일)는 자동으로 인덱싱에서 제외됩니다.

### CSV 로드 옵션

| 옵션 | 설명 | 사용 상황 |
|------|------|----------|
| 기본 (전체 문서) | CSV 전체를 하나의 문서로 처리 | RFP 내용이 여러 행에 걸쳐 정리된 경우 |
| `--csv-row-per-doc` | CSV 각 행을 개별 문서로 처리 | 행 하나가 RFP 한 건인 경우 |
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

AutoRAG는 현재 환경(`sprint_env`)에 설치돼 있어 별도 환경 전환 없이 바로 실행 가능합니다.

### Step 1. 데이터 준비

```bash
# 권장 설정 (RFP 섹션 인식 semantic 청킹)
python scripts/prepare_autorag_data.py \
  --documents-dir data \
  --metadata-csv data/data_list.csv \
  --output-dir data/autorag \
  --chunk-method semantic \
  --chunk-size 600 \
  --chunk-overlap 150
```

CSV 파일이 포함된 경우 옵션 추가:

```bash
python scripts/prepare_autorag_data.py \
  --documents-dir data \
  --output-dir data/autorag \
  --csv-text-columns "사업명,요구사항" \
  --csv-row-per-doc
```

산출물: `data/autorag/corpus.parquet`, `data/autorag/qa.parquet`

### Step 2. 최적화 실행

AutoRAG config 파일은 시나리오별로 분리되어 있습니다.

| 파일 | 시나리오 | Generator | 임베딩 비교 |
|------|---------|-----------|-----------|
| `configs/autorag/tutorial.yaml` | B (OpenAI) | `openai_llm` — gpt-5-mini | text-embedding-3-small |
| `configs/autorag/local.yaml` | A (서버, 22GB) | `vllm` — 5종 모델 | BGE-m3-ko vs ko-sroberta |
| `configs/autorag/local_pc.yaml` | A (PC, 8GB) | `vllm` — 4종 모델 | BAAI/bge-m3 vs ko-sroberta (HF Hub) |

**Scenario B (OpenAI)**
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/tutorial.yaml \
  --project-dir evaluation/autorag_benchmark
```

**Scenario A — GPU 서버**
```bash
nvidia-smi  # 실행 전 GPU 점유 확인
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/local.yaml \
  --project-dir evaluation/autorag_benchmark_local
```

**Scenario A-PC — 로컬 PC (8GB GPU)**
```bash
# 첫 실행 시 HuggingFace에서 모델 자동 다운로드
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
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
cat evaluation/autorag_benchmark_local/0/summary.csv

# 대시보드
autorag dashboard --trial_dir evaluation/autorag_benchmark_local/0

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

# GPU 서버 — LoRA
python scripts/finetune_local.py \
  --model-path /srv/shared_data/models/kanana/kanana-nano-2.1b \
  --output-dir models/finetuned/kanana-nano-rag \
  --epochs 3 --lora-r 16

# 로컬 PC (8GB GPU) — QLoRA (4-bit 양자화)
python scripts/finetune_local.py \
  --model-path kakaocorp/kanana-nano-2.1b \
  --output-dir models/finetuned/kanana-nano-rag \
  --qlora --epochs 3

# EXAONE (trust_remote_code 필요)
python scripts/finetune_local.py \
  --model-path /srv/shared_data/models/exaone/EXAONE-4.0-1.2B \
  --output-dir models/finetuned/exaone-rag \
  --trust-remote-code --epochs 3
```

파인튜닝 완료 후 vLLM 서빙:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/finetuned/kanana-nano-rag/final \
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

## 사용 가능한 로컬 모델 목록 (`/srv/shared_data/models/`)

### 임베딩 모델

| 옵션 (`--hf-embedding-model`) | 경로 | 크기 | 차원 | 특징 |
|-------------------------------|------|------|------|------|
| `bge` (기본값) | `.../embeddings/BGE-m3-ko` | 2.2G | 1024 | 다국어, 고성능 |
| `sroberta` | `.../embeddings/ko-sroberta-multitask` | 846M | 768 | 한국어 특화, 경량 |

### 채팅 모델 전체 목록 (서버 `/srv/shared_data/models/`)

| 모델명 | 경로 | 크기 | 22GB VRAM |
|--------|------|------|-----------|
| EXAONE-4.0-1.2B | `exaone/EXAONE-4.0-1.2B` | 2.4G | ✅ |
| EXAONE-Deep-2.4B | `exaone/EXAONE-Deep-2.4B` | 4.5G | ✅ |
| EXAONE-Deep-7.8B | `exaone/EXAONE-Deep-7.8B` | 15G | ✅ |
| EXAONE-3.5-7.8B | `exaone/EXAONE-3.5-7.8B` | 30G | ✅ |
| Gemma3-4B | `gemma/Gemma3-4B` | 8.1G | ✅ |
| Gemma4-E4B | `gemma/Gemma4-E4B` | 15G | ⚠️ fp8 필요 |
| Gemma4-26B-A4B | `gemma/Gemma4-26B-A4B` | 49G | ❌ VRAM 초과 |
| kanana-nano-2.1b | `kanana/kanana-nano-2.1b` | 4.0G | ✅ |
| kanana-1.5-2.1b | `kanana/kanana-1.5-2.1b` | 4.4G | ✅ |
| Midm-2.0-Mini | `midm/Midm-2.0-Mini` | 4.4G | ✅ |


### AutoRAG 평가 모델 (서버, `local.yaml` 기준)

| 모델명 | 크기 | gpu_memory_utilization | kv_cache_dtype |
|--------|------|----------------------|----------------|
| EXAONE-4.0-1.2B | 2.4G | 0.70 | fp8 |
| kanana-nano-2.1b | 4.0G | 0.70 | fp8 |
| kanana-1.5-2.1b | 4.4G | 0.70 | fp8 |
| Midm-2.0-Mini | 4.4G | 0.70 | fp8 |
| Gemma3-4B | 8.1G | 0.70 | fp8 |

> Gemma4-E4B(15G)는 fp8 KV 캐시 적용 시 추가 가능 — `local.yaml`의 generator.modules에 vllm 블록 추가.  
> Gemma4-26B-A4B(49G)는 단일 GPU 로드 불가.

### AutoRAG 평가 모델 (로컬 PC)

| HF Hub ID | 크기 | gpu_memory_utilization | kv_cache_dtype |
|-----------|------|----------------------|----------------|
| LGAI-EXAONE/EXAONE-4.0-1.2B | 2.4G | 0.80 | auto |
| kakaocorp/kanana-nano-2.1b | 4.0G | 0.80 | auto |
| kakaocorp/kanana-1.5-2.1b | 4.4G | 0.80 | auto |
| skt/Midm-2.0-Mini-Instruct | 4.4G | 0.80 | auto |

> `kv_cache_dtype: auto`: RTX 4070 (Ada sm_89) → fp8 / RTX 3060Ti (Ampere sm_86) → fp16 자동 선택

---

## 컬렉션 관리 (A/B안 분리 운영)

| 컬렉션명 | 시나리오 | 임베딩 | chunk_size |
|---------|---------|--------|-----------|
| `rfp_chunk1200` | B (OpenAI) | text-embedding-3-small (512) | 1200 |
| `rfp_chunk800` | B (OpenAI) | text-embedding-3-small (512) | 800 |
| `rfp_chunk1200_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 1200 |
| `rfp_chunk800_a` | A (BGE-m3-ko) | BGE-m3-ko (1024) | 800 |
| `rfp_chunk1200_a_sroberta` | A (ko-sroberta) | ko-sroberta-multitask (768) | 1200 |

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
