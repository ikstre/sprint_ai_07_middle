# AutoRAG Guide

## 목적
- 수작업 파이프라인 외에 AutoRAG 탐색을 통해 검색/프롬프트/생성 조합을 자동 비교합니다.
- 최적 trial을 API/Web로 바로 배포 가능한 형태로 연결합니다.

## 사전 설치
- 권장 파이썬 버전: `3.11` 또는 `3.12`
- 설치:
```bash
pip install -r requirements-autorag.txt
```

참고:
- AutoRAG는 메인 서비스 스택과 의존성 충돌 가능성이 있어 별도 환경을 권장합니다.

## 1) 데이터 준비

### 1-A) CSV 기반 (권장) — `prepare_autorag_from_csv.py`

CSV 한 행이 곧 한 RFP 문서이며, `사업 요약` 컬럼을 generation_gt로 직접 사용합니다.  
retrieval_gt는 공고번호 기반으로 자동 연결되므로 ground truth 오류가 없습니다.

```bash
# --csv-path와 --output-dir은 .env의 METADATA_CSV, AUTORAG_DATA_DIR이 기본값
python scripts/prepare_autorag_from_csv.py \
  --chunk-size 600 \
  --chunk-overlap 100

# 경로를 명시적으로 지정할 경우
python scripts/prepare_autorag_from_csv.py \
  --csv-path $METADATA_CSV \
  --output-dir $AUTORAG_DATA_DIR \
  --chunk-size 600 \
  --chunk-overlap 100
```

- 산출물
  - `data/autorag_csv/corpus.parquet` — 664청크 (summary 95 + detail 569)
  - `data/autorag_csv/qa.parquet` — 431 QA쌍 (CSV 285 + single_dataset 96 + multi_dataset 50)

청크 구조:
- `chunk_0000`: 발주기관 + 사업명 + 사업 요약 (고정, retrieval_gt가 여기를 가리킴)
- `chunk_0001+`: 상세 텍스트를 naive_chunk로 분할

Ground truth 설계:
- `generation_gt`: 실제 사업 요약 텍스트 → METEOR/ROUGE 정상 계산 가능
- `retrieval_gt`: 공고번호 기반 직접 매핑 → 100% 정확

config: `configs/autorag/local_csv.yaml`

### 1-B) PDF/HWP 파일 기반 — `prepare_autorag_data.py`

```bash
python scripts/prepare_autorag_data.py \
  --documents-dir data \
  --metadata-csv data/data_list.csv \
  --output-dir data/autorag \
  --chunk-method semantic \
  --chunk-size 600 \
  --chunk-overlap 150
```

- 산출물
  - `data/autorag/corpus.parquet`
  - `data/autorag/qa.parquet`

청킹 옵션:
- `--chunk-method`: `semantic`(RFP 섹션 인식, 권장) / `naive`(고정 크기)
- `--chunk-size`: 청크 크기 (문자 수). 권장 500~800
- `--chunk-overlap`: 청크 간 중첩 크기. 권장 chunk-size의 20~25%

semantic 청킹 개선 사항 (`src/chunker.py`):
- 한국어 문장 경계(`다. `, `습니다. ` 등) 구분자 추가
- RFP 섹션 패턴 확장 (조항·별표·부록·표 캡션 등)
- 섹션 간 overlap 삽입으로 경계 손실 방지
- 최소 청크 크기(80자) 미달 시 직전 청크에 자동 병합

> **주의**: PDF/HWP 기반은 retrieval_gt가 토큰 매칭으로 결정되므로 정확도가 낮을 수 있습니다.  
> 평가 신뢰도가 중요한 경우 1-A) CSV 기반을 사용하세요.

## 2) 최적화 실행

### Scenario B (OpenAI API)
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/tutorial.yaml \
  --project-dir evaluation/autorag_benchmark
```

### Scenario A (GPU 서버, 22GB VRAM — NVIDIA L4)

> **주의**: kanana/midm은 transformers 5.x strict 검증 오류 → `PYTHONNOUSERSITE=1` 필수.  
> Gemma4는 user-local transformers 5.x + vLLM 필요 → 별도 실행 후 결과 병합.

```bash
# 사전 준비 — 모델 다운로드 (최초 1회)
python scripts/download_models.py           # 생성모델 + 임베딩 3종
# python scripts/download_models.py --embed-only  # 임베딩만
# python scripts/download_models.py --gen-only    # 생성모델만

# CSV 기반 데이터 준비 (권장 — 1-A 참조, .env의 METADATA_CSV/AUTORAG_DATA_DIR이 기본값)
python scripts/prepare_autorag_from_csv.py

# Step 1 — CSV 기반 실행 (권장)
nvidia-smi  # GPU 점유 확인
PYTHONNOUSERSITE=1 python scripts/run_autorag_optimization.py \
  --qa-path data/autorag_csv/qa.parquet \
  --corpus-path data/autorag_csv/corpus.parquet \
  --config-path configs/autorag/local_csv.yaml \
  --project-dir evaluation/autorag_benchmark_csv

# Step 1 — PDF/HWP 기반 실행 (대안)
PYTHONNOUSERSITE=1 python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/local.yaml \
  --project-dir evaluation/autorag_benchmark_local

# Step 2 — Gemma4-E4B 별도 실행 (user-local transformers 5.x + vLLM)
bash scripts/run_gemma4_optimization.sh

# Step 3 — 결과 병합 (CSV 기반 경로 기준)
python scripts/merge_gemma4_results.py \
  --main-dir evaluation/autorag_benchmark_csv \
  --gemma4-dir evaluation/autorag_benchmark_gemma4
```

Config 파일 목록:

| config | 데이터 | 용도 |
|--------|--------|------|
| `local_csv.yaml` | `data/autorag_csv/` | CSV 기반, **권장** (ground truth 정확, 평가 경로: `autorag_benchmark_csv`) |
| `local_csv_pipeline.yaml` | `data/autorag_csv/` | 통합 파이프라인용 (run_pipeline.py가 자동 생성, 파인튜닝 모델 포함) |
| `local.yaml` | `data/autorag/` | PDF/HWP 기반 대안 (retrieval_gt 토큰 매칭, 신뢰도 낮음) |
| `local_gemma4.yaml` | `data/autorag/` | Gemma4-E4B 전용 별도 실행 (`bash scripts/run_gemma4_optimization.sh`) |
| `local_pc.yaml` | `data/autorag/` | 로컬 PC (8GB GPU) |

`local_csv.yaml` 주요 설정:
- `max_tokens: [256, 512, 1024]` — 출력 토큰 수 비교 (generator 노드)
- 프롬프트 3종: baseline / RFP 구조 명시 / 간결형
- `top_k: [1,2,4,8,16]` — corpus 664개 규모에 최적화

Scenario A 생성 모델 (`configs/autorag/local.yaml` + `local_gemma4.yaml`):

| 모델 | 크기 | gpu_memory_utilization | 특이사항 |
|------|------|------------------------|---------|
| EXAONE-4.0-1.2B | 2.4G | 0.70 | trust_remote_code 필요 |
| kanana-1.5-2.1b | 4.4G | 0.70 | llama 계열, 한국어 특화 |
| Midm-2.0-Mini | 4.4G | 0.70 | llama 계열, 한국어 특화 |
| Gemma3-4B | 8.1G | 0.70 | max_model_len: 16384 |
| Gemma4-E4B | 15G | 0.85 | BF16, dense, 별도 실행 |

Scenario A 임베딩 모델 5종 비교 (`local.yaml` / `local_gemma4.yaml` vectordb):

| 이름 | 모델 경로 | 크기 | 특징 |
|------|----------|------|------|
| local_bge | `embeddings/BGE-m3-ko` | 2.2G | 다국어 BGE, 한국어 도메인 강함 |
| local_sroberta | `embeddings/ko-sroberta-multitask` | 0.8G | 한국어 sRoBERTa |
| local_e5_large | `embeddings/multilingual-e5-large` | 2.2G | 다국어 E5-large, retrieval 최강 |
| local_kosimcse | `embeddings/KoSimCSE-roberta-multitask` | 0.4G | 한국어 SimCSE, 의미 유사도 특화 |
| local_kf_deberta | `embeddings/kf-deberta-multitask` | 0.7G | 한국어 DeBERTa, 문맥 이해 우수 |

- 각 모델 평가 완료 후 VRAM 자동 해제 (`gc.collect` + `cuda.empty_cache`)
- `gpu_memory_utilization: 0.70` (22GB × 0.70 ≈ 15.4GB), Gemma4-E4B: 0.85
- `max_model_len: 16384` — Gemma3-4B, Gemma4-E4B에 적용 (131072 기본값은 KV 캐시 초과)
- `kv_cache_dtype: auto` → 모델 dtype(bfloat16) 자동 사용

### Scenario A-PC (로컬 PC, 8GB GPU)
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/local_pc.yaml \
  --project-dir evaluation/autorag_benchmark_pc
```

로컬 PC 설정 (`configs/autorag/local_pc.yaml`):
- `gpu_memory_utilization: 0.80` (8GB × 0.80 = 6.4GB 예약)
- `kv_cache_dtype: auto` (GPU 아키텍처 자동 감지)
  - RTX 4070 (Ada sm_89): fp8 가능
  - RTX 3060Ti (Ampere sm_86): fp16 fallback
- Gemma3-4B 제외 (8.1G > 6.4GB 한도)
- 임베딩: HuggingFace Hub에서 자동 다운로드 (`BAAI/bge-m3`, `jhgan/ko-sroberta-multitask`)

## 3) 통합 파이프라인 (run_pipeline.py)

데이터 준비 → 파인튜닝 → AutoRAG를 단일 명령으로 실행합니다.  
파인튜닝이 완료되면 학습된 모델이 AutoRAG config(`local_csv_pipeline.yaml`)에 **자동 추가**되어  
원본 모델과 동시에 METEOR/ROUGE 비교 평가됩니다.

### 사용 예

```bash
# 전체: 데이터 + 파인튜닝(2종) + AutoRAG
python scripts/run_pipeline.py --steps all \
  --finetune-models kanana-1.5,exaone

# 데이터 + AutoRAG (파인튜닝 생략)
python scripts/run_pipeline.py --steps data,autorag

# 파인튜닝 + AutoRAG (데이터 이미 준비됨)
python scripts/run_pipeline.py --steps finetune,autorag \
  --finetune-models kanana-1.5 \
  --finetune-epochs 5

# AutoRAG만 재실행
python scripts/run_pipeline.py --steps autorag

# 재학습 강제 (기존 결과 덮어쓰기)
python scripts/run_pipeline.py --steps finetune,autorag \
  --finetune-models kanana-1.5 \
  --force-finetune

# 7.8B 모델 — QLoRA 자동 적용
python scripts/run_pipeline.py --steps finetune,autorag \
  --finetune-models exaone-deep-7.8b
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--steps` | `all` | `data`, `finetune`, `autorag`, `post_eval` 또는 `all` |
| `--finetune-models` | (없음) | 쉼표 구분 모델 short name (아래 표 참조) |
| `--finetune-epochs` | `5` | LoRA 학습 에포크 수 |
| `--finetune-lr` | `2e-4` | 학습률 |
| `--max-seq-length` | `1024` | 최대 시퀀스 길이 |
| `--qlora` | false | 4-bit QLoRA (7.8B 이상, 메모리 제한 환경) |
| `--force-data` | false | corpus/qa 재생성 |
| `--force-finetune` | false | 기존 학습 모델 무시하고 재학습 |
| `--config-path` | `configs/autorag/local_csv.yaml` | 기본 AutoRAG config |
| `--project-dir` | `.env의 AUTORAG_PROJECT_DIR` | AutoRAG 출력 경로 |
| `--csv-path` | `.env의 METADATA_CSV` | CSV 데이터 경로 |
| `--data-dir` | `.env의 AUTORAG_DATA_DIR` | corpus/qa parquet 저장 위치 |
| `--eval-collection` | `rfp_chunk600` | post_eval 단계 ChromaDB 컬렉션 |

### `--finetune-models` 지원 모델

| short name | 실제 모델 | 크기 | 비고 |
|------------|----------|------|------|
| `kanana-1.5` | kanana-1.5-2.1b | 4.4G | |
| `exaone` | EXAONE-4.0-1.2B | 2.4G | trust_remote_code |
| `exaone-deep-2.4b` | EXAONE-Deep-2.4B | 4.5G | trust_remote_code |
| `exaone-deep-7.8b` | EXAONE-Deep-7.8B | 15G | **QLoRA 자동 적용** |
| `midm` | Midm-2.0-Mini | 4.4G | |
| `gemma3` | Gemma3-4B | 8.1G | QLoRA 자동 적용 |
| `gemma4` | Gemma4-E4B | 15G | QLoRA 자동 적용 |

### 파이프라인 동작 흐름

```
data 단계
  CSV → data/autorag_csv/corpus.parquet + qa.parquet

finetune 단계 (모델별 순차 실행)
  kanana-1.5  → models/finetuned/kanana-1.5/final
  exaone      → models/finetuned/exaone/final
  ...

autorag 단계
  ┌ 일반 모델 그룹 (kanana / midm / exaone / gemma3 등)
  │   configs/autorag/local_csv_pipeline.yaml 자동 생성
  │   (base config + 파인튜닝 모델 + eval-only 모델 generator 추가)
  │   → evaluation/autorag_benchmark_csv/0/
  │
  └ Gemma4 그룹 (gemma4 — transformers 5.x 필요)
      configs/autorag/local_csv_pipeline_gemma.yaml 자동 생성
      → evaluation/autorag_benchmark_csv_gemma/0/
```

> **Gemma4 그룹 분리 이유**: transformers 5.x가 필요한 Gemma4-E4B는 user-local 패키지를 허용해야 하므로  
> 별도 project dir(`autorag_benchmark_csv_gemma`)에서 자동으로 분리 실행됩니다.

## 4) 결과 확인

```bash
# CSV 기반 평가 결과 (권장)
cat evaluation/autorag_benchmark_csv/0/retrieve_node_line/*/summary.csv
cat evaluation/autorag_benchmark_csv/0/post_retrieve_node_line/*/summary.csv

# 대시보드 (시각화)
autorag dashboard --trial_dir evaluation/autorag_benchmark_csv/0

# 최적 config 추출
autorag extract_best_config \
  --trial_path evaluation/autorag_benchmark_csv/0 \
  --output_path evaluation/autorag_benchmark_csv/best_config.yaml
```

## 5) 배포

- AutoRAG API
```bash
autorag run_api --trial_dir evaluation/autorag_benchmark/0 --host 0.0.0.0 --port 8000
```

- AutoRAG Web
```bash
autorag run_web --trial_path evaluation/autorag_benchmark/0
```

- 래퍼 스크립트
```bash
python scripts/run_autorag_api.py --trial-dir evaluation/autorag_benchmark/0
python scripts/run_autorag_web.py --trial-path evaluation/autorag_benchmark/0
```

## 6) 앱 통합
- FastAPI 래퍼: `apps/autorag_api.py`
- Streamlit 래퍼: `apps/autorag_streamlit.py`
- 공통 런타임: `src/autorag_runner.py`

## 7) 파인튜닝 (단독 실행)

통합 파이프라인(`run_pipeline.py`) 사용을 권장합니다.  
단독 실행이 필요한 경우 아래 명령을 사용하세요.

### 7-1) 로컬 LoRA/QLoRA (오픈소스 모델)

```bash
# 사전 설치
pip install peft trl bitsandbytes accelerate datasets

# LoRA — CSV 데이터 기반 ($MODEL_DIR은 .env의 MODEL_DIR 또는 SRV_DATA_DIR/models)
python scripts/finetune_local.py \
    --model-path $MODEL_DIR/kanana/kanana-1.5-2.1b \
    --output-dir models/finetuned/kanana-1.5 \
    --epochs 5

# QLoRA (4B 이상 모델, 레지스트리 qlora=True 시 자동 적용)
python scripts/finetune_local.py \
    --model-path $MODEL_DIR/gemma/Gemma3-4B \
    --output-dir models/finetuned/gemma3 \
    --qlora --epochs 5
```

주요 옵션:
| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--qa-path` | `data/autorag_csv/qa.parquet` | QA 데이터 경로 |
| `--corpus-path` | `data/autorag_csv/corpus.parquet` | Corpus 경로 |
| `--qlora` | false | 4-bit 양자화 (8GB GPU용) |
| `--lora-r` | 16 | LoRA rank |
| `--batch-size` | 2 | 배치 크기 |
| `--grad-accum` | 8 | Gradient accumulation |
| `--max-seq-length` | 2048 | 최대 시퀀스 길이 |

학습 후 vLLM으로 서빙:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/finetuned/kanana-1.5/final \
    --port 8001
```

### 6-2) OpenAI Fine-tuning

```bash
# 파인튜닝 시작 (gpt-4o-mini 기준 약 $3~10)
python scripts/finetune_openai.py start \
    --model gpt-4o-mini-2024-07-18 \
    --output-dir models/finetuned/openai

# 상태 확인
python scripts/finetune_openai.py status --job-id ftjob-xxxx

# 작업 목록
python scripts/finetune_openai.py list
```

완료 후 `configs/autorag/tutorial.yaml`의 `llm` 필드를 파인튜닝된 모델 ID로 교체:
```yaml
- module_type: openai_llm
  llm: ft:gpt-4o-mini-2024-07-18:org:rag:xxxx
```

## 8) 주요 버그 패치 (run_autorag_optimization.py 내장 — 모든 config에 자동 적용)

- **Patch 1 — ChromaDB batch 초과**: `add_embedding`을 max_batch_size(5461) 단위로 분할 처리 (11567+ corpus 대응)
- **Patch 2 — VRAM 순차 해제**: `BaseModule.run_evaluator` 후 `gc.collect` + `cuda.empty_cache` 강제 실행
- **Patch 3 — ChromaDB is_exist SQLite 변수 초과**: `is_exist`를 500개 단위 배치로 분할 처리
- **Patch 4 — VectorDB ingestion 후 임베딩 GPU 해제**: ingestion 완료 직후 `vectordb.embedding = None` + `cuda.empty_cache` (임베딩 5종 × ~1.3GB 절감, generator 단계 VRAM 확보)
- **Patch 5 — summary.csv 모델명 정규화**: `module_name`을 `Vllm` 고정값에서 모델 경로 basename으로 치환 (결과 가독성 향상)
- **Patch 6 — HybridCC 정규화 0-division 수정**: `normalize_mm/tmm/z/dbsf` 함수에서 `max == min`일 때 NaN 대신 정상값 반환 (`RuntimeWarning: invalid value encountered in divide` 제거)

## 9) 운영 팁
- 첫 번째 trial(`.../0`)만 고정 사용하지 말고, 지표 기반으로 승자를 선택합니다.
- AutoRAG 결과도 `core` 지표와 함께 재검증해 운영 회귀를 막습니다.
- 파인튜닝 전후 AutoRAG 평가를 비교해 실질적 개선 여부를 확인합니다.
