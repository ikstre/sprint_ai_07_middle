# AutoRAG Guide

## 목적
- 수작업 파이프라인 외에 AutoRAG 탐색을 통해 검색/프롬프트/생성 조합을 자동 비교합니다.
- 최적 trial을 API/Web로 바로 배포 가능한 형태로 연결합니다.

## 사전 설치
- 권장 파이썬 버전: `3.11` 또는 `3.12`
- 설치:
```bash
pip install -r requirements.txt
```

참고:
- `requirements.txt` 내부 마커로 Python `<3.13`에서만 AutoRAG가 설치됩니다.

## 1) 데이터 준비

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

# Step 1 — 메인 실행 (EXAONE / kanana / Midm / Gemma3, 임베딩 5종 비교)
nvidia-smi  # GPU 점유 확인
PYTHONNOUSERSITE=1 python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/local.yaml \
  --project-dir evaluation/autorag_benchmark_local

# Step 2 — Gemma4-E4B 별도 실행 (user-local transformers 5.x + vLLM)
bash scripts/run_gemma4_optimization.sh

# Step 3 — 결과 병합
python scripts/merge_gemma4_results.py \
  --main-dir evaluation/autorag_benchmark_local \
  --gemma4-dir evaluation/autorag_benchmark_gemma4
```

Scenario A 생성 모델 (`configs/autorag/local.yaml` + `local_gemma4.yaml`):

| 모델 | 크기 | gpu_memory_utilization | 특이사항 |
|------|------|------------------------|---------|
| EXAONE-4.0-1.2B | 2.4G | 0.70 | trust_remote_code 필요 |
| kanana-nano-2.1b | 4.0G | 0.70 | llama 계열, 한국어 특화 |
| kanana-1.5-2.1b | 4.4G | 0.70 | llama 계열, 한국어 특화 |
| Midm-2.0-Mini | 4.4G | 0.70 | llama 계열, 한국어 특화 |
| Gemma3-4B | 8.1G | 0.70 | max_model_len: 8192 |
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
- `max_model_len: 8192` — Gemma3-4B, Gemma4-E4B에 적용 (KV 캐시 초과 방지)
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

## 3) 결과 확인

```bash
# 요약 (어떤 모듈이 최적인지)
cat evaluation/autorag_benchmark_local/0/summary.csv

# 대시보드 (시각화)
autorag dashboard --trial_dir evaluation/autorag_benchmark_local/0

# 최적 config 추출
autorag extract_best_config \
  --trial_path evaluation/autorag_benchmark_local/0 \
  --output_path evaluation/autorag_benchmark_local/best_config.yaml
```

## 4) 배포

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

## 5) 앱 통합
- FastAPI 래퍼: `apps/autorag_api.py`
- Streamlit 래퍼: `apps/autorag_streamlit.py`
- 공통 런타임: `src/autorag_runner.py`

## 6) 파인튜닝

AutoRAG 평가 결과를 토대로 RAG에 특화된 모델을 파인튜닝할 수 있습니다.
학습 데이터는 `qa.parquet` + `corpus.parquet`에서 자동 생성됩니다.

### 6-1) 로컬 LoRA/QLoRA (오픈소스 모델)

```bash
# 사전 설치
pip install peft trl bitsandbytes accelerate datasets

# LoRA (서버 GPU, VRAM 충분)
python scripts/finetune_local.py \
    --model-path /srv/shared_data/models/kanana/kanana-nano-2.1b \
    --output-dir models/finetuned/kanana-nano-rag \
    --epochs 3

# QLoRA (로컬 PC, 8GB GPU)
python scripts/finetune_local.py \
    --model-path kakaocorp/kanana-nano-2.1b \
    --output-dir models/finetuned/kanana-nano-rag \
    --qlora \
    --epochs 3
```

주요 옵션:
| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--qlora` | false | 4-bit 양자화 (8GB GPU용) |
| `--lora-r` | 16 | LoRA rank (높을수록 파라미터↑, 성능↑) |
| `--batch-size` | 2 | 배치 크기 |
| `--grad-accum` | 8 | Gradient accumulation (실효 배치 = 2×8=16) |
| `--max-seq-length` | 2048 | 최대 시퀀스 길이 |

학습 후 vLLM으로 서빙:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model models/finetuned/kanana-nano-rag/final \
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

## 7) 주요 버그 패치 (run_autorag_optimization.py 내장)

- **ChromaDB batch 초과**: `add_embedding`을 max_batch_size(5461) 단위로 분할 처리
- **ChromaDB is_exist SQLite 변수 초과**: `is_exist`를 500개 단위 배치로 분할 처리
- **VRAM 순차 해제**: `BaseModule.run_evaluator` 후 `gc.collect` + `cuda.empty_cache` 강제 실행

## 8) 운영 팁
- 첫 번째 trial(`.../0`)만 고정 사용하지 말고, 지표 기반으로 승자를 선택합니다.
- AutoRAG 결과도 `core` 지표와 함께 재검증해 운영 회귀를 막습니다.
- 파인튜닝 전후 AutoRAG 평가를 비교해 실질적 개선 여부를 확인합니다.
