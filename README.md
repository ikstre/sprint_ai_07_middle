# 입찰메이트 기술 1팀 RAG시스템 구현 프로젝트

기업/공공 RFP 문서를 대상으로 한 RAG 시스템 프로젝트입니다.

- **A안**: 로컬 모델 + AutoRAG 최적화 + LoRA 파인튜닝 (GPU 서버)
- **B안**: OpenAI API 기반 서비스형 RAG

## 협업일지

- Notion: https://www.notion.so/AI-7-1-3347356fc5cb8199b3b1c1c83dd1467b?source=copy_link

---

## 디렉토리 구조

```
sprint_ai_07_middle/
├── app.py                          # Streamlit 챗봇 앱 (A/B안 통합)
├── configs/
│   ├── paths.py                    # 중앙 경로 설정 (.env 기반)
│   ├── config.py                   # 런타임 설정
│   └── autorag/
│       ├── local_csv.yaml          # AutoRAG 메인 config (CSV 기반, 권장)
│       ├── local.yaml              # AutoRAG config (PDF/HWP 기반, 대안)
│       ├── local_pc.yaml           # AutoRAG config (8GB GPU PC용)
│       └── tutorial.yaml           # AutoRAG config (OpenAI B안)
├── scripts/
│   ├── run_pipeline.py             # 통합 파이프라인 (A안 전체 자동화)
│   ├── prepare_autorag_from_csv.py # CSV → corpus/qa parquet
│   ├── index_documents.py          # 문서 → ChromaDB 인덱싱
│   ├── run_autorag_optimization.py # AutoRAG 평가 실행
│   ├── merge_gemma4_results.py     # Gemma4 별도 AutoRAG 결과를 메인 trial에 병합
│   ├── finetune_local.py           # LoRA/QLoRA 파인튜닝
│   ├── finetune_openai.py          # OpenAI Fine-tuning API
│   ├── run_evaluation.py           # 질문지 기반 서비스 평가 (A/B안, Chroma 컬렉션 대상)
│   ├── check_release_gate.py       # 릴리즈 게이트
│   ├── download_models.py          # 모델 다운로드
│   ├── repair_finetuned_models.py  # 파인튜닝 모델 복구
│   └── check_env.py                # 환경 진단
├── src/                            # 공통 모듈 (loader/chunker/embedder/retriever/generator)
├── apps/                           # AutoRAG API/Streamlit 앱
└── data/                           # 데이터 (gitignore)
```

---

## 설치

requirements 파일 역할:

- `requirements-integrated.txt`: 통합 환경의 실제 의존성 정의
- `requirements-gemma4.txt`: Gemma4 전용 유저 환경의 실제 의존성 정의

```bash
# 통합 환경(integrated env): B안 실행/평가 + A안 공통 작업
pip install -r requirements-integrated.txt

# Gemma4 전용 유저 환경(user env)
pip install -r requirements-gemma4.txt

# BM25 ko_okt 토크나이저 (Java 필요)
sudo apt-get install -y default-jdk
```

권장 환경 분리:

```bash
# 1) integrated env
conda activate sprint_env
pip install -r requirements-integrated.txt

# 2) gemma4 user env
conda create -n gemma4_env python=3.11
conda activate gemma4_env
pip install -r requirements-gemma4.txt
```

`.env` 설정:

```bash
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
AUTORAG_PYTHON=/path/to/envs/gemma4_env/bin/python
# 서버 경로 (기본: /srv/shared_data)
# SRV_DATA_DIR=/srv/shared_data
```

- `integrated env`는 `B안` 실행/평가와 `A안`의 인덱싱·파인튜닝 같은 공통 작업을 모두 포함합니다.
- `gemma4 env`는 Gemma4 기반 AutoRAG 실험 전용 유저 환경입니다. `scripts/run_autorag_optimization.py`, `scripts/run_autorag_api.py`, `scripts/run_autorag_web.py`는 `AUTORAG_PYTHON`이 설정돼 있으면 해당 인터프리터로 자동 전환합니다.
- 설치 문서와 팀 커뮤니케이션에서는 `requirements-integrated.txt`, `requirements-gemma4.txt` 두 파일만 사용합니다.

---

## A안 실행 가이드

### 방법 1: 통합 파이프라인 (권장)

`run_pipeline.py` 한 명령으로 데이터 준비 → 파인튜닝 → 인덱싱 → AutoRAG 평가 → 최적 임베딩 인덱싱까지 자동 실행합니다.

#### 전체 실행 (처음부터)

```bash
python scripts/run_pipeline.py \
  --steps data,finetune,index,autorag,best_index \
  --chunk-sizes 600,800,1000,1200 \
  --chunk-overlap 100 \
  --index-scenario A \
  --max-seq-length 2048 \
  --force-data \
  --finetune-models kanana-1.5,exaone,exaone-deep-2.4b,midm,gemma3,exaone-deep-7.8b \
  --config-path configs/autorag/local_csv.yaml
```

#### 부분 실행 예시

```bash
# AutoRAG만 (데이터·파인튜닝 이미 완료)
python scripts/run_pipeline.py \
  --steps autorag,best_index \
  --chunk-sizes 600,800,1000,1200 \
  --config-path configs/autorag/local_csv.yaml

# 파인튜닝 없이 데이터+AutoRAG
python scripts/run_pipeline.py \
  --steps data,index,autorag,best_index \
  --chunk-sizes 600,800,1000,1200

# 특정 모델만 파인튜닝 후 AutoRAG
python scripts/run_pipeline.py \
  --steps finetune,autorag,best_index \
  --finetune-models kanana-1.5,exaone
```

#### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--steps` | `all` | 실행 단계: `data` `index` `finetune` `autorag` `best_index` |
| `--chunk-sizes` | (없음) | 다중 청크 크기 (예: `600,800,1000,1200`) — 지정 시 각 크기별 반복 |
| `--chunk-size` | `600` | 단일 청크 크기 (`--chunk-sizes` 미사용 시) |
| `--chunk-overlap` | `100` | 청크 중복 길이 |
| `--finetune-models` | (없음) | 파인튜닝 모델 (아래 표 참조) |
| `--finetune-epochs` | `10` | 최대 에포크 (early stop 포함) |
| `--max-seq-length` | `1024` | 학습 최대 시퀀스 길이 |
| `--force-data` | false | corpus/qa 강제 재생성 |
| `--force-finetune` | false | 기존 모델 무시하고 재학습 |
| `--config-path` | `configs/autorag/local_csv.yaml` | AutoRAG config |
| `--index-scenario` | `A` | 인덱싱 시나리오 (`A`=로컬 HF, `B`=OpenAI) |

#### 지원 파인튜닝 모델

| `--finetune-models` 값 | 실제 모델 | 크기 | 비고 |
|------------------------|----------|------|------|
| `kanana-1.5` | kanana-1.5-2.1b | 4.4G | |
| `exaone` | EXAONE-4.0-1.2B | 2.4G | trust_remote_code |
| `exaone-deep-2.4b` | EXAONE-Deep-2.4B | 4.5G | trust_remote_code |
| `exaone-deep-7.8b` | EXAONE-Deep-7.8B | 15G | QLoRA 자동 적용 |
| `midm` | Midm-2.0-Mini | 4.4G | |
| `gemma3` | Gemma3-4B | 8.1G | QLoRA 자동 적용 |
| `gemma4` | Gemma4-E4B | 15G | QLoRA 자동 적용 |

---

### 방법 2: 단계별 개별 실행

#### Step 1. 데이터 준비

```bash
# 청크 크기별로 각각 실행 (600/800/1000/1200)
python scripts/prepare_autorag_from_csv.py \
  --csv-path /srv/shared_data/datasets/data_list_cleaned.csv \
  --output-dir data/autorag_csv_600 \
  --chunk-size 600 \
  --chunk-overlap 100

python scripts/prepare_autorag_from_csv.py \
  --csv-path /srv/shared_data/datasets/data_list_cleaned.csv \
  --output-dir data/autorag_csv_800 \
  --chunk-size 800 \
  --chunk-overlap 100
# (1000, 1200도 동일)
```

산출물: `data/autorag_csv_{size}/corpus.parquet`, `qa.parquet`

#### Step 2. ChromaDB 인덱싱

```bash
# Scenario A — 로컬 임베딩 (기본: BGE-m3-ko)
python scripts/index_documents.py \
  --scenario A \
  --from-parquet data/autorag_csv_800/corpus.parquet \
  --collection rfp_chunk800_a \
  --hf-embedding-model bge

# 다른 임베딩 모델 옵션: bge / sroberta / e5 / kosimcse / kf_deberta
```

#### Step 3. 파인튜닝 (선택)

```bash
# LoRA (소형 모델)
python scripts/finetune_local.py \
  --model-path /srv/shared_data/models/kanana/kanana-1.5-2.1b \
  --output-dir models/finetuned/kanana-1.5 \
  --epochs 10 \
  --max-seq-length 2048

# QLoRA (대형 모델 — 레지스트리 qlora=True 시 자동, 또는 --qlora 강제)
python scripts/finetune_local.py \
  --model-path /srv/shared_data/models/gemma/Gemma3-4B \
  --output-dir models/finetuned/gemma3 \
  --qlora \
  --epochs 10
```

#### Step 4. AutoRAG 평가

```bash
# 실행 전 GPU 여유 확인
nvidia-smi

# 청크 크기별로 각각 실행
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag_csv_800/qa.parquet \
  --corpus-path data/autorag_csv_800/corpus.parquet \
  --config-path configs/autorag/local_csv.yaml \
  --project-dir evaluation/autorag_benchmark_csv
```

#### Step 5. 결과 확인

```bash
# 요약 CSV
cat evaluation/autorag_benchmark_csv/0/retrieve_node_line/*/summary.csv
cat evaluation/autorag_benchmark_csv/0/post_retrieve_node_line/*/summary.csv

# 대시보드
autorag dashboard --trial_dir evaluation/autorag_benchmark_csv/0
```

#### Step 5-1. Gemma4 결과 병합 (별도 GPU 환경에서 실험한 경우)

```bash
# Gemma4 전용 환경에서 AutoRAG 실행 후 별도 프로젝트 디렉터리에 결과가 있을 때
python scripts/merge_gemma4_results.py \
  --main-dir evaluation/autorag_benchmark_csv \
  --gemma4-dir evaluation/autorag_benchmark_csv_gemma
```

---

## B안 실행 가이드

### Step 1. 인덱싱

```bash
# OpenAI 임베딩 (text-embedding-3-small)
python scripts/index_documents.py \
  --scenario B \
  --collection rfp_chunk800 \
  --chunk-size 800

# CSV 기반 corpus 직접 임베딩
python scripts/index_documents.py \
  --scenario B \
  --from-parquet data/autorag_csv_800/corpus.parquet \
  --collection rfp_chunk800
```

### Step 2. 앱 실행

```bash
streamlit run app.py
```

사이드바에서 `B: OpenAI API` 선택 → 컬렉션 / 모델(gpt-5-mini 등) 선택

### Step 3. 평가

```bash
# core 지표 평가 (hit@5, nDCG@5, grounded ratio 등)
python scripts/run_evaluation.py \
  --scenario B \
  --mode core \
  --collection rfp_chunk800 \
  --output-dir evaluation

# Scenario A core 평가
python scripts/run_evaluation.py \
  --scenario A \
  --mode core \
  --collection rfp_chunk800_a \
  --output-dir evaluation/a_chunk800_core

# 여러 chunk 크기 병렬 평가 (예: B안 600/800/1000/1200)
python scripts/run_evaluation.py \
  --scenario B \
  --mode core \
  --chunk-sizes 600,800,1000,1200 \
  --output-dir evaluation

# LLM judge 포함 상세 평가
python scripts/run_evaluation.py \
  --scenario B \
  --mode detailed \
  --judge on \
  --collection rfp_chunk600 \
  --output-dir evaluation

# 릴리즈 게이트 확인
python scripts/check_release_gate.py
```

주의:
- `run_evaluation.py`는 Chroma 컬렉션을 대상으로 하는 질문지 기반 서비스 평가입니다.
- 저장소에 커밋된 `evaluation/autorag_benchmark_csv`, `evaluation/autorag_benchmark_csv_gemma` 는 A안 AutoRAG 벤치마크 결과이며, `run_evaluation.py`의 출력이 아닙니다.
- 현재 저장소에 가공 청크가 포함된 A안 기준 산출물은 `rfp_chunk600_a`, `rfp_chunk800_a`, `rfp_chunk1000_a`, `rfp_chunk1200_a` 계열입니다.
- A안에서 `--mode detailed` 또는 `--judge on`을 쓰면 judge 단계는 여전히 OpenAI API를 사용합니다.
- `--chunk-sizes`를 쓰면 각 크기별 평가를 별도 하위 디렉터리에 병렬 실행하고, 로그는 각 `run.log`에 저장합니다.

최신 B안 core 재평가 요약:
- 결과 경로: `evaluation/b_chunk{600,800,1000,1200}_full_core/`
- `similarity_k5` 기준 대표 수치:
  - `600`: p95 `20.27s`, hit@5 `0.900`, field_coverage `0.596`, grounded `0.552`
  - `800`: p95 `19.55s`, hit@5 `0.900`, field_coverage `0.584`, grounded `0.588`
  - `1000`: p95 `18.69s`, hit@5 `0.905`, field_coverage `0.586`, grounded `0.557`
  - `1200`: p95 `19.70s`, hit@5 `0.905`, field_coverage `0.608`, grounded `0.590`
- 최신 재평가에서는 4개 chunk 모두 품질 지표는 기준 이상이지만, `p95_elapsed_time`와 `decline_accuracy missing` 때문에 core gate는 모두 미통과입니다.
- 따라서 현재는 특정 chunk를 “최종 운영 기본값”으로 확정하기보다, 속도 개선 또는 gate 기준 재정의가 먼저 필요한 상태입니다.

---

## 컬렉션 목록

| 컬렉션명 | 시나리오 | 임베딩 모델 | chunk_size |
|---------|---------|------------|-----------|
| `rfp_chunk600` | B (OpenAI) | text-embedding-3-small (512-dim) | 600 |
| `rfp_chunk800` | B (OpenAI) | text-embedding-3-small (512-dim) | 800 |
| `rfp_chunk1000` | B (OpenAI) | text-embedding-3-small (512-dim) | 1000 |
| `rfp_chunk1200` | B (OpenAI) | text-embedding-3-small (512-dim) | 1200 |
| `rfp_chunk600_a` | A (BGE-m3-ko) | BGE-m3-ko (1024-dim) | 600 |
| `rfp_chunk800_a` | A (BGE-m3-ko) | BGE-m3-ko (1024-dim) | 800 |
| `rfp_chunk1000_a` | A (BGE-m3-ko) | BGE-m3-ko (1024-dim) | 1000 |
| `rfp_chunk1200_a` | A (BGE-m3-ko) | BGE-m3-ko (1024-dim) | 1200 |

> A안과 B안은 임베딩 차원이 달라 컬렉션을 반드시 분리해야 합니다.

---

## 모델 목록

### AutoRAG 평가 모델

**메인 trial** (`local_csv.yaml`, `evaluation/autorag_benchmark_csv`)

| 모델 | 크기 | gpu_memory_utilization | 비고 |
|------|------|------------------------|------|
| EXAONE-4.0-1.2B | 2.4G | 0.70 | trust_remote_code |
| kanana-1.5-2.1b | 4.4G | 0.70 | |
| Midm-2.0-Mini | 4.4G | 0.70 | |
| EXAONE-Deep-2.4B | 4.5G | 0.70 | trust_remote_code |
| Gemma3-4B | 8.1G | **0.90** | max_model_len=8192, enforce_eager |

**Gemma4 확장 trial** (`local_csv_pipeline_gemma.yaml`, `evaluation/autorag_benchmark_csv_gemma`, 병합 후 포함)

| 모델 | 크기 | gpu_memory_utilization | 비고 |
|------|------|------------------------|------|
| Gemma3 (파인튜닝) | 8.1G | 0.70 | max_model_len=16384 |
| EXAONE-Deep-7.8B (파인튜닝) | 15G | 0.70 | trust_remote_code |
| Gemma4-E4B (파인튜닝) | 15G | 0.70 | QLoRA 적용 |

### 임베딩 모델 (AutoRAG 5종 비교)

| 이름 | 서버 경로 | 차원 |
|------|---------|------|
| BGE-m3-ko | `embeddings/BGE-m3-ko` | 1024 |
| ko-sroberta-multitask | `embeddings/ko-sroberta-multitask` | 768 |
| multilingual-e5-large | `embeddings/multilingual-e5-large` | 1024 |
| KoSimCSE-roberta-multitask | `embeddings/KoSimCSE-roberta-multitask` | 768 |
| kf-deberta-multitask | `embeddings/kf-deberta-multitask` | 768 |

### 앱 채팅 모델 (Scenario A 사이드바)

| 모델명 | 크기 |
|--------|------|
| EXAONE-4.0-1.2B | 2.4G |
| EXAONE-Deep-2.4B | 4.5G |
| EXAONE-Deep-7.8B | 15G |
| Gemma3-4B | 8.1G |
| Gemma4-E4B | 15G |
| kanana-1.5-2.1b | 4.4G |
| Midm-2.0-Mini | 4.4G |

모두 `/srv/shared_data/models/` 하위.

---

## AutoRAG 내장 패치 (`run_autorag_optimization.py`)

모든 config에 자동 적용됩니다.

| 패치 | 내용 |
|------|------|
| Patch 1 | ChromaDB `add_embedding` batch 초과 자동 분할 (5461개 단위) |
| Patch 2 | 모델 평가 후 VRAM 자동 해제 (`gc.collect` + `cuda.empty_cache`) |
| Patch 3 | ChromaDB `is_exist` SQLite 변수 초과 자동 분할 (500개 단위) |
| Patch 4 | VectorDB ingestion 후 임베딩 모델 즉시 GPU 해제 |
| Patch 5 | `summary.csv` module_name → 실제 모델명 치환 |
| Patch 6 | HybridCC 정규화 0-division 수정 (NaN → 정상값) |

---

## 트러블슈팅

### vLLM KV 캐시 부족 (`No available memory for cache blocks`)
- `gpu_memory_utilization`을 높이거나 `max_model_len`을 줄이세요.
- Gemma3-4B: `gpu_memory_utilization: 0.90`, `max_model_len: 8192`, `enforce_eager: true` (기설정)

### AutoRAG 파인튜닝 모델 로드 실패 (`configuration_exaone.py not found`)
- `finetune_local.py`가 저장 시 자동으로 베이스 모델의 설정 파일을 복사합니다.
- 수동 복구: `python scripts/repair_finetuned_models.py`

### ChromaDB batch size 초과 / SQLite 변수 초과
- `run_autorag_optimization.py` Patch 1·3으로 자동 처리됩니다.

### BM25 ko_okt 토크나이저 오류 (`JVM not found`)
- Java 설치 필요: `sudo apt-get install -y default-jdk`

### transformers 버전 충돌 (kanana/midm)
- `run_pipeline.py`는 내부적으로 환경을 자동 처리합니다.
- 직접 실행 시: `PYTHONNOUSERSITE=1 python scripts/run_autorag_optimization.py ...`

### B안 — top-k 검색 결과에 여러 RFP 문서 혼합
- 동일 출처(발주기관+사업명) 청크를 최대 2개로 제한하는 `max_chunks_per_source` 옵션이 기본 적용됩니다.
- 최신 재평가 기준에서는 4개 chunk 모두 품질 지표는 기준 이상이었지만 시간 gate는 미통과였습니다. 따라서 현재는 특정 chunk를 운영 기본값으로 확정하지 않고, `rfp_chunk800`, `rfp_chunk1000`, `rfp_chunk1200`을 비교 후보로 유지하는 것이 적절합니다.

### B안 — LLM judge 응답 없음 또는 JSON 파싱 실패
- gpt-5-nano는 내부 추론 예산이 필요해 `max_completion_tokens=4000`으로 자동 설정됩니다.
- 파싱 실패 시 fallback 점수(3점)로 대체되며 `reasoning` 필드에 원본 응답이 저장됩니다.

### 환경 진단

```bash
python scripts/check_env.py
```
