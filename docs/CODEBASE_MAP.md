# Codebase Map

## 목적
- 모듈별 책임 경계를 명확히 하여 변경 영향 범위를 빠르게 파악합니다.
- 신규 엔지니어가 디버깅/개선 포인트를 빠르게 찾을 수 있게 합니다.

## 엔트리포인트
- `app.py`: 메인 Streamlit 앱. 세션 단위 파이프라인/대화 메모리 유지.
  - 사이드바 **실행 모드** 라디오로 A/B 전환. 시나리오에 따라 컬렉션 목록 자동 전환.
  - B안: `rfp_chunk600`, `rfp_chunk1200`
  - A안: `rfp_chunk1200_a`, `rfp_chunk800_a`, `rfp_chunk1200_a_sroberta`
- `scripts/index_documents.py`: 문서 로딩, 청킹, 임베딩, 벡터스토어 적재.
- `scripts/run_evaluation.py`: `core`/`detailed` 평가 실행 및 리포트 저장.
- `scripts/check_release_gate.py`: `core` 평가 + 게이트 판정을 단일 커맨드로 실행.
- `scripts/prepare_autorag_data.py`: AutoRAG용 `qa.parquet`, `corpus.parquet` 생성.
- `scripts/run_autorag_optimization.py`: AutoRAG 탐색/최적화 실행. ChromaDB 패치 + VRAM 해제 패치 내장.
- `scripts/download_models.py`: 생성 모델(Gemma4-E4B 포함) + 임베딩 모델 다운로드.
- `scripts/finetune_local.py`: 로컬 모델 LoRA/QLoRA 파인튜닝 (peft + trl SFTTrainer).
- `scripts/finetune_openai.py`: OpenAI Fine-tuning API 래퍼 (start / status / list 서브커맨드).

## 핵심 모듈

### `configs/paths.py`
중앙 경로 레지스트리. `.env`의 환경변수를 읽어 **모든 경로를 한 곳에서 관리**합니다.

| 환경변수 | 기본값 | 역할 |
|---------|--------|------|
| `SRV_DATA_DIR` | `/srv/shared_data` | 서버 루트 |
| `METADATA_CSV` | `{SRV_DATA_DIR}/datasets/data_list_cleaned.csv` | 메타데이터 CSV |
| `MODEL_DIR` | `{SRV_DATA_DIR}/models` | 모델 루트 |
| `EMBEDDING_MODEL_PATH` | `{MODEL_DIR}/embeddings/BGE-m3-ko` | 임베딩 모델 |
| `CHAT_MODEL_PATH` | `{MODEL_DIR}/exaone/EXAONE-4.0-1.2B` | 채팅 모델 |
| `VECTORDB_DIR` | `data/vectordb` | 로컬 ChromaDB |
| `AUTORAG_DATA_DIR` | `data/autorag_csv` | AutoRAG parquet |
| `AUTORAG_PROJECT_DIR` | `evaluation/autorag_benchmark_csv` | AutoRAG 결과 |

> 서버 경로 변경 시 `.env`에 `SRV_DATA_DIR=/new/path` 한 줄만 추가하면 모든 스크립트에 즉시 반영됩니다.
> `configs/config.py`, `app.py`, `run_pipeline.py`, `run_evaluation.py`, `index_documents.py`, `prepare_autorag_from_csv.py` 모두 이 모듈을 참조합니다.

---

### `configs/config.py`
전체 런타임 설정. 주요 필드:

- `scenario`: A (HuggingFace 로컬) / B (OpenAI API)
- **Scenario B 전용**
  - `openai_embedding_model`: text-embedding-3-small
  - `openai_embedding_dim`: 512 (원본 1536 대비 67% 절감)
  - `openai_chat_model`: gpt-5-mini (기본값)
  - `reasoning_effort`: low / medium / high (gpt-5 계열 추론 깊이)
  - `auto_model_routing`: 쿼리 복잡도 기반 모델 자동 선택 (bool)
  - `routing_simple_model`: 단순 질문용 경량 모델 (gpt-5-nano)
  - `routing_complexity_threshold`: 단순/복잡 분기 글자 수 기준 (80)
  - `max_tokens`: 3500
  - `retrieval_method`: `similarity` (앱 기본값, 운영 권장 조합은 `rfp_chunk800 + similarity_k5`)
  - `conversation_memory_k`: 3
  - `max_context_chars_per_doc`: 800
- **Scenario A 전용**
  - `hf_embedding_model`: `paths.EMBEDDING_MODEL_PATH` 기본값 (`.env`의 `EMBEDDING_MODEL_PATH` → `{MODEL_DIR}/embeddings/BGE-m3-ko`)
  - `hf_embedding_dim`: 1024 (BGE-m3-ko) / 768 (ko-sroberta)
  - `hf_chat_model`: `paths.CHAT_MODEL_PATH` 기본값 (`.env`의 `CHAT_MODEL_PATH` → `{MODEL_DIR}/exaone/EXAONE-4.0-1.2B`)
  - `hf_token`: 로컬 모델은 빈 문자열로도 동작 (Hub 비공개 모델만 필요)
  - `device`: "auto" — cuda → mps → cpu 자동 감지
  - `hf_max_new_tokens`: 1024 (HF 생성 전용, OpenAI max_tokens와 분리)
  - `hf_load_in_4bit`: False — 4-bit 양자화 활성화 시 bitsandbytes 필요

---

### `src/document_loader.py`
PDF / HWP 파일 로딩 + `data_list.csv` 메타데이터 병합.

현재 구현 기준 일반 파일 로딩 지원 포맷:

| 포맷 | 처리 방식 |
|------|----------|
| `.hwp` | OLE 바이너리 직접 파싱 (olefile), 서로게이트 문자 자동 제거 |
| `.pdf` | pdfplumber 페이지별 텍스트 추출 |

**CSV 처리 모드**:
- `csv_row_per_doc=False` (기본값): 일반 파일 로딩 경로 사용
- `csv_row_per_doc=True`: 메타데이터 CSV 각 행 → 개별 문서 (행 하나가 RFP 한 건인 경우)
- `csv_text_columns`: 본문으로 쓸 컬럼 직접 지정. `None`이면 숫자 전용 컬럼 제외 후 자동 감지

`data_list.csv`(메타데이터 파일)는 인덱싱 대상에서 자동 제외됩니다.

`load_single()` 반환 타입: `dict` — 단일 PDF/HWP 문서 로딩 결과.

---

### `src/chunker.py`
`naive` / `semantic` 청킹 로직.

- `naive`: `RecursiveCharacterTextSplitter` 기반 고정 크기 분할
  - 한국어 문장 경계 구분자 추가: `다. `, `습니다. `, `됩니다. `, `합니다. `, `입니다. `
- `semantic`: RFP 섹션 헤더 인식 → 의미 단위 분할 → 큰 섹션은 `naive`로 재분할
  - `_SECTION_PATTERNS`: 장/절/항 + 조항(`제n조`) + 별표·부록 + 표/그림 캡션 + 인터페이스·보안·유지보수 키워드
  - `_MIN_CHUNK_SIZE = 80`: 최소 크기 미달 청크는 직전 청크에 자동 병합
  - 섹션 간 overlap: `chunk_overlap` 크기의 이전 섹션 꼬리 부분을 다음 섹션 앞에 삽입

---

### `src/embedder.py`
임베딩 생성 (`OpenAI` 또는 `SentenceTransformer`) + ChromaDB/FAISS VectorStore 래퍼.

- **Scenario B**: OpenAI `text-embedding-3-small`, 차원 축소(`dimensions=512`), Batch API 지원 (`use_batch_api=True`)
- **Scenario A**: `SentenceTransformer`로 로컬 경로 직접 로드. HF 토큰 없어도 동작 (토큰이 있을 때만 Hub 로그인)
  - 서버: `/srv/shared_data/models/embeddings/BGE-m3-ko` (dim=1024), `ko-sroberta-multitask` (dim=768)
  - 로컬 PC: HuggingFace Hub에서 자동 다운로드 (`BAAI/bge-m3`, `jhgan/ko-sroberta-multitask`)
- `_sanitize()`: 서로게이트 문자 제거 (HWP 파싱 결과의 잘못된 유니코드 방어)
- `_resolve_device()`: cuda → mps → cpu 자동 감지

---

### `src/retriever.py`
similarity / MMR / hybrid 검색, multi-query, cross-encoder reranker.

- gpt-5 계열 호환: multi-query 생성 시 `max_completion_tokens` 사용, `temperature` 조건부 적용
- `build_metadata_filter()`: 쿼리에서 기관명을 추출해 ChromaDB `where` 필터 구성 (유사 기관명 대응)

---

### `src/generator.py`
RAG 프롬프트 생성, LLM 호출, 슬라이딩 윈도우 대화 메모리.

- **Scenario B** (`_call_openai()`):
  - `_route_model()`: 쿼리 길이 기반 모델 자동 선택 (gpt-5-nano ↔ 선택 모델)
  - `reasoning_effort`, `max_completion_tokens` 적용 (gpt-5 계열 `max_tokens` 미지원)
- **Scenario A** (`_init_hf_model()`, `_call_hf()`):
  - `AutoModelForCausalLM` + `AutoTokenizer`로 로컬 경로 직접 로드
  - `apply_chat_template()`으로 채팅 형식 적용, 입력 토큰 제외 디코딩
  - EXAONE / Gemma3 / Gemma4 / kanana / Midm 등 모두 동일한 인터페이스로 호출
  - `hf_load_in_4bit=True` 시 4-bit 양자화 (bitsandbytes + CUDA 필요)

---

### `src/evaluation/evaluator.py`
평가 실행 오케스트레이터.

- `evaluate_with_llm_judge()`: OpenAI API 호출 시 `max_completion_tokens` 사용 (`max_tokens` → 오류)
- LLM judge 모델: `gpt-5-mini` (gpt-5 계열은 `max_tokens` 미지원)

---

### `src/rag_pipeline.py`
검색/생성/출처 정리를 묶은 오케스트레이션 레이어.

---

### `scripts/run_autorag_optimization.py`
AutoRAG 최적화 실행 + 세 가지 내장 패치:

| 패치 | 대상 | 해결 문제 |
|------|------|----------|
| `_patch_chroma_add_embedding()` | `Chroma.add_embedding` | ChromaDB max_batch_size(5461) 초과 오류 |
| `_patch_chroma_is_exist()` | `Chroma.is_exist` | SQLite 변수 제한(999) 초과 오류 |
| `_patch_run_evaluator()` | `BaseModule.run_evaluator` | 모델별 VRAM 순차 해제 |

> 패치는 `run_autorag_optimization.py`를 통해 실행되는 모든 config에 자동 적용됩니다.

---

### `configs/autorag/`

| 파일 | 시나리오 | 대상 환경 | 임베딩 | LLM |
|------|---------|---------|--------|-----|
| `tutorial.yaml` | B (OpenAI) | 서버/PC | text-embedding-3-small | gpt-5-mini |
| `local.yaml` | A (로컬, 서버) | GCP GPU 22GB (L4) | 5종 (BGE/sroberta/E5/SimCSE/DeBERTa) | 5종 vLLM |
| `local_pc.yaml` | A-PC (로컬, PC) | RTX 4070/3060Ti 8GB | BAAI/bge-m3 + ko-sroberta (HF Hub) | 4종 vLLM |

`local.yaml` 특이 설정:
- `gpu_memory_utilization: 0.70` (22GB GPU)
- `kv_cache_dtype: auto` → 모델 dtype(bfloat16) 자동 사용
- `max_model_len: 16384` — Gemma3-4B에 적용 (131072 기본값 대비 축소, KV 캐시 초과 방지)

`local_pc.yaml` 특이 설정:
- `gpu_memory_utilization: 0.80` (8GB × 0.80 = 6.4GB)
- `kv_cache_dtype: auto` (GPU 아키텍처 자동 감지)
- Gemma3-4B 제외 (8.1G > 한도)

### 서버 저장 모델 현황 (`$MODEL_DIR` — 기본: `/srv/shared_data/models/`)

> `.env`의 `SRV_DATA_DIR` 또는 `MODEL_DIR` 변수로 루트 경로를 변경할 수 있습니다.

| 디렉토리 | 모델명 | 크기 | 22GB 로드 |
|---------|--------|------|----------|
| `exaone/` | EXAONE-4.0-1.2B | 2.4G | ✅ |
| `exaone/` | EXAONE-Deep-2.4B | 4.5G | ✅ |
| `exaone/` | EXAONE-Deep-7.8B | 15G | ✅ |
| `gemma/` | Gemma3-4B | 8.1G | ✅ |
| `gemma/` | Gemma4-E4B | 15G | ✅ |
| `kanana/` | kanana-1.5-2.1b | 4.4G | ✅ |
| `midm/` | Midm-2.0-Mini | 4.4G | ✅ |
| `embeddings/` | BGE-m3-ko | 2.2G | — |
| `embeddings/` | ko-sroberta-multitask | 0.8G | — |
| `embeddings/` | multilingual-e5-large | 2.2G | — (AutoRAG용) |
| `embeddings/` | KoSimCSE-roberta-multitask | 0.4G | — (AutoRAG용) |
| `embeddings/` | kf-deberta-multitask | 0.7G | — (AutoRAG용) |

---

### `scripts/finetune_local.py`
로컬 모델 LoRA/QLoRA 파인튜닝.

- 학습 데이터 자동 생성: `qa.parquet` + `corpus.parquet` → chat 형식 JSONL
- `--qlora`: 4-bit NF4 양자화 (bitsandbytes). 8GB GPU에서도 학습 가능.
- `target_modules`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- 완료 후 `models/finetuned/{name}/final/` 저장 → vLLM으로 바로 서빙 가능
- 저장 방식: GPU 해제 후 **스트리밍 mmap 머지** (`_stream_merge_and_save`) — LoRA 어댑터를 텐서 단위로 적용, 512MB 샤드 분할 저장 (RAM 최소화)
- EarlyStopping 내장: `eval_loss` 기준 patience=3 (기본값), `--early-stop-patience` 옵션으로 조정

### `scripts/finetune_openai.py`
OpenAI Fine-tuning API 래퍼.

- `start`: 데이터 생성 → 파일 업로드 → 파인튜닝 작업 생성 → `job_info.json` 저장
- `status`: 작업 ID로 진행 상태 + 최근 이벤트 조회
- `list`: 파인튜닝 작업 목록 (limit 설정 가능)
- 지원 모델: `gpt-4o-mini-2024-07-18`, `gpt-4.1-mini` 등

---

## 스크립트 CLI 옵션 요약

### `scripts/index_documents.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--scenario` | `B` | A: 로컬 HF / B: OpenAI API |
| `--step` | `all` | all / chunk / embed |
| `--method` | `naive` | naive / semantic |
| `--chunk-size` | `1200` | 청크 크기 |
| `--chunk-overlap` | `200` | 청크 중첩 크기 |
| `--collection` | `rfp_chunk1200` | ChromaDB 컬렉션 이름 |
| `--documents-dir` | `data` | 문서 디렉토리 경로 |
| `--hf-embedding-model` | `bge` | Scenario A 임베딩 모델 (`bge` / `sroberta`) |
| `--use-batch-api` | `False` | OpenAI Batch API 사용 (Scenario B 전용) |
| `--csv-text-columns` | `None` | CSV 본문 컬럼 지정 (쉼표 구분) |
| `--csv-row-per-doc` | `False` | CSV 각 행을 개별 문서로 처리 |
| `--from-parquet` | `None` | corpus.parquet 직접 임베딩 (청킹 생략, rfp_chunk600 생성에 사용) |

### `scripts/prepare_autorag_data.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--documents-dir` | `data` | 문서 디렉토리 경로 |
| `--metadata-csv` | `data/data_list.csv` | 메타데이터 CSV |
| `--output-dir` | `data/autorag` | 출력 디렉토리 |
| `--chunk-method` | `semantic` | naive / semantic |
| `--chunk-size` | `600` | 청크 크기 (권장) |
| `--chunk-overlap` | `150` | 청크 중첩 크기 (권장) |
| `--csv-text-columns` | `None` | CSV 본문 컬럼 지정 |
| `--csv-row-per-doc` | `False` | CSV 각 행을 개별 문서로 처리 |

### `scripts/run_autorag_optimization.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--qa-path` | `data/autorag/qa.parquet` | QA 데이터 경로 |
| `--corpus-path` | `data/autorag/corpus.parquet` | 코퍼스 경로 |
| `--config-path` | `configs/autorag/tutorial.yaml` | AutoRAG 설정 파일 |
| `--project-dir` | `evaluation/autorag_benchmark` | 결과 저장 디렉토리 |
| `--run-dashboard` | `False` | 완료 후 대시보드 자동 실행 |

### `scripts/finetune_local.py`

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model-path` | (필수) | 모델 경로 또는 HF Hub ID |
| `--output-dir` | (필수) | 결과 저장 디렉토리 |
| `--qlora` | False | 4-bit QLoRA 활성화 (8GB GPU용) |
| `--trust-remote-code` | False | EXAONE 등 커스텀 코드 모델 |
| `--epochs` | 5 | 학습 epoch |
| `--early-stop-patience` | 3 | EarlyStopping patience (eval_loss 기준) |
| `--batch-size` | 2 | 배치 크기 |
| `--grad-accum` | 8 | Gradient accumulation (실효 배치 = 2×8=16) |
| `--lora-r` | 16 | LoRA rank |
| `--max-seq-length` | 2048 | 최대 시퀀스 길이 |

---

## 평가 모듈
- `src/evaluation/dataset.py`: 평가 질문셋 (단일/다중/후속/out-of-scope)
- `src/evaluation/retrieval_metrics.py`: hit@k, mrr, ndcg 등
- `src/evaluation/generation_metrics.py`: keyword recall, field coverage, rouge, meteor, bertscore
- `src/evaluation/grounding_metrics.py`: grounded token ratio, decline accuracy
- `src/evaluation/runtime_metrics.py`: latency, token usage 집계
- `src/evaluation/evaluator.py`: 평가 실행 오케스트레이터 (`max_completion_tokens` 적용)
- `src/evaluator.py`: 하위호환 re-export

## API/앱 확장
- `apps/autorag_api.py`: AutoRAG 결과를 감싼 FastAPI 서버
- `apps/autorag_streamlit.py`: AutoRAG 결과를 감싼 Streamlit 앱
- `src/autorag_runner.py`: trial 폴더 로딩 및 질의 래퍼

## 변경 가이드
- 검색 로직 변경: `src/retriever.py` + `scripts/run_evaluation.py`로 회귀 확인
- 생성 프롬프트 변경: `src/generator.py` + `detailed` 평가로 품질 확인
- 메타데이터 필터 변경: `src/retriever.py`, `src/rag_pipeline.py` 동시 점검
- 평가 지표 추가: `src/evaluation/*` 모듈 추가 후 `summary_report`와 CLI 출력 반영
- 새 문서 포맷 추가: `src/document_loader.py`의 `load_single()` 및 `load_all()`의 확장자 분기 확장
- AutoRAG 새 모델 추가: `configs/autorag/local.yaml` 또는 `local_pc.yaml`의 `generator.modules`에 vllm 블록 추가
- AutoRAG Gemma4 모델 변경: `configs/autorag/local.yaml`의 Gemma4 관련 설정과 `run_pipeline.py`의 Gemma4 그룹 로직을 함께 점검
- AutoRAG 새 임베딩 추가: `configs/autorag/local.yaml`의 `vectordb` + `semantic_retrieval.modules`에 추가, `scripts/download_models.py`에 다운로드 항목 추가
- AutoRAG ChromaDB 관련 오류: `scripts/run_autorag_optimization.py`의 패치 함수 확인
- 서버 경로 변경: `.env`의 `SRV_DATA_DIR` 수정 → `configs/paths.py`가 모든 하위 경로 자동 파생
- 모델 경로만 변경: `.env`의 `MODEL_DIR` 또는 개별 `CHAT_MODEL_PATH`/`EMBEDDING_MODEL_PATH` 수정
- AutoRAG YAML 모델 경로 변경: `${SRV_DATA_DIR}` 플레이스홀더가 `run_pipeline.py`의 `_resolve_yaml_env()`에서 자동 치환됨
