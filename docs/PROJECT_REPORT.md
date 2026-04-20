# 입찰메이트 기술 1팀 RAG시스템 구현 프로젝트 — 최종 보고서

## 1. 프로젝트 개요

공공·기업 RFP(Request for Proposal) 문서를 대상으로, 컨설턴트가 다수의 제안요청서를 빠르게 분류·요약·질의할 수 있도록 지원하는 RAG(Retrieval-Augmented Generation) 시스템을 구현하였습니다.

### 핵심 목표

| 관점 | 목표 |
|---|---|
| 정확도 | 상위 검색(hit@5) ≥ 0.80, 생성 grounding ≥ 0.55 |
| 속도 | p95 응답시간 ≤ 12초 |
| 비용 | core 평가 중심 운영, 토큰 상한 관리 |
| 운영성 | 대화형 챗봇 UI, 다국어/다형식 문서 지원 |

### 시스템 설계 이원화

| 구분 | A안 (로컬) | B안 (API) |
|---|---|---|
| 임베딩 | BGE-m3-ko (1024-dim) | text-embedding-3-small (512-dim) |
| 생성 | EXAONE / Gemma / kanana / Midm | gpt-5-mini / nano |
| 검색 | ChromaDB + AutoRAG 최적화 | ChromaDB + hybrid retrieval |
| 용도 | 비용 최적화, 온프레미스 대응, 파인튜닝 | 서비스형 운영, 빠른 프로토타이핑 |

---

## 2. 시스템 아키텍처

### 공통 파이프라인

```
문서 로드 → 청킹 → 임베딩 → 벡터DB 인덱싱 → 검색 → 프롬프트 구성 → LLM 생성 → 평가
```

### 주요 모듈

| 모듈 | 책임 |
|---|---|
| `src/document_loader.py` | PDF/HWP 로딩 + 메타데이터 CSV 병합 |
| `src/chunker.py` | naive / semantic 청킹, RFP 섹션 인식 |
| `src/embedder.py` | OpenAI / SentenceTransformer 임베딩 래퍼 |
| `src/retriever.py` | similarity / MMR / hybrid 검색, 메타필터, 다중 쿼리 |
| `src/generator.py` | 프롬프트 구성, LLM 호출, 대화 메모리, 필드 추출 |
| `src/rag_pipeline.py` | 검색/생성/출처 정리 오케스트레이션 |
| `src/evaluation/*` | retrieval/generation/grounding/runtime 지표 |

### 데이터 흐름

1. `data_list_cleaned.csv` 기반 CSV → corpus/qa parquet (`scripts/prepare_autorag_from_csv.py`)
2. ChromaDB 인덱싱 (`scripts/index_documents.py`) — 청크 크기별 별도 컬렉션
3. 질문지 기반 평가 (`scripts/run_evaluation.py`) — 4개 retrieval config 순회
4. AutoRAG 자동 최적화 (`scripts/run_autorag_optimization.py`) — 임베딩·생성 모델 조합 탐색

---

## 3. 실험 설정

### 데이터셋

- **문서 원천**: `/srv/shared_data/datasets/data_list_cleaned.csv` 기반 RFP 집합
- **청크 크기 실험**: 600 / 800 / 1000 / 1200 자
- **평가 질문지**: 총 200문항
  - `single_doc` 100문항 (단일 RFP 질의)
  - `follow_up` 100문항 (이전 질문 컨텍스트 연결)

### 평가 지표 및 게이트

| 지표 | 기준 | 의미 |
|---|---|---|
| `p95_elapsed_time` | ≤ 12.0s | 응답 지연 상한 |
| `avg_hit_at_5` | ≥ 0.80 | 정답 문서 상위 5개 포함률 |
| `avg_ndcg_at_5` | ≥ 0.65 | 순위 가중 검색 품질 |
| `avg_field_coverage` | ≥ 0.55 | 핵심 필드(발주기관/사업명/금액/일정 등) 포함률 |
| `avg_grounded_token_ratio` | ≥ 0.55 | 답변 토큰의 문서 근거 비율 |
| `decline_accuracy` | ≥ 0.90 | 답할 수 없는 질문 거절 정확도 |

### 비교 구성

| config | retrieval | top_k |
|---|---|---|
| similarity_k5 | dense similarity | 5 |
| mmr_k5 | MMR(λ=0.5) | 5 |
| hybrid_k5 | BM25 + dense | 5 |
| similarity_k10 | dense similarity | 10 |

---

## 4. 핵심 결과

### 4.1 B안 청크 크기별 성능 (similarity_k5 기준)

| chunk | p95(s) | hit@5 | nDCG@5 | field_cov | grounded | gate |
|---|---:|---:|---:|---:|---:|---:|
| 600 | 8.56 | 0.880 | 0.844 | 0.569 | 0.547 | FAIL |
| **800** | **8.85** | **0.880** | **0.840** | **0.570** | **0.561** | **PASS** (5/5) |
| 1000 | 9.04 | 0.860 | 0.824 | 0.562 | 0.568 | PASS (5/5) |
| 1200 | 8.46 | 0.865 | 0.832 | 0.556 | 0.560 | PASS (5/5) |

> `decline_accuracy`는 질문지에 거절 유형이 없어 모든 config에서 `missing`. gate 계산에서 제외됩니다.
> 최신 재평가 기준 산출물 경로: `evaluation/parallel_b_fieldcov/b_chunk*_full_core`

### 4.2 4개 retrieval config 비교 (chunk800 기준)

| config | p95(s) | hit@5 | nDCG@5 | field_cov | grounded | prompt_tokens |
|---|---:|---:|---:|---:|---:|---:|
| **similarity_k5** | **8.85** | 0.880 | 0.840 | **0.570** | 0.561 | 3,073 |
| mmr_k5 | 10.92 | 0.860 | 0.827 | 0.558 | 0.560 | 2,901 |
| hybrid_k5 | 10.75 | **0.885** | **0.846** | 0.562 | 0.556 | 3,155 |
| similarity_k10 | 10.60 | 0.880 | 0.813 | 0.567 | **0.578** | 4,425 |

- `similarity_k5`: 가장 빠르고 field coverage도 가장 높아 운영 기본값으로 유지하기 적합
- `hybrid_k5`: hit@5와 nDCG@5는 최고지만 latency가 더 길고 grounded는 약간 낮음
- `similarity_k10`: grounded는 가장 높지만 토큰 비용이 크게 증가해 운영 효율은 떨어짐
- `mmr_k5`: 최신 재평가에서는 gate는 통과했지만 속도와 품질 모두 우위가 뚜렷하지 않음

### 4.3 A안 AutoRAG 자동 최적화 결과

| 노드 | best module | 주요 파라미터 |
|---|---|---|
| lexical_retrieval | BM25 | tokenizer=ko_okt, top_k=1 |
| semantic_retrieval | VectorDB | local_bge, top_k=1 |
| hybrid_retrieval | HybridRRF | top_k=8, weight=4.0 |
| generator | vLLM | kanana-1.5-2.1b, temperature=0.2 |

### 4.4 카테고리별 성능 (chunk800 similarity_k5)

| 카테고리 | count | hit@5 | nDCG@5 | field_cov | grounded |
|---|---:|---:|---:|---:|---:|
| single_doc | 100 | 0.90 | 0.824 | 0.551 | 0.441 |
| follow_up | 100 | 0.86 | 0.856 | — | 0.782 |

- `follow_up` 질문에서 `grounded_token_ratio`가 크게 상승 — 팔로우업이 메인 질문 컨텍스트를 재활용하기 때문
- `single_doc`의 grounded가 낮은 이유: LLM이 문서 외 일반 지식을 혼입하는 경향

---

## 5. 핵심 의사결정

### 5.1 채택 구성

**B안 권장 설정**
- 청크 크기: **800자** (속도·품질·토큰 비용 균형이 가장 안정적)
- retrieval: `similarity_k5` (속도·품질 균형)
- 임베딩: `text-embedding-3-small` (512-dim 축소)
- 생성: `gpt-5-mini` (복잡 질문) + `gpt-5-nano` (단순 질문 라우팅)

최신 병렬 재평가 기준:
- 600: `hybrid_k5`, `similarity_k10` PASS
- 800: 4개 config 전부 PASS
- 1000: 4개 config 전부 PASS
- 1200: `mmr_k5`만 FAIL, 나머지 PASS

**A안 권장 설정**
- 청크 크기: 600자 (AutoRAG 최적화 결과)
- retrieval: `HybridRRF` (top_k=8, weight=4.0)
- 임베딩: `BGE-m3-ko` (1024-dim)
- 생성: `kanana-1.5-2.1b` (LoRA 파인튜닝 적용)

### 5.2 개선 이력 (PR #26~#36)

| PR | 개선 항목 | 효과 |
|---|---|---|
| #26 | B안 응답 속도 최적화 (컨텍스트 캐싱, 메타데이터 인덱싱) | 평균 응답시간 단축 |
| #32 | 문서 전반 현행화 | 스크립트 사용성 개선 |
| #33 | A안 AutoRAG 평가 결과, Gemma4 병합 | 최적 조합 확정 |
| #34 | `run_evaluation.py` A/B안 통합, 버그 수정 | 평가 파이프라인 일원화 |
| #35 | field_coverage 개선 (`[필드 후보]` 주입), 병렬 평가, A안 800 인덱싱 | 핵심 필드 추출 강화 |
| #36 | B안 chunk1000/1200 평가 결과 추가 | 청크 크기 민감도 분석 완료 |

### 5.3 보류·미충족 구성

**600 similarity_k5 / 600 mmr_k5 / 1200 mmr_k5**
- 최신 재평가에서도 일부 config는 gate 미달이 남아 있습니다.
- 600 `similarity_k5`: `avg_grounded_token_ratio` 0.547로 임계값 0.55 소폭 미달
- 600 `mmr_k5`: `avg_grounded_token_ratio` 0.546으로 미달
- 1200 `mmr_k5`: `avg_field_coverage` 0.547로 미달
- 대응 계획: 운영 기본값은 계속 `similarity_k5@800`으로 두고, 보조 후보는 `hybrid_k5@600`, `similarity_k5@1000`, `similarity_k5@1200` 위주로 비교 유지

**`decline_accuracy` — missing**
- 원인: 현재 질문지(single_doc / follow_up)에 거절 유형 문항이 없음
- 대응 계획: `out_of_scope` 유형 질문셋 추가 후 재측정

---

## 6. 리스크 및 대응

| 리스크 | 현황 | 대응 |
|---|---|---|
| HWP 파싱 노이즈 (서로게이트 문자) | `_sanitize()`로 선제 방어 중 | 새 포맷 추가 시 `document_loader.py` 확장 |
| 환각 위험 | `grounded_token_ratio` 모니터링 | core 게이트로 회귀 감시 |
| API 비용 급등 | gpt-5-nano 자동 라우팅 (쿼리 ≤80자) | `core` 모드 중심 운영, max_tokens 3500 상한 |
| VRAM 부족 (A안) | `gpu_memory_utilization` 조정 + Patch 2·4 자동 해제 | Gemma3-4B는 0.90 고정 |
| ChromaDB batch/SQLite 한계 | Patch 1·3 자동 적용 (5461 / 500 단위 분할) | AutoRAG 실행 시 자동 처리 |
| 병렬 평가 시 OOM (A안) | `--max-parallel` 기본값 순차 실행 권장 | HF 모델 동시 로드 주의 |

---

## 7. 다음 단계

1. **field_coverage/grounded 미세 조정** — 600 similarity/mmr, 1200 mmr에서 임계값 근처 미달 원인 재분석
2. **거절 질문지 추가** — `src/evaluation/single_dataset.py`에 `out_of_scope` 카테고리 확장 → `decline_accuracy` 실측
3. **LLM judge 상세 평가** — `--mode detailed --judge on` 으로 relevance/accuracy/faithfulness/completeness/conciseness 5축 측정
4. **실사용 질문 로그 확보** — 사내 파일럿 운영 후 질문 로그 반영한 회귀 평가셋 확장
5. **API 서버 분리** — Streamlit 단일 프로세스 → FastAPI + 멀티 워커 전환 (`apps/autorag_api.py` 기반 확장)
6. **메타데이터 fuzzy filter 정교화** — 유사 기관명 매칭 정확도 추가 개선

---

## 8. 부록

### 8.1 컬렉션 목록

| 컬렉션명 | 시나리오 | 임베딩 | chunk_size |
|---|---|---|---|
| `rfp_chunk600` / `rfp_chunk800` / `rfp_chunk1000` / `rfp_chunk1200` | B | text-embedding-3-small (512-dim) | 600~1200 |
| `rfp_chunk600_a` / `rfp_chunk800_a` | A | BGE-m3-ko (1024-dim) | 600 / 800 |

### 8.2 AutoRAG 내장 패치

| 패치 | 해결 |
|---|---|
| 1 | ChromaDB `add_embedding` batch(5461) 초과 자동 분할 |
| 2 | 모델 평가 후 VRAM 자동 해제 |
| 3 | ChromaDB `is_exist` SQLite 변수(500) 초과 자동 분할 |
| 4 | VectorDB ingestion 후 임베딩 모델 GPU 즉시 해제 |
| 5 | `summary.csv` module_name → 실제 모델명 치환 |
| 6 | HybridCC 정규화 0-division 수정 |

### 8.3 관련 문서

- [`README.md`](../README.md) — 설치·실행 가이드
- [`docs/ENGINEERING_GUIDE.md`](ENGINEERING_GUIDE.md) — 엔지니어링 규약
- [`docs/CODEBASE_MAP.md`](CODEBASE_MAP.md) — 모듈별 책임 경계
- [`docs/EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) — 평가 실행·해석
- [`docs/AUTORAG_GUIDE.md`](AUTORAG_GUIDE.md) — AutoRAG 최적화 가이드
- [`docs/DATA_CLEANER_GUIDE.md`](DATA_CLEANER_GUIDE.md) — 데이터 정제
- [`docs/OPS_SECURITY_MULTIUSER.md`](OPS_SECURITY_MULTIUSER.md) — 운영·보안
