# Engineering Guide

## Purpose
This project provides:
1. A production-facing RAG service for RFP analysis.
2. An AutoRAG-based optimization workflow.
3. A two-layer evaluation strategy (`core` for operations, `detailed` for tuning).

## Recommended Reading Order
1. `docs/CODEBASE_MAP.md`
2. `docs/EVALUATION_GUIDE.md`
3. `docs/OPS_SECURITY_MULTIUSER.md`
4. `docs/AUTORAG_GUIDE.md`

## Directory Map
- `app.py`: Main Streamlit chat app (service UI).
- `src/`: Core RAG modules.
- `src/evaluation/`: Modular evaluation package.
- `scripts/`: CLI entrypoints for indexing, evaluation, AutoRAG workflows.
- `apps/`: Optional API/UI wrappers for AutoRAG runtime.
- `configs/autorag/`: AutoRAG search config.
- `notebooks/`: Reproducible notebook workflows.

## Operating Principle
- Production decision: optimize for stable `core` metrics and gate pass.
- Tuning decision: use `detailed` mode for metric decomposition and model/prompt iteration.

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
4. Apply retrieval/generation changes.
5. Re-run `core` for regression safety.

## Scenario A / B 선택 가이드

| 항목 | Scenario B (OpenAI) | Scenario A (HuggingFace) |
|------|--------------------|-----------------------|
| 실행 환경 | 로컬/GCP 모두 가능 | GCP GPU 권장 |
| 모델 | gpt-5-mini / nano / gpt-5 | google/gemma-3-4b-it |
| 임베딩 | text-embedding-3-small (dim=512) | intfloat/multilingual-e5-large |
| 비용 | 토큰 기반 과금 | 인프라 비용만 |
| 파인튜닝 | 불가 (API 전용) | 가능 (LoRA/QLoRA) |
| 필수 환경변수 | `OPENAI_API_KEY` | `HF_TOKEN` |

앱 실행 시 사이드바 **실행 모드** 라디오 버튼으로 실시간 전환 가능.

## 인덱싱 2단계 구조

```
[1단계: chunk] 문서 로딩 → 청킹 → data/processed/{collection}_chunks.json 저장
[2단계: embed] 청크 파일 로딩 → 임베딩 → ChromaDB 저장
```

- 임베딩 중 오류 시 1단계 재실행 없이 `--step embed`만 재실행
- 청크 크기별 컬렉션을 여러 개 운영해 앱에서 실시간 비교 가능

## 컬렉션 관리

벡터스토어는 컬렉션 단위로 독립 관리됩니다.

```bash
# 컬렉션 생성
python scripts/index_documents.py --chunk-size 1200 --collection rfp_chunk1200
python scripts/index_documents.py --chunk-size 800  --collection rfp_chunk800

# 앱에서 선택 → 사이드바 "컬렉션 (청크 크기)" selectbox
```

## gpt-5 계열 파라미터 제한사항

| 파라미터 | gpt-4 계열 | gpt-5 계열 |
|---------|-----------|-----------|
| `temperature` | 지원 | **미지원** (Scenario A 전용) |
| `top_p` | 지원 | **미지원** |
| `max_tokens` | 지원 | **미지원** |
| `max_completion_tokens` | 미지원 | **필수** |
| `reasoning_effort` | 미지원 | low/medium/high |

## Complexity Note
- Current logic is more modular than the initial single-file version.
- Remaining complexity hotspots:
  - retrieval strategy combination (`multi-query + rerank + hybrid`)
  - metadata filtering quality for near-match institution names
  - scenario A/B branching and local environment differences
- Mitigation:
  - keep `core` regression fixed and small,
  - add feature flags by config (not hard-coded branches),
  - document each extension in `docs/CODEBASE_MAP.md` first.
