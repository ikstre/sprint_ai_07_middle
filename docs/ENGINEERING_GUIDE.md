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
