# Evaluation Guide

## 모드 설계
- `core` 모드
  - 운영 모니터링/회귀 검증용.
  - 기본값: `LLM Judge OFF`, `BERTScore OFF`, `Gate ON(auto)`.
  - 빠르게 반복 실행 가능한 지표 위주.
- `detailed` 모드
  - 모델/프롬프트/검색 전략 튜닝용.
  - 기본값: `LLM Judge ON`, `BERTScore ON`, `Gate OFF(auto)`.
  - 비용이 더 들지만 원인 분석 가능.

## 현재 범위

- `scripts/run_evaluation.py`는 질문지 기반 서비스 평가를 A/B안 모두에 대해 실행할 수 있습니다.
- `scripts/check_release_gate.py`는 기본적으로 `run_evaluation.py`가 생성한 core gate 산출물을 점검합니다.
- A안의 AutoRAG 자체 평가는 여전히 `scripts/run_autorag_optimization.py` 또는 `scripts/run_pipeline.py` 경로가 중심입니다.
- 따라서 질문지 기반 서비스 평가 결과와 AutoRAG 벤치마크 결과는 같은 의미로 섞어 해석하면 안 됩니다.
- 저장소에 커밋된 `evaluation/autorag_benchmark_csv`, `evaluation/autorag_benchmark_csv_gemma` 는 A안 AutoRAG 벤치마크 결과물입니다.
- `run_evaluation.py`는 이 디렉터리를 직접 읽지 않고, 지정한 Chroma 컬렉션(`--collection`)을 대상으로 평가합니다.

## 실행 커맨드
- 릴리즈 게이트 원커맨드 (권장)
```bash
python scripts/check_release_gate.py
```

- core
```bash
python scripts/run_evaluation.py --scenario B --mode core --collection rfp_chunk600 --output-dir evaluation
```

- detailed
```bash
python scripts/run_evaluation.py --scenario B --mode detailed --collection rfp_chunk600 --output-dir evaluation
```

- 테스트 샘플 축소
```bash
python scripts/run_evaluation.py --scenario B --mode detailed --collection rfp_chunk600 --test-limit 2 --output-dir evaluation
```

- A안 질문지 평가
```bash
python scripts/run_evaluation.py --scenario A --mode core --collection rfp_chunk600_a --output-dir evaluation/a_chunk600_core
```

- A안 상세 평가
```bash
python scripts/run_evaluation.py --scenario A --mode detailed --collection rfp_chunk600_a --output-dir evaluation/a_chunk600_detailed
```
  - `detailed` 또는 `--judge on`은 judge 단계에서 OpenAI API를 사용하므로 `OPENAI_API_KEY`가 필요합니다.

- 여러 chunk 크기 병렬 평가
```bash
python scripts/run_evaluation.py \
  --scenario B \
  --mode core \
  --chunk-sizes 600,800,1000,1200 \
  --output-dir evaluation
```
  - 각 크기는 별도 하위 디렉터리에서 실행됩니다.
  - 예: `evaluation/b_chunk600_full_core/run.log`
  - `--max-parallel N`으로 동시 실행 개수를 제한할 수 있습니다.

## Gate 리포트
- 기본 Gate 임계값은 코드 내 `DEFAULT_CORE_GATE_THRESHOLDS`를 사용합니다.
- 기본 평가 컬렉션은 코드상 `rfp_documents`/B안 설정에 가까우므로, 실행 전 실제 인덱스 상태를 확인해야 합니다.
- 파일 기반 임계값 오버라이드:
```bash
python scripts/run_evaluation.py \
  --scenario B \
  --mode core \
  --gate on \
  --collection rfp_chunk600 \
  --gate-thresholds configs/evaluation/core_gate.default.json \
  --output-dir evaluation
```

- 출력물
  - `gate_report_core.json`
  - `gate_report_core.md`

## 스크립트 옵션 (릴리즈 게이트)
- `--test-limit N`: 빠른 스모크 체크.
- `--no-run`: 기존 `gate_report_core.json`만 판정.
- `--allow-fail`: 로컬 진단용(실패여도 종료코드 0).

## AutoRAG 평가 데이터 품질 주의사항

AutoRAG는 `qa.parquet`의 `retrieval_gt` / `generation_gt`를 ground truth로 사용합니다.

| 항목 | 구 방식 (`data/autorag/`) | 현행 방식 (`data/autorag_csv/`) |
|------|--------------------------|--------------------------------|
| `retrieval_gt` | 토큰 매칭 (부정확, 오매핑 다수) | 공고번호 직접 연결 (100% 정확) |
| `generation_gt` | 키워드 목록 문자열 | 실제 사업 요약 텍스트 |
| QA 수 | 9개 | 285개 (문서당 3종) |
| METEOR/ROUGE 신뢰도 | 낮음 (gt가 키워드라서) | 정상 |

AutoRAG 평가 시 반드시 `data/autorag_csv/` + `configs/autorag/local_csv.yaml` 조합을 사용하세요.  
결과 경로: `evaluation/autorag_benchmark_csv/0/`

Gemma4 병합 결과는 별도 경로에 저장됩니다.
- `evaluation/autorag_benchmark_csv_gemma/`
- 필요 시 `scripts/merge_gemma4_results.py`로 메인 trial과 병합

## 지표 해석 요약
- Retrieval
  - `hit_at_5`: 정답 관련 문서가 상위 5개에 있는 비율.
  - `ndcg_at_5`: 상위 랭크 품질 반영 점수.
- Generation
  - `keyword_recall`: 기대 키워드 반영률.
  - `field_coverage`: 필수 필드 커버리지.
  - `rougeL_f1`, `meteor`, `bertscore_f1`: 참조답안 근접도.
- Grounding/신뢰성
  - `grounded_token_ratio`: 검색 근거와 연결된 토큰 비율.
  - `decline_accuracy`: out-of-scope 질의에서 적절히 거절하는 정확도.
- Runtime/비용
  - `p95_elapsed_time`: 사용자 체감 상한 지연.
  - `avg_total_tokens`: 토큰 비용 추정 기준.

## 권장 운영 정책
- 배포 전: `core` + Gate PASS를 필수 기준으로 사용.
- 튜닝 중: `detailed`에서 후보군 비교 후 `core`로 회귀 확인.
- 지표 상충 시 우선순위
  1. `decline_accuracy`, `grounded_token_ratio`
  2. `hit_at_5`, `ndcg_at_5`, `field_coverage`
  3. `p95_elapsed_time`, `avg_total_tokens`
