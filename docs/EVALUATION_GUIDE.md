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

## 실행 커맨드
- 릴리즈 게이트 원커맨드 (권장)
```bash
python scripts/check_release_gate.py
```

- core
```bash
python scripts/run_evaluation.py --mode core --output-dir evaluation
```

- detailed
```bash
python scripts/run_evaluation.py --mode detailed --output-dir evaluation
```

- 테스트 샘플 축소
```bash
python scripts/run_evaluation.py --mode detailed --test-limit 2 --output-dir evaluation
```

## Gate 리포트
- 기본 Gate 임계값은 코드 내 `DEFAULT_CORE_GATE_THRESHOLDS`를 사용합니다.
- 파일 기반 임계값 오버라이드:
```bash
python scripts/run_evaluation.py \
  --mode core \
  --gate on \
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
