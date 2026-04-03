# Report Template

## 1. 문제 정의
- 대상: 다수 기관 RFP를 빠르게 분류/요약해야 하는 컨설턴트
- 목표: 정확도, 속도, 비용의 균형
- 제약: 문서 형식 다양성(PDF/HWP), 대화형 질의 필요, 동시 사용자 고려

## 2. 시스템 설계
- 시나리오 A: 로컬(HuggingFace + Chroma/FAISS)
- 시나리오 B: API(OpenAI + Chroma/FAISS)
- 공통: 문서 로딩 -> 청킹 -> 임베딩 -> 검색 -> 생성 -> 평가

## 3. 실험 설정
- 데이터: 제공 문서 + `data_list.csv`
- 질문셋: `src/evaluation/dataset.py`의 단일/다중/후속/out-of-scope
- 비교군: retrieval method, top-k, multi-query, rerank
- 모드:
  - 운영 관점 `core`
  - 튜닝 관점 `detailed`

## 4. 핵심 결과 표
- 운영 지표:
  - `p95_elapsed_time`, `avg_hit_at_5`, `avg_ndcg_at_5`
  - `avg_field_coverage`, `avg_grounded_token_ratio`, `decline_accuracy`
- 튜닝 지표:
  - `avg_bertscore_f1`, `avg_rougeL_f1`, `avg_meteor`, `avg_llm_*`

## 5. 의사결정
- 채택 구성:
  - Gate PASS 여부
  - 운영 기준(지연/정확성) 만족 여부
- 보류 구성:
  - 어떤 지표에서 임계값 미달인지
  - 개선 계획(검색/프롬프트/모델/메타필터)

## 6. 리스크 및 대응
- 데이터 파싱 노이즈(HWP): 전처리 개선
- 환각 위험: grounding 지표 기반 회귀 감시
- 비용 급등: `core` 모드 중심 모니터링과 토큰 상한 설정

## 7. 다음 단계
1. 실제 사용자 질문 로그 반영한 평가셋 확장
2. 메타데이터 fuzzy filter 정교화
3. API 서버 분리 및 멀티 워커 운영 전환
