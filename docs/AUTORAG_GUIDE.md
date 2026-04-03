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
  --chunk-size 800 \
  --chunk-overlap 200
```

- 산출물
  - `data/autorag/corpus.parquet`
  - `data/autorag/qa.parquet`

## 2) 최적화 실행
```bash
python scripts/run_autorag_optimization.py \
  --qa-path data/autorag/qa.parquet \
  --corpus-path data/autorag/corpus.parquet \
  --config-path configs/autorag/tutorial.yaml \
  --project-dir evaluation/autorag_benchmark
```

## 3) 결과 사용
- AutoRAG 기본 API
```bash
autorag run_api --trial_dir evaluation/autorag_benchmark/0 --host 0.0.0.0 --port 8000
```

- AutoRAG 기본 Web
```bash
autorag run_web --trial_path evaluation/autorag_benchmark/0
```

- 래퍼 스크립트 사용
```bash
python scripts/run_autorag_api.py --trial-dir evaluation/autorag_benchmark/0
python scripts/run_autorag_web.py --trial-path evaluation/autorag_benchmark/0
```

## 4) 앱 통합
- FastAPI 래퍼: `apps/autorag_api.py`
- Streamlit 래퍼: `apps/autorag_streamlit.py`
- 공통 런타임: `src/autorag_runner.py`

## 5) 운영 팁
- 첫 번째 trial(`.../0`)만 고정 사용하지 말고, 지표 기반으로 승자를 선택합니다.
- AutoRAG 결과도 `core` 지표와 함께 재검증해 운영 회귀를 막습니다.
