# Notebook Guide

## 목적
- CLI 스크립트 실행 전후를 셀 단위로 확인하고 실험 로그를 남기기 위한 노트북 가이드입니다.

## 노트북 파일
- `notebooks/autorag_workflow.ipynb`

## 실행
```bash
jupyter notebook
```

노트북에서 권장 순서:
1. 환경 변수 확인 (`OPENAI_API_KEY`).
2. 문서 로딩/청킹 샘플 확인.
3. AutoRAG 데이터셋 생성 실행.
4. AutoRAG 최적화 커맨드 실행.
5. `core`/`detailed` 평가 결과 파일 확인.

## 주의사항
- 노트북 실행 시 커널과 CLI 파이썬 버전이 다르면 패키지 충돌이 발생할 수 있습니다.
- 대규모 실험은 노트북보다 CLI 스크립트 실행을 권장합니다 (재현성/자동화 용이).
