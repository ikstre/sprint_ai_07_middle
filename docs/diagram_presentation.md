# Presentation Diagram Notes

발표 자료용으로 바로 옮기기 쉬운 레이아웃 초안입니다.

## 슬라이드 1. 프로젝트 전체 구조

제목:
- `프로젝트 전체 구조`

좌측 박스:
- `A안`
- `목표: 로컬 모델 기반 AutoRAG 최적화`
- `엔트리: run_pipeline.py`

우측 박스:
- `B안`
- `목표: OpenAI 기반 서비스형 RAG`
- `엔트리: index_documents.py -> app.py`

중앙 공통 박스:
- `DocumentLoader`
- `Chunker`
- `VectorStore`
- `Retriever`
- `Generator`
- `RAGPipeline`

하단 산출물:
- `A안: corpus/qa, finetuned model, autorag result`
- `B안: vectordb, app response, evaluation report`

## 슬라이드 2. B안 서비스형 RAG

제목:
- `B안 서비스형 RAG 흐름`

흐름:
1. `원본 문서 + metadata`
2. `index_documents.py`
3. `DocumentLoader`
4. `Chunking`
5. `Embedding`
6. `Vector DB 저장`
7. `app.py`
8. `RAGPipeline`
9. `Retriever`
10. `Generator`
11. `응답 출력`

보조 흐름:
- `run_evaluation.py`
- `check_release_gate.py`

발표 포인트:
- `B안은 서비스 운영과 사용자 응답이 중심`
- `평가 체계는 B안 기준으로 가장 명확하게 분리되어 있음`

## 슬라이드 3. A안 AutoRAG/파인튜닝

제목:
- `A안 AutoRAG/파인튜닝 흐름`

메인 흐름:
1. `CSV 원천 데이터`
2. `run_pipeline.py`
3. `prepare_autorag_from_csv.py`
4. `corpus.parquet / qa.parquet`
5. `finetune_local.py`
6. `models/finetuned/*`
7. `run_autorag_optimization.py`
8. `retrieval / prompt / generator 탐색`
9. `최적 조건 선정`

발표 포인트:
- `A안은 서비스 파이프라인보다 실험/최적화 파이프라인에 가깝다`
- `최종 실행 엔트리는 run_pipeline.py 하나로 보는 것이 맞다`

## 슬라이드 4. A안/B안 차이

제목:
- `A안과 B안의 역할 차이`

비교표:
- `A안`
- `로컬 모델`
- `AutoRAG 중심`
- `파인튜닝 포함`
- `최적 조건 탐색`

- `B안`
- `OpenAI API`
- `서비스형 RAG 중심`
- `즉시 응답`
- `운영 평가/게이트`

## 슬라이드 5. 주의사항

제목:
- `현재 구현 기준 주의사항`

항목:
- `A안의 최종 엔트리는 run_pipeline.py`
- `B안 평가는 run_evaluation.py / check_release_gate.py 중심`
- `DocumentLoader 일반 파일 로딩은 PDF/HWP 중심`
- `A안 AutoRAG 결과와 B안 gate 결과는 같은 평가 축으로 바로 비교하면 안 됨`
