# Project Diagrams

현재 코드 기준으로 정리한 구조 도식입니다.

렌더링 산출물:
- `docs/diagram_architecture.png`
- `docs/diagram_paths.png`
- `docs/diagram.jpg` (동기화 파일)
- `docs/diagram1.jpg` (동기화 파일)

생성 스크립트:
```bash
python3 scripts/generate_diagram_pngs.py
cp docs/diagram_architecture.png docs/diagram.jpg
cp docs/diagram_paths.png docs/diagram1.jpg
```

## 1. 전체 구조

```mermaid
flowchart LR
    A[프로젝트 목표] --> B[B안 서비스형 RAG]
    A --> C[A안 AutoRAG/파인튜닝]

    subgraph Common[공통 모듈]
        D[configs/config.py]
        E[src/document_loader.py]
        F[src/chunker.py]
        G[src/embedder.py / VectorStore]
        H[src/retriever.py]
        I[src/generator.py]
        J[src/rag_pipeline.py]
    end

    B --> B1[scripts/index_documents.py]
    B1 --> E
    B1 --> F
    B1 --> G
    B --> B2[app.py]
    B2 --> J
    J --> H
    J --> I
    B --> B3[scripts/run_evaluation.py]
    B --> B4[scripts/check_release_gate.py]

    C --> C1[scripts/run_pipeline.py]
    C1 --> C2[prepare_autorag_from_csv.py]
    C1 --> C3[finetune_local.py]
    C1 --> C4[run_autorag_optimization.py]
    C4 --> C5[AutoRAG 결과 리포트]
    C3 --> C6[models/finetuned/*]
    C2 --> C7[data/autorag_csv/corpus.parquet]
    C2 --> C8[data/autorag_csv/qa.parquet]
```

## 2. B안 구조

```mermaid
flowchart TD
    A[원본 문서 PDF/HWP] --> B[scripts/index_documents.py]
    M[data/data_list.csv] --> B

    B --> C[DocumentLoader]
    C --> D[Chunking]
    D --> E[Embedding]
    E --> F[Chroma Vector DB]

    U[사용자 질문] --> G[app.py]
    G --> H[RAGPipeline]
    H --> I[Retriever]
    H --> J[Generator]
    F --> I
    J --> K[응답 출력]

    H -. 운영 평가 .-> L[scripts/run_evaluation.py]
    L --> N[평가 리포트]
    N --> O[scripts/check_release_gate.py]
```

## 3. A안 구조

```mermaid
flowchart TD
    A[CSV 원천 데이터] --> B[scripts/run_pipeline.py]

    B --> C[Step 1: prepare_autorag_from_csv.py]
    C --> C1[data/autorag_csv/corpus.parquet]
    C --> C2[data/autorag_csv/qa.parquet]

    B --> D[Step 2: finetune_local.py]
    D --> D1[models/finetuned/*]

    B --> E[Step 3: run_autorag_optimization.py]
    C1 --> E
    C2 --> E
    D1 --> E

    E --> F[retrieval / prompt / generator 탐색]
    F --> G[AutoRAG summary.csv / trial 결과]
    G --> H[최적 조건 선정]
```

## 4. 역할 구분

```mermaid
flowchart LR
    A[A안] --> A1[목표: 로컬 모델 기반 AutoRAG 최적화]
    A --> A2[엔트리: scripts/run_pipeline.py]
    A --> A3[산출물: corpus/qa, finetuned models, autorag results]

    B[B안] --> B1[목표: OpenAI 기반 서비스형 RAG]
    B --> B2[엔트리: index_documents.py -> app.py]
    B --> B3[평가: run_evaluation.py / check_release_gate.py]
```
