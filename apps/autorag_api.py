"""
FastAPI wrapper for AutoRAG optimized trial.

Usage:
  set AUTORAG_TRIAL_DIR=evaluation/autorag_benchmark/0
  uvicorn apps.autorag_api:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.autorag_runner import AutoRAGRuntime


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


app = FastAPI(title="AutoRAG API", version="1.0.0")


@lru_cache(maxsize=1)
def get_runtime() -> AutoRAGRuntime:
    trial_dir = os.getenv("AUTORAG_TRIAL_DIR", "evaluation/autorag_benchmark/0")
    return AutoRAGRuntime(trial_dir=trial_dir)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")

    try:
        result = get_runtime().ask(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(answer=str(result))


