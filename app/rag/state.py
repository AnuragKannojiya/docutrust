"""LangGraph state for the Corrective RAG pipeline.

The graph is intentionally serialisable: every node reads from and writes to
this dict-shaped state, which is what allows ``tracing.Trace`` to record each
step in plain Python.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    doc_id: str
    section_path: str
    text: str
    score: float
    source: str  # "corpus" or "web"
    url: Optional[str]


class GradedChunk(TypedDict):
    chunk_id: str
    doc_id: str
    section_path: str
    text: str
    score: float
    graded_score: float
    source: str
    url: Optional[str]


class GraphState(TypedDict, total=False):
    question: str
    rewritten_question: Optional[str]
    retrieved: list[RetrievedChunk]
    relevant: list[GradedChunk]
    web_results: list[RetrievedChunk]
    documents: list[GradedChunk]  # what the generator sees (relevant + web)
    generation: Optional[str]
    citations: list[dict]  # list of Citation.model_dump() dicts
    refused: bool
    refusal_reason: Optional[str]
    trace: list[dict]  # populated by the runner, not the graph itself
