"""Pydantic models for the DocuTrust domain."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---- Clients ----------------------------------------------------------------

class ClientIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    industry: Optional[str] = Field(default=None, max_length=200)
    notes: Optional[str] = None


class Client(ClientIn):
    id: str = Field(..., alias="_id")
    created_at: datetime
    policies: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ---- Documents --------------------------------------------------------------

class Document(BaseModel):
    id: str = Field(..., alias="_id")
    client_id: str
    filename: str
    mime: str
    sha256: str
    uploaded_at: datetime
    section_index: list[str] = Field(default_factory=list)
    total_chunks: int = 0

    model_config = {"populate_by_name": True}


class IngestResult(BaseModel):
    doc_id: str
    filename: str
    section_count: int
    chunk_count: int


# ---- Chunks -----------------------------------------------------------------

class ChunkMeta(BaseModel):
    """Metadata for a single chunk. The vector lives in the VectorStore."""

    id: str = Field(..., alias="_id")
    doc_id: str
    section_path: str
    text: str
    char_range: tuple[int, int]
    source: str = "corpus"  # or "web"

    model_config = {"populate_by_name": True}


# ---- Citations --------------------------------------------------------------

class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    section_path: str
    quote: str
    source: str = "corpus"  # or "web"
    url: Optional[str] = None


# ---- Ask request/response ---------------------------------------------------

class AskRequest(BaseModel):
    client_id: Optional[str] = None
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    trace_id: str
    answer: Optional[str]
    refused: bool
    refusal_reason: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    steps: list[dict] = Field(default_factory=list)


# ---- Trace log --------------------------------------------------------------

class TraceStep(BaseModel):
    node: str
    input: dict
    output: dict
    duration_ms: int
    started_at: datetime
    finished_at: datetime


class TraceLog(BaseModel):
    id: str = Field(..., alias="_id")
    client_id: Optional[str]
    question: str
    final_answer: Optional[str]
    refused: bool
    refusal_reason: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    model_config = {"populate_by_name": True}
