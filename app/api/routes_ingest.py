"""POST /ingest: upload a document, parse, chunk, embed, and index it."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.db.models import IngestResult
from app.deps import get_embedder, get_mongo, get_vector_store
from app.ingestion.pipeline import IngestionPipeline

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=IngestResult)
async def ingest(
    client_id: str = Form(...),
    file: UploadFile = File(...),
    mongo=Depends(get_mongo),
    vector_store=Depends(get_vector_store),
    embedder=Depends(get_embedder),
) -> IngestResult:
    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="empty file")

    pipeline = IngestionPipeline(mongo=mongo, vector_store=vector_store, embedder=embedder)
    try:
        result = await pipeline.ingest(
            client_id=client_id,
            filename=file.filename or "untitled",
            mime=file.content_type or "application/octet-stream",
            blob=blob,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
