"""POST /ask: run the CRAG graph; GET /ask/{trace_id}: fetch a previous trace."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.db.models import AskRequest, AskResponse, Citation
from app.deps import (
    get_chat_model,
    get_embedder,
    get_grader,
    get_graph,
    get_mongo,
    get_settings_dep,
    get_tracer,
    get_vector_store,
)
from app.rag.grader import CrossEncoderGrader
from app.rag.runner import run_crag
from app.tracing.logger import Trace

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=AskResponse)
async def ask(
    body: AskRequest,
    request: Request,
) -> AskResponse:
    settings = get_settings_dep(request)
    mongo = get_mongo(request)
    embedder = get_embedder(request)
    chat = get_chat_model(request)
    tracer = get_tracer(request)
    graph = get_graph(request)
    grader_model = get_grader(request)
    vector_store = get_vector_store(request)

    grader = CrossEncoderGrader(grader_model, settings.grader_threshold)
    trace = Trace(question=body.question, client_id=body.client_id)

    result = await run_crag(
        question=body.question,
        client_id=body.client_id,
        settings=settings,
        mongo=mongo,
        chat=chat,
        embedder=embedder,
        graph=graph,
        trace=trace,
        grader=grader,
    )

    citations = [
        Citation(
            chunk_id=c["chunk_id"],
            doc_id=c["doc_id"],
            section_path=c["section_path"],
            quote=c.get("quote", ""),
            source=c.get("source", "corpus"),
            url=c.get("url"),
        )
        for c in result.get("citations", [])
    ]
    answer = result.get("answer")
    refused = result.get("refused", False)
    refusal_reason = result.get("refusal_reason")
    trace.set_final(
        answer=answer,
        refused=refused,
        refusal_reason=refusal_reason,
        citations=citations,
    )
    await tracer.save(trace)

    return AskResponse(
        trace_id=trace.id,
        answer=answer,
        refused=refused,
        refusal_reason=refusal_reason,
        citations=citations,
        steps=[s.to_model().model_dump(mode="json") for s in trace.steps],
    )


@router.get("/{trace_id}")
async def get_trace(trace_id: str, request: Request) -> dict[str, Any]:
    tracer = get_tracer(request)
    doc = await tracer.get(trace_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="trace not found")
    for k in ("started_at", "finished_at"):
        if isinstance(doc.get(k), datetime):
            doc[k] = doc[k].isoformat()
    return doc
