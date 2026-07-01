"""FastAPI entry point.

Wires the lifespan (Mongo, vector store, cross-encoder, graph) and the routers.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_ask import router as ask_router
from app.api.routes_clients import router as clients_router
from app.api.routes_ingest import router as ingest_router
from app.config import get_settings
from app.db.mongo import MongoStore
from app.rag.graph import build_graph
from app.tracing.logger import TraceLogger
from app.vectors import build_vector_store

log = logging.getLogger("docutrust")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    # --- MongoDB ---
    mongo = MongoStore(settings.mongo_uri, settings.mongo_db)
    await mongo.connect()
    await mongo.ensure_indexes()
    app.state.mongo = mongo
    log.info("Mongo connected at %s", settings.mongo_uri)

    # --- Vector store ---
    app.state.vector_store = build_vector_store(settings)
    log.info("Vector store backend: %s", settings.vector_backend)

    # --- Cross-encoder grader (lazy import; sentence-transformers is heavy) ---
    from sentence_transformers import CrossEncoder

    log.info("Loading cross-encoder: %s", settings.grader_model)
    app.state.grader = CrossEncoder(
        settings.grader_model, device=settings.grader_device
    )
    log.info("Cross-encoder ready")

    # --- OpenAI clients ---
    # Lazy import so missing key surfaces here rather than at import time.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY is empty — generation calls will fail")
    app.state.chat_model = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key or None,
        temperature=0.0,
    )
    app.state.embedder = OpenAIEmbeddings(
        model=settings.openai_embed_model,
        api_key=settings.openai_api_key or None,
    )

    # --- Tracer ---
    app.state.tracer = TraceLogger(mongo)

    # --- Compiled LangGraph ---
    app.state.graph = build_graph(
        mongo=mongo,
        vector_store=app.state.vector_store,
        chat_model=app.state.chat_model,
        embedder=app.state.embedder,
        grader=app.state.grader,
        settings=settings,
    )
    log.info("CRAG graph compiled")

    try:
        yield
    finally:
        await mongo.close()
        log.info("Mongo closed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuTrust",
        version="0.1.0",
        description=(
            "Self-correcting enterprise RAG with cross-encoder grading, "
            "web fallback, and strict citations."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(ask_router, prefix="/ask", tags=["ask"])
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(clients_router, prefix="/clients", tags=["clients"])
    return app


app = create_app()
