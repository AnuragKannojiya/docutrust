"""CRAG runner tests with stubbed LLM, grader, and vector store."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.rag.grader import CrossEncoderGrader
from app.rag.runner import run_crag
from app.rag.state import GraphState
from app.tracing.logger import Trace
from app.vectors.base import ScoredChunk
from tests.conftest import FakeChat, FakeEmbedder, FakeGrader, fake_mongo, faiss_store  # noqa: F401


def _settings() -> Settings:
    return Settings(
        retrieval_k=4,
        grader_threshold=0.5,
        web_max_results=3,
        web_domains_allowlist="example.com,test.org",
    )


def _seed_corpus(mongo, vector_store, dim=4):
    """Add a single fake document with one chunk that contains a relevant match."""
    doc_id = "doc_test::1"
    chunk_id = "doc_test::1::intro::0"
    text = "Report security incidents within one hour of detection."
    mongo.chunks_meta.docs[chunk_id] = {
        "_id": chunk_id,
        "doc_id": doc_id,
        "section_path": "intro",
        "text": text,
        "char_range": [0, len(text)],
        "source": "corpus",
    }
    fake_embedding = [1.0, 0.0, 0.0, 0.0][:dim]
    vector_store.add(ids=[chunk_id], vectors=[fake_embedding], metadatas=[{"doc_id": doc_id}])


def _stub_graph(vector_store, mongo, embedder, chat, grader):
    """A no-op graph stand-in that exposes the stashed attributes the runner needs."""

    class _G:
        pass

    g = _G()
    g._docutrust_grader = grader
    g._docutrust_vector_store = vector_store
    g._docutrust_mongo = mongo
    g._docutrust_embedder = embedder
    g._docutrust_chat = chat
    g._docutrust_settings = _settings()
    return g


@pytest.mark.asyncio
async def test_happy_path_emits_citations(fake_mongo, faiss_store):  # noqa: F811
    _seed_corpus(fake_mongo, faiss_store)
    chat = FakeChat(
        [
            "You must report incidents within 1 hour. [doc_test::1::intro::0]",
        ]
    )
    trace = Trace(question="How long do I have to report a security incident?", client_id=None)
    result = await run_crag(
        question="How long do I have to report a security incident?",
        client_id=None,
        settings=_settings(),
        mongo=fake_mongo,
        chat=chat,
        embedder=FakeEmbedder(),
        graph=_stub_graph(faiss_store, fake_mongo, FakeEmbedder(), chat, FakeGrader()),
        trace=trace,
        grader=CrossEncoderGrader(FakeGrader(), 0.5),
    )
    assert result["refused"] is False
    assert result["answer"] is not None
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "doc_test::1::intro::0"
    node_names = [s.node for s in trace.steps]
    assert "retrieve" in node_names
    assert "grade" in node_names
    assert "generate" in node_names
    # No rewrite or web_search on the happy path.
    assert "rewrite" not in node_names
    assert "web_search" not in node_names


@pytest.mark.asyncio
async def test_refusal_when_no_citations(fake_mongo, faiss_store):  # noqa: F811
    _seed_corpus(fake_mongo, faiss_store)
    # Chat returns a plausible-looking answer but without citation markers.
    chat = FakeChat(["You should report incidents promptly."])
    trace = Trace(question="How long?", client_id=None)
    result = await run_crag(
        question="How long?",
        client_id=None,
        settings=_settings(),
        mongo=fake_mongo,
        chat=chat,
        embedder=FakeEmbedder(),
        graph=_stub_graph(faiss_store, fake_mongo, FakeEmbedder(), chat, FakeGrader()),
        trace=trace,
        grader=CrossEncoderGrader(FakeGrader(), 0.5),
    )
    assert result["refused"] is True
    assert "no citations" in (result["refusal_reason"] or "")
    assert result["answer"] is None
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_rewrite_path_when_no_relevant_chunks(fake_mongo, faiss_store, monkeypatch):  # noqa: F811
    # Seed an irrelevant chunk.
    doc_id = "doc_unrelated::1"
    chunk_id = "doc_unrelated::1::cafe::0"
    text = "The cafeteria serves fresh sushi on Tuesdays."
    fake_mongo.chunks_meta.docs[chunk_id] = {
        "_id": chunk_id,
        "doc_id": doc_id,
        "section_path": "cafe",
        "text": text,
        "char_range": [0, len(text)],
        "source": "corpus",
    }
    faiss_store.add(ids=[chunk_id], vectors=[[0.0, 1.0, 0.0, 0.0]], metadatas=[{"doc_id": doc_id}])

    # Stub web search to return a known chunk.
    async def _fake_web(state, settings=None):
        return [
            {
                "chunk_id": "web::example::nist.gov",
                "doc_id": "web::example::nist.gov",
                "section_path": "NIST example",
                "text": "Incidents must be reported within 1 hour.",
                "score": 0.0,
                "source": "web",
                "url": "https://nist.gov/example",
            }
        ]

    from app.rag import nodes as rag_nodes
    from app.rag import runner as rag_runner
    monkeypatch.setattr(rag_nodes, "do_web_search", _fake_web)
    monkeypatch.setattr(rag_runner, "do_web_search", _fake_web)

    chat = FakeChat(
        [
            "incident reporting window nist",
            "Report within 1 hour per [web::example::nist.gov]",
        ]
    )
    trace = Trace(question="How long do I have to report a security incident?", client_id=None)
    result = await run_crag(
        question="How long do I have to report a security incident?",
        client_id=None,
        settings=_settings(),
        mongo=fake_mongo,
        chat=chat,
        embedder=FakeEmbedder(),
        graph=_stub_graph(faiss_store, fake_mongo, FakeEmbedder(), chat, FakeGrader()),
        trace=trace,
        grader=CrossEncoderGrader(FakeGrader(), 0.99),  # Force nothing to clear the threshold
    )

    assert result["refused"] is False
    assert result["answer"] is not None
    node_names = [s.node for s in trace.steps]
    assert "rewrite" in node_names
    assert "web_search" in node_names
    # The web citation should be the only citation.
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "web"
