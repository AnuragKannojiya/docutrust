"""FastAPI smoke tests.

We don't start a real Mongo or load a real cross-encoder; instead we
override the app's lifespan-bound dependencies with in-memory stubs and
exercise the routes via ``httpx.AsyncClient``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import io
from datetime import datetime, timezone

import httpx
import pytest

from app.config import Settings
from app.main import create_app
from app.rag.grader import CrossEncoderGrader
from app.tracing.logger import Trace
from app.vectors.base import ScoredChunk
from app.vectors.faiss_store import LocalFaissStore
from tests.conftest import FakeChat, FakeEmbedder, FakeGrader


class _StubMongo:
    def __init__(self):
        self.clients = _StubColl()
        self.documents = _StubColl()
        self.chunks_meta = _StubColl()
        self.trace_logs = _StubColl()

    async def connect(self):
        return None

    async def close(self):
        return None

    async def ensure_indexes(self):
        return None


class _StubColl:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    def find(self, q=None, **kwargs):
        return _AsyncIter([d for d in self.docs.values() if _match(d, q or {})])

    async def find_one(self, q):
        for d in self.docs.values():
            if _match(d, q or {}):
                return d
        return None

    async def insert_one(self, doc):
        self.docs[doc["_id"]] = doc
        return _StubResult()

    async def insert_many(self, docs):
        for d in docs:
            self.docs[d["_id"]] = d
        return _StubResult()

    async def update_one(self, q, update, upsert=False):
        doc = await self.find_one(q)
        if doc is None and upsert:
            doc = dict(q)
            self.docs[doc.get("_id", "stub")] = doc
        for op, fields in update.items():
            if op == "$set":
                doc.update(fields)
            elif op == "$addToSet":
                for k, v in fields.items():
                    doc.setdefault(k, [])
                    if v not in doc[k]:
                        doc[k].append(v)
        return _StubResult()


class _StubResult:
    inserted_id = None
    acknowledged = True


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _match(doc, q):
    for k, v in q.items():
        if isinstance(v, dict) and "$in" in v:
            if doc.get(k) not in v["$in"]:
                return False
        elif doc.get(k) != v:
            return False
    return True


@pytest.fixture
async def app_with_stubs(tmp_path, monkeypatch):
    """A FastAPI app with all heavy deps replaced by fakes."""
    from sentence_transformers import CrossEncoder  # noqa: F401  (typing only)

    app = create_app()

    @asynccontextmanager
    async def _fake_lifespan(_app):
        settings = Settings(retrieval_k=4, grader_threshold=0.5)
        _app.state.settings = settings
        _app.state.mongo = _StubMongo()
        faiss = LocalFaissStore(path=str(tmp_path / "faiss"), dim=4)
        _app.state.vector_store = faiss
        _app.state.grader = FakeGrader()
        _app.state.chat_model = FakeChat(
            ["You must report incidents within 1 hour. [doc_abc::intro::0]"]
        )
        _app.state.embedder = FakeEmbedder()

        class _Graph:
            pass

        g = _Graph()
        g._docutrust_grader = _app.state.grader
        g._docutrust_vector_store = faiss
        g._docutrust_mongo = _app.state.mongo
        g._docutrust_embedder = _app.state.embedder
        g._docutrust_chat = _app.state.chat_model
        g._docutrust_settings = settings
        _app.state.graph = g

        from app.tracing.logger import TraceLogger

        _app.state.tracer = TraceLogger(_app.state.mongo)
        yield

    async with _fake_lifespan(app):
        yield app


@pytest.mark.asyncio
async def test_healthz(app_with_stubs):
    transport = httpx.ASGITransport(app=app_with_stubs)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_clients_create_and_list(app_with_stubs):
    transport = httpx.ASGITransport(app=app_with_stubs)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/clients", json={"name": "ACME"})
        assert r.status_code == 201
        cid = r.json()["_id"]

        r2 = await client.get("/clients")
        assert r2.status_code == 200
        names = [c["name"] for c in r2.json()]
        assert "ACME" in names

        r3 = await client.get(f"/clients/{cid}")
        assert r3.status_code == 200
        assert r3.json()["name"] == "ACME"


@pytest.mark.asyncio
async def test_ask_happy_path_persists_trace(app_with_stubs):
    transport = httpx.ASGITransport(app=app_with_stubs)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Seed a relevant chunk in the stub mongo.
        mongo = app_with_stubs.state.mongo
        mongo.chunks_meta.docs["doc_abc::intro::0"] = {
            "_id": "doc_abc::intro::0",
            "doc_id": "doc_abc",
            "section_path": "intro",
            "text": "Report security incidents within 1 hour.",
            "char_range": [0, 100],
            "source": "corpus",
        }
        # Make sure the embedder's "hash" of the question matches the seeded vector.
        app_with_stubs.state.vector_store.add(
            ids=["doc_abc::intro::0"],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            metadatas=[{"doc_id": "doc_abc"}],
        )

        r = await client.post(
            "/ask", json={"client_id": None, "question": "How long to report?"}
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["refused"] is False
        assert data["answer"]
        assert len(data["citations"]) == 1

        # Trace should be retrievable.
        r2 = await client.get(f"/ask/{data['trace_id']}")
        assert r2.status_code == 200
        trace = r2.json()
        node_names = [s["node"] for s in trace["steps"]]
        assert "retrieve" in node_names
        assert "generate" in node_names


@pytest.mark.asyncio
async def test_ask_refusal_when_no_citations(app_with_stubs):
    transport = httpx.ASGITransport(app=app_with_stubs)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Override chat to return a citation-less answer.
        app_with_stubs.state.chat_model = FakeChat(
            ["You should report incidents promptly."]
        )
        app_with_stubs.state.graph._docutrust_chat = app_with_stubs.state.chat_model

        mongo = app_with_stubs.state.mongo
        mongo.chunks_meta.docs["doc_x::intro::0"] = {
            "_id": "doc_x::intro::0",
            "doc_id": "doc_x",
            "section_path": "intro",
            "text": "lunch menu",
            "char_range": [0, 10],
            "source": "corpus",
        }
        app_with_stubs.state.vector_store.add(
            ids=["doc_x::intro::0"],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            metadatas=[{"doc_id": "doc_x"}],
        )
        r = await client.post(
            "/ask", json={"client_id": None, "question": "What is for lunch?"}
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["refused"] is True
        assert "no citations" in (data["refusal_reason"] or "")
        assert data["answer"] is None
        assert data["citations"] == []
