"""Shared pytest fixtures.

We don't start a real FastAPI server or Mongo in unit tests; instead, tests
stub the heavy dependencies (LLM, grader, vector store) and exercise the
imperative CRAG runner directly.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_faiss_path(tmp_path) -> str:
    d = tmp_path / "faiss"
    d.mkdir()
    return str(d)


@pytest_asyncio.fixture
async def faiss_store(tmp_faiss_path):
    from app.vectors.faiss_store import LocalFaissStore

    store = LocalFaissStore(path=tmp_faiss_path, dim=4)
    return store


@pytest_asyncio.fixture
async def fake_mongo():
    """In-memory stub of the few Mongo collections we touch in tests."""
    from app.db.mongo import MongoStore

    class _StubStore:
        def __init__(self) -> None:
            self.chunks_meta = _StubColl()
            self.documents = _StubColl()
            self.clients = _StubColl()
            self.trace_logs = _StubColl()

        async def ensure_indexes(self):
            return None

    class _StubColl:
        def __init__(self):
            self.docs: dict[str, dict] = {}

        def find(self, q=None, **kwargs):
            q = q or {}
            results = []
            for d in self.docs.values():
                if all(_match(d.get(k), v) for k, v in q.items()):
                    results.append(d)
            return _AsyncIter(results)

        async def find_one(self, q):
            for d in self.docs.values():
                if all(_match(d.get(k), v) for k, v in (q or {}).items()):
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

        async def delete_many(self, q):
            for k in list(self.docs.keys()):
                if all(_match(self.docs[k].get(f), v) for f, v in (q or {}).items()):
                    del self.docs[k]
            return _StubResult()

    class _StubResult:
        inserted_id = None
        acknowledged = True

    class _AsyncIter:
        def __init__(self, items):
            self._items = items

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    return _StubStore()


def _match(needle, haystack):
    if isinstance(haystack, dict) and "$in" in haystack:
        return needle in haystack["$in"]
    return needle == haystack


class FakeChat:
    """Stub chat model: returns canned answers based on the user message."""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)
        self._i = 0

    async def ainvoke(self, prompt_or_messages):
        if not self._scripted:
            return _FakeMsg("REFUSE: insufficient evidence in the provided sources.")
        out = self._scripted[self._i % len(self._scripted)]
        self._i += 1
        return _FakeMsg(out)


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content


class FakeEmbedder:
    def __init__(self, dim: int = 4):
        self.dim = dim

    async def aembed_query(self, text: str) -> list[float]:
        # Deterministic: hash the text to a unit vector.
        return _to_vec(text, self.dim)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_to_vec(t, self.dim) for t in texts]


def _to_vec(text: str, dim: int) -> list[float]:
    h = abs(hash(text)) or 1
    return [((h >> i) & 0xFF) / 255.0 for i in range(0, dim * 8, 8)][:dim]


class FakeGrader:
    """Cross-encoder stand-in: returns scores based on keyword overlap."""

    def predict(self, pairs, show_progress_bar=False):
        stop_words = {
            "how", "do", "i", "have", "to", "a", "the", "of", "within", "one", "hour",
            "on", "is", "open", "today", "and", "or", "in", "for", "with", "at", "by",
            "an", "about", "your", "we", "you", "long"
        }
        out = []
        for q, d in pairs:
            q_words = [w for w in re.findall(r'\b\w+\b', q.lower()) if w not in stop_words]
            d_words = [w for w in re.findall(r'\b\w+\b', d.lower()) if w not in stop_words]
            if not q_words:
                out.append(-1.0)
                continue
            
            matches = 0
            for qw in q_words:
                for dw in d_words:
                    if qw == dw or (len(qw) >= 4 and len(dw) >= 4 and (qw.startswith(dw[:4]) or dw.startswith(qw[:4]))):
                        matches += 1
                        break
            overlap = matches / len(q_words)
            out.append(2 * overlap - 1)  # raw score in [-1, 1]
        return out


@pytest.fixture
def fake_chat():
    return FakeChat
