"""Local FAISS vector store tests."""
from __future__ import annotations

import numpy as np
import pytest

from app.vectors.faiss_store import LocalFaissStore


@pytest.mark.asyncio
async def test_add_and_search_roundtrip(tmp_faiss_path):
    store = LocalFaissStore(path=tmp_faiss_path, dim=4)
    store.add(
        ids=["a", "b", "c"],
        vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    assert store.count() == 3

    # Query close to vector "a".
    hits = store.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "a"
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_filter_by_metadata(tmp_faiss_path):
    store = LocalFaissStore(path=tmp_faiss_path, dim=2)
    store.add(
        ids=["x", "y"],
        vectors=[[1.0, 0.0], [0.9, 0.1]],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    hits = store.search([1.0, 0.0], k=2, filter={"doc_id": "d2"})
    assert [h.chunk_id for h in hits] == ["y"]


@pytest.mark.asyncio
async def test_persistence_across_instances(tmp_faiss_path):
    store = LocalFaissStore(path=tmp_faiss_path, dim=2)
    store.add(ids=["p"], vectors=[[1.0, 0.0]], metadatas=[{"doc_id": "d1"}])
    store2 = LocalFaissStore(path=tmp_faiss_path, dim=2)
    assert store2.count() == 1
    assert store2.search([1.0, 0.0], k=1)[0].chunk_id == "p"


@pytest.mark.asyncio
async def test_delete_by_document_rebuilds(tmp_faiss_path):
    store = LocalFaissStore(path=tmp_faiss_path, dim=2)
    store.add(
        ids=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    store.delete_by_document("d1")
    assert store.count() == 1
    assert [h.chunk_id for h in store.search([0.5, 0.5], k=1)] == ["c"]


@pytest.mark.asyncio
async def test_empty_search_returns_empty(tmp_faiss_path):
    store = LocalFaissStore(path=tmp_faiss_path, dim=2)
    assert store.search([1.0, 0.0], k=3) == []
