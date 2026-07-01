"""Atlas Vector Search backend.

Stores vectors in a dedicated MongoDB collection (``docutrust_vectors``) and
queries them via the ``$vectorSearch`` aggregation stage. The Atlas cluster
must have a vector index configured with the name from ``ATLAS_VECTOR_INDEX``
(default ``vector_index``) covering the ``embedding`` field with ``cosine``
similarity.

This module is not exercised in the dev path (``VECTOR_BACKEND=faiss``); the
class is here so the production deployment is one env var away.
"""
from __future__ import annotations

import logging
from typing import Optional

from pymongo import MongoClient

from app.vectors.base import ScoredChunk, VectorStore

log = logging.getLogger(__name__)


class AtlasVectorStore(VectorStore):
    """MongoDB Atlas ``$vectorSearch`` backend."""

    COLLECTION = "docutrust_vectors"

    def __init__(self, mongo_client: MongoClient, db_name: str, index_name: str, dim: int) -> None:
        self._client = mongo_client
        self._db = mongo_client[db_name]
        self._index_name = index_name
        self.dim = dim
        # Ensure the vector index path on every chunk document is consistent.
        self._db[self.COLLECTION].create_index("doc_id")

    def _coll(self):
        return self._db[self.COLLECTION]

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors, metadatas must have equal length")
        docs = []
        for cid, vec, meta in zip(ids, vectors, metadatas):
            if len(vec) != self.dim:
                raise ValueError(f"vector dim {len(vec)} != configured {self.dim}")
            docs.append({**meta, "chunk_id": cid, "embedding": vec})
        self._coll().insert_many(docs, ordered=False)

    def search(
        self,
        query_vector: list[float],
        k: int,
        filter: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        if len(query_vector) != self.dim:
            raise ValueError(f"query dim {len(query_vector)} != configured {self.dim}")

        # $vectorSearch requires the path/numCandidates/limit trio and an index
        # name. ``filter`` is the Atlas pre-filter syntax (Mongo match expression).
        stage: dict = {
            "$vectorSearch": {
                "index": self._index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(50, k * 10),
                "limit": k,
            }
        }
        if filter:
            stage["$vectorSearch"]["filter"] = filter

        pipeline = [stage, {"$project": {"embedding": 0}}]
        cursor = self._coll().aggregate(pipeline)
        results: list[ScoredChunk] = []
        for doc in cursor:
            results.append(
                ScoredChunk(
                    chunk_id=doc["chunk_id"],
                    score=float(doc.get("score", 0.0)),
                    metadata={k: v for k, v in doc.items() if k not in ("_id", "embedding")},
                )
            )
        return results

    def delete_by_document(self, doc_id: str) -> None:
        self._coll().delete_many({"doc_id": doc_id})

    def count(self) -> int:
        return self._coll().estimated_document_count()
