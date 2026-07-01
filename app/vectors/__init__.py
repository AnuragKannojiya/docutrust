"""Vector store implementations.

Two backends share a common Protocol (``base.VectorStore``) so the rest of the
code never knows which one is in use:

- ``LocalFaissStore`` (default for dev) — file-backed FAISS index with
  jsonl-serialised metadata. No Atlas required.
- ``AtlasVectorStore`` (production) — uses MongoDB Atlas ``$vectorSearch``
  against a configured vector index.

Selection happens in :func:`build_vector_store` based on the
``VECTOR_BACKEND`` env var.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.config import Settings
from app.vectors.atlas_store import AtlasVectorStore
from app.vectors.base import ScoredChunk, VectorStore
from app.vectors.faiss_store import LocalFaissStore

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

__all__ = ["VectorStore", "ScoredChunk", "LocalFaissStore", "AtlasVectorStore", "build_vector_store"]


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "atlas":
        from pymongo import MongoClient

        client = MongoClient(settings.mongo_uri)
        log.info("Using AtlasVectorStore on %s", settings.mongo_uri)
        return AtlasVectorStore(
            mongo_client=client,
            db_name=settings.mongo_db,
            index_name=settings.atlas_vector_index,
            dim=settings.atlas_vector_dim,
        )

    log.info("Using LocalFaissStore at %s", settings.faiss_index_path)
    return LocalFaissStore(
        path=settings.faiss_index_path,
        dim=settings.openai_embed_dim,
    )
