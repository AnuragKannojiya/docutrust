"""Ingestion pipeline: parse → chunk → embed → upsert."""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone

from app.db.models import IngestResult
from app.db.mongo import MongoStore
from app.ingestion.chunker import ChunkRecord, chunk_sections, make_chunk_id
from app.ingestion.parsers import detect_and_parse
from app.vectors.base import VectorStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        *,
        mongo: MongoStore,
        vector_store: VectorStore,
        embedder,
    ) -> None:
        self.mongo = mongo
        self.vector_store = vector_store
        self.embedder = embedder

    async def ingest(
        self,
        *,
        client_id: str,
        filename: str,
        mime: str,
        blob: bytes,
    ) -> IngestResult:
        if not blob:
            raise ValueError("empty document")

        sha = hashlib.sha256(blob).hexdigest()
        doc_id = f"doc_{uuid.uuid5(uuid.NAMESPACE_OID, sha)}"

        # 1) Parse
        sections = detect_and_parse(filename, mime, blob)
        if not sections:
            raise ValueError("no extractable text in document")

        # 2) Chunk
        chunk_pairs = chunk_sections(sections)
        if not chunk_pairs:
            raise ValueError("chunker produced zero chunks")

        records: list[ChunkRecord] = []
        for idx, (path, text) in enumerate(chunk_pairs):
            records.append(
                ChunkRecord(
                    chunk_id=make_chunk_id(doc_id, path, idx),
                    doc_id=doc_id,
                    section_path=path,
                    text=text,
                    char_range=(0, len(text)),
                )
            )

        # 3) Embed
        texts = [r.text for r in records]
        vectors = await self.embedder.aembed_documents(texts)

        # 4) Upsert document metadata
        await self.mongo.documents.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    "_id": doc_id,
                    "client_id": client_id,
                    "filename": filename,
                    "mime": mime,
                    "sha256": sha,
                    "uploaded_at": datetime.now(tz=timezone.utc),
                    "section_index": [r.section_path for r in records],
                    "total_chunks": len(records),
                }
            },
            upsert=True,
        )

        # 5) Replace chunk metadata
        if records:
            await self.mongo.chunks_meta.delete_many({"doc_id": doc_id})
            await self.mongo.chunks_meta.insert_many(
                [
                    {
                        "_id": r.chunk_id,
                        "doc_id": r.doc_id,
                        "section_path": r.section_path,
                        "text": r.text,
                        "char_range": list(r.char_range),
                        "source": r.source,
                    }
                    for r in records
                ]
            )

        # 6) Push to vector store (overwrite any existing vectors for this doc)
        self.vector_store.delete_by_document(doc_id)
        metadatas = [
            {
                "doc_id": r.doc_id,
                "client_id": client_id,
                "section_path": r.section_path,
                "source": r.source,
            }
            for r in records
        ]
        self.vector_store.add(
            ids=[r.chunk_id for r in records],
            vectors=vectors,
            metadatas=metadatas,
        )

        # 7) Link doc to client
        await self.mongo.clients.update_one(
            {"_id": client_id}, {"$addToSet": {"policies": doc_id}}
        )

        return IngestResult(
            doc_id=doc_id,
            filename=filename,
            section_count=len({r.section_path for r in records}),
            chunk_count=len(records),
        )
