"""MongoDB client and collection helpers."""
from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING

log = logging.getLogger(__name__)


class MongoStore:
    """Thin wrapper around motor: one client, one database, named collections."""

    def __init__(self, uri: str, db_name: str) -> None:
        self.uri = uri
        self.db_name = db_name
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("MongoStore.connect() has not been called")
        return self._db

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=5000)
        # Force a round-trip so we fail fast on bad URIs.
        await self._client.admin.command("ping")
        self._db = self._client[self.db_name]
        log.info("Mongo connected to %s/%s", self.uri, self.db_name)

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None

    async def ensure_indexes(self) -> None:
        assert self._db is not None
        await self._db["clients"].create_index([("name", ASCENDING)], unique=False)
        await self._db["documents"].create_index([("client_id", ASCENDING)])
        await self._db["documents"].create_index([("sha256", ASCENDING)])
        await self._db["chunks_meta"].create_index([("doc_id", ASCENDING)])
        await self._db["trace_logs"].create_index([("client_id", ASCENDING)])
        await self._db["trace_logs"].create_index([("started_at", ASCENDING)])

    # Convenience accessors ----------------------------------------------------
    @property
    def clients(self):
        return self.db["clients"]

    @property
    def documents(self):
        return self.db["documents"]

    @property
    def chunks_meta(self):
        return self.db["chunks_meta"]

    @property
    def trace_logs(self):
        return self.db["trace_logs"]
