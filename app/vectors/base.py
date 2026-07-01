"""Vector store Protocol and shared types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class ScoredChunk:
    """One retrieved chunk with its similarity score and metadata."""

    chunk_id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: Optional[str] = None  # populated lazily by the graph, not by the store


@runtime_checkable
class VectorStore(Protocol):
    """Backend-agnostic interface used by the CRAG graph and the ingestion
    pipeline. Implementations: :class:`LocalFaissStore`, :class:`AtlasVectorStore`.
    """

    def add(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None: ...

    def search(
        self,
        query_vector: list[float],
        k: int,
        filter: Optional[dict] = None,
    ) -> list[ScoredChunk]: ...

    def delete_by_document(self, doc_id: str) -> None: ...

    def count(self) -> int: ...
