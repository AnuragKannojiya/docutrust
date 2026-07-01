"""File-backed FAISS vector store for local development.

Persists two files under ``path``:
- ``index.faiss`` — the FAISS index.
- ``meta.jsonl``  — one JSON record per vector, in the same order as the index.

The store is intentionally simple — it loads the entire index into memory on
first access. For real workloads, use ``AtlasVectorStore`` instead.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import faiss  # type: ignore
import numpy as np

from app.vectors.base import ScoredChunk, VectorStore

log = logging.getLogger(__name__)


class LocalFaissStore(VectorStore):
    """In-process FAISS index with a sidecar jsonl metadata file."""

    def __init__(self, path: str, dim: int) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.index_path = self.path / "index.faiss"
        self.meta_path = self.path / "meta.jsonl"
        self._lock = threading.RLock()

        self._index: Optional[faiss.Index] = None
        self._metas: list[dict] = []
        self._load()

    # ---- persistence ---------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            if self.index_path.exists() and self.meta_path.exists():
                log.info("Loading FAISS index from %s", self.index_path)
                self._index = faiss.read_index(str(self.index_path))
                self._metas = [
                    json.loads(line)
                    for line in self.meta_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if self._index.ntotal != len(self._metas):
                    raise RuntimeError(
                        f"FAISS index / metadata mismatch: "
                        f"{self._index.ntotal} vectors vs {len(self._metas)} metas"
                    )
                # Reject silently-drifted dimensions rather than corrupting writes.
                if self._index.d != self.dim:
                    raise RuntimeError(
                        f"FAISS dim {self._index.d} != configured {self.dim}; "
                        "delete the index or update OPENAI_EMBED_DIM"
                    )
            else:
                log.info("Creating new FAISS index (dim=%d)", self.dim)
                self._index = faiss.IndexFlatIP(self.dim)  # cosine via normalised vectors
                self._metas = []
                self._save()

    def _save(self) -> None:
        assert self._index is not None
        faiss.write_index(self._index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as fh:
            for m in self._metas:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---- VectorStore interface ---------------------------------------------

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
        with self._lock:
            assert self._index is not None
            arr = np.asarray(vectors, dtype="float32")
            # Normalise rows so IndexFlatIP behaves like cosine similarity.
            faiss.normalize_L2(arr)
            self._index.add(arr)
            for cid, meta in zip(ids, metadatas):
                # Embed the id into the metadata so a hit tells us the chunk_id
                # without a separate lookup table.
                self._metas.append({**meta, "chunk_id": cid})
            self._save()

    def search(
        self,
        query_vector: list[float],
        k: int,
        filter: Optional[dict] = None,
    ) -> list[ScoredChunk]:
        with self._lock:
            assert self._index is not None
            if self._index.ntotal == 0:
                return []
            arr = np.asarray([query_vector], dtype="float32")
            faiss.normalize_L2(arr)
            k_eff = min(k, self._index.ntotal)
            scores, idxs = self._index.search(arr, k_eff)
            results: list[ScoredChunk] = []
            for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
                if idx < 0 or idx >= len(self._metas):
                    continue
                meta = self._metas[idx]
                if filter and not _matches_filter(meta, filter):
                    continue
                results.append(
                    ScoredChunk(
                        chunk_id=meta["chunk_id"],
                        score=float(score),
                        metadata=meta,
                    )
                )
            return results

    def delete_by_document(self, doc_id: str) -> None:
        """Rebuild the index without chunks belonging to ``doc_id``.

        FAISS has no native per-id delete for ``IndexFlat``, so we rebuild.
        This is fine for the dev workflow (delete-and-reingest a document) but
        would need a different structure for production. With ``IndexFlat`` the
        cost is O(N) per delete, which is acceptable while the index is small.
        """
        with self._lock:
            assert self._index is not None
            if self._index.ntotal == 0:
                return
            keep_idx = [i for i, m in enumerate(self._metas) if m.get("doc_id") != doc_id]
            if len(keep_idx) == self._index.ntotal:
                return
            kept_vectors = np.zeros((len(keep_idx), self.dim), dtype="float32")
            for new_i, old_i in enumerate(keep_idx):
                kept_vectors[new_i] = self._index.reconstruct(old_i)
            self._metas = [self._metas[i] for i in keep_idx]
            self._index = faiss.IndexFlatIP(self.dim)
            if len(kept_vectors):
                faiss.normalize_L2(kept_vectors)
                self._index.add(kept_vectors)
            self._save()

    def count(self) -> int:
        with self._lock:
            return 0 if self._index is None else int(self._index.ntotal)


def _matches_filter(meta: dict, filt: dict) -> bool:
    """Tiny filter language: ``{"doc_id": "abc"}`` or ``{"client_id": "x", "source": "web"}``."""
    for k, v in filt.items():
        if meta.get(k) != v:
            return False
    return True
