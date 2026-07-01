"""Structural chunker.

Splits parsed document text into chunks of approximately ``target_tokens`` with
``overlap_tokens`` of overlap. Chunking is performed per-section, so a single
section is never split across chunks unless it is too large.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import tiktoken

log = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


def _n_tokens(s: str) -> int:
    return len(_ENC.encode(s or ""))


def _split_long_section(
    section_path: str,
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str]]:
    """Token-window a section that exceeds ``target_tokens``."""
    tokens = _ENC.encode(text)
    out: list[tuple[str, str]] = []
    step = max(1, target_tokens - overlap_tokens)
    i = 0
    idx = 0
    while i < len(tokens):
        window = tokens[i : i + target_tokens]
        chunk_text = _ENC.decode(window)
        out.append((f"{section_path}::chunk_{idx}", chunk_text))
        idx += 1
        if i + target_tokens >= len(tokens):
            break
        i += step
    return out


def chunk_sections(
    sections: Iterable[tuple[str, str]],
    *,
    target_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[tuple[str, str]]:
    """Yield ``(chunk_path, text)`` pairs from parsed sections."""
    out: list[tuple[str, str]] = []
    for section_path, text in sections:
        text = (text or "").strip()
        if not text:
            continue
        if _n_tokens(text) <= target_tokens:
            out.append((section_path, text))
        else:
            out.extend(_split_long_section(section_path, text, target_tokens, overlap_tokens))
    return out


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    section_path: str
    text: str
    char_range: tuple[int, int]
    source: str = "corpus"


def make_chunk_id(doc_id: str, section_path: str, idx: int) -> str:
    """Stable, human-readable chunk id used as the citation marker."""
    safe = section_path.replace(" ", "_").replace("/", "_")[:60]
    return f"{doc_id}::{safe}::{idx}"
