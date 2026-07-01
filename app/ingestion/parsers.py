"""Document parsers for PDF, DOCX, and Markdown.

Each parser returns a flat list of ``(section_path, text)`` pairs in document
order. The chunker downstream joins adjacent pieces under a single section.
"""
from __future__ import annotations

import io
import logging
import re
from typing import Iterable

log = logging.getLogger(__name__)


def parse_pdf(blob: bytes) -> list[tuple[str, str]]:
    """Parse a PDF into ``(section_path, text)`` pairs using ``pypdf``."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(blob))
    out: list[tuple[str, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        out.append((f"page_{i + 1}", text))
    return out


def parse_docx(blob: bytes) -> list[tuple[str, str]]:
    """Parse a DOCX into ``(section_path, text)`` pairs using ``python-docx``."""
    from docx import Document

    doc = Document(io.BytesIO(blob))
    out: list[tuple[str, str]] = []
    current_heading = "document"
    buffer: list[str] = []
    for p in doc.paragraphs:
        style = (p.style.name or "").lower() if p.style else ""
        txt = p.text.strip()
        if not txt:
            continue
        if "heading" in style or style.startswith("title"):
            if buffer:
                out.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = txt[:120]
        else:
            buffer.append(txt)
    if buffer:
        out.append((current_heading, "\n".join(buffer).strip()))
    return out


def parse_markdown(blob: bytes) -> list[tuple[str, str]]:
    """Parse Markdown into ``(section_path, text)`` pairs split on headings."""
    text = blob.decode("utf-8", errors="replace")
    parts = re.split(r"^(#{1,6})\s+(.+)$", text, flags=re.MULTILINE)
    out: list[tuple[str, str]] = []
    if parts[0].strip():
        out.append(("preamble", parts[0].strip()))
    # parts is: [pre, hashes, title, body, hashes, title, body, ...]
    for i in range(1, len(parts), 3):
        hashes, title, body = parts[i], parts[i + 1], parts[i + 2] if i + 2 < len(parts) else ""
        level = len(hashes)
        out.append((f"h{level}_{title.strip()[:80]}", body.strip()))
    return [(p, t) for p, t in out if t]


def parse_text(blob: bytes) -> list[tuple[str, str]]:
    return [("text", blob.decode("utf-8", errors="replace").strip())]


PARSERS = {
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
    "application/msword": parse_docx,
    "text/markdown": parse_markdown,
    "text/x-markdown": parse_markdown,
    "text/plain": parse_text,
}


def detect_and_parse(filename: str, mime: str, blob: bytes) -> list[tuple[str, str]]:
    """Pick a parser by mime type first, then filename extension, then text."""
    if mime in PARSERS:
        return PARSERS[mime](blob)
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return parse_pdf(blob)
    if fn.endswith(".docx"):
        return parse_docx(blob)
    if fn.endswith(".md") or fn.endswith(".markdown"):
        return parse_markdown(blob)
    return parse_text(blob)
