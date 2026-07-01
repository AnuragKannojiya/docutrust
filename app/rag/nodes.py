"""CRAG graph nodes.

These are the building blocks of the LangGraph :class:`StateGraph`. Each
function takes a :class:`GraphState` and returns a partial update. They are
intentionally small and side-effect-light — the actual orchestration happens
in :func:`app.rag.graph.build_graph` and the trace recording in
:func:`app.rag.runner.run_crag`.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.embeddings import Embeddings

from app.db.mongo import MongoStore
from app.rag.grader import CrossEncoderGrader
from app.rag.prompts import GENERATION_SYSTEM, REWRITE_PROMPT
from app.rag.state import GradedChunk, GraphState, RetrievedChunk
from app.rag.web import make_web_chunk_id, web_search
from app.vectors.base import VectorStore

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


# ---- retrieve ---------------------------------------------------------------

async def retrieve(
    state: GraphState,
    *,
    settings,
    mongo: MongoStore,
    vector_store: VectorStore,
    embedder: Embeddings,
) -> list[RetrievedChunk]:
    """Embed the question, hit the vector store, hydrate chunk text from Mongo."""
    question = state.get("question") or ""
    if not question:
        return []
    vec = await embedder.aembed_query(question)
    hits = vector_store.search(vec, k=settings.retrieval_k)
    if not hits:
        return []
    ids = [h.chunk_id for h in hits]
    meta_by_id: dict[str, dict] = {}
    cursor = mongo.chunks_meta.find({"_id": {"$in": ids}})
    async for d in cursor:
        meta_by_id[d["_id"]] = d
    out: list[RetrievedChunk] = []
    for h in hits:
        m = meta_by_id.get(h.chunk_id)
        if m is None:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=h.chunk_id,
                doc_id=m["doc_id"],
                section_path=m.get("section_path", ""),
                text=m.get("text", ""),
                score=h.score,
                source=m.get("source", "corpus"),
                url=m.get("url"),
            )
        )
    return out


# ---- grade ------------------------------------------------------------------

def grade_chunks(
    state: GraphState,
    *,
    grader: CrossEncoderGrader,
) -> list[GradedChunk]:
    """Score each retrieved chunk against the question with the cross-encoder."""
    question = state.get("rewritten_question") or state.get("question") or ""
    retrieved: list[RetrievedChunk] = state.get("retrieved") or []
    if not retrieved:
        return []
    docs = [r["text"] for r in retrieved]
    scores = grader.score(question, docs)
    out: list[GradedChunk] = []
    for r, s in zip(retrieved, scores):
        out.append(
            GradedChunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                section_path=r["section_path"],
                text=r["text"],
                score=r["score"],
                graded_score=s,
                source=r["source"],
                url=r.get("url"),
            )
        )
    return out


# ---- rewrite ----------------------------------------------------------------

async def rewrite(state: GraphState, *, chat) -> str:
    """Rewrite the question for a web search via the chat LLM."""
    question = state["question"]
    prompt = REWRITE_PROMPT.format(question=question)
    resp = await chat.ainvoke(prompt)
    return resp.content.strip().strip('"').strip()


# ---- web_search -------------------------------------------------------------

async def do_web_search(state: GraphState, *, settings) -> list[RetrievedChunk]:
    """Run the web fallback. Returns a list of chunks the generator can use."""
    query = state.get("rewritten_question") or state["question"]
    raw = web_search(query, settings)
    out: list[RetrievedChunk] = []
    for i, r in enumerate(raw):
        text = (r.get("body") or "").strip()
        if not text:
            continue
        domain = r.get("domain") or ""
        cid = make_web_chunk_id(domain) if domain else f"web::{i}"
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=cid,
                section_path=r.get("title") or domain or "web",
                text=text,
                score=0.0,
                source="web",
                url=r.get("url"),
            )
        )
    return out


# ---- generate ---------------------------------------------------------------

def _format_context(chunks: list[GradedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, 1):
        marker = c["chunk_id"]
        src = c.get("source", "corpus")
        url = c.get("url")
        header = f"[{i}] {marker} (source: {src}"
        if url:
            header += f", url: {url}"
        header += ")"
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n".join(blocks)


async def generate(state: GraphState, *, chat) -> tuple[Optional[str], list[dict]]:
    """Generate an answer with citations or return None if refused."""
    docs: list[GradedChunk] = state.get("documents") or []
    if not docs:
        return None, []
    context = _format_context(docs)
    system = GENERATION_SYSTEM.format(context=context)
    question = state["question"]
    resp = await chat.ainvoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
    )
    answer = resp.content.strip()
    if answer.upper().startswith("REFUSE"):
        return None, []

    citations: list[dict] = []
    seen: set[str] = set()
    by_id = {c["chunk_id"]: c for c in docs}
    for marker in CITATION_RE.findall(answer):
        marker = marker.strip()
        if marker in seen or marker not in by_id:
            continue
        seen.add(marker)
        c = by_id[marker]
        citations.append(
            {
                "chunk_id": marker,
                "doc_id": c["doc_id"],
                "section_path": c["section_path"],
                "quote": (c["text"][:300] + "…") if len(c["text"]) > 300 else c["text"],
                "source": c.get("source", "corpus"),
                "url": c.get("url"),
            }
        )
    return answer, citations
