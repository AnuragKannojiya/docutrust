"""Compiled LangGraph for the CRAG pipeline.

The graph has a linear flow with one conditional edge:

    retrieve → grade → decide ──┬── generate → END
                               └── rewrite → web_search → generate → END

In practice we don't need the full LangGraph machinery to express this: the
real branching is "if no relevant chunks, then rewrite + web". We use the
graph for two reasons:

1. It makes each node independently testable via :func:`build_graph`.
2. It serialises neatly into the trace logger (one node = one step).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from sentence_transformers import CrossEncoder

from app.config import Settings
from app.db.mongo import MongoStore
from app.rag.grader import CrossEncoderGrader
from app.rag.nodes import (
    do_web_search,
    generate,
    grade_chunks,
    retrieve,
    rewrite,
)
from app.rag.state import GraphState
from app.vectors.base import VectorStore

log = logging.getLogger(__name__)


def build_graph(
    *,
    mongo: MongoStore,
    vector_store: VectorStore,
    chat_model: BaseChatModel,
    embedder: Embeddings,
    grader: CrossEncoder,
    settings: Settings,
) -> Any:
    """Build a StateGraph and compile it.

    Returns a compiled graph (callable that takes a state dict and returns the
    updated state) but also stashes the heavy dependencies on the returned
    object so :func:`app.rag.nodes.run_crag` can reach them.
    """
    grader_obj = CrossEncoderGrader(grader, settings.grader_threshold)

    async def node_retrieve(state: GraphState) -> dict:
        retrieved = await retrieve(
            state, settings=settings, mongo=mongo, vector_store=vector_store, embedder=embedder
        )
        return {"retrieved": retrieved}

    async def node_grade(state: GraphState) -> dict:
        graded = grade_chunks(state, grader=grader_obj)
        relevant = [g for g in graded if grader_obj.is_relevant(g["graded_score"])]
        return {"relevant": relevant, "documents": relevant}

    def decide(state: GraphState) -> str:
        return "generate" if state.get("relevant") else "rewrite"

    async def node_rewrite(state: GraphState) -> dict:
        rewritten = await rewrite(state, chat=chat_model)
        return {"rewritten_question": rewritten}

    async def node_web(state: GraphState) -> dict:
        web_hits = await do_web_search(state, settings=settings)
        from app.rag.state import GradedChunk

        web_graded: list[GradedChunk] = [
            GradedChunk(
                chunk_id=h["chunk_id"],
                doc_id=h["doc_id"],
                section_path=h["section_path"],
                text=h["text"],
                score=h["score"],
                graded_score=1.0,
                source="web",
                url=h.get("url"),
            )
            for h in web_hits
        ]
        return {"web_results": web_hits, "documents": web_graded}

    async def node_generate(state: GraphState) -> dict:
        answer, citations = await generate(state, chat=chat_model)
        if not citations:
            return {
                "generation": answer,
                "citations": [],
                "refused": True,
                "refusal_reason": "no citations in generated answer",
            }
        return {
            "generation": answer,
            "citations": citations,
            "refused": False,
            "refusal_reason": None,
        }

    g = StateGraph(GraphState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("rewrite", node_rewrite)
    g.add_node("web_search", node_web)
    g.add_node("generate", node_generate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "web_search")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)

    compiled = g.compile()
    # Stash heavy dependencies on the compiled object so the imperative
    # runner can reach them without re-deriving from ``app.state``.
    compiled._docutrust_grader = grader  # type: ignore[attr-defined]
    compiled._docutrust_grader_obj = grader_obj  # type: ignore[attr-defined]
    compiled._docutrust_vector_store = vector_store  # type: ignore[attr-defined]
    compiled._docutrust_mongo = mongo  # type: ignore[attr-defined]
    compiled._docutrust_embedder = embedder  # type: ignore[attr-defined]
    compiled._docutrust_chat = chat_model  # type: ignore[attr-defined]
    compiled._docutrust_settings = settings  # type: ignore[attr-defined]
    return compiled
