"""Top-level runner that orchestrates the CRAG flow and records trace steps.

This is what :func:`app.api.routes_ask.ask` calls. It uses the same
node-level building blocks the LangGraph in :mod:`app.rag.graph` is built
from, but orchestrates them imperatively so we can record a clean trace step
per node and refuse cleanly when the generator fails to cite.

The LangGraph compiled graph is still available on ``app.state.graph`` for
the unit tests in ``tests/test_graph.py`` to exercise the wiring.
"""
from __future__ import annotations

import logging
from typing import Optional

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
from app.rag.state import GradedChunk, GraphState
from app.tracing.logger import Trace

log = logging.getLogger(__name__)


async def run_crag(
    *,
    question: str,
    client_id: Optional[str],
    settings: Settings,
    mongo: MongoStore,
    chat,
    embedder,
    graph,  # noqa: ARG001 — kept for symmetry with API; not used in the imperative path
    trace: Trace,
    grader: CrossEncoderGrader,
) -> dict:
    """Imperative CRAG run with per-step tracing and a strict-citation gate."""
    state: GraphState = GraphState(question=question)

    # ---- 1) retrieve ----------------------------------------------------
    step = trace.begin_step("retrieve", input={"question": question, "client_id": client_id})
    retrieved = await retrieve(
        state,
        settings=settings,
        mongo=mongo,
        vector_store=graph._docutrust_vector_store,  # type: ignore[attr-defined]
        embedder=embedder,
    )
    state["retrieved"] = retrieved
    step.set_output({"count": len(retrieved), "ids": [r["chunk_id"] for r in retrieved]})

    # ---- 2) grade -------------------------------------------------------
    step = trace.begin_step("grade", input={"count": len(retrieved)})
    graded = grade_chunks(state, grader=grader)
    relevant = [g for g in graded if grader.is_relevant(g["graded_score"])]
    state["relevant"] = relevant
    state["graded"] = graded
    state["documents"] = relevant
    step.set_output(
        {
            "total_graded": len(graded),
            "relevant_count": len(relevant),
            "graded_scores": [round(g["graded_score"], 3) for g in graded],
        }
    )

    # ---- 3) decide → rewrite + web if needed -----------------------------
    if not relevant:
        step = trace.begin_step("rewrite", input={"question": question})
        rewritten = await rewrite(state, chat=chat)
        state["rewritten_question"] = rewritten
        step.set_output({"rewritten_question": rewritten})

        step = trace.begin_step("web_search", input={"query": rewritten})
        web_hits = await do_web_search(state, settings=settings)
        state["web_results"] = web_hits
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
        state["documents"] = web_graded
        step.set_output(
            {
                "count": len(web_hits),
                "urls": [h.get("url") for h in web_hits],
            }
        )

    # ---- 4) generate (citation-enforced) --------------------------------
    step = trace.begin_step("generate", input={"doc_count": len(state.get("documents") or [])})
    answer, citations = await generate(state, chat=chat)
    if not citations:
        refusal_reason = "no citations in generated answer"
        step.set_output(
            {
                "answer_preview": (answer or "")[:200],
                "refused": True,
                "reason": refusal_reason,
            }
        )
        return {
            "answer": None,
            "refused": True,
            "refusal_reason": refusal_reason,
            "citations": [],
        }
    step.set_output(
        {
            "answer_preview": (answer or "")[:200],
            "citation_count": len(citations),
        }
    )
    return {
        "answer": answer,
        "refused": False,
        "refusal_reason": None,
        "citations": citations,
    }
