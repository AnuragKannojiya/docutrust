"""Structured trace logger for CRAG runs.

A ``Trace`` accumulates ``TraceStep`` records as the graph runs and is
persisted to MongoDB ``trace_logs`` when ``finish()`` is called.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.db.mongo import MongoStore
from app.db.models import Citation, TraceStep

log = logging.getLogger(__name__)


@dataclass
class _StepTimer:
    node: str
    started_at: datetime
    _t0: float = field(default_factory=time.perf_counter)
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    def set_output(self, output: dict) -> None:
        self.output = output

    def to_model(self) -> TraceStep:
        return TraceStep(
            node=self.node,
            input=self.input,
            output=self.output,
            duration_ms=int((time.perf_counter() - self._t0) * 1000),
            started_at=self.started_at,
            finished_at=datetime.now(tz=timezone.utc),
        )


class Trace:
    """A single /ask invocation's structured log."""

    def __init__(self, question: str, client_id: Optional[str]) -> None:
        self.id = str(uuid.uuid4())
        self.question = question
        self.client_id = client_id
        self.started_at = datetime.now(tz=timezone.utc)
        self.finished_at: Optional[datetime] = None
        self.steps: list[_StepTimer] = []
        self.final_answer: Optional[str] = None
        self.refused: bool = False
        self.refusal_reason: Optional[str] = None
        self.citations: list[Citation] = []

    def begin_step(self, node: str, *, input: Optional[dict] = None) -> _StepTimer:
        step = _StepTimer(
            node=node,
            started_at=datetime.now(tz=timezone.utc),
            input=input or {},
        )
        self.steps.append(step)
        return step

    def set_final(
        self,
        answer: Optional[str],
        refused: bool,
        refusal_reason: Optional[str] = None,
        citations: Optional[list[Citation]] = None,
    ) -> None:
        self.final_answer = answer
        self.refused = refused
        self.refusal_reason = refusal_reason
        self.citations = citations or []
        self.finished_at = datetime.now(tz=timezone.utc)


class TraceLogger:
    """Persists :class:`Trace` objects into the ``trace_logs`` collection."""

    def __init__(self, mongo: MongoStore) -> None:
        self.mongo = mongo

    async def save(self, trace: Trace) -> None:
        if trace.finished_at is None:
            trace.finished_at = datetime.now(tz=timezone.utc)
        doc = {
            "_id": trace.id,
            "client_id": trace.client_id,
            "question": trace.question,
            "final_answer": trace.final_answer,
            "refused": trace.refused,
            "refusal_reason": trace.refusal_reason,
            "citations": [c.model_dump() for c in trace.citations],
            "steps": [s.to_model().model_dump(mode="json") for s in trace.steps],
            "started_at": trace.started_at,
            "finished_at": trace.finished_at,
        }
        await self.mongo.trace_logs.insert_one(doc)
        log.info("Trace %s persisted", trace.id)

    async def get(self, trace_id: str) -> Optional[dict[str, Any]]:
        return await self.mongo.trace_logs.find_one({"_id": trace_id})
