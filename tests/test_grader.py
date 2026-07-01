"""Tests for the cross-encoder grader wrapper."""
from __future__ import annotations

import math

import numpy as np

from app.rag.grader import CrossEncoderGrader
from tests.conftest import FakeGrader


def test_score_sigmoid_normalised():
    grader = CrossEncoderGrader(FakeGrader(), threshold=0.5)
    scores = grader.score("password rotation", ["rotate passwords every 90 days", "lunch menu"])
    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s <= 1.0
    # The first document mentions rotation/90 days, the second doesn't.
    assert scores[0] > scores[1]


def test_threshold_routes_relevant_vs_not():
    grader = CrossEncoderGrader(FakeGrader(), threshold=0.5)
    high = grader.score("password rotation", ["rotate passwords every 90 days"])[0]
    low = grader.score("password rotation", ["the cafeteria is open today"])[0]
    assert grader.is_relevant(high) is True
    assert grader.is_relevant(low) is False


def test_empty_documents_returns_empty():
    grader = CrossEncoderGrader(FakeGrader(), threshold=0.5)
    assert grader.score("anything", []) == []
