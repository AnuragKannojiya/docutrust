"""Cross-encoder relevance grader.

Wraps a sentence-transformers CrossEncoder model behind a single ``score()``
function that returns sigmoid-normalised values in [0, 1]. The threshold for
"relevant" is configured via ``Settings.grader_threshold``.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class CrossEncoderGrader:
    """Thin wrapper so the rest of the code doesn't import sentence-transformers."""

    def __init__(self, model, threshold: float) -> None:
        self.model = model
        self.threshold = threshold

    def score(self, question: str, documents: Sequence[str]) -> list[float]:
        """Return a normalised relevance score in [0, 1] for each document."""
        if not documents:
            return []
        pairs = [(question, d) for d in documents]
        raw = self.model.predict(pairs, show_progress_bar=False)
        # Some CrossEncoder models output one float per pair; older ones output
        # a 2-class softmax. Handle both.
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Take the "relevant" class probability (column 1) as the score.
            return [float(p) for p in arr[:, 1]]
        return [_sigmoid(float(x)) for x in arr.flatten()]

    def is_relevant(self, score: float) -> bool:
        return score >= self.threshold
