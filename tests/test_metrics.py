"""Tests for set metrics on a three-frame toy example."""

from __future__ import annotations

import numpy as np
import pytest

from scaling_surgical_ai.metrics import (
    encode_multilabel,
    exact_match_from_indices,
    jaccard_from_indices,
    macro_f1_from_indices,
    micro_f1_from_indices,
    per_frame_correctness,
)

TOOL_INDEX = {"drill": 0, "suction": 1}
PREDICTED = [["drill", "suction"], ["suction"], []]
TRUTH = [["drill", "suction"], ["drill"], ["suction"]]


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    """Encode the toy predictions and ground truth.

    Returns:
        Prediction matrix and ground-truth matrix, both shape (3, 2).
    """
    pred = encode_multilabel(PREDICTED, TOOL_INDEX)
    truth = encode_multilabel(TRUTH, TOOL_INDEX)
    return pred, truth


def test_encode_multilabel_ignores_unknown_tools() -> None:
    """Known tools become columns and an unknown name is dropped."""
    matrix = encode_multilabel([["drill", "missing"], []], TOOL_INDEX)
    assert matrix.dtype == np.uint8
    assert matrix.tolist() == [[1, 0], [0, 0]]


def test_exact_match_and_micro_f1_on_three_frames() -> None:
    """Exact match is 1/3 and micro-F1 is 4/7 on the toy frames.

    Frame 0 matches both tools. Frame 1 adds a false-positive suction and
    misses drill. Frame 2 misses suction. That is TP=2, FP=1, FN=2.
    """
    pred, truth = _matrices()
    indices = np.arange(3)
    assert exact_match_from_indices(indices, pred, truth) == 1.0 / 3.0
    assert micro_f1_from_indices(indices, pred, truth) == 4.0 / 7.0


def test_jaccard_and_macro_f1_on_three_frames() -> None:
    """Jaccard averages to 1/3 and macro-F1 to 7/12.

    Per-tool F1 is 2/3 for drill (TP=1, FN=1) and 1/2 for suction
    (TP=1, FP=1, FN=1).
    """
    pred, truth = _matrices()
    indices = np.arange(3)
    assert jaccard_from_indices(indices, pred, truth) == 1.0 / 3.0
    assert macro_f1_from_indices(indices, pred, truth) == pytest.approx(7.0 / 12.0)


def test_duplicated_correct_frame_is_perfect() -> None:
    """Resampling the matching frame twice scores 1 on both metrics."""
    pred, truth = _matrices()
    indices = np.array([0, 0])
    assert exact_match_from_indices(indices, pred, truth) == 1.0
    assert micro_f1_from_indices(indices, pred, truth) == 1.0


def test_empty_resample_is_zero() -> None:
    """An empty index set does not divide by zero."""
    pred, truth = _matrices()
    indices = np.array([], dtype=np.int64)
    assert exact_match_from_indices(indices, pred, truth) == 0.0
    assert micro_f1_from_indices(indices, pred, truth) == 0.0
    assert jaccard_from_indices(indices, pred, truth) == 0.0


def test_per_frame_correctness_matches_exact_set_equality() -> None:
    """The block-length vector is the per-frame exact-match indicator."""
    pred, truth = _matrices()
    assert per_frame_correctness(pred, truth).tolist() == [1, 0, 0]
