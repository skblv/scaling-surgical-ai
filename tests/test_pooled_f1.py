"""Tests for pooled micro-F1 and exact match on a three-frame toy example."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scaling_surgical_ai.pooled_f1 import (
    majority_tool_baseline,
    micro_f1_em,
    yolo_micro_f1_em,
)

TOOLS = ["Suction", "Drill", "Curette"]
TRUTH_BY_FRAME = {
    "frame_1": ["Suction", "Drill"],
    "frame_2": ["Suction"],
    "frame_3": [],
}
PREDICTED_BY_FRAME = {
    "frame_1": ["Suction"],
    "frame_2": ["Suction", "Curette", "Grasper"],
    "frame_3": [],
}
EXPECTED_MICRO_F1 = 100.0 * 4 / 6
EXPECTED_EXACT_MATCH = 100.0 / 3


def test_micro_f1_em_pools_counts_over_frames() -> None:
    """Two hits, one miss and one false alarm give F1 4/6; one frame is exact."""
    micro_f1, exact_match, frame_count = micro_f1_em(
        TRUTH_BY_FRAME, PREDICTED_BY_FRAME, TOOLS
    )
    assert micro_f1 == pytest.approx(EXPECTED_MICRO_F1)
    assert exact_match == pytest.approx(EXPECTED_EXACT_MATCH)
    assert frame_count == len(TRUTH_BY_FRAME)


def test_micro_f1_em_scores_only_covered_frames() -> None:
    """A frame without a prediction is left out of both scores."""
    covered = {"frame_3": []}
    micro_f1, exact_match, frame_count = micro_f1_em(TRUTH_BY_FRAME, covered, TOOLS)
    assert micro_f1 == 0.0
    assert exact_match == 100.0
    assert frame_count == 1


def test_majority_tool_baseline_predicts_the_most_frequent_tool() -> None:
    """Suction is in two of three frames; predicting it everywhere gives F1 4/6."""
    majority, micro_f1 = majority_tool_baseline(TRUTH_BY_FRAME, TOOLS)
    assert majority == "Suction"
    assert micro_f1 == pytest.approx(EXPECTED_MICRO_F1)


def test_yolo_micro_f1_em_reads_detailed_predictions(tmp_path: Path) -> None:
    """Records in ``predictions_detailed.json`` are scored like any other model."""
    records = [
        {"ground_truth": TRUTH_BY_FRAME[frame], "predicted": PREDICTED_BY_FRAME[frame]}
        for frame in TRUTH_BY_FRAME
    ]
    (tmp_path / "predictions_detailed.json").write_text(json.dumps(records))
    micro_f1, exact_match, frame_count = yolo_micro_f1_em(tmp_path, TOOLS)
    assert micro_f1 == pytest.approx(EXPECTED_MICRO_F1)
    assert exact_match == pytest.approx(EXPECTED_EXACT_MATCH)
    assert frame_count == len(records)
