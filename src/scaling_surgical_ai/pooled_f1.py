"""Pooled micro-F1 and exact match over a fixed tool set.

These are the scores behind the zero-shot scaling figure and the
cross-dataset figure: true positives, false positives and false negatives
are pooled over frames and over the tools in the set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

EEA_15_TOOLS = [
    "Suction",
    "Curette",
    "Scissor",
    "Rongeur",
    "Rhoton Dissector",
    "Bipolar Forceps",
    "Grasper",
    "Surgicel",
    "Suction Coagulator",
    "Surgical Knife",
    "Doppler",
    "Drill",
    "Sonopet- pineapple tip",
    "Cotton Patty",
    "Straight Forceps",
]


def micro_f1_em(
    gt_by_key: Dict[str, List[str]],
    pred_by_key: Dict[str, List[str]],
    tools: List[str],
) -> Tuple[float, float, int]:
    """Micro F1 and exact match over ``tools`` on the model's covered frames.

    Args:
        gt_by_key: Frame key to ground-truth tool names.
        pred_by_key: Frame key to predicted tool names.
        tools: Tool names included in the score.

    Returns:
        Micro F1, exact match, and the number of covered frames, as percents
        and a count.
    """
    tool_set = set(tools)
    keys = [k for k in pred_by_key if k in gt_by_key]
    tp = fp = fn = exact = 0
    for k in keys:
        g = set(gt_by_key[k]) & tool_set
        p = set(pred_by_key[k]) & tool_set
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        if g == p:
            exact += 1
    denom = 2 * tp + fp + fn
    f1 = 100.0 * (2 * tp) / denom if denom > 0 else 0.0
    em = 100.0 * exact / len(keys) if keys else 0.0
    return f1, em, len(keys)


def yolo_micro_f1_em(yolo_dir: Path, tools: List[str]) -> Tuple[float, float, int]:
    """Micro F1 and exact match from a YOLO ``predictions_detailed.json``.

    Args:
        yolo_dir: Directory containing ``predictions_detailed.json``.
        tools: Tool names included in the score.

    Returns:
        Micro F1, exact match, and the number of records.
    """
    records = json.loads((yolo_dir / "predictions_detailed.json").read_text())
    gt_by_key = {str(i): r["ground_truth"] for i, r in enumerate(records)}
    pred_by_key = {str(i): r["predicted"] for i, r in enumerate(records)}
    return micro_f1_em(gt_by_key, pred_by_key, tools)


def majority_tool_baseline(gt_by_key, tools) -> Tuple[str, float]:
    """Micro F1 of always predicting the single most frequent tool.

    Args:
        gt_by_key: Frame key to ground-truth tool names.
        tools: Tool names included in the score.

    Returns:
        The majority tool and its micro F1.
    """
    tool_set = set(tools)
    support = {t: 0 for t in tools}
    for g in gt_by_key.values():
        for t in set(g) & tool_set:
            support[t] += 1
    majority = max(support, key=support.get)
    tp = fp = fn = 0
    for g in gt_by_key.values():
        truth = set(g) & tool_set
        pred = {majority}
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)
    denom = 2 * tp + fp + fn
    f1 = 100.0 * (2 * tp) / denom if denom > 0 else 0.0
    return majority, f1
