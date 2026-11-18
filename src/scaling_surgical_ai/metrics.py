"""Set-based metrics recomputed from resampled frame indices.

The resampled indices are applied to the stored predictions and ground
truth multi-label sets and the metric is recomputed on the resampled
multiset of frames. Correctness vectors are never used as the source of
truth for non-trivial metrics; they only appear as inputs to block-length
estimation.

Metrics implemented:

    - Exact match accuracy
    - Mean Jaccard
    - Per-tool TP / FP / FN -> precision / recall / F1
    - Macro-averaged F1
    - Micro-averaged F1
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def encode_multilabel(
    label_lists: List[List[str]],
    tool_index: Dict[str, int],
) -> np.ndarray:
    """Encode a list of tool-name lists into a 0/1 matrix.

    Args:
        label_lists: Length-N list, each entry is a list of tool name strings.
        tool_index: Mapping tool_name -> column index, length T.

    Returns:
        (N, T) uint8 array with 1 where the tool appears.
    """
    n_frames = len(label_lists)
    n_tools = len(tool_index)
    matrix = np.zeros((n_frames, n_tools), dtype=np.uint8)
    for row_idx, tools in enumerate(label_lists):
        for tool in tools:
            col = tool_index.get(tool)
            if col is not None:
                matrix[row_idx, col] = 1
    return matrix


def exact_match_from_indices(
    indices: np.ndarray,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
) -> float:
    """Exact-match accuracy on the resampled frames.

    Args:
        indices: Length-N* array of resampled frame indices.
        pred_matrix: Pre-encoded (N, T) prediction matrix.
        gt_matrix: Pre-encoded (N, T) ground truth matrix.

    Returns:
        Mean exact-match accuracy in [0, 1].
    """
    if indices.size == 0:
        return 0.0
    matches = np.all(pred_matrix[indices] == gt_matrix[indices], axis=1)
    return float(matches.mean())


def jaccard_from_indices(
    indices: np.ndarray,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
) -> float:
    """Mean per-frame Jaccard similarity on the resampled frames.

    Frames where prediction and ground truth are both empty are counted as
    Jaccard = 1, matching the existing evaluation convention.

    Args:
        indices: Length-N* array of resampled frame indices.
        pred_matrix: Pre-encoded (N, T) prediction matrix.
        gt_matrix: Pre-encoded (N, T) ground truth matrix.

    Returns:
        Mean Jaccard similarity in [0, 1].
    """
    if indices.size == 0:
        return 0.0
    p = pred_matrix[indices]
    g = gt_matrix[indices]
    inter = np.sum(p & g, axis=1)
    union = np.sum(p | g, axis=1)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return float(jaccard.mean())


def per_tool_counts_from_indices(
    indices: np.ndarray,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
) -> np.ndarray:
    """Per-tool TP / FP / FN / TN counts on the resampled frames.

    Args:
        indices: Length-N* array of resampled frame indices.
        pred_matrix: Pre-encoded (N, T) prediction matrix.
        gt_matrix: Pre-encoded (N, T) ground truth matrix.

    Returns:
        (T, 4) int64 array with columns [TP, FP, FN, TN].
    """
    if indices.size == 0:
        return np.zeros((pred_matrix.shape[1], 4), dtype=np.int64)
    p = pred_matrix[indices].astype(np.int64)
    g = gt_matrix[indices].astype(np.int64)
    tp = np.sum((p == 1) & (g == 1), axis=0)
    fp = np.sum((p == 1) & (g == 0), axis=0)
    fn = np.sum((p == 0) & (g == 1), axis=0)
    tn = np.sum((p == 0) & (g == 0), axis=0)
    return np.stack([tp, fp, fn, tn], axis=1)


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """Per-tool F1 from TP / FP / FN counts.

    Tools with no positives in either prediction or ground truth get F1 = 0
    by the standard convention to avoid undefined-metric inflation.

    Args:
        tp: Length-T int array of true positives.
        fp: Length-T int array of false positives.
        fn: Length-T int array of false negatives.

    Returns:
        Length-T float array of F1 scores.
    """
    denom = (2 * tp + fp + fn).astype(np.float64)
    f1 = np.where(denom > 0, (2 * tp) / np.maximum(denom, 1), 0.0)
    return f1.astype(np.float64)


def macro_f1_from_indices(
    indices: np.ndarray,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Macro-averaged F1 across tools.

    Args:
        indices: Length-N* array of resampled frame indices.
        pred_matrix: Pre-encoded (N, T) prediction matrix.
        gt_matrix: Pre-encoded (N, T) ground truth matrix.
        valid_mask: Optional length-T bool array selecting which tools to
            include in the macro mean.

    Returns:
        Mean per-tool F1.
    """
    counts = per_tool_counts_from_indices(indices, pred_matrix, gt_matrix)
    f1 = f1_from_counts(counts[:, 0], counts[:, 1], counts[:, 2])
    if valid_mask is not None:
        f1 = f1[valid_mask]
    if f1.size == 0:
        return 0.0
    return float(f1.mean())


def micro_f1_from_indices(
    indices: np.ndarray,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
) -> float:
    """Micro-averaged F1 (pooled TP/FP/FN across tools).

    Args:
        indices: Length-N* array of resampled frame indices.
        pred_matrix: Pre-encoded (N, T) prediction matrix.
        gt_matrix: Pre-encoded (N, T) ground truth matrix.

    Returns:
        Micro-averaged F1.
    """
    counts = per_tool_counts_from_indices(indices, pred_matrix, gt_matrix)
    tp = float(counts[:, 0].sum())
    fp = float(counts[:, 1].sum())
    fn = float(counts[:, 2].sum())
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def per_frame_correctness(
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
) -> np.ndarray:
    """Per-frame exact-match correctness vector (used only for block length).

    Args:
        pred_matrix: (N, T) prediction matrix.
        gt_matrix: (N, T) ground truth matrix.

    Returns:
        Length-N int8 0/1 vector of exact-match indicators.
    """
    return np.all(pred_matrix == gt_matrix, axis=1).astype(np.int8)
