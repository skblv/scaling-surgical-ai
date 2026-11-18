"""Per-cluster block length selection for the moving-block bootstrap.

Block length is chosen model-agnostically from the per-frame mean correctness
across all available models on a given video, then passed through the
Politis-White (2004) automatic optimal block-length selector. A single L_k is
attached to each cluster and reused for every model (necessary for the paired
bootstrap interpretation).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from arch.bootstrap import optimal_block_length

MIN_BLOCK_LENGTH = 1
MAX_BLOCK_LENGTH = 5_000


def _politis_white_l(series: np.ndarray) -> int:
    """Run Politis-White on a single 1D series and return an int block length.

    The arch implementation returns recommendations for both the stationary
    bootstrap (geometric block lengths) and the circular block bootstrap
    (constant block lengths). We use the circular block recommendation
    because the moving-block bootstrap implemented in
    ``scaling_surgical_ai.bootstrap`` uses constant-length blocks.

    Args:
        series: 1D float array, length n_k.

    Returns:
        Integer block length in [MIN_BLOCK_LENGTH, n_k].
    """
    n = series.shape[0]
    if n < 4 or float(np.std(series)) == 0.0:
        return MIN_BLOCK_LENGTH
    block_lengths_df = optimal_block_length(series.astype(np.float64))
    raw = float(block_lengths_df["circular"].iloc[0])
    if not np.isfinite(raw) or raw <= 0:
        return MIN_BLOCK_LENGTH
    rounded = int(np.ceil(raw))
    return int(min(max(rounded, MIN_BLOCK_LENGTH), min(n, MAX_BLOCK_LENGTH)))


def select_block_lengths(
    cluster_to_ordered_indices: Dict[str, np.ndarray],
    correctness_per_model: List[np.ndarray],
) -> Dict[str, int]:
    """Pick one block length per cluster from across-model mean correctness.

    For each cluster k, we align the per-frame correctness from every model
    on the cluster's time-ordered indices, average them across models to get
    a model-agnostic per-frame "ease" sequence, and run Politis-White on it.

    Args:
        cluster_to_ordered_indices: Mapping cluster_id -> length-n_k int array
            of global frame indices ordered by within-cluster time.
        correctness_per_model: List of length-N int8 0/1 correctness vectors,
            one per model on the same N-row prediction table.

    Returns:
        Mapping cluster_id -> integer block length L_k.
    """
    if not correctness_per_model:
        return {cid: MIN_BLOCK_LENGTH for cid in cluster_to_ordered_indices}

    stacked = np.stack(correctness_per_model, axis=0).astype(np.float64)
    mean_correctness = stacked.mean(axis=0)

    block_lengths = {}
    for cluster_id, ordered in cluster_to_ordered_indices.items():
        series = mean_correctness[ordered]
        block_lengths[cluster_id] = _politis_white_l(series)

    return block_lengths
