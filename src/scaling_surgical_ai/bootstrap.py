"""Hierarchical paired moving-block bootstrap over frame indices.

The bootstrap operates exclusively on global frame indices: each replicate
returns a list of integer indices into a frame table. Metric functions are
free to consume those indices however they wish (recompute from stored
predictions and ground truth, never collapse to a correctness vector).

References:
    Kunsch, H. R. (1989). The jackknife and the bootstrap for general
        stationary observations. Annals of Statistics, 17(3), 1217-1241.
    Davison, A. C., & Hinkley, D. V. (1997). Bootstrap Methods and Their
        Applications. Cambridge University Press, sec. 8.2 (two-stage / nested).
    Field, C. A., & Welsh, A. H. (2007). Bootstrapping clustered data.
        Journal of the Royal Statistical Society B, 69(3), 369-390.
    Politis, D. N., & White, H. (2004). Automatic block-length selection for
        the dependent bootstrap. Econometric Reviews, 23(1), 53-70.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

RANDOM_SEED = 42


@dataclass(frozen=True)
class FrameTable:
    """Container for the frame-level data used by the bootstrap.

    Attributes:
        cluster_ids: Per-frame cluster (video / procedure) identifiers, length N.
        within_cluster_index: Per-frame integer index within its cluster
            (sorted by time), used for diagnostics only.
        cluster_to_ordered_indices: Mapping from cluster_id to the list of
            global frame indices in that cluster, ordered by time.
    """

    cluster_ids: np.ndarray
    within_cluster_index: np.ndarray
    cluster_to_ordered_indices: Dict[str, np.ndarray]

    @property
    def n_frames(self) -> int:
        """Total number of frames in the table."""
        return int(self.cluster_ids.shape[0])

    @property
    def cluster_keys(self) -> List[str]:
        """Sorted cluster identifiers (deterministic order)."""
        return sorted(self.cluster_to_ordered_indices.keys())

    @property
    def n_clusters(self) -> int:
        """Number of distinct clusters."""
        return len(self.cluster_to_ordered_indices)


def build_frame_table(
    cluster_ids: Sequence[str],
    within_cluster_index: Sequence[int],
) -> FrameTable:
    """Build a FrameTable from per-frame cluster IDs and time indices.

    Args:
        cluster_ids: Length-N sequence of cluster IDs (one per row of the
            prediction table).
        within_cluster_index: Length-N sequence of integer time indices used
            to sort frames within each cluster.

    Returns:
        FrameTable with deterministic per-cluster ordering.
    """
    cluster_array = np.asarray(cluster_ids)
    time_array = np.asarray(within_cluster_index, dtype=np.int64)
    if cluster_array.shape[0] != time_array.shape[0]:
        raise ValueError("cluster_ids and within_cluster_index length mismatch")

    cluster_to_ordered = {}
    for cid in np.unique(cluster_array):
        mask = cluster_array == cid
        global_idx = np.flatnonzero(mask)
        order = np.argsort(time_array[mask], kind="stable")
        cluster_to_ordered[str(cid)] = global_idx[order]

    return FrameTable(
        cluster_ids=cluster_array,
        within_cluster_index=time_array,
        cluster_to_ordered_indices=cluster_to_ordered,
    )


def hierarchical_paired_indices(
    table: FrameTable,
    block_lengths: Dict[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one hierarchical paired bootstrap replicate of frame indices.

    Stage 1 resamples K* = K clusters with replacement (each cluster is a
    surgery / video). Stage 2 applies a circular moving-block bootstrap of
    length L_k within each drawn cluster occurrence to preserve within-video
    serial correlation. Block start positions are drawn uniformly with
    replacement; the concatenated indices are trimmed to the original
    cluster length.

    The same draws (cluster sequence and block start positions) are reused
    across models when callers seed identically and call this function once
    per replicate, which produces the paired clustered bootstrap.

    Args:
        table: FrameTable describing per-frame cluster membership and order.
        block_lengths: Mapping cluster_id -> integer block length L_k.
        rng: Numpy random generator (seed externally for reproducibility).

    Returns:
        1D int64 array of resampled global frame indices.
    """
    keys = table.cluster_keys
    n_clusters = len(keys)
    drawn_cluster_positions = rng.integers(0, n_clusters, size=n_clusters)

    pieces = []
    for pos in drawn_cluster_positions:
        cluster_id = keys[int(pos)]
        ordered = table.cluster_to_ordered_indices[cluster_id]
        n_k = ordered.shape[0]
        if n_k == 0:
            continue
        L_k = max(1, int(block_lengths.get(cluster_id, 1)))
        L_k = min(L_k, n_k)
        n_blocks = math.ceil(n_k / L_k)
        starts = rng.integers(0, n_k, size=n_blocks)
        block_offsets = (starts[:, None] + np.arange(L_k)[None, :]) % n_k
        sampled = ordered[block_offsets].reshape(-1)[:n_k]
        pieces.append(sampled)

    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Standard percentile bootstrap CI.

    Args:
        values: 1D array of bootstrap replicate values.
        confidence: Coverage probability.

    Returns:
        (lower, upper) percentile bounds.
    """
    alpha = 1.0 - confidence
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lower, upper


def bca_ci(
    point_estimate: float,
    bootstrap_values: np.ndarray,
    jackknife_values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI.

    Args:
        point_estimate: The metric evaluated on the original (non-resampled) data.
        bootstrap_values: Length-B array of bootstrap replicates of the metric.
        jackknife_values: Length-K array of leave-one-cluster-out values of
            the metric (clusters are the resampling unit, not frames).
        confidence: Coverage probability.

    Returns:
        (lower, upper) BCa bounds.
    """
    n_boot = bootstrap_values.shape[0]
    proportion_below = float(np.mean(bootstrap_values < point_estimate))
    proportion_below = min(
        max(proportion_below, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1)
    )
    z0 = stats.norm.ppf(proportion_below)

    jk_mean = float(np.mean(jackknife_values))
    deviations = jk_mean - jackknife_values
    numerator = float(np.sum(deviations**3))
    denominator = 6.0 * (float(np.sum(deviations**2)) ** 1.5)
    acceleration = numerator / denominator if denominator > 0 else 0.0

    alpha = 1.0 - confidence
    z_lower = stats.norm.ppf(alpha / 2.0)
    z_upper = stats.norm.ppf(1.0 - alpha / 2.0)

    def adjust(z: float) -> float:
        """Map a normal quantile through the BCa adjustment.

        Args:
            z: Standard-normal quantile of the nominal tail.

        Returns:
            The adjusted percentile in (0, 1).
        """
        adjusted = z0 + (z0 + z) / (1.0 - acceleration * (z0 + z))
        return float(stats.norm.cdf(adjusted))

    p_lower = adjust(z_lower)
    p_upper = adjust(z_upper)
    lower = float(np.quantile(bootstrap_values, p_lower))
    upper = float(np.quantile(bootstrap_values, p_upper))
    return lower, upper


def two_sided_pvalue(differences: np.ndarray) -> float:
    """Two-sided bootstrap p-value for the null delta = 0.

    Args:
        differences: Length-B array of bootstrap replicates of (metric_A - metric_B).

    Returns:
        2 * min(P(delta <= 0), P(delta >= 0)), with the standard +1 / B+1 correction.
    """
    n = differences.shape[0]
    p_le = (np.sum(differences <= 0) + 1) / (n + 1)
    p_ge = (np.sum(differences >= 0) + 1) / (n + 1)
    return float(2.0 * min(p_le, p_ge))
