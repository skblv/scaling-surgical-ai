"""Tests for the paired moving-block bootstrap on a toy frame table."""

from __future__ import annotations

import numpy as np
import pytest

from scaling_surgical_ai.bootstrap import (
    RANDOM_SEED,
    bca_ci,
    build_frame_table,
    hierarchical_paired_indices,
    percentile_ci,
    two_sided_pvalue,
)

CLUSTER_IDS = ["b", "a", "b", "a"]
WITHIN_CLUSTER_INDEX = [1, 0, 0, 1]


def test_frame_table_orders_clusters_by_time() -> None:
    """Frames in a cluster are sorted by time, and keys are sorted."""
    table = build_frame_table(CLUSTER_IDS, WITHIN_CLUSTER_INDEX)
    assert table.cluster_keys == ["a", "b"]
    assert table.cluster_to_ordered_indices["a"].tolist() == [1, 3]
    assert table.cluster_to_ordered_indices["b"].tolist() == [2, 0]
    assert table.n_frames == 4
    assert table.n_clusters == 2


def test_percentile_ci_matches_hand_computed_quantiles() -> None:
    """The 95% percentile interval of 1..5 is [1.1, 4.9], and low <= high."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower, upper = percentile_ci(values, 0.95)
    assert lower <= upper
    assert lower == pytest.approx(1.1)
    assert upper == pytest.approx(4.9)


def test_two_sided_pvalue_uses_the_plus_one_correction() -> None:
    """All-positive differences give 2 * 1 / (B + 1)."""
    differences = np.array([1.0, 2.0, 3.0])
    assert two_sided_pvalue(differences) == pytest.approx(0.5)
    mixed = np.array([-1.0, 0.0, 1.0])
    assert two_sided_pvalue(mixed) == pytest.approx(1.5)


def test_hierarchical_draw_is_deterministic_at_seed_42() -> None:
    """The same seed reproduces the resampled frame indices."""
    table = build_frame_table(CLUSTER_IDS, WITHIN_CLUSTER_INDEX)
    block_lengths = {"a": 2, "b": 1}
    first = hierarchical_paired_indices(
        table, block_lengths, np.random.default_rng(RANDOM_SEED)
    )
    second = hierarchical_paired_indices(
        table, block_lengths, np.random.default_rng(RANDOM_SEED)
    )
    assert RANDOM_SEED == 42
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert first.shape == (4,)


def test_bca_bounds_are_ordered_and_deterministic() -> None:
    """BCa bounds stay ordered and do not depend on a hidden seed."""
    point = 0.5
    bootstrap_values = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
    jackknife_values = np.array([0.45, 0.50, 0.55, 0.60])
    first = bca_ci(point, bootstrap_values, jackknife_values, 0.95)
    second = bca_ci(point, bootstrap_values, jackknife_values, 0.95)
    assert first == second
    assert first[0] <= first[1]
