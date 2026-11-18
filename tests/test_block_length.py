"""Tests for Politis-White block lengths on synthetic series."""

from __future__ import annotations

import numpy as np

from scaling_surgical_ai.block_length import (
    MAX_BLOCK_LENGTH,
    MIN_BLOCK_LENGTH,
    select_block_lengths,
)


def test_short_or_constant_series_uses_the_minimum() -> None:
    """Fewer than four frames, or zero variance, selects length 1."""
    short = {"clip": np.array([0, 1, 2])}
    constant = [np.array([1, 1, 1], dtype=np.int8)]
    assert select_block_lengths(short, constant) == {"clip": MIN_BLOCK_LENGTH}

    varying_but_short = [np.array([0, 1, 0], dtype=np.int8)]
    assert select_block_lengths(short, varying_but_short) == {"clip": MIN_BLOCK_LENGTH}


def test_no_models_uses_the_minimum_for_every_cluster() -> None:
    """With no correctness vectors, every cluster gets the minimum length."""
    clusters = {
        "a": np.arange(6),
        "b": np.arange(6, 10),
    }
    assert select_block_lengths(clusters, []) == {
        "a": MIN_BLOCK_LENGTH,
        "b": MIN_BLOCK_LENGTH,
    }


def test_block_length_is_deterministic_and_inside_bounds() -> None:
    """A varying series yields one integer length inside [1, n]."""
    indices = np.arange(32)
    correctness = [np.array([i % 2 for i in range(32)], dtype=np.int8)]
    first = select_block_lengths({"clip": indices}, correctness)
    second = select_block_lengths({"clip": indices}, correctness)
    length = first["clip"]
    assert first == second
    assert isinstance(length, int)
    assert MIN_BLOCK_LENGTH <= length <= min(32, MAX_BLOCK_LENGTH)


def test_lengths_follow_the_across_model_mean() -> None:
    """Two complementary models average to a constant series of length 1.

    Model A is 1 on even frames and model B is 1 on odd frames, so the
    across-model mean is 0.5 everywhere and the variance is zero.
    """
    indices = np.arange(8)
    even = np.array([i % 2 == 0 for i in range(8)], dtype=np.int8)
    odd = np.array([i % 2 == 1 for i in range(8)], dtype=np.int8)
    lengths = select_block_lengths({"clip": indices}, [even, odd])
    assert lengths == {"clip": MIN_BLOCK_LENGTH}
