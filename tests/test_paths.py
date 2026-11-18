"""Tests for the shared filesystem helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scaling_surgical_ai import paths


def test_roots_follow_environment() -> None:
    """Required roots are exactly the directories named by the environment."""
    assert paths.data_root() == Path(os.environ["DATA_ROOT"])
    assert paths.output_root() == Path(os.environ["OUTPUT_ROOT"])
    assert paths.models_root() == Path(os.environ["MODELS_ROOT"])
    assert paths.hf_cache_root() == Path(os.environ["HF_HOME"])


def test_experiment_output_dir_appends_run_id() -> None:
    """A run id is joined onto the experiment name with one underscore."""
    root = Path(os.environ["OUTPUT_ROOT"])
    assert paths.experiment_output_dir("clustered_bootstrap") == (
        root / "clustered_bootstrap"
    )
    assert paths.experiment_output_dir("clustered_bootstrap", "42") == (
        root / "clustered_bootstrap_42"
    )


def test_logs_and_labels_sit_under_the_roots() -> None:
    """Logs and EEA labels use the documented subdirectories."""
    assert paths.logs_dir() == paths.output_root() / "logs"
    assert paths.eea_labels_root() == paths.data_root() / "eea" / "video_labels"
    assert paths.public_data_root() == paths.data_root() / "public"


def test_dataset_root_uses_override_then_default() -> None:
    """An override replaces the folder under the public data root."""
    assert paths.cholect50_root() == Path(os.environ["CHOLECT50_ROOT"])
    previous = os.environ.pop("CHOLECT50_ROOT")
    try:
        assert paths.cholect50_root() == paths.public_data_root() / "CholecT50"
    finally:
        os.environ["CHOLECT50_ROOT"] = previous


def test_hf_hub_snapshot_layout() -> None:
    """Snapshot paths follow the Hugging Face hub cache layout."""
    snapshot = paths.hf_hub_snapshot("google/gemma-3-27b-it", "abc")
    expected = (
        Path(os.environ["HF_HOME"])
        / "hub"
        / "models--google--gemma-3-27b-it"
        / "snapshots"
        / "abc"
    )
    assert snapshot == expected


def test_repo_files_resolve_from_the_checkout() -> None:
    """Shipped files are resolved from the repository checkout."""
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert paths.FIGURE_DATA_DIR == paths.REPO_ROOT / "figures" / "data"


def test_class_mapping_comes_from_the_environment() -> None:
    """The SDSC-EEA class mapping is read from its environment variable."""
    expected = Path(os.environ["SDSC_EEA_CLASS_MAPPING"])
    assert paths.eea_class_mapping_path() == expected


def test_missing_variable_raises() -> None:
    """An unset required variable raises before any path is used."""
    previous = os.environ.pop("DATA_ROOT")
    try:
        with pytest.raises(RuntimeError, match="DATA_ROOT"):
            paths.data_root()
    finally:
        os.environ["DATA_ROOT"] = previous
