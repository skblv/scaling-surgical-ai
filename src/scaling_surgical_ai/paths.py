"""Filesystem locations shared by every experiment.

Machine-specific roots come from environment variables (see ``.env.example``).
A missing variable raises immediately, so a job never writes to a default
location by accident. Files shipped with the repository are resolved relative
to the checkout, which assumes an editable install.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DATA_DIR = REPO_ROOT / "figures" / "data"

DATA_ROOT_VARIABLE = "DATA_ROOT"
OUTPUT_ROOT_VARIABLE = "OUTPUT_ROOT"
MODELS_ROOT_VARIABLE = "MODELS_ROOT"
HF_HOME_VARIABLE = "HF_HOME"
CHOLECT50_ROOT_VARIABLE = "CHOLECT50_ROOT"
PITVIS_ROOT_VARIABLE = "PITVIS_ROOT"
SURGVU_ROOT_VARIABLE = "SURGVU_ROOT"
PITVIS_RECORDED_ROOT_VARIABLE = "PITVIS_RECORDED_ROOT"
EEA_CLASS_MAPPING_VARIABLE = "SDSC_EEA_CLASS_MAPPING"


def env_path(variable: str) -> Path:
    """Read a path from a required environment variable.

    Args:
        variable: Name of the environment variable.

    Returns:
        The path stored in the variable.

    Raises:
        RuntimeError: If the variable is unset or empty.
    """
    value = os.environ.get(variable, "")
    if not value:
        raise RuntimeError(
            f"Environment variable {variable} is not set; see .env.example."
        )
    return Path(value)


def eea_class_mapping_path() -> Path:
    """Return the JSON file that maps SDSC-EEA label class ids to tool names.

    The file comes with the dataset. Its ``mapping`` entry holds one tool name
    per class id.
    """
    return env_path(EEA_CLASS_MAPPING_VARIABLE)


def data_root() -> Path:
    """Return the root directory that holds every input dataset."""
    return env_path(DATA_ROOT_VARIABLE)


def output_root() -> Path:
    """Return the root directory for experiment outputs, logs, and artifacts."""
    return env_path(OUTPUT_ROOT_VARIABLE)


def models_root() -> Path:
    """Return the directory with local Hugging Face model checkouts."""
    return env_path(MODELS_ROOT_VARIABLE)


def hf_cache_root() -> Path:
    """Return the persistent Hugging Face cache directory."""
    return env_path(HF_HOME_VARIABLE)


def hf_hub_snapshot(repo_id: str, revision: str) -> Path:
    """Return the cached snapshot directory of a Hugging Face model revision.

    Args:
        repo_id: Model identifier such as ``google/gemma-3-27b-it``.
        revision: Commit hash of the snapshot.

    Returns:
        The snapshot directory inside the Hugging Face hub cache.
    """
    folder = "models--" + repo_id.replace("/", "--")
    return hf_cache_root() / "hub" / folder / "snapshots" / revision


def eea_labels_root() -> Path:
    """Return the SDSC-EEA frames and YOLO labels directory."""
    return data_root() / "eea" / "video_labels"


def public_data_root() -> Path:
    """Return the parent directory of the three public datasets."""
    return data_root() / "public"


def _dataset_root(variable: str, folder: str) -> Path:
    """Return a public dataset root, honoring its override variable.

    Args:
        variable: Environment variable that overrides the default location.
        folder: Folder name under the public data root.

    Returns:
        The override when the variable is set, otherwise the default folder.
    """
    override = os.environ.get(variable, "")
    if override:
        return Path(override)
    return public_data_root() / folder


def cholect50_root() -> Path:
    """Return the CholecT50 dataset root."""
    return _dataset_root(CHOLECT50_ROOT_VARIABLE, "CholecT50")


def pitvis_root() -> Path:
    """Return the PitVis-2023 dataset root."""
    return _dataset_root(PITVIS_ROOT_VARIABLE, "PitVis2023")


def surgvu_root() -> Path:
    """Return the SurgVU dataset root."""
    return _dataset_root(SURGVU_ROOT_VARIABLE, "SurgVU")


def pitvis_recorded_root() -> Path:
    """Return the PitVis-2023 root written inside ``dataset.csv`` frame paths.

    The dataset table stores absolute frame paths. When the frames have moved
    since the table was written, the override variable holds the old root and
    the loaders rewrite it to the current root.

    Returns:
        The recorded root when the variable is set, otherwise the current
        PitVis-2023 root, which makes the rewrite a no-op.
    """
    override = os.environ.get(PITVIS_RECORDED_ROOT_VARIABLE, "")
    if override:
        return Path(override)
    return pitvis_root()


def logs_dir() -> Path:
    """Return the shared log directory."""
    return output_root() / "logs"


def experiment_output_dir(name: str, run_id: str | None = None) -> Path:
    """Return the output directory of an experiment.

    Args:
        name: Experiment output name, for example ``sdsc_eea_zero_shot_vlm``.
        run_id: Optional run identifier (for example a job ID) appended
            after an underscore.

    Returns:
        ``<OUTPUT_ROOT>/<name>`` or ``<OUTPUT_ROOT>/<name>_<run_id>``.
    """
    if run_id is None:
        return output_root() / name
    return output_root() / f"{name}_{run_id}"
