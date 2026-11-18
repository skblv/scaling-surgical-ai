"""Loaders for per-frame predictions across the four datasets and all model families.

Every loader returns a per-(dataset, model) `pandas.DataFrame` with the canonical
schema:

    frame_path: str  (unique row identifier, joined to the dataset CSV)
    cluster_id: str  (procedure_id / video_id / case_id)
    within_cluster_index: int  (time-order within cluster)
    pred_tools: List[str]
    gt_tools: List[str]

Parse failures from VLMs are mapped to ``pred_tools = []`` to match the
``experiments/sdsc_eea/recalculate_zero_shot_metrics.py`` policy. These tables
feed the hierarchical clustered bootstrap used for the paper's confidence
intervals.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from scaling_surgical_ai import paths

EEA_DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
EEA_ZERO_SHOT_VLM_DIR = paths.experiment_output_dir("sdsc_eea_zero_shot_vlm")
EEA_ZERO_SHOT_LLAMA_DIR = (
    paths.experiment_output_dir("sdsc_eea_zero_shot_llama_vision")
    / "llama-3.2-90b-vision_temp0_0"
)
EEA_ZERO_SHOT_MEDGEMMA_DIR = (
    paths.experiment_output_dir("sdsc_eea_zero_shot_medgemma")
    / "medgemma-27b-it_temp0_0"
)
EEA_ZERO_SHOT_LLAVA_DIR = (
    paths.experiment_output_dir("sdsc_eea_zero_shot_llava") / "llava-v1.5-13b_temp0_2"
)
EEA_ZERO_SHOT_QWEN3_235B_SPLIT_DIRS = [
    EEA_ZERO_SHOT_VLM_DIR / f"qwen3-vl-235b_temp0_0_split{i}" for i in range(1, 5)
]
EEA_LORA_JSON_HEAD_EVAL_GLOB = str(
    paths.output_root() / "sdsc_eea_lora_json_head_eval_*"
)
EEA_YOLO_MAPPING = paths.eea_class_mapping_path()
EEA_EXCLUDE_DIR_NAMES = {
    "OLD_Llama-3.2-11B-Vision_temp1_0",
    "Qwen3-VL-235B-A22B-Thinking_temp0_0",
    "qwen3-vl-235b_temp0_0",
    "kimi-k2.5_temp0_6",
    "gemma-4-31b-it_temp1_0",
}
EEA_EXCLUDE_DIR_NAMES.update({d.name for d in EEA_ZERO_SHOT_QWEN3_235B_SPLIT_DIRS})

PITVIS_DATASET_CSV = paths.pitvis_root() / "dataset.csv"
PITVIS_ZERO_SHOT_VLM_DIR = paths.experiment_output_dir("pitvis_zero_shot_vlm")
PITVIS_FRONTIER_VLM_DIR = paths.experiment_output_dir("pitvis_frontier_vlm_openrouter")
PITVIS_LORA_CLASSIFICATION_HEAD_GLOB = str(
    paths.output_root() / "pitvis_lora_classification_head_*"
)

CHOLECT_DATASET_CSV = paths.experiment_output_dir("cholect50_dataset") / "dataset.csv"
CHOLECT_ZERO_SHOT_VLM_DIR = paths.experiment_output_dir("cholect50_zero_shot_vlm")
CHOLECT_FRONTIER_VLM_DIR = paths.experiment_output_dir(
    "cholect50_frontier_vlm_openrouter"
)
CHOLECT_LORA_CLASSIFICATION_HEAD_GLOB = str(
    paths.output_root() / "cholect50_lora_classification_head_*"
)

SURGVU_DATASET_CSV = paths.surgvu_root() / "dataset.csv"
SURGVU_ZERO_SHOT_VLM_DIR = paths.experiment_output_dir("surgvu_zero_shot_vlm")
SURGVU_FRONTIER_VLM_DIR = paths.experiment_output_dir("surgvu_frontier_vlm_openrouter")
SURGVU_LORA_CLASSIFICATION_HEAD_GLOB = str(
    paths.output_root() / "surgvu_lora_classification_head_*"
)

PITVIS_TOOLS = sorted(
    [
        "bipolar_forceps",
        "cottle",
        "cup_forceps",
        "dural_scissors",
        "freer_elevator",
        "haemostatic_foam",
        "irrigation_syringe",
        "kerrisons",
        "micro_doppler",
        "nasal_cutting_forceps",
        "pituitary_rongeurs",
        "retractable_knife",
        "ring_curette",
        "spatula_dissector",
        "stealth_pointer",
        "suction",
        "surgical_drill",
        "tissue_glue",
    ]
)

CHOLECT_TOOLS = sorted(
    ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
)

EEA_FRAME_INDEX_RES = (
    re.compile(r"_frame_(\d+)\.png$"),
    re.compile(r"-(\d+)-mp4\.png$", re.IGNORECASE),
)


@dataclass(frozen=True)
class DatasetSpec:
    """All per-dataset constants needed by the bootstrap drivers.

    Attributes:
        name: Short dataset identifier ("eea", "pitvis", "cholect50", "surgvu").
        dataset_csv: Path to the canonical dataset.csv for the dataset.
        cluster_column: Name of the cluster identifier column in dataset.csv.
        valid_tools: Sorted list of canonical tool names.
    """

    name: str
    dataset_csv: Path
    cluster_column: str
    valid_tools: List[str]


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the DatasetSpec for one of the four supported datasets.

    Args:
        name: Dataset name ("eea", "pitvis", "cholect50", "surgvu").

    Returns:
        DatasetSpec with paths and tool list.
    """
    if name == "eea":
        with EEA_YOLO_MAPPING.open("r") as f:
            mapping = json.load(f)["mapping"]
        return DatasetSpec(
            name="eea",
            dataset_csv=EEA_DATASET_CSV,
            cluster_column="procedure_id",
            valid_tools=sorted(set(mapping.values())),
        )
    if name == "pitvis":
        return DatasetSpec(
            name="pitvis",
            dataset_csv=PITVIS_DATASET_CSV,
            cluster_column="video_id",
            valid_tools=list(PITVIS_TOOLS),
        )
    if name == "cholect50":
        return DatasetSpec(
            name="cholect50",
            dataset_csv=CHOLECT_DATASET_CSV,
            cluster_column="video_id",
            valid_tools=list(CHOLECT_TOOLS),
        )
    if name == "surgvu":
        return DatasetSpec(
            name="surgvu",
            dataset_csv=SURGVU_DATASET_CSV,
            cluster_column="video_id",
            valid_tools=_load_surgvu_tools(),
        )
    raise ValueError(f"Unknown dataset: {name}")


def _load_surgvu_tools() -> List[str]:
    """Load SurgVU canonical tool list from the dataset CSV header.

    Returns:
        Sorted unique tool name list parsed from any non-null `truth_label`
        across the val split.
    """
    frame_truth_df = pd.read_csv(SURGVU_DATASET_CSV, usecols=["split", "truth_label"])
    frame_truth_trainval_df = frame_truth_df[
        frame_truth_df["split"].isin({"train", "val"})
    ]
    tools = set()
    for raw in frame_truth_trainval_df["truth_label"].dropna():
        for tool in json.loads(raw):
            tools.add(tool)
    return sorted(tools)


def canonical_frame_key(frame_path: str) -> str:
    """Return a path-prefix-independent key for a frame.

    For all datasets, this is the last two path components
    (``<video_or_case_dir>/<basename>``). Different scripts in this codebase
    write prediction files using different absolute prefixes for the same
    underlying frame; the tail two components uniquely identify the frame.

    Args:
        frame_path: Absolute or relative frame path.

    Returns:
        ``"<parent_dir>/<basename>"`` string.
    """
    p = Path(frame_path)
    return f"{p.parent.name}/{p.name}"


def load_val_frame_table(spec: DatasetSpec) -> pd.DataFrame:
    """Load the validation split with cluster IDs and within-cluster time index.

    Args:
        spec: DatasetSpec describing the dataset.

    Returns:
        DataFrame with columns frame_key (canonical join key), frame_path
        (canonical absolute path from dataset.csv), cluster_id,
        within_cluster_index, gt_tools (List[str]).
    """
    columns = ["frame_path", "split", "truth_label", spec.cluster_column]
    if spec.name in {"pitvis", "cholect50", "surgvu"}:
        columns.append("frame_id")
    frame_dataset_df = pd.read_csv(spec.dataset_csv, usecols=columns)
    frame_dataset_val_df = frame_dataset_df[frame_dataset_df["split"] == "val"].copy()

    frame_dataset_val_df["cluster_id"] = frame_dataset_val_df[
        spec.cluster_column
    ].astype(str)
    if spec.name == "eea":
        frame_dataset_val_df["within_cluster_index"] = frame_dataset_val_df[
            "frame_path"
        ].map(_eea_frame_index)
    else:
        frame_dataset_val_df["within_cluster_index"] = frame_dataset_val_df[
            "frame_id"
        ].astype(int)

    frame_dataset_val_df["gt_tools"] = frame_dataset_val_df["truth_label"].apply(
        _parse_json_list
    )
    frame_dataset_val_df["frame_key"] = frame_dataset_val_df["frame_path"].apply(
        canonical_frame_key
    )
    cols = ["frame_key", "frame_path", "cluster_id", "within_cluster_index", "gt_tools"]
    return frame_dataset_val_df[cols].reset_index(drop=True)


def _parse_json_list(raw: str) -> List[str]:
    """Parse a JSON-encoded list of tool names; missing -> []."""
    if pd.isna(raw):
        return []
    return list(json.loads(raw))


def _eea_frame_index(frame_path: str) -> int:
    """Extract the integer frame index from an EEA frame path.

    Supports the two filename conventions in the dataset:
        - ``..._frame_<N>.png`` (videos extracted via the standard pipeline)
        - ``..._<id>-<N>-MP4.png`` / ``resized_<id>-<N>-MP4.png`` (older
          per-frame exports). The 5-digit number before the ``-MP4`` suffix
          encodes frame position in the source video.
    """
    s = str(frame_path)
    for pattern in EEA_FRAME_INDEX_RES:
        match = pattern.search(s)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Could not extract frame index from {frame_path}")


def load_results_csv(results_csv: Path) -> Dict[str, List[str]]:
    """Read frame_path -> predicted_tools list from a standard results.csv.

    Empty / NaN / unparseable ``predicted_tools`` values are mapped to ``[]``
    (treated as a parse failure). The "no silent failures" policy applies to
    structural problems (missing file, missing required columns); content-level
    parse failures are an explicit, expected case for these CSVs.

    Args:
        results_csv: Path to a `results.csv` produced by VLM eval scripts.

    Returns:
        Mapping frame_path -> sorted prediction list.
    """
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    frame_pred_df = pd.read_csv(
        results_csv,
        usecols=["frame_path", "predicted_tools"],
        dtype=str,
    )
    out: Dict[str, List[str]] = {}
    for row in frame_pred_df.itertuples(index=False):
        raw = row.predicted_tools
        if raw is None or (isinstance(raw, float) and pd.isna(raw)) or raw == "":
            out[str(row.frame_path)] = []
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Unparseable predicted_tools for {row.frame_path}: {exc}")
            out[str(row.frame_path)] = []
            continue
        out[str(row.frame_path)] = sorted(parsed) if isinstance(parsed, list) else []
    return out


def _load_eea_one_dir(model_dir: Path, val_frame_keys: set) -> Dict[str, List[str]]:
    """Load EEA predictions from a single directory, mapping parse failures to [].

    Two layouts are supported:
        - "Standard zero-shot" dirs that have ``debug_responses.jsonl`` (and possibly
          ``debug_responses_rank_*.jsonl``). Predictions are taken from
          ``results.csv`` only when the matching debug entry has
          ``parse_success=True``; all other attempted frames are mapped to
          ``predicted_tools = []`` (matches ``recalculate_zero_shot_metrics``).
        - "Single-CSV" dirs (Llama-3.2-90B / MedGemma) that have only
          ``results.csv``. The CSV already encodes parse failures as
          ``predicted_tools = "[]"``.

    Args:
        model_dir: Per-shard directory.
        val_frame_keys: Set of canonical val frame keys (parent_dir/basename).

    Returns:
        Mapping frame_key -> sorted predicted tool list, restricted to val.
    """
    debug_files = sorted(model_dir.glob("debug_responses_rank_*.jsonl"))
    if not debug_files:
        single_debug = model_dir / "debug_responses.jsonl"
        if single_debug.exists():
            debug_files = [single_debug]
    results_files = sorted(model_dir.glob("results_rank_*.csv"))
    if not results_files:
        results_files = [model_dir / "results.csv"]
    results_files = [f for f in results_files if f.exists()]
    if not results_files:
        raise FileNotFoundError(f"No results CSVs in {model_dir}")

    parsed_predictions: Dict[str, List[str]] = {}
    for results_file in results_files:
        for fp, tools in load_results_csv(results_file).items():
            parsed_predictions[canonical_frame_key(fp)] = tools

    if not debug_files:
        return {k: t for k, t in parsed_predictions.items() if k in val_frame_keys}

    final_predictions: Dict[str, List[str]] = {}
    for debug_file in debug_files:
        with debug_file.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                key = canonical_frame_key(entry["frame_path"])
                if key not in val_frame_keys:
                    continue
                if entry.get("parse_success") and key in parsed_predictions:
                    final_predictions[key] = parsed_predictions[key]
                else:
                    final_predictions[key] = []
    return final_predictions


def load_eea_zero_shot_predictions(
    model_dirs: List[Path],
    val_frame_keys: Sequence[str],
) -> Dict[str, List[str]]:
    """Load EEA zero-shot predictions across one or more shard directories.

    Multi-shard merging is used for the four ``qwen3-vl-235b_temp0_0_split{1..4}``
    directories that together cover the full val split.

    Args:
        model_dirs: One or more per-shard directories for the same canonical model.
        val_frame_keys: Canonical val frame keys to keep.

    Returns:
        Mapping frame_key -> sorted predicted tool list, merged across shards.
    """
    val_frame_keys_set = set(val_frame_keys)
    merged: Dict[str, List[str]] = {}
    for model_dir in model_dirs:
        per_dir = _load_eea_one_dir(model_dir, val_frame_keys_set)
        merged.update(per_dir)
    return merged


def discover_eea_zero_shot_models(
    input_base: Path = EEA_ZERO_SHOT_VLM_DIR,
) -> Dict[str, List[Path]]:
    """List paper-facing EEA zero-shot model directories with multi-shard merging.

    Standard models map to a single directory under ``sdsc_eea_zero_shot_vlm``.
    The following are special-cased to match
    ``experiments/sdsc_eea/appendix_tables.py`` and
    ``experiments/sdsc_eea/summarize_metrics.py``:
        - "Llama-3.2-90B-Vision": single dir under ``sdsc_eea_zero_shot_llama_vision``
        - "MedGemma-3-27B-it": single dir under ``sdsc_eea_zero_shot_medgemma``
        - "LLaVA-1.5-13B": single dir under ``sdsc_eea_zero_shot_llava``
        - "Qwen3-VL-235B-A22B-Thinking": four split dirs whose predictions
          must be concatenated.

    Args:
        input_base: Root directory of standard zero-shot dirs.

    Returns:
        Mapping canonical model name -> list of one or more directories.
    """
    models: Dict[str, List[Path]] = {}
    for child in sorted(input_base.iterdir()):
        if not child.is_dir() or child.name in EEA_EXCLUDE_DIR_NAMES:
            continue
        if child.name.startswith("OLD_"):
            continue
        debug_present = (child / "debug_responses.jsonl").exists() or list(
            child.glob("debug_responses_rank_*.jsonl")
        )
        if debug_present:
            models[child.name] = [child]

    if all(d.exists() for d in EEA_ZERO_SHOT_QWEN3_235B_SPLIT_DIRS):
        models["Qwen3-VL-235B-A22B-Thinking"] = list(
            EEA_ZERO_SHOT_QWEN3_235B_SPLIT_DIRS
        )

    if EEA_ZERO_SHOT_LLAMA_DIR.exists():
        models["Llama-3.2-90B-Vision"] = [EEA_ZERO_SHOT_LLAMA_DIR]
    if EEA_ZERO_SHOT_MEDGEMMA_DIR.exists():
        models["MedGemma-3-27B-it"] = [EEA_ZERO_SHOT_MEDGEMMA_DIR]
    if EEA_ZERO_SHOT_LLAVA_DIR.exists():
        models["LLaVA-1.5-13B"] = [EEA_ZERO_SHOT_LLAVA_DIR]

    models.pop("Llama-3.2-90B-Vision_temp1_0", None)
    return models


def discover_other_vlm_models(input_base: Path) -> Dict[str, Path]:
    """List VLM model directories that have a `results.csv` directly inside.

    Args:
        input_base: Root directory to scan (PitVis / CholecT50 / SurgVU).

    Returns:
        Mapping model_name -> model directory path.
    """
    if not input_base.exists():
        return {}
    return {
        child.name: child
        for child in sorted(input_base.iterdir())
        if child.is_dir() and (child / "results.csv").exists()
    }


def discover_lora_jobs(glob_pattern: str) -> Dict[str, Path]:
    """List LoRA job directories that have a `val_predictions_after_training.jsonl`.

    Args:
        glob_pattern: Glob string matching parent directories like
            ``<OUTPUT_ROOT>/pitvis_lora_classification_head_*``.

    Returns:
        Mapping `{script_name}_{job_id}` -> directory path, sorted by job ID.
    """
    matches = sorted(Path(p) for p in glob.glob(glob_pattern))
    jobs = {}
    for path in matches:
        if (path / "val_predictions_after_training.jsonl").exists():
            jobs[path.name] = path
    return jobs


def discover_eea_lora_jobs(
    glob_pattern: str = EEA_LORA_JSON_HEAD_EVAL_GLOB,
) -> Dict[str, Path]:
    """List EEA LoRA evaluation directories that have a `results.csv`.

    Args:
        glob_pattern: Glob string for ``sdsc_eea_lora_json_head_eval_*`` directories.

    Returns:
        Mapping job_dir_name -> directory path.
    """
    matches = sorted(Path(p) for p in glob.glob(glob_pattern))
    return {p.name: p for p in matches if (p / "results.csv").exists()}


def load_pitvis_cholect_surgvu_vlm_predictions(
    model_dir: Path,
    val_frame_keys: Sequence[str],
) -> Dict[str, List[str]]:
    """Load predictions from a per-model `results.csv` (PitVis/CholecT50/SurgVU VLMs).

    Args:
        model_dir: Per-model directory with a `results.csv`.
        val_frame_keys: Canonical val frame keys to keep.

    Returns:
        Mapping frame_key -> sorted predicted tool list, restricted to val.
    """
    raw = load_results_csv(model_dir / "results.csv")
    val_set = set(val_frame_keys)
    return {
        canonical_frame_key(fp): tools
        for fp, tools in raw.items()
        if canonical_frame_key(fp) in val_set
    }


def load_lora_jsonl_predictions(
    job_dir: Path,
    val_frame_keys: Sequence[str],
) -> Dict[str, List[str]]:
    """Load predictions from `val_predictions_after_training.jsonl`.

    Args:
        job_dir: LoRA job directory.
        val_frame_keys: Canonical val frame keys to keep.

    Returns:
        Mapping frame_key -> sorted predicted tool list.
    """
    predictions_file = job_dir / "val_predictions_after_training.jsonl"
    val_set = set(val_frame_keys)
    out = {}
    with predictions_file.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            key = canonical_frame_key(entry["frame_path"])
            if key not in val_set:
                continue
            out[key] = sorted(entry["predicted"])
    return out


def load_eea_lora_predictions(
    job_dir: Path,
    val_frame_keys: Sequence[str],
) -> Dict[str, List[str]]:
    """Load predictions from an EEA LoRA evaluation ``results.csv``.

    Args:
        job_dir: ``sdsc_eea_lora_json_head_eval_<JOB>`` directory.
        val_frame_keys: Canonical val frame keys to keep.

    Returns:
        Mapping frame_key -> sorted predicted tool list.
    """
    raw = load_results_csv(job_dir / "results.csv")
    val_set = set(val_frame_keys)
    return {
        canonical_frame_key(fp): tools
        for fp, tools in raw.items()
        if canonical_frame_key(fp) in val_set
    }
