"""Hierarchical paired moving-block bootstrap for one dataset.

Produces the clustered confidence intervals that accompany the exact-match
results in the camera-ready paper.

For one dataset:

    1. Load the validation frame table (cluster_id, within-cluster time index, GT).
    2. Discover all eligible model output directories (zero-shot VLM,
       OpenRouter VLM, LoRA-finetuned).
    3. Load each model's per-frame predictions.
    4. Restrict to the intersection of frames present across every model so
       the bootstrap operates on a single canonical FrameTable shared by all
       models. This is what makes the bootstrap genuinely paired across
       models: per replicate, one draw of (cluster sequence + block-start
       offsets) is applied to every model's stored predictions and labels.
    5. Encode each model's predictions and ground truth as multi-label
       (N, T) matrices.
    6. Estimate one block length per cluster from the across-models mean
       per-frame correctness (Politis-White, model-agnostic).
    7. Run B = 10,000 hierarchical paired MBB replicates, evaluating every
       (model, metric) pair on each replicate.
    8. Persist headline summary, per-tool F1, paired Delta-CIs across model
       pairs, block lengths, and the raw replicate matrix for downstream
       plotting / sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scaling_surgical_ai import paths
from scaling_surgical_ai.block_length import select_block_lengths
from scaling_surgical_ai.bootstrap import (
    FrameTable,
    bca_ci,
    build_frame_table,
    hierarchical_paired_indices,
    percentile_ci,
    two_sided_pvalue,
)
from scaling_surgical_ai.metrics import (
    encode_multilabel,
    f1_from_counts,
    per_frame_correctness,
)
from scaling_surgical_ai.prediction_loaders import (
    CHOLECT_FRONTIER_VLM_DIR,
    CHOLECT_LORA_CLASSIFICATION_HEAD_GLOB,
    CHOLECT_ZERO_SHOT_VLM_DIR,
    EEA_LORA_JSON_HEAD_EVAL_GLOB,
    EEA_ZERO_SHOT_VLM_DIR,
    PITVIS_FRONTIER_VLM_DIR,
    PITVIS_LORA_CLASSIFICATION_HEAD_GLOB,
    PITVIS_ZERO_SHOT_VLM_DIR,
    SURGVU_FRONTIER_VLM_DIR,
    SURGVU_LORA_CLASSIFICATION_HEAD_GLOB,
    SURGVU_ZERO_SHOT_VLM_DIR,
    DatasetSpec,
    discover_eea_lora_jobs,
    discover_eea_zero_shot_models,
    discover_lora_jobs,
    discover_other_vlm_models,
    get_dataset_spec,
    load_eea_lora_predictions,
    load_eea_zero_shot_predictions,
    load_lora_jsonl_predictions,
    load_pitvis_cholect_surgvu_vlm_predictions,
    load_val_frame_table,
)

RANDOM_SEED = 42
N_BOOTSTRAP = 10_000
OUTPUT_ROOT = paths.experiment_output_dir("clustered_bootstrap")
DEFAULT_MIN_COVERAGE = 0.80


def discover_models_for_dataset(dataset_name: str) -> List[Tuple[str, str, List[Path]]]:
    """List all (family, model_name, model_dirs) tuples for a dataset.

    Multi-shard models (e.g. EEA Qwen3-VL-235B) get a list of >1 directories.

    Args:
        dataset_name: Dataset name.

    Returns:
        List of (family, model_name, directories). `family` is one of
        "zero_shot", "openrouter", "lora".
    """
    found: List[Tuple[str, str, List[Path]]] = []
    if dataset_name == "eea":
        for name, dirs in discover_eea_zero_shot_models(EEA_ZERO_SHOT_VLM_DIR).items():
            found.append(("zero_shot", name, dirs))
        for name, path in discover_eea_lora_jobs(EEA_LORA_JSON_HEAD_EVAL_GLOB).items():
            found.append(("lora", name, [path]))
    elif dataset_name == "pitvis":
        for name, path in discover_other_vlm_models(PITVIS_ZERO_SHOT_VLM_DIR).items():
            found.append(("zero_shot", name, [path]))
        for name, path in discover_other_vlm_models(PITVIS_FRONTIER_VLM_DIR).items():
            found.append(("openrouter", name, [path]))
        for name, path in discover_lora_jobs(
            PITVIS_LORA_CLASSIFICATION_HEAD_GLOB
        ).items():
            found.append(("lora", name, [path]))
    elif dataset_name == "cholect50":
        for name, path in discover_other_vlm_models(CHOLECT_ZERO_SHOT_VLM_DIR).items():
            found.append(("zero_shot", name, [path]))
        for name, path in discover_other_vlm_models(CHOLECT_FRONTIER_VLM_DIR).items():
            found.append(("openrouter", name, [path]))
        for name, path in discover_lora_jobs(
            CHOLECT_LORA_CLASSIFICATION_HEAD_GLOB
        ).items():
            found.append(("lora", name, [path]))
    elif dataset_name == "surgvu":
        for name, path in discover_other_vlm_models(SURGVU_ZERO_SHOT_VLM_DIR).items():
            found.append(("zero_shot", name, [path]))
        for name, path in discover_other_vlm_models(SURGVU_FRONTIER_VLM_DIR).items():
            found.append(("openrouter", name, [path]))
        for name, path in discover_lora_jobs(
            SURGVU_LORA_CLASSIFICATION_HEAD_GLOB
        ).items():
            found.append(("lora", name, [path]))
    else:
        raise ValueError(dataset_name)
    return found


def load_predictions_for_model(
    dataset_name: str,
    family: str,
    model_dirs: List[Path],
    val_frame_keys: List[str],
) -> Dict[str, List[str]]:
    """Dispatch to the right per-frame prediction loader.

    Args:
        dataset_name: Dataset name.
        family: Model family ("zero_shot", "openrouter", "lora").
        model_dirs: One or more model output directories (multi-shard supported
            for EEA zero-shot only).
        val_frame_keys: Canonical val frame keys.

    Returns:
        Mapping frame_key -> predicted tool list.
    """
    if dataset_name == "eea" and family == "zero_shot":
        return load_eea_zero_shot_predictions(model_dirs, val_frame_keys)
    if len(model_dirs) != 1:
        raise ValueError(
            f"Multi-shard not supported for dataset={dataset_name} family={family}"
        )
    model_dir = model_dirs[0]
    if dataset_name == "eea" and family == "lora":
        return load_eea_lora_predictions(model_dir, val_frame_keys)
    if family in {"zero_shot", "openrouter"}:
        return load_pitvis_cholect_surgvu_vlm_predictions(model_dir, val_frame_keys)
    if family == "lora":
        return load_lora_jsonl_predictions(model_dir, val_frame_keys)
    raise ValueError(f"Unknown family: {family}")


def encode_predictions_aligned(
    frame_table_common_df: pd.DataFrame,
    predictions_by_model: Dict[str, Dict[str, List[str]]],
    valid_tools: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Encode predictions and ground truth on the common val frame ordering.

    Args:
        frame_table_common_df: DataFrame restricted to the intersection of frames,
            in the canonical row order. Must contain `frame_path` and
            `gt_tools` columns.
        predictions_by_model: Mapping model_name -> frame_path -> pred list.
        valid_tools: Sorted canonical tool list.

    Returns:
        (pred_matrices_by_model, gt_matrix) where each pred matrix and the gt
        matrix are (N_common, T) uint8 with rows aligned to
        ``frame_table_common_df`` order.
    """
    tool_index = {tool: i for i, tool in enumerate(valid_tools)}
    gt_matrix = encode_multilabel(
        frame_table_common_df["gt_tools"].tolist(), tool_index
    )

    pred_matrices: Dict[str, np.ndarray] = {}
    common_frame_keys = frame_table_common_df["frame_key"].tolist()
    for model_name, predictions in predictions_by_model.items():
        ordered = [predictions[k] for k in common_frame_keys]
        pred_matrices[model_name] = encode_multilabel(ordered, tool_index)
    return pred_matrices, gt_matrix


def evaluate_all_metrics_for_model(
    indices: np.ndarray,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    valid_tools: List[str],
) -> Dict[str, float]:
    """Evaluate every metric for one model from a single counts matrix.

    Computes the (T, 4) per-tool TP/FP/FN/TN counts matrix on the resampled
    indices once, then derives exact match, Jaccard, micro / macro
    precision / recall / F1, and per-tool precision / recall / F1 plus
    raw counts (TP / FP / FN / support).

    Args:
        indices: Length-N* resampled frame indices.
        pred_matrix: (N, T) prediction matrix.
        gt_matrix: (N, T) ground truth matrix.
        valid_tools: Sorted canonical tool list.

    Returns:
        Mapping metric_name -> float.
    """
    headline_zero = {
        name: 0.0
        for name in (
            "exact_match",
            "jaccard",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    }
    if indices.size == 0:
        per_tool_zero = {
            f"{prefix}::{t}": 0.0
            for prefix in ("precision", "recall", "f1", "tp", "fp", "fn", "support")
            for t in valid_tools
        }
        return {**headline_zero, **per_tool_zero}

    p = pred_matrix[indices]
    g = gt_matrix[indices]
    matches = ~np.any(p != g, axis=1)
    exact_match = float(matches.mean())

    inter = np.sum((p & g), axis=1, dtype=np.int64)
    union = np.sum((p | g), axis=1, dtype=np.int64)
    jaccard_per_frame = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    jaccard_mean = float(jaccard_per_frame.mean())

    p64 = p.astype(np.int64)
    g64 = g.astype(np.int64)
    tp = np.sum((p64 == 1) & (g64 == 1), axis=0)
    fp = np.sum((p64 == 1) & (g64 == 0), axis=0)
    fn = np.sum((p64 == 0) & (g64 == 1), axis=0)

    precision_per_tool_denom = (tp + fp).astype(np.float64)
    precision_per_tool = np.where(
        precision_per_tool_denom > 0,
        tp / np.maximum(precision_per_tool_denom, 1),
        0.0,
    )
    recall_per_tool_denom = (tp + fn).astype(np.float64)
    recall_per_tool = np.where(
        recall_per_tool_denom > 0,
        tp / np.maximum(recall_per_tool_denom, 1),
        0.0,
    )
    f1_per_tool = f1_from_counts(tp, fp, fn)

    macro_precision = (
        float(precision_per_tool.mean()) if precision_per_tool.size > 0 else 0.0
    )
    macro_recall = float(recall_per_tool.mean()) if recall_per_tool.size > 0 else 0.0
    macro_f1 = float(f1_per_tool.mean()) if f1_per_tool.size > 0 else 0.0

    tp_sum = float(tp.sum())
    fp_sum = float(fp.sum())
    fn_sum = float(fn.sum())
    micro_precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    micro_recall = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    micro_denom = 2 * tp_sum + fp_sum + fn_sum
    micro_f1 = ((2 * tp_sum) / micro_denom) if micro_denom > 0 else 0.0

    metrics: Dict[str, float] = {
        "exact_match": exact_match,
        "jaccard": jaccard_mean,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    for tool, p_tool, r_tool, f1_tool, tp_tool, fp_tool, fn_tool in zip(
        valid_tools,
        precision_per_tool,
        recall_per_tool,
        f1_per_tool,
        tp,
        fp,
        fn,
    ):
        metrics[f"precision::{tool}"] = float(p_tool)
        metrics[f"recall::{tool}"] = float(r_tool)
        metrics[f"f1::{tool}"] = float(f1_tool)
        metrics[f"tp::{tool}"] = float(tp_tool)
        metrics[f"fp::{tool}"] = float(fp_tool)
        metrics[f"fn::{tool}"] = float(fn_tool)
        metrics[f"support::{tool}"] = float(tp_tool + fn_tool)
    return metrics


def compute_point_estimates(
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    valid_tools: List[str],
) -> Dict[str, float]:
    """Evaluate every metric on the original (un-resampled) data.

    Args:
        pred_matrix: (N, T) prediction matrix.
        gt_matrix: (N, T) ground truth matrix.
        valid_tools: Sorted canonical tool list.

    Returns:
        Mapping metric_name -> point estimate.
    """
    full_indices = np.arange(pred_matrix.shape[0], dtype=np.int64)
    return evaluate_all_metrics_for_model(
        full_indices, pred_matrix, gt_matrix, valid_tools
    )


def run_paired_replicates(
    table: FrameTable,
    block_lengths: Dict[str, int],
    matrices_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    valid_tools: List[str],
    metric_names: List[str],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run B replicates with one shared draw per replicate, applied to all models.

    The single per-replicate draw of (cluster sequence + block-start offsets)
    is applied to every model's stored predictions and ground truth, which
    is what makes Delta-CIs across models genuinely paired.

    Args:
        table: Common FrameTable for the dataset.
        block_lengths: Per-cluster block lengths.
        matrices_by_model: Mapping model -> (pred_matrix, gt_matrix). Note
            that gt_matrix is identical across models on the common frame set
            but is passed per model to keep the function signature symmetric.
        valid_tools: Sorted canonical tool list.
        metric_names: Order of metric names to populate per replicate (for
            allocating output arrays).
        n_bootstrap: Number of bootstrap replicates.
        seed: Master RNG seed.

    Returns:
        Mapping model -> metric -> length-B float array.
    """
    rng = np.random.default_rng(seed)
    replicates: Dict[str, Dict[str, np.ndarray]] = {
        model_name: {
            name: np.empty(n_bootstrap, dtype=np.float64) for name in metric_names
        }
        for model_name in matrices_by_model
    }

    for b in tqdm(range(n_bootstrap), desc="bootstrap replicates"):
        sampled = hierarchical_paired_indices(table, block_lengths, rng)
        for model_name, (pred_matrix, gt_matrix) in matrices_by_model.items():
            metric_values = evaluate_all_metrics_for_model(
                sampled,
                pred_matrix,
                gt_matrix,
                valid_tools,
            )
            for name in metric_names:
                replicates[model_name][name][b] = metric_values[name]

    return replicates


def build_marginal_summary(
    point_estimates_by_model: Dict[str, Dict[str, float]],
    replicates_by_model: Dict[str, Dict[str, np.ndarray]],
    matrices_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    valid_tools: List[str],
    table: FrameTable,
    family_by_model: Dict[str, str],
    n_common_frames: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tabulate marginal CIs for headline metrics and per-tool F1.

    Args:
        point_estimates_by_model: Mapping model -> metric -> point estimate.
        replicates_by_model: Mapping model -> metric -> bootstrap replicates.
        matrices_by_model: Mapping model -> (pred_matrix, gt_matrix), used by
            the BCa jackknife to recompute metrics on leave-one-cluster-out.
        valid_tools: Sorted canonical tool list.
        table: Common FrameTable (defines the jackknife clusters).
        family_by_model: Mapping model -> family.
        n_common_frames: Number of frames in the common val set.

    Returns:
        (model_headline_df, model_tool_f1_df).
    """
    headline_metric_set = {"exact_match", "jaccard", "micro_f1", "macro_f1"}
    headline_rows = []
    per_tool_rows = []

    for model_name in tqdm(point_estimates_by_model, desc="summarising models"):
        point_estimates = point_estimates_by_model[model_name]
        replicate_arrays = replicates_by_model[model_name]
        pred_matrix, gt_matrix = matrices_by_model[model_name]

        jackknife_values_by_metric = _jackknife_all_metrics(
            table,
            pred_matrix,
            gt_matrix,
            valid_tools,
        )

        for metric_name, point in point_estimates.items():
            replicate_values = replicate_arrays[metric_name]
            pct_low, pct_high = percentile_ci(replicate_values)
            bca_low, bca_high = bca_ci(
                point,
                replicate_values,
                jackknife_values_by_metric[metric_name],
            )
            row = {
                "model_name": model_name,
                "family": family_by_model[model_name],
                "n_val_frames": n_common_frames,
                "metric": metric_name,
                "point_estimate": point,
                "percentile_ci_low": pct_low,
                "percentile_ci_high": pct_high,
                "bca_ci_low": bca_low,
                "bca_ci_high": bca_high,
            }
            if metric_name in headline_metric_set:
                headline_rows.append(row)
            elif metric_name.startswith("f1::"):
                per_tool_rows.append(
                    {
                        "model_name": model_name,
                        "family": family_by_model[model_name],
                        "tool_name": metric_name.replace("f1::", "", 1),
                        "f1": point,
                        "percentile_ci_low": pct_low,
                        "percentile_ci_high": pct_high,
                        "bca_ci_low": bca_low,
                        "bca_ci_high": bca_high,
                    }
                )

    return pd.DataFrame(headline_rows), pd.DataFrame(per_tool_rows)


def _jackknife_all_metrics(
    table: FrameTable,
    pred_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    valid_tools: List[str],
) -> Dict[str, np.ndarray]:
    """Leave-one-cluster-out jackknife values for every metric.

    Args:
        table: Common FrameTable.
        pred_matrix: (N, T) prediction matrix.
        gt_matrix: (N, T) ground truth matrix.
        valid_tools: Sorted canonical tool list.

    Returns:
        Mapping metric_name -> length-K array of metric values.
    """
    keys = table.cluster_keys
    out: Dict[str, List[float]] = {}
    for drop_key in keys:
        kept = np.concatenate(
            [table.cluster_to_ordered_indices[k] for k in keys if k != drop_key]
        )
        metric_values = evaluate_all_metrics_for_model(
            kept, pred_matrix, gt_matrix, valid_tools
        )
        for name, value in metric_values.items():
            out.setdefault(name, []).append(value)
    return {name: np.asarray(values, dtype=np.float64) for name, values in out.items()}


def build_paired_comparisons(
    replicates_by_model: Dict[str, Dict[str, np.ndarray]],
    point_estimates_by_model: Dict[str, Dict[str, float]],
    family_by_model: Dict[str, str],
    metrics_for_pairs: Tuple[str, ...] = (
        "exact_match",
        "jaccard",
        "micro_f1",
        "macro_f1",
    ),
) -> pd.DataFrame:
    """Build paired Delta-CIs and p-values for every model pair x headline metric.

    Args:
        replicates_by_model: Mapping model -> metric -> length-B array.
        point_estimates_by_model: Mapping model -> metric -> point estimate.
        family_by_model: Mapping model -> family.
        metrics_for_pairs: Headline metrics to compare.

    Returns:
        DataFrame with one row per (model_a, model_b, metric) combination.
    """
    model_names = sorted(replicates_by_model)
    rows = []
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1 :]:
            for metric in metrics_for_pairs:
                a_replicates = replicates_by_model[model_a][metric]
                b_replicates = replicates_by_model[model_b][metric]
                deltas = a_replicates - b_replicates
                point_delta = (
                    point_estimates_by_model[model_a][metric]
                    - point_estimates_by_model[model_b][metric]
                )
                pct_low, pct_high = percentile_ci(deltas)
                rows.append(
                    {
                        "model_a": model_a,
                        "family_a": family_by_model[model_a],
                        "model_b": model_b,
                        "family_b": family_by_model[model_b],
                        "metric": metric,
                        "point_delta": point_delta,
                        "delta_ci_low": pct_low,
                        "delta_ci_high": pct_high,
                        "p_value_two_sided": two_sided_pvalue(deltas),
                    }
                )
    return pd.DataFrame(rows)


def write_outputs(
    dataset_name: str,
    spec: DatasetSpec,
    table: FrameTable,
    block_lengths: Dict[str, int],
    model_headline_df: pd.DataFrame,
    model_tool_f1_df: pd.DataFrame,
    model_pair_delta_df: pd.DataFrame,
    replicates_by_model: Dict[str, Dict[str, np.ndarray]],
    family_by_model: Dict[str, str],
    n_common_frames: int,
    output_dir: Path,
) -> None:
    """Persist all per-dataset reanalysis outputs.

    Args:
        dataset_name: Dataset name.
        spec: DatasetSpec.
        table: Validation FrameTable.
        block_lengths: Per-cluster block lengths.
        model_headline_df: Long-form per-model headline metric CIs.
        model_tool_f1_df: Long-form per-model per-tool F1 CIs.
        model_pair_delta_df: Paired Delta-CIs across model pairs.
        replicates_by_model: Raw bootstrap replicates per model x metric.
        family_by_model: Mapping model_name -> family.
        n_common_frames: Number of val frames in the common set.
        output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    block_length_rows = [
        {
            "cluster_id": cid,
            "n_frames": int(table.cluster_to_ordered_indices[cid].shape[0]),
            "block_length": int(L),
        }
        for cid, L in sorted(block_lengths.items())
    ]
    pd.DataFrame(block_length_rows).to_csv(
        output_dir / "block_lengths.csv", index=False
    )

    model_headline_df.to_csv(output_dir / "headline_summary.csv", index=False)
    model_tool_f1_df.to_csv(output_dir / "per_tool_f1.csv", index=False)
    model_pair_delta_df.to_csv(output_dir / "paired_deltas.csv", index=False)

    np.savez_compressed(
        output_dir / "bootstrap_replicates.npz",
        **{
            f"{model_name}__{metric}": values
            for model_name, replicate_dict in replicates_by_model.items()
            for metric, values in replicate_dict.items()
        },
    )

    metadata = {
        "dataset": dataset_name,
        "n_clusters": int(table.n_clusters),
        "n_val_frames_common": int(n_common_frames),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": RANDOM_SEED,
        "valid_tools": spec.valid_tools,
        "models": [
            {"model_name": m, "family": family_by_model[m]}
            for m in sorted(family_by_model)
        ],
        "block_length_summary": {
            "min": int(min(block_lengths.values())) if block_lengths else 0,
            "median": int(np.median(list(block_lengths.values())))
            if block_lengths
            else 0,
            "max": int(max(block_lengths.values())) if block_lengths else 0,
        },
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def restrict_to_common_frames(
    frame_table_val_df: pd.DataFrame,
    predictions_by_model: Dict[str, Dict[str, List[str]]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[str]]], Dict[str, int]]:
    """Restrict the val table and predictions to frame_keys present in every model.

    Args:
        frame_table_val_df: Full val frame table.
        predictions_by_model: Mapping model -> frame_key -> pred list.

    Returns:
        (frame_table_common_df, predictions_by_model_pruned, dropped_per_model)
        where dropped_per_model reports frames each model contributed beyond
        the intersection (informational only).
    """
    full_val_keys = set(frame_table_val_df["frame_key"])
    common = full_val_keys.copy()
    for predictions in predictions_by_model.values():
        common &= set(predictions.keys())

    common_mask = frame_table_val_df["frame_key"].isin(common)
    frame_table_common_df = frame_table_val_df[common_mask].reset_index(drop=True)

    pruned = {
        model_name: {k: tools for k, tools in predictions.items() if k in common}
        for model_name, predictions in predictions_by_model.items()
    }
    dropped = {
        model_name: len(predictions) - len(common)
        for model_name, predictions in predictions_by_model.items()
    }
    return frame_table_common_df, pruned, dropped


def run_dataset(
    dataset_name: str,
    output_dir: Optional[Path] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    only_models: Optional[List[str]] = None,
) -> None:
    """End-to-end driver for one dataset.

    Args:
        dataset_name: Dataset name.
        output_dir: Where to write outputs (defaults to OUTPUT_ROOT/<dataset>).
        n_bootstrap: Number of bootstrap replicates.
        seed: RNG seed.
        only_models: Optional whitelist of model_name strings.
    """
    spec = get_dataset_spec(dataset_name)
    print(f"=== {dataset_name}: loading val frame table ===")
    frame_table_val_df = load_val_frame_table(spec)
    print(
        f"  full val frames: {len(frame_table_val_df)}  "
        f"clusters: {frame_table_val_df['cluster_id'].nunique()}"
    )

    val_frame_keys = frame_table_val_df["frame_key"].tolist()
    n_val_full = len(val_frame_keys)
    discovered_models = discover_models_for_dataset(dataset_name)
    if only_models is not None:
        discovered_models = [m for m in discovered_models if m[1] in set(only_models)]
    print(f"  discovered models: {len(discovered_models)}")

    print(f"=== {dataset_name}: loading per-model predictions ===")
    predictions_by_model: Dict[str, Dict[str, List[str]]] = {}
    family_by_model: Dict[str, str] = {}
    for family, model_name, model_dirs in tqdm(discovered_models, desc="loading"):
        try:
            predictions = load_predictions_for_model(
                dataset_name,
                family,
                model_dirs,
                val_frame_keys,
            )
        except FileNotFoundError as exc:
            print(f"  SKIP (missing file): {model_name}  {exc}")
            continue
        if not predictions:
            print(f"  SKIP (zero predictions): {model_name}")
            continue
        coverage = len(predictions) / n_val_full
        if coverage < DEFAULT_MIN_COVERAGE:
            print(
                f"  SKIP (coverage {coverage:.1%} < {DEFAULT_MIN_COVERAGE:.0%}): "
                f"{model_name}  n={len(predictions)} / {n_val_full}"
            )
            continue
        predictions_by_model[model_name] = predictions
        family_by_model[model_name] = family
        print(
            f"  {model_name:60s} family={family:10s} n={len(predictions):6d} "
            f"({coverage:.1%})"
        )

    if not predictions_by_model:
        raise RuntimeError(f"No models loaded for {dataset_name}")

    print(
        f"=== {dataset_name}: intersecting frames across {len(predictions_by_model)} models ==="
    )
    frame_table_common_df, pruned_predictions, dropped = restrict_to_common_frames(
        frame_table_val_df,
        predictions_by_model,
    )
    n_common = len(frame_table_common_df)
    print(f"  common frames: {n_common}")
    for model_name, n_dropped in dropped.items():
        if n_dropped:
            print(f"    {model_name}: dropped {n_dropped} frames not in common set")

    if n_common == 0:
        raise RuntimeError(
            f"Empty common frame set for {dataset_name}; cannot run paired bootstrap."
        )

    print(f"=== {dataset_name}: encoding matrices ===")
    pred_matrices_by_model, gt_matrix = encode_predictions_aligned(
        frame_table_common_df,
        pruned_predictions,
        spec.valid_tools,
    )
    correctness_per_model = [
        per_frame_correctness(pm, gt_matrix).astype(np.float64)
        for pm in pred_matrices_by_model.values()
    ]

    print(f"=== {dataset_name}: building common FrameTable ===")
    common_table = build_frame_table(
        cluster_ids=frame_table_common_df["cluster_id"].tolist(),
        within_cluster_index=frame_table_common_df["within_cluster_index"].tolist(),
    )

    print(f"=== {dataset_name}: estimating block lengths (Politis-White) ===")
    block_lengths = select_block_lengths(
        common_table.cluster_to_ordered_indices,
        correctness_per_model,
    )
    block_length_summary = {
        "min": min(block_lengths.values()),
        "median": int(np.median(list(block_lengths.values()))),
        "max": max(block_lengths.values()),
    }
    print(f"  block lengths: {block_length_summary}")

    print(f"=== {dataset_name}: computing point estimates ===")
    point_estimates_by_model: Dict[str, Dict[str, float]] = {}
    matrices_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for model_name, pred_matrix in pred_matrices_by_model.items():
        matrices_by_model[model_name] = (pred_matrix, gt_matrix)
        point_estimates_by_model[model_name] = compute_point_estimates(
            pred_matrix,
            gt_matrix,
            spec.valid_tools,
        )

    headline_metric_set = ["exact_match", "jaccard", "micro_f1", "macro_f1"]
    metric_names = headline_metric_set + [f"f1::{t}" for t in spec.valid_tools]

    print(
        f"=== {dataset_name}: running B={n_bootstrap} paired replicates "
        f"({len(matrices_by_model)} models, all paired) ==="
    )
    replicates_by_model = run_paired_replicates(
        common_table,
        block_lengths,
        matrices_by_model,
        spec.valid_tools,
        metric_names,
        n_bootstrap,
        seed,
    )

    print(f"=== {dataset_name}: building marginal summary ===")
    model_headline_df, model_tool_f1_df = build_marginal_summary(
        point_estimates_by_model,
        replicates_by_model,
        matrices_by_model,
        spec.valid_tools,
        common_table,
        family_by_model,
        n_common,
    )

    print(f"=== {dataset_name}: building paired Delta-CIs ===")
    model_pair_delta_df = build_paired_comparisons(
        replicates_by_model,
        point_estimates_by_model,
        family_by_model,
    )

    if output_dir is None:
        output_dir = OUTPUT_ROOT / dataset_name
    print(f"=== {dataset_name}: writing outputs to {output_dir} ===")
    write_outputs(
        dataset_name=dataset_name,
        spec=spec,
        table=common_table,
        block_lengths=block_lengths,
        model_headline_df=model_headline_df,
        model_tool_f1_df=model_tool_f1_df,
        model_pair_delta_df=model_pair_delta_df,
        replicates_by_model=replicates_by_model,
        family_by_model=family_by_model,
        n_common_frames=n_common,
        output_dir=output_dir,
    )
    print(f"=== {dataset_name}: done ===")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["eea", "pitvis", "cholect50", "surgvu"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: OUTPUT_ROOT/<dataset>).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--only-model",
        action="append",
        default=None,
        help="Restrict to specific model_name(s) (smoke test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        only_models=args.only_model,
    )
