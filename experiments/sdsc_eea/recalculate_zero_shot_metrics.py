"""Recompute zero-shot metrics with parse failures as empty predictions.

For each model under the zero-shot VLM output, reads the debug logs of every
attempted frame and the result CSVs of successful parses. A parse failure
becomes ``predicted_tools = []`` for per-tool precision, recall, and F1, and
counts as a miss for exact match. This is the protocol behind the published
zero-shot tables. Original result directories are not overwritten.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from scaling_surgical_ai import paths

RANDOM_SEED = 42
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
INPUT_BASE = paths.experiment_output_dir("sdsc_eea_zero_shot_vlm")
OUTPUT_BASE = paths.experiment_output_dir("sdsc_eea_recalculate_zero_shot_metrics")

MODEL_DIRS = [
    "Llama-3.2-11B-Vision_temp1_0",
    "Llama-3.2-90B-Vision_temp1_0",
    "Qwen2-VL-2B-Instruct_temp1_0",
    "Qwen2-VL-7B-Instruct_temp1_0",
    "Qwen2-VL-72B-Instruct_temp1_0",
    "Qwen2.5-VL-3B-Instruct_temp1_0",
    "Qwen2.5-VL-7B-Instruct_temp1_0",
    "Qwen2.5-VL-32B-Instruct_temp1_0",
    "Qwen2.5-VL-72B-Instruct_temp1_0",
    "Qwen3-VL-2B-Instruct_temp1_0",
    "Qwen3-VL-4B-Instruct_temp1_0",
    "Qwen3-VL-8B-Instruct_temp1_0",
    "Qwen3-VL-32B-Instruct_temp1_0",
    "Qwen3-VL-235B-A22B-Thinking_temp1_0",
    "gemma-3-4b-it_temp1_0",
    "gemma-3-12b-it_temp1_0",
    "gemma-3-27b-it_temp1_0",
    "llava-v1.5-13b_temp1_0",
]


def load_yolo_mapping() -> dict:
    """Load the YOLO class id to tool name mapping."""
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["mapping"].items()}


def load_val_frame_set() -> set:
    """Load set of validation frame paths from dataset CSV."""
    val_frames = set()
    with DATASET_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "val":
                val_frames.add(row["frame_path"])
    print(f"Loaded {len(val_frames)} validation frame paths")
    return val_frames


def get_ground_truth_tools(frame_path: Path, yolo_mapping: dict) -> list:
    """Look up ground truth tools from YOLO label file."""
    label_path = frame_path.parent.parent / "labels" / f"{frame_path.stem}.txt"
    if not label_path.exists():
        return None
    class_ids = set()
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.add(int(parts[0]))
    tool_names = sorted(yolo_mapping[cid] for cid in class_ids if cid in yolo_mapping)
    return tool_names


def read_debug_logs(model_dir: Path) -> list:
    """Read all debug log entries, supporting both rank-based and single-file formats."""
    entries = []
    rank_files = sorted(model_dir.glob("debug_responses_rank_*.jsonl"))
    if rank_files:
        for f_path in rank_files:
            with f_path.open("r", encoding="utf-8") as f:
                for line in f:
                    entries.append(json.loads(line))
    else:
        single = model_dir / "debug_responses.jsonl"
        if single.exists():
            with single.open("r", encoding="utf-8") as f:
                for line in f:
                    entries.append(json.loads(line))
    return entries


def read_results_csvs(model_dir: Path) -> dict:
    """Read predictions from results CSVs. Returns {frame_path: predicted_tools_list}."""
    predictions = {}
    rank_files = sorted(model_dir.glob("results_rank_*.csv"))
    if rank_files:
        for f_path in rank_files:
            with f_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions[row["frame_path"]] = json.loads(row["predicted_tools"])
    else:
        single = model_dir / "results.csv"
        if single.exists():
            with single.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions[row["frame_path"]] = json.loads(row["predicted_tools"])
    return predictions


def bootstrap_ci(
    correct_array: list,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple:
    """Return a percentile bootstrap interval for a 0/1 vector."""
    if len(correct_array) == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(correct_array)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(correct_array) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)
    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def process_model(
    model_name: str,
    model_dir: Path,
    yolo_mapping: dict,
    valid_tools: list,
    val_frame_set: set,
    output_dir: Path,
):
    """Recompute one model's metrics and write its summary files."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {model_name}")
    print(f"{'=' * 70}")

    debug_entries = read_debug_logs(model_dir)
    if not debug_entries:
        print(f"  SKIP: No debug logs found in {model_dir}")
        return None

    predictions = read_results_csvs(model_dir)
    print(f"  Debug log entries: {len(debug_entries)}")
    print(f"  Successful predictions (results CSV): {len(predictions)}")

    total_attempted = 0
    parse_successes = 0
    parse_failures = 0
    skipped_no_label = 0

    tool_tp = defaultdict(int)
    tool_fp = defaultdict(int)
    tool_fn = defaultdict(int)
    tool_tn = defaultdict(int)
    tool_gt_counts = defaultdict(int)
    tool_pred_counts = defaultdict(int)

    all_exact_match = []
    all_jaccard = []
    val_exact_match = []
    val_jaccard = []
    train_exact_match = []
    train_jaccard = []

    for entry in tqdm(debug_entries, desc=model_name):
        frame_path_str = entry["frame_path"]
        frame_path = Path(frame_path_str)
        parse_success = entry.get("parse_success", False)

        gt_tools = get_ground_truth_tools(frame_path, yolo_mapping)
        if gt_tools is None:
            skipped_no_label += 1
            continue

        total_attempted += 1

        if parse_success and frame_path_str in predictions:
            predicted_tools = sorted(predictions[frame_path_str])
            parse_successes += 1
            is_parse_failure = False
        else:
            predicted_tools = []
            parse_failures += 1
            is_parse_failure = True

        gt_set = set(gt_tools)
        pred_set = set(predicted_tools)

        if is_parse_failure:
            exact_match = False
            jaccard = 0.0
        else:
            exact_match = gt_set == pred_set
            union = gt_set | pred_set
            jaccard = len(gt_set & pred_set) / len(union) if union else 1.0

        all_exact_match.append(1 if exact_match else 0)
        all_jaccard.append(jaccard)

        is_val = frame_path_str in val_frame_set
        if is_val:
            val_exact_match.append(1 if exact_match else 0)
            val_jaccard.append(jaccard)
        else:
            train_exact_match.append(1 if exact_match else 0)
            train_jaccard.append(jaccard)

        for tool in valid_tools:
            in_gt = tool in gt_set
            in_pred = tool in pred_set

            if in_gt:
                tool_gt_counts[tool] += 1
            if in_pred:
                tool_pred_counts[tool] += 1

            if in_gt and in_pred:
                tool_tp[tool] += 1
            elif in_pred and not in_gt:
                tool_fp[tool] += 1
            elif in_gt and not in_pred:
                tool_fn[tool] += 1
            else:
                tool_tn[tool] += 1

    parse_failure_rate = (
        parse_failures / (parse_successes + parse_failures) * 100
        if (parse_successes + parse_failures) > 0
        else 0.0
    )

    all_em_acc = (
        sum(all_exact_match) / len(all_exact_match) * 100 if all_exact_match else 0.0
    )
    all_em_ci = bootstrap_ci(all_exact_match)

    val_em_acc = (
        sum(val_exact_match) / len(val_exact_match) * 100 if val_exact_match else 0.0
    )
    val_em_ci = bootstrap_ci(val_exact_match)

    train_em_acc = (
        sum(train_exact_match) / len(train_exact_match) * 100
        if train_exact_match
        else 0.0
    )
    train_em_ci = bootstrap_ci(train_exact_match)

    print(f"  Total attempted (with GT): {total_attempted}")
    print(f"  Parse successes: {parse_successes}")
    print(f"  Parse failures (-> [] for P/R/F1, wrong for EM): {parse_failures}")
    print(f"  Parse failure rate: {parse_failure_rate:.1f}%")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(
        f"  Train exact match: {train_em_acc:.2f}% "
        f"(95% CI: {train_em_ci[0] * 100:.2f}%--{train_em_ci[1] * 100:.2f}%) "
        f"n={len(train_exact_match)}"
    )
    print(
        f"  Val exact match:   {val_em_acc:.2f}% "
        f"(95% CI: {val_em_ci[0] * 100:.2f}%--{val_em_ci[1] * 100:.2f}%) "
        f"n={len(val_exact_match)}"
    )
    print(
        f"  All exact match:   {all_em_acc:.2f}% "
        f"(95% CI: {all_em_ci[0] * 100:.2f}%--{all_em_ci[1] * 100:.2f}%) "
        f"n={len(all_exact_match)}"
    )

    model_output = output_dir / model_name
    model_output.mkdir(parents=True, exist_ok=True)

    summary_path = model_output / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "total_attempted",
                "parse_successes",
                "parse_failures",
                "parse_failure_rate_pct",
                "skipped_no_label",
                "train_frames",
                "train_exact_matches",
                "train_exact_match_accuracy_pct",
                "train_exact_match_ci_low_pct",
                "train_exact_match_ci_high_pct",
                "val_frames",
                "val_exact_matches",
                "val_exact_match_accuracy_pct",
                "val_exact_match_ci_low_pct",
                "val_exact_match_ci_high_pct",
                "all_exact_matches",
                "all_exact_match_accuracy_pct",
                "all_exact_match_ci_low_pct",
                "all_exact_match_ci_high_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": model_name,
                "total_attempted": total_attempted,
                "parse_successes": parse_successes,
                "parse_failures": parse_failures,
                "parse_failure_rate_pct": f"{parse_failure_rate:.2f}",
                "skipped_no_label": skipped_no_label,
                "train_frames": len(train_exact_match),
                "train_exact_matches": sum(train_exact_match),
                "train_exact_match_accuracy_pct": f"{train_em_acc:.2f}",
                "train_exact_match_ci_low_pct": f"{train_em_ci[0] * 100:.2f}",
                "train_exact_match_ci_high_pct": f"{train_em_ci[1] * 100:.2f}",
                "val_frames": len(val_exact_match),
                "val_exact_matches": sum(val_exact_match),
                "val_exact_match_accuracy_pct": f"{val_em_acc:.2f}",
                "val_exact_match_ci_low_pct": f"{val_em_ci[0] * 100:.2f}",
                "val_exact_match_ci_high_pct": f"{val_em_ci[1] * 100:.2f}",
                "all_exact_matches": sum(all_exact_match),
                "all_exact_match_accuracy_pct": f"{all_em_acc:.2f}",
                "all_exact_match_ci_low_pct": f"{all_em_ci[0] * 100:.2f}",
                "all_exact_match_ci_high_pct": f"{all_em_ci[1] * 100:.2f}",
            }
        )

    tool_metrics_path = model_output / "tool_metrics.csv"
    with tool_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tool_name",
            "ground_truth_count",
            "predicted_count",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tool in sorted(valid_tools):
            tp = tool_tp[tool]
            fp = tool_fp[tool]
            fn = tool_fn[tool]
            tn = tool_tn[tool]
            total_tool = tp + fp + fn + tn

            accuracy = (tp + tn) / total_tool if total_tool > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            writer.writerow(
                {
                    "tool_name": tool,
                    "ground_truth_count": tool_gt_counts[tool],
                    "predicted_count": tool_pred_counts[tool],
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn,
                    "accuracy": f"{accuracy:.4f}",
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}",
                    "f1_score": f"{f1:.4f}",
                }
            )

    print(f"  Saved: {summary_path}")
    print(f"  Saved: {tool_metrics_path}")

    return {
        "model_name": model_name,
        "total_attempted": total_attempted,
        "parse_successes": parse_successes,
        "parse_failures": parse_failures,
        "parse_failure_rate_pct": parse_failure_rate,
        "train_n": len(train_exact_match),
        "train_exact_match_pct": train_em_acc,
        "train_ci_low": train_em_ci[0] * 100,
        "train_ci_high": train_em_ci[1] * 100,
        "val_n": len(val_exact_match),
        "val_exact_match_pct": val_em_acc,
        "val_ci_low": val_em_ci[0] * 100,
        "val_ci_high": val_em_ci[1] * 100,
        "all_exact_match_pct": all_em_acc,
        "all_ci_low": all_em_ci[0] * 100,
        "all_ci_high": all_em_ci[1] * 100,
    }


def main() -> None:
    """Recompute every configured model and write the combined summary."""
    print("=" * 70)
    print("Recalculating zero-shot metrics")
    print("Per-tool P/R/F1: parse failures -> predicted_tools = []")
    print("Exact match: parse failures -> always wrong (y != yhat)")
    print("=" * 70)

    yolo_mapping = load_yolo_mapping()
    valid_tools = sorted(set(yolo_mapping.values()))
    print(f"Loaded {len(valid_tools)} valid tool names")

    val_frame_set = load_val_frame_set()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for model_name in MODEL_DIRS:
        model_dir = INPUT_BASE / model_name
        if not model_dir.exists():
            print(f"\nSKIP: {model_dir} does not exist")
            continue

        summary = process_model(
            model_name,
            model_dir,
            yolo_mapping,
            valid_tools,
            val_frame_set,
            OUTPUT_BASE,
        )
        if summary:
            all_summaries.append(summary)

    combined_path = OUTPUT_BASE / "all_models_summary.csv"
    if all_summaries:
        with combined_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(all_summaries[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in sorted(all_summaries, key=lambda x: -x["val_exact_match_pct"]):
                writer.writerow(
                    {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in s.items()}
                )

    print(f"\n{'=' * 100}")
    print("COMBINED RESULTS (sorted by val exact match accuracy)")
    print("EM: parse failures = always wrong | P/R/F1: parse failures = []")
    print(f"{'=' * 100}")
    header = (
        f"{'Model':<42} {'PF%':>5} "
        f"{'Train%':>7} {'Val%':>7} {'Val 95% CI':>20} {'All%':>7}"
    )
    print(header)
    print("-" * len(header))
    for s in sorted(all_summaries, key=lambda x: -x["val_exact_match_pct"]):
        print(
            f"{s['model_name']:<42} "
            f"{s['parse_failure_rate_pct']:>4.1f}% "
            f"{s['train_exact_match_pct']:>6.2f}% "
            f"{s['val_exact_match_pct']:>6.2f}% "
            f"({s['val_ci_low']:>5.2f}%--{s['val_ci_high']:>5.2f}%) "
            f"{s['all_exact_match_pct']:>6.2f}%"
        )

    print(f"\nAll results saved to: {OUTPUT_BASE}")
    print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
