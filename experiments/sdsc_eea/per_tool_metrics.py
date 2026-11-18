"""Per-tool precision, recall, and F1 for three zero-shot VLMs.

Reads ``results.csv`` from the MedGemma, LLaVA, and Llama-3.2-90B runs and
writes ``tool_metrics.csv`` beside each one. Those tables feed the appendix
per-tool metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tqdm.auto import tqdm

from scaling_surgical_ai import paths

VALID_TOOLS = [
    "Suction Coagulator",
    "Straight Curette",
    "Surgical Knife",
    "Suction microdebrider",
    "Local Anesthesia Needle",
    "Suction",
    "Grasper",
    "Scissor",
    "Rongeur",
    "Rhoton Dissector",
    "Curette",
    "Irrigation",
    "Doppler",
    "Straight Forceps",
    "Drill",
    "Sonopet- pineapple tip",
    "unknown",
    "Tissue shaver",
    "Bipolar Forceps",
    "Backbiting rongeur",
    "Cotton Patty",
    "Aspirating dissector",
    "Through cutting forceps",
    "Floseal Applicator",
    "Floseal Gel",
    "Surgicel",
    "Tisseel Applicator",
    "Cottle Elevator",
    "Monopolar Electrocautery",
    "Collagen Matrix",
    "Needle",
]

RESULTS_DIRS = [
    (
        paths.experiment_output_dir("sdsc_eea_zero_shot_medgemma")
        / "medgemma-27b-it_temp0_0",
        "MedGemma-3-27B-it",
    ),
    (
        paths.experiment_output_dir("sdsc_eea_zero_shot_llava")
        / "llava-v1.5-13b_temp0_2",
        "LLaVA-1.5-13B",
    ),
    (
        paths.experiment_output_dir("sdsc_eea_zero_shot_llama_vision")
        / "llama-3.2-90b-vision_temp0_0",
        "Llama-3.2-90B-Vision",
    ),
]


def compute_tool_metrics(results_csv: Path) -> tuple[dict, int, int]:
    """Compute per-tool confusion counts from a results CSV.

    Args:
        results_csv: ``results.csv`` with JSON tool lists.

    Returns:
        Per-tool counts, the number of scored frames, and skipped rows.
    """
    tool_stats = {
        tool: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt": 0, "pred": 0}
        for tool in VALID_TOOLS
    }
    total_frames = 0
    skipped_rows = 0

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=results_csv.parent.name):
            try:
                gt_str = row.get("ground_truth_tools", "")
                pred_str = row.get("predicted_tools", "")
                if not gt_str or not pred_str:
                    skipped_rows += 1
                    continue
                gt_tools = set(json.loads(gt_str))
                pred_tools = set(json.loads(pred_str))
            except (json.JSONDecodeError, TypeError) as exc:
                print(f"Skipping unparseable row in {results_csv}: {exc}")
                skipped_rows += 1
                continue

            total_frames += 1

            for tool in VALID_TOOLS:
                in_gt = tool in gt_tools
                in_pred = tool in pred_tools

                if in_gt:
                    tool_stats[tool]["gt"] += 1
                if in_pred:
                    tool_stats[tool]["pred"] += 1

                if in_gt and in_pred:
                    tool_stats[tool]["tp"] += 1
                elif in_pred and not in_gt:
                    tool_stats[tool]["fp"] += 1
                elif in_gt and not in_pred:
                    tool_stats[tool]["fn"] += 1
                else:
                    tool_stats[tool]["tn"] += 1

    return tool_stats, total_frames, skipped_rows


def main() -> None:
    """Write ``tool_metrics.csv`` for each configured results directory."""
    for results_dir, model_name in RESULTS_DIRS:
        results_csv = Path(results_dir) / "results.csv"
        if not results_csv.exists():
            print(f"Skipping {model_name}: results.csv not found")
            continue

        print(f"Processing {model_name}...")
        tool_stats, total_frames, skipped_rows = compute_tool_metrics(results_csv)

        output_csv = Path(results_dir) / "tool_metrics.csv"
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

        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tool in sorted(VALID_TOOLS):
                stats = tool_stats[tool]
                tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
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
                        "ground_truth_count": stats["gt"],
                        "predicted_count": stats["pred"],
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

        print(f"  Saved: {output_csv}")
        print(f"  Total frames: {total_frames}")
        if skipped_rows > 0:
            print(f"  Skipped rows: {skipped_rows}")


if __name__ == "__main__":
    main()
