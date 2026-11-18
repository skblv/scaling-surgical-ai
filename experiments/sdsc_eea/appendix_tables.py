"""Write the appendix LaTeX tables of per-tool zero-shot metrics.

Combines ``tool_metrics.csv`` from the recalculated zero-shot runs and from
the Llama, MedGemma, LLaVA, and Qwen3-VL-235B split directories. Parse
failures stay in those CSVs as empty predictions, which is the paper's
per-tool protocol.
"""

from __future__ import annotations

import csv
from pathlib import Path

from scaling_surgical_ai import paths

RECALCULATED_BASE = paths.experiment_output_dir(
    "sdsc_eea_recalculate_zero_shot_metrics"
)
ZERO_SHOT_VLM_DIR = paths.experiment_output_dir("sdsc_eea_zero_shot_vlm")

SINGLE_DIR_MODELS = {
    "Llama-3.2-90B-Vision": (
        paths.experiment_output_dir("sdsc_eea_zero_shot_llama_vision")
        / "llama-3.2-90b-vision_temp0_0"
    ),
    "MedGemma-3-27B-it": (
        paths.experiment_output_dir("sdsc_eea_zero_shot_medgemma")
        / "medgemma-27b-it_temp0_0"
    ),
    "LLaVA-1.5-13B": (
        paths.experiment_output_dir("sdsc_eea_zero_shot_llava")
        / "llava-v1.5-13b_temp0_2"
    ),
}

MULTI_DIR_MODELS = {
    "Qwen3-VL-235B-A22B-Thinking": [
        ZERO_SHOT_VLM_DIR / f"qwen3-vl-235b_temp0_0_split{i}" for i in range(1, 5)
    ],
}

MODEL_ORDER = [
    ("Qwen3-VL-235B-A22B-Thinking_temp1_0", "Qwen3-VL-235B-A22B-Thinking", "235B"),
    ("Qwen3-VL-32B-Instruct_temp1_0", "Qwen3-VL-32B-Instruct", "32B"),
    ("Qwen3-VL-8B-Instruct_temp1_0", "Qwen3-VL-8B-Instruct", "8B"),
    ("Qwen3-VL-4B-Instruct_temp1_0", "Qwen3-VL-4B-Instruct", "4B"),
    ("Qwen3-VL-2B-Instruct_temp1_0", "Qwen3-VL-2B-Instruct", "2B"),
    ("Qwen2.5-VL-72B-Instruct_temp1_0", "Qwen2.5-VL-72B-Instruct", "72B"),
    ("Qwen2.5-VL-32B-Instruct_temp1_0", "Qwen2.5-VL-32B-Instruct", "32B"),
    ("Qwen2.5-VL-7B-Instruct_temp1_0", "Qwen2.5-VL-7B-Instruct", "7B"),
    ("Qwen2.5-VL-3B-Instruct_temp1_0", "Qwen2.5-VL-3B-Instruct", "3B"),
    ("Qwen2-VL-72B-Instruct_temp1_0", "Qwen2-VL-72B-Instruct", "72B"),
    ("Qwen2-VL-7B-Instruct_temp1_0", "Qwen2-VL-7B-Instruct", "7B"),
    ("Qwen2-VL-2B-Instruct_temp1_0", "Qwen2-VL-2B-Instruct", "2B"),
    ("gemma-3-27b-it_temp1_0", "Gemma 3 27B-it", "27B"),
    ("gemma-3-12b-it_temp1_0", "Gemma 3 12B-it", "12B"),
    ("gemma-3-4b-it_temp1_0", "Gemma 3 4B-it", "4B"),
    ("MedGemma-3-27B-it", "MedGemma 3 27B-it", "27B"),
    ("Llama-3.2-90B-Vision", "Llama-3.2-90B-Vision", "90B"),
    ("Llama-3.2-11B-Vision_temp1_0", "Llama-3.2-11B-Vision", "11B"),
    ("LLaVA-1.5-13B", "LLaVA-1.5-13B", "13B"),
]

LABEL_MAP = {
    "Qwen3-VL-235B-A22B-Thinking": "qwen3_vl_235b_a22b_thinking",
    "Qwen3-VL-32B-Instruct": "qwen3_vl_32b_instruct",
    "Qwen3-VL-8B-Instruct": "qwen3_vl_8b_instruct",
    "Qwen3-VL-4B-Instruct": "qwen3_vl_4b_instruct",
    "Qwen3-VL-2B-Instruct": "qwen3_vl_2b_instruct",
    "Qwen2.5-VL-72B-Instruct": "qwen2_5_vl_72b_instruct",
    "Qwen2.5-VL-32B-Instruct": "qwen2_5_vl_32b_instruct",
    "Qwen2.5-VL-7B-Instruct": "qwen2_5_vl_7b_instruct",
    "Qwen2.5-VL-3B-Instruct": "qwen2_5_vl_3b_instruct",
    "Qwen2-VL-72B-Instruct": "qwen2_vl_72b_instruct",
    "Qwen2-VL-7B-Instruct": "qwen2_vl_7b_instruct",
    "Qwen2-VL-2B-Instruct": "qwen2_vl_2b_instruct",
    "Gemma 3 27B-it": "gemma_3_27b_it",
    "Gemma 3 12B-it": "gemma_3_12b_it",
    "Gemma 3 4B-it": "gemma_3_4b_it",
    "MedGemma 3 27B-it": "medgemma_3_27b_it",
    "Llama-3.2-90B-Vision": "llama_3_2_90b_vision",
    "Llama-3.2-11B-Vision": "llama_3_2_11b_vision",
    "LLaVA-1.5-13B": "llava_1_5_13b",
}


def load_tool_metrics(csv_paths) -> list:
    """Load and combine tool metrics from one or more CSV files.

    Args:
        csv_paths: One ``tool_metrics.csv`` path, or a list of them.

    Returns:
        Rows with pooled counts and precision, recall, and F1, sorted by F1.
    """
    if isinstance(csv_paths, Path):
        csv_paths = [csv_paths]

    combined = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tool = row["tool_name"]
                if tool not in combined:
                    combined[tool] = {"tp": 0, "fp": 0, "fn": 0}
                combined[tool]["tp"] += int(row["true_positives"])
                combined[tool]["fp"] += int(row["false_positives"])
                combined[tool]["fn"] += int(row["false_negatives"])

    rows = []
    for tool, stats in combined.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        if tp > 0 or fp > 0 or fn > 0:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append(
                {
                    "tool": tool,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "p": p,
                    "r": r,
                    "f1": f1,
                }
            )
    rows.sort(key=lambda x: -x["f1"])
    return rows


def generate_table(model_display, params, label_key, rows):
    """Render one appendix table.

    Args:
        model_display: Model name used in the caption and label map.
        params: Parameter-count string used in the caption.
        label_key: Directory key from the model order. Unused by the table.
        rows: Per-tool metric rows from ``load_tool_metrics``.

    Returns:
        A LaTeX table.
    """
    label = LABEL_MAP[model_display]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Per-tool metrics for "
        + model_display
        + " ("
        + params
        + r") zero-shot evaluation. P = precision, R = recall.}"
    )
    lines.append(r"\label{tab:zero_shot_" + label + "}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Tool & TP & FP & FN & P & R & F1 \\")
    lines.append(r"\midrule")

    for row in rows:
        tool_name = row["tool"].replace(
            "Sonopet- pineapple tip", "Sonopet pineapple tip"
        )
        lines.append(
            f"{tool_name} & {row['tp']} & {row['fp']} & {row['fn']} "
            f"& {row['p']:.3f} & {row['r']:.3f} & {row['f1']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    """Write ``appendix_zero_shot_tables.tex`` under the recalculated output."""
    output_lines = []
    output_lines.append(r"% Auto-generated zero-shot per-tool metrics tables")
    output_lines.append(r"\section{Per-Tool Metrics for Zero-Shot VLM Evaluation}")
    output_lines.append(r"\label{sec:appendix_zero_shot_metrics}")
    output_lines.append("")
    output_lines.append(
        r"The following tables present per-tool classification metrics "
        r"for each zero-shot vision-language model evaluated on the full "
        r"dataset ($n\approx 67{,}634$ frames). Parse failures are treated "
        r"as empty predictions (\texttt{detected\_tools: []}) for the "
        r"purpose of computing per-tool precision, recall, and F1. "
        r"Only tools with at least one true positive, false positive, or "
        r"false negative are shown. Tools are sorted by F1 score in "
        r"descending order."
    )
    output_lines.append("")

    for dir_key, display_name, params in MODEL_ORDER:
        if display_name in MULTI_DIR_MODELS:
            csv_paths = [d / "tool_metrics.csv" for d in MULTI_DIR_MODELS[display_name]]
            missing = [p for p in csv_paths if not p.exists()]
            if missing:
                print(f"WARNING: missing splits for {display_name}: {missing}")
                continue
        elif dir_key in SINGLE_DIR_MODELS:
            csv_paths = [SINGLE_DIR_MODELS[dir_key] / "tool_metrics.csv"]
            if not csv_paths[0].exists():
                print(f"WARNING: {csv_paths[0]} not found")
                continue
        else:
            csv_paths = [RECALCULATED_BASE / dir_key / "tool_metrics.csv"]
            if not csv_paths[0].exists():
                print(f"WARNING: {csv_paths[0]} not found")
                continue

        rows = load_tool_metrics(csv_paths)
        table = generate_table(display_name, params, dir_key, rows)
        output_lines.append("")
        output_lines.append(table)
        output_lines.append("")
        print(f"Generated table for {display_name} ({len(rows)} tools)")

    output_path = RECALCULATED_BASE / "appendix_zero_shot_tables.tex"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
