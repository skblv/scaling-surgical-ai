"""Per-model zero-shot micro F1 on the 15-tool SDSC-EEA set for Fig. 2.

Uses the shared prediction loaders and the same 15 evaluable EEA tools.
Computes micro-averaged F1, and exact match as a self-check, for every
zero-shot open-weight VLM in Fig. 2, the fine-tuned Gemma 3 classification
head, YOLOv12-m, and the majority-tool baseline. CPU only.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from scaling_surgical_ai import paths
from scaling_surgical_ai.pooled_f1 import (
    EEA_15_TOOLS,
    majority_tool_baseline,
    micro_f1_em,
    yolo_micro_f1_em,
)
from scaling_surgical_ai.prediction_loaders import (
    EEA_ZERO_SHOT_LLAMA_DIR,
    EEA_ZERO_SHOT_LLAVA_DIR,
    EEA_ZERO_SHOT_MEDGEMMA_DIR,
    EEA_ZERO_SHOT_QWEN3_235B_SPLIT_DIRS,
    EEA_ZERO_SHOT_VLM_DIR,
    get_dataset_spec,
    load_eea_zero_shot_predictions,
    load_lora_jsonl_predictions,
    load_val_frame_table,
)

ZERO_SHOT_VLM_DIR = EEA_ZERO_SHOT_VLM_DIR
CLASSIFICATION_HEAD_DIR = paths.experiment_output_dir(
    "sdsc_eea_lora_classification_head",
    "74900",
)
YOLO_EEA_DIR = paths.experiment_output_dir(
    "sdsc_eea_yolo_full_ontology",
    "58525",
)

ZS_MODEL_DIRS: Dict[str, List[Path]] = {
    "Qwen3-VL-235B-A22B-Thinking": list(EEA_ZERO_SHOT_QWEN3_235B_SPLIT_DIRS),
    "Qwen3-VL-32B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen3-VL-32B-Instruct_temp1_0"],
    "Qwen3-VL-8B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen3-VL-8B-Instruct_temp1_0"],
    "Qwen3-VL-4B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen3-VL-4B-Instruct_temp1_0"],
    "Qwen3-VL-2B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen3-VL-2B-Instruct_temp1_0"],
    "Qwen2.5-VL-72B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2.5-VL-72B-Instruct_temp1_0"],
    "Qwen2.5-VL-32B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2.5-VL-32B-Instruct_temp1_0"],
    "Qwen2.5-VL-7B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2.5-VL-7B-Instruct_temp1_0"],
    "Qwen2.5-VL-3B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2.5-VL-3B-Instruct_temp1_0"],
    "Qwen2-VL-72B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2-VL-72B-Instruct_temp1_0"],
    "Qwen2-VL-7B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2-VL-7B-Instruct_temp1_0"],
    "Qwen2-VL-2B-Instruct": [ZERO_SHOT_VLM_DIR / "Qwen2-VL-2B-Instruct_temp1_0"],
    "Gemma 3 27B-it": [ZERO_SHOT_VLM_DIR / "gemma-3-27b-it_temp1_0"],
    "Gemma 3 12B-it": [ZERO_SHOT_VLM_DIR / "gemma-3-12b-it_temp1_0"],
    "Gemma 3 4B-it": [ZERO_SHOT_VLM_DIR / "gemma-3-4b-it_temp1_0"],
    "MedGemma 3 27B-it": [EEA_ZERO_SHOT_MEDGEMMA_DIR],
    "Llama-3.2-90B-Vision": [EEA_ZERO_SHOT_LLAMA_DIR],
    "Llama-3.2-11B-Vision": [ZERO_SHOT_VLM_DIR / "Llama-3.2-11B-Vision_temp1_0"],
    "LLaVA-1.5-13B": [EEA_ZERO_SHOT_LLAVA_DIR],
    "Gemma 4 31B-it": [ZERO_SHOT_VLM_DIR / "gemma-4-31b-it_temp1_0"],
}

OUT = paths.output_root() / "sdsc_eea_zero_shot_micro_f1.csv"


def main() -> None:
    """Write Fig. 2 micro F1 for the EEA zero-shot and specialist models."""
    spec = get_dataset_spec("eea")
    frame_label_val_df = load_val_frame_table(spec)
    gt_by_key = dict(
        zip(frame_label_val_df["frame_key"], frame_label_val_df["gt_tools"])
    )
    val_keys = list(gt_by_key.keys())
    tools = EEA_15_TOOLS

    rows: List[dict] = []

    majority, base_f1 = majority_tool_baseline(gt_by_key, tools)
    print(f"baseline (majority='{majority}') micro_f1={base_f1:.2f}")
    rows.append(
        {
            "model": f"__baseline__ ({majority})",
            "micro_f1": base_f1,
            "exact_match": "",
            "n_frames": len(gt_by_key),
        }
    )

    cls_preds = load_lora_jsonl_predictions(CLASSIFICATION_HEAD_DIR, val_keys)
    f1, em, n = micro_f1_em(gt_by_key, cls_preds, tools)
    rows.append(
        {
            "model": "Fine-tuned Gemma 3 (ours)",
            "micro_f1": f1,
            "exact_match": em,
            "n_frames": n,
        }
    )
    print(f"{'Fine-tuned Gemma 3 (ours)':32s} F1={f1:6.2f} EM={em:6.2f} n={n}")

    f1, em, n = yolo_micro_f1_em(YOLO_EEA_DIR, tools)
    rows.append(
        {
            "model": "YOLOv12-m (ours)",
            "micro_f1": f1,
            "exact_match": em,
            "n_frames": n,
        }
    )
    print(f"{'YOLOv12-m (ours)':32s} F1={f1:6.2f} EM={em:6.2f} n={n}")

    for name, dirs in ZS_MODEL_DIRS.items():
        missing = [d for d in dirs if not d.exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing dirs {missing}")
        preds = load_eea_zero_shot_predictions(dirs, val_keys)
        f1, em, n = micro_f1_em(gt_by_key, preds, tools)
        rows.append({"model": name, "micro_f1": f1, "exact_match": em, "n_frames": n})
        print(f"{name:32s} F1={f1:6.2f} EM={em:6.2f} n={n}")

    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "micro_f1", "exact_match", "n_frames"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
