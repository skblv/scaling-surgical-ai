"""Point-estimate micro-F1 and exact match for the Figure 4 model set.

Micro-F1 is pooled over a tool set, matching Figure 2:
    - SDSC-EEA: the 15 EEA TD v4.2.4 evaluable tools.
    - CholecT50 / PitVis-2023 / SurgVU: each dataset's full canonical ontology.

Exact match is recomputed over the same tool set as a self-check against the
published numbers. The manuscript keeps its existing exact-match values, and
the exact-match counterpart of this comparison is Supplementary Figure S3.

YOLO predictions are read from each run's ``predictions_detailed.json`` (which
already carries per-frame ground_truth/predicted tool lists). All other model
predictions are loaded through ``scaling_surgical_ai.prediction_loaders`` and
joined to the validation ground truth on the canonical frame key.

Run on an HPC compute node (CPU only).
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

from scaling_surgical_ai import paths
from scaling_surgical_ai.pooled_f1 import (
    EEA_15_TOOLS,
    majority_tool_baseline,
    micro_f1_em,
    yolo_micro_f1_em,
)
from scaling_surgical_ai.prediction_loaders import (
    CHOLECT_ZERO_SHOT_VLM_DIR,
    EEA_ZERO_SHOT_VLM_DIR,
    PITVIS_ZERO_SHOT_VLM_DIR,
    SURGVU_ZERO_SHOT_VLM_DIR,
    get_dataset_spec,
    load_eea_zero_shot_predictions,
    load_lora_jsonl_predictions,
    load_pitvis_cholect_surgvu_vlm_predictions,
    load_val_frame_table,
)

EEA_CLASSIFICATION_HEAD_DIR = paths.experiment_output_dir(
    "sdsc_eea_lora_classification_head", "74900"
)

YOLO_DIRS = {
    "eea": paths.experiment_output_dir("sdsc_eea_yolo_full_ontology", "58525"),
    "cholect50": paths.experiment_output_dir("cholect50_yolo", "60464"),
    "pitvis": paths.experiment_output_dir("pitvis_yolo", "92920"),
    "surgvu": paths.experiment_output_dir("surgvu_yolo", "102001"),
}

FRONTIER_INPUTS = {
    "cholect50": paths.experiment_output_dir("cholect50_frontier_vlm_openrouter"),
    "pitvis": paths.experiment_output_dir("pitvis_frontier_vlm_openrouter"),
    "surgvu": paths.experiment_output_dir("surgvu_frontier_vlm_openrouter"),
}
FRONTIER_NAMES = [
    "gemini-3_1-pro",
    "gemini-3-flash",
    "claude-opus-4_6",
    "claude-sonnet-4_6",
    "gpt-5_4",
]

ZERO_SHOT_GEMMA = "gemma-3-27b-it_temp1_0"

LORA_GLOBS = {
    "cholect50": str(paths.output_root() / "cholect50_lora_classification_head_*"),
    "pitvis": str(paths.output_root() / "pitvis_lora_classification_head_*"),
    "surgvu": str(paths.output_root() / "surgvu_lora_classification_head_*"),
}
LORA_EM_TARGET = {"cholect50": 83.02, "pitvis": 84.77, "surgvu": 50.61}


def load_cls_head_jsonl(job_dir: Path, val_keys: List[str]) -> Dict[str, List[str]]:
    """Load a LoRA/classification-head ``val_predictions_after_training.jsonl``."""
    return load_lora_jsonl_predictions(job_dir, val_keys)


def pick_lora_job(
    glob_pat: str, gt_by_key, tools, em_target
) -> Tuple[Path, float, float, int]:
    """Among rank-sweep LoRA jobs, pick the one whose EM matches the published value."""
    best = None
    for path in sorted(Path(p) for p in glob(glob_pat)):
        jsonl = path / "val_predictions_after_training.jsonl"
        if not jsonl.exists():
            continue
        preds = load_lora_jsonl_predictions(path, list(gt_by_key.keys()))
        if len(preds) < 0.5 * len(gt_by_key):
            continue
        f1, em, n = micro_f1_em(gt_by_key, preds, tools)
        if best is None or abs(em - em_target) < abs(best[2] - em_target):
            best = (path, f1, em, n)
    return best


def run_dataset(name: str) -> List[dict]:
    """Compute F1 + EM for every Fig. 3 model on one dataset."""
    spec = get_dataset_spec(name)
    tools = EEA_15_TOOLS if name == "eea" else spec.valid_tools
    frame_label_val_df = load_val_frame_table(spec)
    gt_by_key = dict(
        zip(frame_label_val_df["frame_key"], frame_label_val_df["gt_tools"])
    )
    val_keys = list(gt_by_key.keys())
    rows: List[dict] = []

    majority, base_f1 = majority_tool_baseline(gt_by_key, tools)
    print(f"  {name:10s} [majority-tool baseline = '{majority}': F1={base_f1:.2f}]")
    rows.append(
        {
            "dataset": name,
            "model": f"__baseline__ ({majority})",
            "micro_f1": base_f1,
            "exact_match": None,
            "n_frames": len(gt_by_key),
        }
    )

    def add(model: str, f1: float, em: float, n: int) -> None:
        """Record one model and print its scores.

        Args:
            model: Display name.
            f1: Micro-F1 in percent.
            em: Exact match in percent.
            n: Number of frames scored.
        """
        rows.append(
            {
                "dataset": name,
                "model": model,
                "micro_f1": f1,
                "exact_match": em,
                "n_frames": n,
            }
        )
        print(f"  {name:10s} {model:34s} F1={f1:6.2f}  EM={em:6.2f}  n={n}")

    f1, em, n = yolo_micro_f1_em(YOLO_DIRS[name], tools)
    add("YOLOv12-m", f1, em, n)

    if name == "eea":
        preds = load_eea_zero_shot_predictions(
            [EEA_ZERO_SHOT_VLM_DIR / ZERO_SHOT_GEMMA], val_keys
        )
        add("Gemma 3 27B-it (zero-shot)", *micro_f1_em(gt_by_key, preds, tools))
        cls_preds = load_cls_head_jsonl(EEA_CLASSIFICATION_HEAD_DIR, val_keys)
        add("Gemma 3 27B + LoRA", *micro_f1_em(gt_by_key, cls_preds, tools))
    else:
        zs_base = {
            "cholect50": CHOLECT_ZERO_SHOT_VLM_DIR,
            "pitvis": PITVIS_ZERO_SHOT_VLM_DIR,
            "surgvu": SURGVU_ZERO_SHOT_VLM_DIR,
        }[name]
        zs_model_dir = zs_base / ZERO_SHOT_GEMMA
        preds = load_pitvis_cholect_surgvu_vlm_predictions(zs_model_dir, val_keys)
        add("Gemma 3 27B-it (zero-shot)", *micro_f1_em(gt_by_key, preds, tools))

        job, f1, em, n = pick_lora_job(
            LORA_GLOBS[name], gt_by_key, tools, LORA_EM_TARGET[name]
        )
        print(f"    [LoRA pick: {job.name}]")
        add("Gemma 3 27B + LoRA", f1, em, n)

        for fname in FRONTIER_NAMES:
            mdir = FRONTIER_INPUTS[name] / fname
            if not (mdir / "results.csv").exists():
                print(f"  {name}: MISSING frontier dir {fname}")
                continue
            preds = load_pitvis_cholect_surgvu_vlm_predictions(mdir, val_keys)
            add(fname, *micro_f1_em(gt_by_key, preds, tools))
    return rows


def main() -> None:
    """Score every dataset and write ``cross_dataset_f1.json``."""
    all_rows: List[dict] = []
    for name in ["eea", "cholect50", "pitvis", "surgvu"]:
        print(
            f"=== {name} (tools={'15 v4.2.4' if name == 'eea' else 'full ontology'}) ==="
        )
        all_rows.extend(run_dataset(name))
    out = paths.output_root() / "cross_dataset_f1.json"
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
