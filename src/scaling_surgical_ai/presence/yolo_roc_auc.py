"""Compute ROC-AUC for the PitVis-2023 YOLOv12m model.

Uses the maximum detection confidence of each class in a frame as the
continuous score. This is the PitVis-2023 YOLO ROC-AUC result in the
paper. SurgVU reuses this module by overriding the instrument names
and the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from ultralytics import YOLO

from scaling_surgical_ai import paths

INSTRUMENT_NAMES = {
    0: "bipolar_forceps",
    1: "cottle",
    2: "cup_forceps",
    3: "dural_scissors",
    4: "freer_elevator",
    5: "haemostatic_foam",
    6: "irrigation_syringe",
    7: "kerrisons",
    8: "micro_doppler",
    9: "nasal_cutting_forceps",
    10: "pituitary_rongeurs",
    11: "retractable_knife",
    12: "ring_curette",
    13: "spatula_dissector",
    14: "stealth_pointer",
    15: "suction",
    16: "surgical_drill",
    17: "tissue_glue",
}
NUM_CLASSES = len(INSTRUMENT_NAMES)
IMGSZ = 640

OUTPUT_DIR = paths.experiment_output_dir("pitvis_yolo_roc_auc")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Score the PitVis YOLO checkpoint and write per-tool ROC-AUC."""
    parser = argparse.ArgumentParser(
        description="Compute ROC-AUC for PitVis-2023 YOLO model"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Job ID of the PitVis YOLO training run",
    )
    args = parser.parse_args()

    yolo_output = paths.experiment_output_dir("pitvis_yolo", args.job_id)
    model_path = yolo_output / "train" / "weights" / "best.pt"
    dataset_yaml_path = yolo_output / "dataset" / "dataset.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {dataset_yaml_path}")

    model = YOLO(str(model_path))

    with dataset_yaml_path.open("r") as f:
        cfg = yaml.safe_load(f)

    val_images_dir = Path(cfg["path"]) / cfg["val"]
    val_labels_dir = val_images_dir.parent.parent / "labels" / "val"

    image_files = sorted(
        list(val_images_dir.glob("*.png")) + list(val_images_dir.glob("*.jpg"))
    )
    logger.info(f"Found {len(image_files)} validation images")

    gt_matrix = []
    score_matrix = []

    for img_path in tqdm(image_files, desc="Inference"):
        label_path = val_labels_dir / f"{img_path.stem}.txt"

        gt_vec = np.zeros(NUM_CLASSES)
        if label_path.exists():
            with label_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < NUM_CLASSES:
                            gt_vec[cls_id] = 1.0

        results = model.predict(
            source=str(img_path),
            conf=0.001,
            verbose=False,
            imgsz=IMGSZ,
        )

        score_vec = np.zeros(NUM_CLASSES)
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id < NUM_CLASSES:
                    score_vec[cls_id] = max(score_vec[cls_id], conf)

        gt_matrix.append(gt_vec)
        score_matrix.append(score_vec)

    gt_matrix = np.array(gt_matrix)
    score_matrix = np.array(score_matrix)

    logger.info("\n" + "=" * 60)
    logger.info("PER-TOOL ROC-AUC")
    logger.info("=" * 60)

    per_tool_auc = {}
    auc_values = []
    for i in range(NUM_CLASSES):
        tool = INSTRUMENT_NAMES[i]
        n_pos = gt_matrix[:, i].sum()
        n_neg = len(gt_matrix) - n_pos
        if n_pos > 0 and n_neg > 0:
            auc = roc_auc_score(gt_matrix[:, i], score_matrix[:, i])
            auc_values.append(auc)
            per_tool_auc[tool] = auc
            logger.info(
                f"  {tool:<30} ROC-AUC: {auc:.4f}  (pos={int(n_pos)}, neg={int(n_neg)})"
            )
        else:
            per_tool_auc[tool] = None
            logger.info(
                f"  {tool:<30} ROC-AUC: N/A  (pos={int(n_pos)}, neg={int(n_neg)})"
            )

    macro_auc = float(np.mean(auc_values)) if auc_values else 0.0
    logger.info(f"\nMacro ROC-AUC: {macro_auc:.4f}")

    results_out = {
        "per_tool_roc_auc": per_tool_auc,
        "macro_roc_auc": macro_auc,
        "n_images": len(image_files),
        "source_job_id": args.job_id,
    }

    output_file = OUTPUT_DIR / "roc_auc_results.json"
    with output_file.open("w") as f:
        json.dump(results_out, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
