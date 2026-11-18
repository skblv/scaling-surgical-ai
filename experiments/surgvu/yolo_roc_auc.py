"""Compute ROC-AUC for the SurgVU YOLOv12m model.

Uses the maximum detection confidence of each class in a frame as the
continuous score. Overrides the PitVis ROC-AUC module with the SurgVU
instrument names and output directory. This is the SurgVU YOLO ROC-AUC
result in the paper.
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
from scaling_surgical_ai.presence import yolo_roc_auc as base

INSTRUMENT_NAMES = {
    0: "bipolar dissector",
    1: "bipolar forceps",
    2: "cadiere forceps",
    3: "clip applier",
    4: "force bipolar",
    5: "grasping retractor",
    6: "monopolar curved scissors",
    7: "needle driver",
    8: "permanent cautery hook/spatula",
    9: "potts scissors",
    10: "prograsp forceps",
    11: "stapler",
    12: "suction irrigator",
    13: "synchroseal",
    14: "tenaculum forceps",
    15: "tip-up fenestrated grasper",
    16: "vessel sealer",
}

OUTPUT_DIR = paths.experiment_output_dir("surgvu_yolo_roc_auc")

base.INSTRUMENT_NAMES = INSTRUMENT_NAMES
base.NUM_CLASSES = len(INSTRUMENT_NAMES)
base.OUTPUT_DIR = OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Score the SurgVU YOLO checkpoint and write per-tool ROC-AUC."""
    parser = argparse.ArgumentParser(
        description="Compute ROC-AUC for SurgVU YOLO model"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Job ID of the SurgVU YOLO training run",
    )
    args = parser.parse_args()

    yolo_output = paths.experiment_output_dir("surgvu_yolo", args.job_id)
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
    num_classes = len(INSTRUMENT_NAMES)

    for img_path in tqdm(image_files, desc="Inference"):
        label_path = val_labels_dir / f"{img_path.stem}.txt"

        gt_vec = np.zeros(num_classes)
        if label_path.exists():
            with label_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < num_classes:
                            gt_vec[cls_id] = 1.0

        results = model.predict(
            source=str(img_path),
            conf=0.001,
            verbose=False,
            imgsz=base.IMGSZ,
        )

        score_vec = np.zeros(num_classes)
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id < num_classes:
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
    for i in range(num_classes):
        tool = INSTRUMENT_NAMES[i]
        n_pos = gt_matrix[:, i].sum()
        n_neg = len(gt_matrix) - n_pos
        if n_pos > 0 and n_neg > 0:
            auc = roc_auc_score(gt_matrix[:, i], score_matrix[:, i])
            auc_values.append(auc)
            per_tool_auc[tool] = auc
            logger.info(
                f"  {tool:<35} ROC-AUC: {auc:.4f}  (pos={int(n_pos)}, neg={int(n_neg)})"
            )
        else:
            per_tool_auc[tool] = None
            logger.info(
                f"  {tool:<35} ROC-AUC: N/A  (pos={int(n_pos)}, neg={int(n_neg)})"
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
