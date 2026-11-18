"""Compute per-tool and macro ROC-AUC for YOLOv12-m on SDSC-EEA.

Uses the maximum detection confidence of each class in a frame as the
continuous score, on the reported full-ontology checkpoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from ultralytics import YOLO

from scaling_surgical_ai.paths import experiment_output_dir

NUM_CLASSES = 31
IMGSZ = 1280

MODEL_PATH = (
    experiment_output_dir("sdsc_eea_yolo_full_ontology", "58525")
    / "train"
    / "weights"
    / "best.pt"
)
DATASET_YAML = (
    experiment_output_dir("sdsc_eea_yolo_full_ontology", "58525")
    / "dataset"
    / "dataset.yaml"
)
OUTPUT_DIR = experiment_output_dir("sdsc_eea_yolo_roc_auc")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Score the validation set and write per-tool ROC-AUC."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Dataset: {DATASET_YAML}")

    model = YOLO(str(MODEL_PATH))

    with DATASET_YAML.open("r") as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["names"]
    logger.info(f"Classes: {len(class_names)}")

    val_images_dir = Path(cfg["path"]) / cfg["val"]
    val_labels_dir = val_images_dir.parent.parent / "labels" / "val"

    image_files = sorted(
        list(val_images_dir.glob("*.png"))
        + list(val_images_dir.glob("*.jpg"))
        + list(val_images_dir.glob("*.jpeg"))
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
        tool = class_names[i]
        n_pos = gt_matrix[:, i].sum()
        n_neg = len(gt_matrix) - n_pos
        if n_pos > 0 and n_neg > 0:
            auc = roc_auc_score(gt_matrix[:, i], score_matrix[:, i])
            auc_values.append(auc)
            per_tool_auc[tool] = round(auc, 4)
            logger.info(
                f"  {tool:<30} ROC-AUC: {auc:.4f}  (pos={int(n_pos)}, neg={int(n_neg)})"
            )
        else:
            per_tool_auc[tool] = None
            logger.info(
                f"  {tool:<30} ROC-AUC: N/A  (pos={int(n_pos)}, neg={int(n_neg)})"
            )

    macro_auc = float(np.mean(auc_values))
    logger.info(f"\nMacro ROC-AUC: {macro_auc:.4f}")
    logger.info(f"Tools with ROC-AUC: {len(auc_values)} / {NUM_CLASSES}")

    results_out = {
        "per_tool_roc_auc": per_tool_auc,
        "macro_roc_auc": round(macro_auc, 4),
        "n_images": len(image_files),
        "model": str(MODEL_PATH),
    }

    output_file = OUTPUT_DIR / "roc_auc_results.json"
    with output_file.open("w") as f:
        json.dump(results_out, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
