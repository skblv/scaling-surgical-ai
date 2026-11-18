"""Train the YOLOv12m presence baseline on SurgVU.

SurgVU has 17 instrument classes and binary presence labels. This
script points the PitVis YOLO module at SurgVU, builds synthetic
full-frame boxes, and trains YOLOv12m. It produces the SurgVU YOLO
exact-match, Jaccard, top-1, and micro-F1 results in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import pandas as pd
from ultralytics import YOLO

from scaling_surgical_ai import paths
from scaling_surgical_ai.presence import yolo as base

SURGVU_ROOT = paths.surgvu_root()

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

base.PITVIS_ROOT = SURGVU_ROOT
base.DATASET_CSV = SURGVU_ROOT / "dataset.csv"
base.INSTRUMENT_NAMES = INSTRUMENT_NAMES
base.BATCH_SIZE = 64
base.EPOCHS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train YOLOv12m on SurgVU and write detection and presence metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID to evaluate (for --eval-only mode)",
    )
    args = parser.parse_args()

    job_id = args.job_id or os.environ.get("SLURM_JOB_ID", "local")
    OUTPUT_DIR = paths.experiment_output_dir("surgvu_yolo", job_id)

    logger.info("=" * 80)
    logger.info("SurgVU YOLOv12m baseline training")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_yaml, dataset_dir = base.create_yolo_dataset_structure(OUTPUT_DIR)

    best_model = OUTPUT_DIR / "train" / "weights" / "best.pt"
    results_csv = OUTPUT_DIR / "train" / "results.csv"

    training_complete = best_model.exists() and results_csv.exists()
    if training_complete:
        epoch_metric_df = pd.read_csv(results_csv)
        if len(epoch_metric_df) >= base.EPOCHS:
            logger.info(
                "\nTraining already complete! "
                f"Found {len(epoch_metric_df)} epochs in results.csv"
            )
            logger.info("Skipping training, proceeding to evaluation...")
            last_row = epoch_metric_df.iloc[-1]
            metrics = {
                "map50": float(last_row.get("metrics/mAP50(B)", 0)),
                "map50_95": float(last_row.get("metrics/mAP50-95(B)", 0)),
                "precision": float(last_row.get("metrics/precision(B)", 0)),
                "recall": float(last_row.get("metrics/recall(B)", 0)),
            }
        else:
            training_complete = False

    if not training_complete:
        model = YOLO(base.MODEL_NAME)

        if base.NUM_GPUS > 1:
            device_str = ",".join(str(i) for i in range(base.NUM_GPUS))
        else:
            device_str = "0"

        effective_batch = base.BATCH_SIZE

        logger.info("\nStarting training:")
        logger.info(f"  Model: {base.MODEL_NAME}")
        logger.info(f"  Epochs: {base.EPOCHS}")
        logger.info(
            f"  Batch size: {effective_batch} "
            f"({effective_batch // max(base.NUM_GPUS, 1)}/GPU "
            f"x {base.NUM_GPUS} GPUs)"
        )
        logger.info(f"  Image size: {base.IMGSZ}")
        logger.info(f"  Device(s): {device_str}")

        results = model.train(
            data=str(dataset_yaml),
            epochs=base.EPOCHS,
            batch=effective_batch,
            imgsz=base.IMGSZ,
            project=str(OUTPUT_DIR),
            name="train",
            exist_ok=True,
            save=True,
            plots=True,
            workers=base.WORKERS,
            seed=base.RANDOM_SEED,
            verbose=True,
            device=device_str,
            amp=True,
            cache=False,
            patience=20,
            cos_lr=True,
            close_mosaic=10,
        )

        if (
            results is not None
            and hasattr(results, "results_dict")
            and results.results_dict
        ):
            metrics = {
                "map50": results.results_dict.get("metrics/mAP50(B)", 0.0),
                "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0.0),
                "precision": results.results_dict.get("metrics/precision(B)", 0.0),
                "recall": results.results_dict.get("metrics/recall(B)", 0.0),
            }
        elif results_csv.exists():
            epoch_metric_df = pd.read_csv(results_csv)
            last_row = epoch_metric_df.iloc[-1]
            metrics = {
                "map50": float(last_row.get("metrics/mAP50(B)", 0)),
                "map50_95": float(last_row.get("metrics/mAP50-95(B)", 0)),
                "precision": float(last_row.get("metrics/precision(B)", 0)),
                "recall": float(last_row.get("metrics/recall(B)", 0)),
            }
        else:
            logger.warning("Training results not available")
            metrics = {
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

    logger.info("\nTraining complete!")
    logger.info(f"  mAP50: {metrics['map50']:.4f}")
    logger.info(f"  mAP50-95: {metrics['map50_95']:.4f}")

    best_model = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if best_model.exists():
        eval_results = base.compute_all_metrics(
            model_path=best_model,
            dataset_yaml=dataset_yaml,
            output_dir=OUTPUT_DIR,
        )
    else:
        eval_results = {}

    eval_results_for_json = {
        k: v for k, v in eval_results.items() if k != "per_tool_metrics"
    }

    results_summary = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "model": base.MODEL_NAME,
        "epochs": base.EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "imgsz": base.IMGSZ,
        "training_metrics": metrics,
        "eval_results": eval_results_for_json,
    }

    with (OUTPUT_DIR / "results_summary.json").open("w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"mAP50: {metrics['map50']:.4f}")
    if eval_results:
        logger.info(
            f"Exact Match Accuracy: {eval_results['exact_match_accuracy']:.2f}%"
        )
        logger.info(f"Jaccard Similarity: {eval_results['jaccard_accuracy']:.2f}%")
        logger.info(f"Top-1 Accuracy: {eval_results['top1_accuracy']:.2f}%")
        logger.info(f"Micro F1: {eval_results['micro_f1']:.4f}")
    logger.info(f"\nBest model: {best_model}")
    logger.info("\nSurgVU YOLO complete")


if __name__ == "__main__":
    main()
