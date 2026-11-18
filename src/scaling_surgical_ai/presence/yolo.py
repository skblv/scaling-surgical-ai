"""Train the YOLOv12m presence baseline on PitVis-2023.

PitVis-2023 only has binary instrument labels, so this module writes
synthetic full-frame boxes and trains YOLOv12m. It produces the
PitVis-2023 YOLO exact-match, Jaccard, top-1, and micro-F1 results in
the paper. SurgVU reuses it by overriding the dataset paths, instrument
names, batch size, and epoch count.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml
from tqdm.auto import tqdm
from ultralytics import YOLO

from scaling_surgical_ai import paths

RANDOM_SEED = 42

PITVIS_ROOT = paths.pitvis_root()
DATASET_CSV = PITVIS_ROOT / "dataset.csv"

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

MODEL_NAME = "yolo12m.pt"
EPOCHS = 300
BATCH_SIZE = 32
IMGSZ = 640
WORKERS = 8

NUM_GPUS = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_yolo_dataset_structure(output_base: Path) -> Tuple[Path, Path]:
    """Create YOLO-compatible dataset structure with synthetic full-frame labels.

    PitVis-2023 only has binary presence labels, no bounding boxes.
    We create synthetic full-frame boxes (0.5 0.5 1.0 1.0) for each present instrument.
    """
    logger.info(f"Loading PitVis-2023 dataset from {DATASET_CSV}")
    frame_label_df = pd.read_csv(DATASET_CSV)

    frame_stride = int(os.environ.get("FRAME_STRIDE", "1"))
    if frame_stride > 1:
        before = len(frame_label_df)
        frame_label_df = frame_label_df[
            frame_label_df["frame_id"] % frame_stride == 0
        ].reset_index(drop=True)
        logger.info(
            f"FRAME_STRIDE={frame_stride}: filtered "
            f"{before} -> {len(frame_label_df)} frames"
        )

    frame_label_train_df = frame_label_df[frame_label_df["split"] == "train"]
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"]

    logger.info(
        f"Train: {len(frame_label_train_df)} frames, "
        f"Val: {len(frame_label_val_df)} frames"
    )

    class_names = [INSTRUMENT_NAMES[i] for i in sorted(INSTRUMENT_NAMES.keys())]
    num_classes = len(class_names)

    name_to_idx = {name: idx for idx, name in INSTRUMENT_NAMES.items()}

    dataset_dir = output_base / "dataset"
    train_images_dir = dataset_dir / "images" / "train"
    val_images_dir = dataset_dir / "images" / "val"
    train_labels_dir = dataset_dir / "labels" / "train"
    val_labels_dir = dataset_dir / "labels" / "val"

    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def create_data(
        split_frame_label_df: pd.DataFrame,
        images_dir: Path,
        labels_dir: Path,
        desc: str,
    ) -> int:
        """Create image symlinks and synthetic full-frame YOLO labels.

        Args:
            split_frame_label_df: Frames belonging to one split.
            images_dir: Directory that receives image symlinks.
            labels_dir: Directory that receives YOLO label files.
            desc: Progress-bar description.

        Returns:
            Number of frames written.
        """
        count = 0
        for _, row in tqdm(
            split_frame_label_df.iterrows(),
            total=len(split_frame_label_df),
            desc=desc,
        ):
            frame_path = Path(row["frame_path"])

            if not frame_path.exists():
                logger.warning("Missing frame skipped: %s", frame_path)
                continue

            unique_name = f"{row['video_id']}_{frame_path.name}"
            img_dst = images_dir / unique_name
            lbl_dst = labels_dir / f"{Path(unique_name).stem}.txt"

            if not img_dst.exists():
                img_dst.symlink_to(frame_path)

            tools = json.loads(row["truth_label"])
            with lbl_dst.open("w") as f:
                for tool in tools:
                    if tool in name_to_idx:
                        class_idx = name_to_idx[tool]
                        f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

            count += 1
        return count

    train_count = create_data(
        frame_label_train_df, train_images_dir, train_labels_dir, "Train data"
    )
    val_count = create_data(
        frame_label_val_df, val_images_dir, val_labels_dir, "Val data"
    )

    logger.info(f"Created {train_count} train, {val_count} val samples")

    dataset_yaml = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": num_classes,
    }

    dataset_yaml_path = dataset_dir / "dataset.yaml"
    with dataset_yaml_path.open("w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    return dataset_yaml_path, dataset_dir


def compute_all_metrics(
    model_path: Path,
    dataset_yaml: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> Dict:
    """Compute exact-match, Jaccard, top-1, and per-tool metrics.

    Args:
        model_path: Trained YOLO checkpoint.
        dataset_yaml: YOLO dataset configuration.
        output_dir: Directory for prediction and metric files.
        conf_threshold: Detection confidence threshold.

    Returns:
        Exact-match accuracy, Jaccard, top-1 accuracy, micro-F1, and
        per-tool precision, recall, and F1.
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING ALL METRICS")
    logger.info("=" * 80)

    model = YOLO(str(model_path))
    all_tool_names = set(INSTRUMENT_NAMES.values())

    with dataset_yaml.open("r") as f:
        dataset_config = yaml.safe_load(f)

    val_images_dir = Path(dataset_config["path"]) / dataset_config["val"]
    val_labels_dir = val_images_dir.parent.parent / "labels" / "val"

    image_files = list(val_images_dir.glob("*.jpg")) + list(
        val_images_dir.glob("*.png")
    )
    logger.info(f"Found {len(image_files)} validation images")

    exact_matches = 0
    jaccard_sum = 0.0
    top1_correct = 0
    top1_total = 0
    total = 0
    predictions_list = []

    per_tool_tp = defaultdict(int)
    per_tool_fp = defaultdict(int)
    per_tool_fn = defaultdict(int)
    per_tool_tn = defaultdict(int)

    for img_path in tqdm(image_files, desc="Computing metrics"):
        label_path = val_labels_dir / f"{img_path.stem}.txt"

        gt_class_ids = set()
        if label_path.exists():
            with label_path.open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        gt_class_ids.add(int(parts[0]))

        gt_tools = set(
            INSTRUMENT_NAMES.get(cid, f"class_{cid}") for cid in gt_class_ids
        )

        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False,
            imgsz=IMGSZ,
        )

        pred_class_ids = set()
        top1_class_id = None
        top1_conf = -1.0

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                pred_class_ids.add(class_id)
                if conf > top1_conf:
                    top1_conf = conf
                    top1_class_id = class_id

        pred_tools = set(
            INSTRUMENT_NAMES.get(cid, f"class_{cid}") for cid in pred_class_ids
        )

        is_exact_match = pred_tools == gt_tools
        if is_exact_match:
            exact_matches += 1

        union = pred_tools | gt_tools
        jaccard = len(pred_tools & gt_tools) / len(union) if union else 1.0
        jaccard_sum += jaccard

        if top1_class_id is not None:
            top1_tool = INSTRUMENT_NAMES.get(top1_class_id, f"class_{top1_class_id}")
            top1_in_gt = top1_tool in gt_tools
            if top1_in_gt:
                top1_correct += 1
            top1_total += 1
        elif len(gt_tools) == 0:
            top1_correct += 1
            top1_total += 1
        else:
            top1_total += 1

        for tool in all_tool_names:
            in_gt = tool in gt_tools
            in_pred = tool in pred_tools
            if in_gt and in_pred:
                per_tool_tp[tool] += 1
            elif in_pred and not in_gt:
                per_tool_fp[tool] += 1
            elif in_gt and not in_pred:
                per_tool_fn[tool] += 1
            else:
                per_tool_tn[tool] += 1

        total += 1

        predictions_list.append(
            {
                "image": img_path.name,
                "ground_truth": sorted(gt_tools),
                "predicted": sorted(pred_tools),
                "exact_match": is_exact_match,
                "jaccard": jaccard,
                "top1_tool": (
                    INSTRUMENT_NAMES.get(top1_class_id, None)
                    if top1_class_id is not None
                    else None
                ),
                "top1_conf": top1_conf if top1_class_id is not None else None,
            }
        )

    exact_accuracy = (exact_matches / total * 100) if total > 0 else 0.0
    jaccard_accuracy = (jaccard_sum / total * 100) if total > 0 else 0.0
    top1_accuracy = (top1_correct / top1_total * 100) if top1_total > 0 else 0.0

    logger.info(f"\nExact Match Accuracy: {exact_accuracy:.2f}%")
    logger.info(f"Jaccard Similarity: {jaccard_accuracy:.2f}%")
    logger.info(f"Top-1 Accuracy: {top1_accuracy:.2f}%")

    per_tool_metrics = {}
    logger.info("\n" + "-" * 60)
    logger.info("PER-TOOL METRICS")
    logger.info("-" * 60)
    logger.info(f"{'Tool':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    logger.info("-" * 60)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for tool in sorted(all_tool_names):
        tp = per_tool_tp[tool]
        fp = per_tool_fp[tool]
        fn = per_tool_fn[tool]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = tp + fn

        per_tool_metrics[tool] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        if support > 0:
            logger.info(
                f"{tool:<30} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f} {support:>8}"
            )

    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    logger.info("-" * 60)
    logger.info(
        f"{'MICRO AVG':<30} {micro_precision:>8.3f} "
        f"{micro_recall:>8.3f} {micro_f1:>8.3f}"
    )

    predictions_file = output_dir / "predictions_detailed.json"
    with predictions_file.open("w") as f:
        json.dump(predictions_list, f, indent=2)

    per_tool_csv = output_dir / "per_tool_metrics.csv"
    per_tool_rows = []
    for tool, m in per_tool_metrics.items():
        per_tool_rows.append(
            {
                "tool": tool,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "support": m["support"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
    tool_metric_df = pd.DataFrame(per_tool_rows)
    tool_metric_df.to_csv(per_tool_csv, index=False)
    logger.info(f"\nPer-tool metrics saved to: {per_tool_csv}")

    return {
        "exact_match_accuracy": exact_accuracy,
        "jaccard_accuracy": jaccard_accuracy,
        "top1_accuracy": top1_accuracy,
        "total_images": total,
        "exact_matches": exact_matches,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "per_tool_metrics": per_tool_metrics,
    }


def main() -> None:
    """Train YOLOv12m and write detection and presence metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID to evaluate (for --eval-only mode)",
    )
    args = parser.parse_args()

    if args.job_id:
        job_id = args.job_id
    else:
        job_id = os.environ.get("SLURM_JOB_ID", "local")

    OUTPUT_DIR = paths.experiment_output_dir("pitvis_yolo", job_id)

    logger.info("=" * 80)
    logger.info("PitVis-2023 YOLOv12m baseline training")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_yaml, dataset_dir = create_yolo_dataset_structure(OUTPUT_DIR)

    best_model = OUTPUT_DIR / "train" / "weights" / "best.pt"
    results_csv = OUTPUT_DIR / "train" / "results.csv"

    training_complete = best_model.exists() and results_csv.exists()
    if training_complete:
        epoch_metric_df = pd.read_csv(results_csv)
        if len(epoch_metric_df) >= EPOCHS:
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
        model = YOLO(MODEL_NAME)

        if NUM_GPUS > 1:
            device_str = ",".join(str(i) for i in range(NUM_GPUS))
        else:
            device_str = "0"

        effective_batch = BATCH_SIZE

        logger.info("\nStarting training:")
        logger.info(f"  Model: {MODEL_NAME}")
        logger.info(f"  Epochs: {EPOCHS}")
        logger.info(
            f"  Batch size: {effective_batch} "
            f"({effective_batch // max(NUM_GPUS, 1)}/GPU x {NUM_GPUS} GPUs)"
        )
        logger.info(f"  Image size: {IMGSZ}")
        logger.info(f"  Device(s): {device_str}")

        results = model.train(
            data=str(dataset_yaml),
            epochs=EPOCHS,
            batch=effective_batch,
            imgsz=IMGSZ,
            project=str(OUTPUT_DIR),
            name="train",
            exist_ok=True,
            save=True,
            plots=True,
            workers=WORKERS,
            seed=RANDOM_SEED,
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
        else:
            if results_csv.exists():
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
        eval_results = compute_all_metrics(
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
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "imgsz": IMGSZ,
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

    logger.info("\nPitVis-2023 YOLO complete")


if __name__ == "__main__":
    main()
