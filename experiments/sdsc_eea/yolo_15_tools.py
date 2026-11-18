"""Train YOLOv12-m on the 15 evaluable SDSC-EEA tools.

Label files are filtered to the 15-tool taxonomy and remapped to class
ids 0-14. Frames that lose every label stay as background images, so the
full-ontology run is left untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yaml
from tqdm.auto import tqdm
from ultralytics import YOLO

from scaling_surgical_ai.paths import (
    eea_class_mapping_path,
    experiment_output_dir,
)

RANDOM_SEED = 42

DATASET_CSV = experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
YOLO_MAPPING_FILE = eea_class_mapping_path()

V424_TOOLS_ORDERED = [
    "Suction",
    "Suction Coagulator",
    "Grasper",
    "Scissor",
    "Rongeur",
    "Rhoton Dissector",
    "Curette",
    "Doppler",
    "Straight Forceps",
    "Surgical Knife",
    "Drill",
    "Sonopet- pineapple tip",
    "Bipolar Forceps",
    "Cotton Patty",
    "Surgicel",
]

MODEL_NAME = "yolo12m.pt"
EPOCHS = 300
BATCH_SIZE = 32
IMGSZ = 1280
WORKERS = 8

NUM_GPUS = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_old_id_to_tool() -> Dict[int, str]:
    """Load the original YOLO class id to tool name mapping.

    Returns:
        Class id keyed to the tool name.
    """
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["mapping"].items()}


def build_class_remap() -> Tuple[Dict[int, int], List[str]]:
    """Map the 15 evaluable tools onto sequential class ids.

    Returns:
        Old class id keyed to the new id, and tool names in new-id order.

    Raises:
        RuntimeError: If a listed tool is missing from the YOLO mapping.
    """
    old_id_to_tool = load_old_id_to_tool()
    tool_to_old_id = {v: k for k, v in old_id_to_tool.items()}
    missing = [t for t in V424_TOOLS_ORDERED if t not in tool_to_old_id]
    if missing:
        raise RuntimeError(f"v4.2.4 tools not present in YOLO mapping: {missing}")
    old_to_new: Dict[int, int] = {}
    new_id_to_tool: List[str] = []
    for new_id, tool in enumerate(V424_TOOLS_ORDERED):
        old_to_new[tool_to_old_id[tool]] = new_id
        new_id_to_tool.append(tool)
    return old_to_new, new_id_to_tool


def get_label_path(frame_path: Path) -> Path:
    """Return the YOLO label path that corresponds to a frame.

    Args:
        frame_path: Path of the frame image.

    Returns:
        Path of the label file with the same stem.
    """
    if frame_path.parent.name == "images":
        label_dir = frame_path.parent.parent / "labels"
    else:
        label_dir = frame_path.parent / "labels"
    return label_dir / f"{frame_path.stem}.txt"


def filter_label_lines(
    src_lines: Iterable[str],
    old_to_new: Dict[int, int],
) -> List[str]:
    """Keep labels in the 15-tool set and remap them to sequential ids.

    Args:
        src_lines: Lines of one YOLO label file.
        old_to_new: Old class id keyed to the new class id.

    Returns:
        Label lines whose class id was rewritten.
    """
    out: List[str] = []
    for line in src_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id not in old_to_new:
            continue
        parts[0] = str(old_to_new[old_id])
        out.append(" ".join(parts))
    return out


def create_yolo_dataset_structure(output_base: Path) -> Tuple[Path, Path]:
    """Create a YOLO dataset whose labels contain only the 15 tools.

    Args:
        output_base: Experiment output directory that will hold ``dataset``.

    Returns:
        The dataset yaml path and the dataset directory.
    """
    logger.info(f"Loading dataset from {DATASET_CSV}")
    frame_dataset_df = pd.read_csv(DATASET_CSV)
    frame_dataset_train_df = frame_dataset_df[frame_dataset_df["split"] == "train"]
    frame_dataset_val_df = frame_dataset_df[frame_dataset_df["split"] == "val"]
    logger.info(
        f"Train: {len(frame_dataset_train_df)} frames, "
        f"Val: {len(frame_dataset_val_df)} frames"
    )

    old_to_new, class_names = build_class_remap()
    num_classes = len(class_names)
    logger.info(f"Filtered taxonomy: {num_classes} classes")
    for new_id, tool in enumerate(class_names):
        logger.info(f"  {new_id:2d}: {tool}")

    dataset_dir = output_base / "dataset"
    train_images_dir = dataset_dir / "images" / "train"
    val_images_dir = dataset_dir / "images" / "val"
    train_labels_dir = dataset_dir / "labels" / "train"
    val_labels_dir = dataset_dir / "labels" / "val"
    for d in (train_images_dir, val_images_dir, train_labels_dir, val_labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    def materialize(
        frame_split_df: pd.DataFrame,
        images_dir: Path,
        labels_dir: Path,
        desc: str,
    ) -> Tuple[int, int, Dict[str, int]]:
        """Write filtered labels and symlink images for one split.

        Args:
            frame_split_df: Frames that belong to this split.
            images_dir: Destination directory for image symlinks.
            labels_dir: Destination directory for filtered label files.
            desc: Progress-bar description.

        Returns:
            Frame count, count of frames that still have a label, and the
            number of boxes of each tool.
        """
        n_frames = 0
        n_kept_frames = 0
        per_tool_counts = defaultdict(int)
        for _, row in tqdm(
            frame_split_df.iterrows(),
            total=len(frame_split_df),
            desc=desc,
        ):
            frame_path = Path(row["frame_path"])
            label_path = get_label_path(frame_path)
            if not frame_path.exists() or not label_path.exists():
                continue

            unique_name = f"{row['procedure_id']}_{frame_path.name}"
            img_dst = images_dir / unique_name
            lbl_dst = labels_dir / f"{Path(unique_name).stem}.txt"

            if not img_dst.exists():
                img_dst.symlink_to(frame_path)

            if not lbl_dst.exists():
                with label_path.open("r") as f:
                    new_lines = filter_label_lines(f.readlines(), old_to_new)
                with lbl_dst.open("w") as f:
                    if new_lines:
                        f.write("\n".join(new_lines) + "\n")
                for line in new_lines:
                    new_id = int(line.split()[0])
                    per_tool_counts[class_names[new_id]] += 1
            else:
                with lbl_dst.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        new_id = int(line.split()[0])
                        per_tool_counts[class_names[new_id]] += 1

            n_frames += 1
            if lbl_dst.stat().st_size > 0:
                n_kept_frames += 1

        skipped = len(frame_split_df) - n_frames
        if skipped:
            logger.warning(
                f"{desc}: skipped {skipped} frames with a missing image or label file"
            )
        return n_frames, n_kept_frames, dict(per_tool_counts)

    train_total, train_with_labels, train_per_tool = materialize(
        frame_dataset_train_df,
        train_images_dir,
        train_labels_dir,
        "Train (filtered)",
    )
    val_total, val_with_labels, val_per_tool = materialize(
        frame_dataset_val_df,
        val_images_dir,
        val_labels_dir,
        "Val (filtered)",
    )
    logger.info(f"Train: {train_total} frames ({train_with_labels} with kept labels)")
    logger.info(f"Val:   {val_total} frames ({val_with_labels} with kept labels)")

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

    stats_path = output_base / "label_filter_stats.json"
    with stats_path.open("w") as f:
        json.dump(
            {
                "v424_tools_ordered": V424_TOOLS_ORDERED,
                "train_total_frames": train_total,
                "train_frames_with_labels": train_with_labels,
                "val_total_frames": val_total,
                "val_frames_with_labels": val_with_labels,
                "train_per_tool_instances": train_per_tool,
                "val_per_tool_instances": val_per_tool,
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote {stats_path}")

    return dataset_yaml_path, dataset_dir


def compute_all_metrics(
    model_path: Path,
    dataset_yaml: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> Dict:
    """Score the validation set on the 15-tool taxonomy.

    Args:
        model_path: Checkpoint passed to Ultralytics.
        dataset_yaml: Dataset yaml written for this run.
        output_dir: Directory for the prediction and metric files.
        conf_threshold: Detection confidence threshold.

    Returns:
        Exact match, Jaccard, top-1, micro and macro F1, and per-tool metrics.
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING ALL METRICS (15-class taxonomy)")
    logger.info("=" * 80)

    model = YOLO(str(model_path))
    _, class_names = build_class_remap()
    new_id_to_tool = {i: t for i, t in enumerate(class_names)}
    all_tool_names = set(class_names)

    with dataset_yaml.open("r") as f:
        dataset_config = yaml.safe_load(f)
    val_images_dir = Path(dataset_config["path"]) / dataset_config["val"]
    val_labels_dir = val_images_dir.parent.parent / "labels" / "val"
    image_files = sorted(
        list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
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
        gt_tools = {new_id_to_tool[c] for c in gt_class_ids if c in new_id_to_tool}

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
        pred_tools = {new_id_to_tool[c] for c in pred_class_ids if c in new_id_to_tool}

        is_exact_match = pred_tools == gt_tools
        if is_exact_match:
            exact_matches += 1
        union = pred_tools | gt_tools
        jaccard = len(pred_tools & gt_tools) / len(union) if union else 1.0
        jaccard_sum += jaccard

        if top1_class_id is not None:
            top1_tool = new_id_to_tool.get(top1_class_id)
            if top1_tool in gt_tools:
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
                "top1_tool": new_id_to_tool.get(top1_class_id)
                if top1_class_id is not None
                else None,
                "top1_conf": top1_conf if top1_class_id is not None else None,
            }
        )

    exact_accuracy = (exact_matches / total * 100) if total > 0 else 0.0
    jaccard_accuracy = (jaccard_sum / total * 100) if total > 0 else 0.0
    top1_accuracy = (top1_correct / top1_total * 100) if top1_total > 0 else 0.0

    logger.info(f"\nExact Match Accuracy (15-tool): {exact_accuracy:.2f}%")
    logger.info(f"Jaccard Similarity (15-tool):   {jaccard_accuracy:.2f}%")
    logger.info(f"Top-1 Accuracy:                 {top1_accuracy:.2f}%")

    per_tool_metrics = {}
    total_tp = total_fp = total_fn = 0
    logger.info(f"\n{'Tool':<28} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    logger.info("-" * 60)
    for tool in class_names:
        tp = per_tool_tp[tool]
        fp = per_tool_fp[tool]
        fn = per_tool_fn[tool]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp + fn
        per_tool_metrics[tool] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        logger.info(f"{tool:<28} {p:>8.3f} {r:>8.3f} {f1:>8.3f} {support:>8}")

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )
    logger.info("-" * 60)
    logger.info(f"{'MICRO AVG':<28} {micro_p:>8.3f} {micro_r:>8.3f} {micro_f1:>8.3f}")

    macro_p = sum(m["precision"] for m in per_tool_metrics.values()) / len(
        per_tool_metrics
    )
    macro_r = sum(m["recall"] for m in per_tool_metrics.values()) / len(
        per_tool_metrics
    )
    macro_f1 = sum(m["f1"] for m in per_tool_metrics.values()) / len(per_tool_metrics)
    logger.info(f"{'MACRO AVG':<28} {macro_p:>8.3f} {macro_r:>8.3f} {macro_f1:>8.3f}")

    with (output_dir / "predictions_detailed.json").open("w") as f:
        json.dump(predictions_list, f, indent=2)
    pd.DataFrame([{"tool": t, **m} for t, m in per_tool_metrics.items()]).to_csv(
        output_dir / "per_tool_metrics.csv", index=False
    )

    return {
        "exact_match_accuracy": exact_accuracy,
        "jaccard_accuracy": jaccard_accuracy,
        "top1_accuracy": top1_accuracy,
        "total_images": total,
        "exact_matches": exact_matches,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_tool_metrics": per_tool_metrics,
    }


def main() -> None:
    """Train YOLOv12-m on the 15-tool taxonomy and save metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=str, default=None)
    args = parser.parse_args()

    job_id = args.job_id or os.environ.get("SLURM_JOB_ID", "local")
    output_dir = experiment_output_dir("sdsc_eea_yolo_15_tools", job_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("YOLOv12-m on the 15 evaluable SDSC-EEA tools")
    logger.info("=" * 80)
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Output: {output_dir}")

    dataset_yaml, _dataset_dir = create_yolo_dataset_structure(output_dir)

    best_model = output_dir / "train" / "weights" / "best.pt"
    results_csv = output_dir / "train" / "results.csv"

    training_complete = best_model.exists() and results_csv.exists()
    if training_complete:
        epoch_metric_df = pd.read_csv(results_csv)
        if len(epoch_metric_df) >= EPOCHS:
            logger.info(
                "\nTraining already complete: "
                f"{len(epoch_metric_df)} epochs in results.csv. "
                "Skipping training, proceeding to evaluation..."
            )
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
        device_str = ",".join(str(i) for i in range(NUM_GPUS)) if NUM_GPUS > 1 else "0"
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
            project=str(output_dir),
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

        if results is not None and getattr(results, "results_dict", None):
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
            raise RuntimeError("Training did not produce results")

    logger.info("\nTraining complete!")
    logger.info(f"  mAP50: {metrics['map50']:.4f}")
    logger.info(f"  mAP50-95: {metrics['map50_95']:.4f}")

    if best_model.exists():
        eval_results = compute_all_metrics(
            model_path=best_model,
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
        )
    else:
        raise RuntimeError(f"Best model not found at {best_model}")

    eval_results_for_json = {
        k: v for k, v in eval_results.items() if k != "per_tool_metrics"
    }
    results_summary = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "experiment": "sdsc_eea_yolo_15_tools",
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "imgsz": IMGSZ,
        "v424_tools_ordered": V424_TOOLS_ORDERED,
        "training_metrics": metrics,
        "eval_results": eval_results_for_json,
    }
    with (output_dir / "results_summary.json").open("w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY (15-tool taxonomy)")
    logger.info("=" * 80)
    logger.info(f"mAP50: {metrics['map50']:.4f}")
    logger.info(
        f"Exact Match: {eval_results['exact_match_accuracy']:.2f}%  "
        f"Jaccard: {eval_results['jaccard_accuracy']:.2f}%  "
        f"Top-1: {eval_results['top1_accuracy']:.2f}%"
    )
    logger.info(
        f"Micro F1: {eval_results['micro_f1']:.4f}  "
        f"Macro F1: {eval_results['macro_f1']:.4f}"
    )
    logger.info(f"\nBest model: {best_model}")
    logger.info("\nYOLOv12-m 15-tool run complete!")


if __name__ == "__main__":
    main()
