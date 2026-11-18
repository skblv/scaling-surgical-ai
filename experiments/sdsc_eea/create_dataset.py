"""Build the SDSC-EEA frame table used by every EEA experiment.

Collects labeled frames, splits procedures 80/20 into train and validation
so no procedure leaks across splits, and writes ``dataset.csv``. The
``truth_label`` column is a JSON list of tool names from the YOLO labels.
This table is the frame index behind the paper's SDSC-EEA results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from scaling_surgical_ai import paths

RANDOM_SEED = 42
LABELS_ROOT = paths.eea_labels_root()
CSV_PATH = LABELS_ROOT / "eea_09_24_2025.csv"
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()
OUTPUT_DIR = paths.experiment_output_dir("sdsc_eea_dataset")

TRAIN_RATIO = 0.8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yolo_mapping() -> Dict[int, str]:
    """Load YOLO class ID to tool name mapping."""
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    return {int(k): v for k, v in mapping.items()}


def build_filename_to_path_lookup() -> Dict[str, Path]:
    """Build a lookup from filename to full path by scanning all subdirectories."""
    lookup = {}
    for subdir in tqdm(list(LABELS_ROOT.iterdir()), desc="Scanning label directories"):
        if not subdir.is_dir():
            continue
        images_dir = subdir / "images"
        if images_dir.exists():
            for img_path in images_dir.iterdir():
                if img_path.suffix in (".png", ".jpg", ".jpeg"):
                    lookup[img_path.name] = img_path
        else:
            for img_path in subdir.iterdir():
                if img_path.suffix in (".png", ".jpg", ".jpeg"):
                    lookup[img_path.name] = img_path
    return lookup


def load_csv_data() -> pd.DataFrame:
    """Load the main CSV with frame metadata."""
    return pd.read_csv(CSV_PATH)


def parse_yolo_label(label_path: Path) -> List[int]:
    """Parse YOLO label file and return list of class IDs."""
    if not label_path.exists():
        return []
    class_ids = []
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))
    return class_ids


def get_label_path(frame_path: Path) -> Path:
    """Get the corresponding label path for a frame."""
    if frame_path.parent.name == "images":
        label_dir = frame_path.parent.parent / "labels"
    else:
        label_dir = frame_path.parent / "labels"
    return label_dir / f"{frame_path.stem}.txt"


def get_ground_truth_tools(frame_path: Path, yolo_mapping: Dict[int, str]) -> List[str]:
    """Get ground truth tool names for a frame from YOLO labels."""
    label_path = get_label_path(frame_path)
    class_ids = parse_yolo_label(label_path)
    tools = [yolo_mapping[cid] for cid in class_ids if cid in yolo_mapping]
    return sorted(set(tools))


def is_valid_image(frame_path: Path) -> bool:
    """Check if image file exists and can be opened."""
    if not frame_path.exists():
        return False
    try:
        with Image.open(frame_path) as img:
            img.verify()
        return True
    except Exception as exc:
        logger.warning("Invalid image %s: %s", frame_path, exc)
        return False


def generate_frame_id(frame_path: Path, procedure_id: str) -> str:
    """Generate a unique frame ID based on filename and procedure."""
    unique_str = f"{procedure_id}_{frame_path.name}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:12]


def split_by_procedure(
    procedure_to_frames: Dict[str, List[dict]],
    train_ratio: float = 0.8,
) -> Tuple[List[dict], List[dict]]:
    """Split frames by procedure to prevent data leakage."""
    procedures = list(procedure_to_frames.keys())
    random.shuffle(procedures)
    split_idx = int(len(procedures) * train_ratio)

    train_procedures = set(procedures[:split_idx])

    train_frames = []
    val_frames = []

    for proc_id, frames in procedure_to_frames.items():
        if proc_id in train_procedures:
            for frame in frames:
                frame["split"] = "train"
            train_frames.extend(frames)
        else:
            for frame in frames:
                frame["split"] = "val"
            val_frames.extend(frames)

    return train_frames, val_frames


def main() -> None:
    """Create the unified dataset CSV."""
    logger.info("=" * 80)
    logger.info("Create unified SDSC-EEA dataset CSV")
    logger.info("=" * 80)

    random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading YOLO mapping from {YOLO_MAPPING_FILE}")
    yolo_mapping = load_yolo_mapping()
    valid_tools = sorted(set(yolo_mapping.values()))
    logger.info(f"Loaded {len(valid_tools)} tool classes")

    logger.info(f"\nBuilding filename->path lookup from {LABELS_ROOT}")
    filename_lookup = build_filename_to_path_lookup()
    logger.info(f"Found {len(filename_lookup)} image files")

    logger.info(f"\nLoading CSV from {CSV_PATH}")
    frame_metadata_df = load_csv_data()
    logger.info(f"CSV has {len(frame_metadata_df)} rows")

    procedure_to_frames: Dict[str, List[dict]] = {}
    skipped_missing_path = 0
    skipped_invalid_image = 0
    skipped_no_labels = 0

    logger.info("\nCollecting frames and labels...")
    for _, row in tqdm(
        frame_metadata_df.iterrows(),
        total=len(frame_metadata_df),
        desc="Processing frames",
    ):
        filename = row["name"]
        procedure_id = str(row["procedure_id"])

        if filename not in filename_lookup:
            skipped_missing_path += 1
            continue

        frame_path = filename_lookup[filename]

        if not is_valid_image(frame_path):
            skipped_invalid_image += 1
            continue

        label_path = get_label_path(frame_path)
        if not label_path.exists():
            skipped_no_labels += 1
            continue

        tools = get_ground_truth_tools(frame_path, yolo_mapping)
        frame_id = generate_frame_id(frame_path, procedure_id)

        frame_record = {
            "frame_id": frame_id,
            "procedure_id": procedure_id,
            "frame_path": str(frame_path),
            "truth_label": json.dumps(tools),
        }

        procedure_to_frames.setdefault(procedure_id, []).append(frame_record)

    total_frames = sum(len(frames) for frames in procedure_to_frames.values())
    num_procedures = len(procedure_to_frames)

    logger.info(
        f"\nCollected {total_frames} valid frames from {num_procedures} procedures"
    )
    logger.info(f"Skipped - missing path: {skipped_missing_path}")
    logger.info(f"Skipped - invalid image: {skipped_invalid_image}")
    logger.info(f"Skipped - no labels: {skipped_no_labels}")

    logger.info(
        f"\nSplitting by procedure ({TRAIN_RATIO * 100:.0f}/{(1 - TRAIN_RATIO) * 100:.0f} train/val)"
    )
    train_frames, val_frames = split_by_procedure(procedure_to_frames, TRAIN_RATIO)

    train_procedures = {f["procedure_id"] for f in train_frames}
    val_procedures = {f["procedure_id"] for f in val_frames}
    overlap = train_procedures & val_procedures

    if overlap:
        raise RuntimeError(
            f"PROCEDURE LEAKAGE DETECTED! {len(overlap)} procedures in both splits"
        )

    logger.info(
        f"Train: {len(train_frames)} frames from {len(train_procedures)} procedures"
    )
    logger.info(f"Val: {len(val_frames)} frames from {len(val_procedures)} procedures")
    logger.info("✓ No procedure overlap between train and val")

    all_records = train_frames + val_frames

    frame_dataset_df = pd.DataFrame(all_records)
    dataset_columns = [
        "frame_id",
        "procedure_id",
        "split",
        "frame_path",
        "truth_label",
    ]
    frame_dataset_df = frame_dataset_df[dataset_columns]

    output_csv = OUTPUT_DIR / "dataset.csv"
    frame_dataset_df.to_csv(output_csv, index=False)
    logger.info(f"\nSaved dataset to {output_csv}")

    metadata = {
        "total_frames": len(frame_dataset_df),
        "train_frames": len(train_frames),
        "val_frames": len(val_frames),
        "num_procedures": num_procedures,
        "train_procedures": len(train_procedures),
        "val_procedures": len(val_procedures),
        "train_ratio": TRAIN_RATIO,
        "random_seed": RANDOM_SEED,
        "labels_root": str(LABELS_ROOT),
        "csv_path": str(CSV_PATH),
        "yolo_mapping_file": str(YOLO_MAPPING_FILE),
        "valid_tools": valid_tools,
        "skipped_missing_path": skipped_missing_path,
        "skipped_invalid_image": skipped_invalid_image,
        "skipped_no_labels": skipped_no_labels,
    }

    metadata_file = OUTPUT_DIR / "metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

    logger.info("\n" + "=" * 80)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total frames: {len(frame_dataset_df)}")
    logger.info(
        f"  Train: {len(train_frames)} "
        f"({len(train_frames) / len(frame_dataset_df) * 100:.1f}%)"
    )
    logger.info(
        f"  Val: {len(val_frames)} "
        f"({len(val_frames) / len(frame_dataset_df) * 100:.1f}%)"
    )
    logger.info(f"Total procedures: {num_procedures}")
    logger.info(f"  Train: {len(train_procedures)}")
    logger.info(f"  Val: {len(val_procedures)}")

    tool_counts = {}
    for record in all_records:
        tools = json.loads(record["truth_label"])
        for tool in tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    logger.info("\nTool distribution (top 10):")
    sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    for tool, count in sorted_tools[:10]:
        logger.info(f"  {tool}: {count} ({count / len(all_records) * 100:.1f}%)")

    logger.info("\nOutput files:")
    logger.info(f"  - {output_csv}")
    logger.info(f"  - {metadata_file}")

    logger.info("\nDataset creation complete!")


if __name__ == "__main__":
    main()
