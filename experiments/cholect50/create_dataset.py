"""Create the CholecT50 video-level train/validation split.

Each video goes entirely to train (80%) or validation (20%). The script
writes ``dataset.csv`` and ``split_info.json``, including the majority-class
instrument baseline reported for CholecT50.

CholecT50 annotation groups in the source labels:
- Instruments (6): grasper, bipolar, hook, scissors, clipper, irrigator
- Verbs (10): grasp, retract, dissect, coagulate, clip, cut, aspirate,
  irrigate, pack, null_verb
- Targets (15): gallbladder, cystic_plate, cystic_duct, cystic_artery,
  cystic_pedicle, blood_vessel, fluid, abdominal_wall_cavity, liver,
  adhesion, omentum, peritoneum, gut, specimen_bag, null_target
- Phases (7): preparation, carlot-triangle-dissection, clipping-and-cutting,
  gallbladder-dissection, gallbladder-packaging, cleaning-and-coagulation,
  gallbladder-extraction
- Triplets (100): instrument, verb, target combinations
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from scaling_surgical_ai import paths

RANDOM_SEED = 42
CHOLECT50_ROOT = paths.cholect50_root()
OUTPUT_DIR = paths.experiment_output_dir("cholect50_dataset")

INSTRUMENT_NAMES = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
}


def get_video_ids() -> List[str]:
    """Get all video IDs from CholecT50."""
    videos_dir = CHOLECT50_ROOT / "videos"
    return sorted([d.name for d in videos_dir.iterdir() if d.is_dir()])


def split_videos_by_id(
    video_ids: List[str], train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    """Split video IDs into train and val sets."""
    random.seed(RANDOM_SEED)
    shuffled = video_ids.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_ids = sorted(shuffled[:split_idx])
    val_ids = sorted(shuffled[split_idx:])

    return train_ids, val_ids


def parse_instrument_labels(label_file: Path) -> Dict[str, List[int]]:
    """Parse CholecT50 JSON label file and extract instrument presence per frame.

    Label format per frame: list of triplet annotations
    Each annotation: [triplet_id, instrument_id, verb_id, ...]
    Index 1 is instrument_id (0-5)

    Returns:
        Dict mapping frame_id -> list of instrument IDs present
    """
    with label_file.open("r") as f:
        data = json.load(f)

    frame_instruments = {}
    annotations = data.get("annotations", {})

    for frame_id, triplet_list in annotations.items():
        instruments_in_frame = set()
        for triplet in triplet_list:
            if len(triplet) > 1:
                instrument_id = triplet[1]
                if instrument_id >= 0 and instrument_id < 6:
                    instruments_in_frame.add(instrument_id)
        frame_instruments[frame_id] = sorted(instruments_in_frame)

    return frame_instruments


def create_dataset_csv() -> None:
    """Create the dataset.csv for CholecT50 with 80/20 train/val split."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_ids = get_video_ids()
    print(f"Found {len(video_ids)} videos")

    train_ids, val_ids = split_videos_by_id(video_ids)
    print(f"Train videos ({len(train_ids)}): {train_ids}")
    print(f"Val videos ({len(val_ids)}): {val_ids}")

    rows = []

    for video_id in tqdm(video_ids, desc="Processing videos"):
        split = "train" if video_id in train_ids else "val"

        video_dir = CHOLECT50_ROOT / "videos" / video_id
        label_file = CHOLECT50_ROOT / "labels" / f"{video_id}.json"

        if not label_file.exists():
            print(f"Warning: Label file not found for {video_id}")
            continue

        frame_instruments = parse_instrument_labels(label_file)

        frame_files = sorted(video_dir.glob("*.png"))
        for frame_file in frame_files:
            frame_id = frame_file.stem.lstrip("0") or "0"

            instruments = frame_instruments.get(frame_id, [])
            tool_names = [INSTRUMENT_NAMES[i] for i in instruments]

            rows.append(
                {
                    "frame_path": str(frame_file),
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "split": split,
                    "truth_label": json.dumps(tool_names),
                    "instrument_ids": json.dumps(instruments),
                }
            )

    frame_label_df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "dataset.csv"
    frame_label_df.to_csv(output_path, index=False)

    print(f"\nDataset saved to {output_path}")
    print(f"Total frames: {len(frame_label_df)}")
    print(f"Train frames: {len(frame_label_df[frame_label_df['split'] == 'train'])}")
    print(f"Val frames: {len(frame_label_df[frame_label_df['split'] == 'val'])}")

    majority_baseline = compute_majority_class_baseline(frame_label_df)

    split_info = {
        "train_videos": train_ids,
        "val_videos": val_ids,
        "instrument_names": {str(k): v for k, v in INSTRUMENT_NAMES.items()},
        "total_frames": len(frame_label_df),
        "train_frames": len(frame_label_df[frame_label_df["split"] == "train"]),
        "val_frames": len(frame_label_df[frame_label_df["split"] == "val"]),
        "majority_class_baseline": majority_baseline,
    }
    with (OUTPUT_DIR / "split_info.json").open("w") as f:
        json.dump(split_info, f, indent=2)


def compute_majority_class_baseline(frame_label_df: pd.DataFrame) -> Dict:
    """Compute majority class baseline for multi-label classification.

    For each instrument, compute:
    - Frequency in train set
    - If we always predict the most common label set, what's the accuracy?
    - Per-tool majority prediction (present/absent based on >50% occurrence)
    """
    frame_label_train_df = frame_label_df[frame_label_df["split"] == "train"]
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"]

    tool_names = list(INSTRUMENT_NAMES.values())

    train_tool_freq = {tool: 0 for tool in tool_names}
    train_label_sets = Counter()

    for _, row in tqdm(
        frame_label_train_df.iterrows(),
        total=len(frame_label_train_df),
        desc="Majority baseline train",
    ):
        tools = json.loads(row["truth_label"])
        label_set = tuple(sorted(tools))
        train_label_sets[label_set] += 1
        for tool in tools:
            train_tool_freq[tool] += 1

    n_train = len(frame_label_train_df)
    tool_presence_rate = {
        tool: count / n_train for tool, count in train_tool_freq.items()
    }

    majority_label_set, majority_count = train_label_sets.most_common(1)[0]
    majority_set = set(majority_label_set)

    exact_match = 0
    jaccard_sum = 0.0

    for _, row in tqdm(
        frame_label_val_df.iterrows(),
        total=len(frame_label_val_df),
        desc="Majority baseline validation",
    ):
        gt_tools = set(json.loads(row["truth_label"]))

        if majority_set == gt_tools:
            exact_match += 1

        union = majority_set | gt_tools
        jaccard = len(majority_set & gt_tools) / len(union) if union else 1.0
        jaccard_sum += jaccard

    n_val = len(frame_label_val_df)
    exact_match_acc = exact_match / n_val * 100 if n_val > 0 else 0.0
    jaccard_acc = jaccard_sum / n_val * 100 if n_val > 0 else 0.0

    per_tool_baseline = {}
    for tool in tool_names:
        predict_present = tool_presence_rate[tool] > 0.5

        tp = fp = fn = tn = 0
        for _, row in tqdm(
            frame_label_val_df.iterrows(),
            total=len(frame_label_val_df),
            desc="Per-tool baseline",
            leave=False,
        ):
            gt_tools = set(json.loads(row["truth_label"]))
            in_gt = tool in gt_tools

            if predict_present and in_gt:
                tp += 1
            elif predict_present and not in_gt:
                fp += 1
            elif not predict_present and in_gt:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / n_val * 100 if n_val > 0 else 0.0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_tool_baseline[tool] = {
            "train_presence_rate": round(tool_presence_rate[tool] * 100, 2),
            "predict_present": predict_present,
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
        }

    baseline = {
        "majority_label_set": list(majority_label_set),
        "majority_set_train_frequency": round(majority_count / n_train * 100, 2),
        "exact_match_accuracy": round(exact_match_acc, 2),
        "jaccard_accuracy": round(jaccard_acc, 2),
        "per_tool_baseline": per_tool_baseline,
    }

    print("\n" + "=" * 60)
    print("MAJORITY CLASS BASELINE")
    print("=" * 60)
    print(f"Most common label set in train: {list(majority_label_set)}")
    print(f"  Frequency in train: {majority_count / n_train * 100:.2f}%")
    print(f"Validation Exact Match Accuracy: {exact_match_acc:.2f}%")
    print(f"Validation Jaccard Accuracy: {jaccard_acc:.2f}%")
    print("\nPer-tool baseline (predict present if >50% in train):")
    for tool, metrics in per_tool_baseline.items():
        print(
            f"  {tool:10}: train={metrics['train_presence_rate']:5.1f}%, "
            f"pred={'Y' if metrics['predict_present'] else 'N'}, "
            f"acc={metrics['accuracy']:5.1f}%, F1={metrics['f1']:5.1f}%"
        )
    print("=" * 60)

    return baseline


def main() -> None:
    """Create the CholecT50 dataset CSV and split summary."""
    create_dataset_csv()


if __name__ == "__main__":
    main()
