"""Check the PitVis-2023 dataset CSV, split file, and sampled frames.

Prints frame counts, tool coverage, and whether a random sample of the
frame paths written by the dataset builder still exist on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scaling_surgical_ai import paths

RANDOM_SEED = 42
PITVIS_ROOT = paths.pitvis_root()
DATASET_PATH = PITVIS_ROOT / "dataset.csv"
SPLIT_INFO_PATH = PITVIS_ROOT / "split_info.json"


def main() -> None:
    """Print dataset counts and check that sampled frame files exist."""
    frame_label_df = pd.read_csv(DATASET_PATH)

    print("=== CSV Summary ===")
    print(f"Total rows: {len(frame_label_df)}")
    print(f"Unique videos: {frame_label_df['video_id'].nunique()}")

    n_train = len(frame_label_df[frame_label_df["split"] == "train"])
    n_val = len(frame_label_df[frame_label_df["split"] == "val"])
    print(f"Split: train={n_train}, val={n_val}")

    print("\n=== Per-video row counts ===")
    vid_counts = frame_label_df.groupby("video_id").size()
    for vid, n in vid_counts.items():
        print(f"  {vid}: {n}")

    frame_label_with_tools_df = frame_label_df[frame_label_df["truth_label"] != "[]"]
    pct = len(frame_label_with_tools_df) / len(frame_label_df) * 100
    print(
        f"\nFrames with at least one tool: "
        f"{len(frame_label_with_tools_df)} / {len(frame_label_df)} ({pct:.1f}%)"
    )

    print("\n=== File existence check (200 random) ===")
    sample = frame_label_df["frame_path"].sample(200, random_state=RANDOM_SEED)
    missing = [p for p in sample if not Path(p).exists()]
    print(f"Checked 200 random paths: {200 - len(missing)}/200 exist")
    if missing:
        print(f"WARNING: {len(missing)} MISSING files!")
        for m in missing[:5]:
            print(f"  {m}")
    else:
        print("All sampled frame files verified OK")

    print("\n=== Sample annotated rows ===")
    cols = ["video_id", "frame_id", "split", "truth_label", "step_label"]
    print(frame_label_with_tools_df[cols].head(5).to_string(index=False))

    with SPLIT_INFO_PATH.open() as f:
        info = json.load(f)
    print("\n=== split_info.json ===")
    print(f"Train videos: {info['train_videos']}")
    print(f"Val videos: {info['val_videos']}")
    print(f"Skipped: {info['skipped_videos']}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
