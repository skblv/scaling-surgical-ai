"""Majority tool-set baseline for SDSC-EEA exact match.

Always predicting the tool combination that is most common in the training
frames gives the majority-class exact-match baseline reported next to the
SDSC-EEA models. Frames, splits and ground-truth tool sets come from the
dataset table written by ``experiments/sdsc_eea/create_dataset.py``.
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from scaling_surgical_ai import paths

DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
OUTPUT_DIR = paths.experiment_output_dir("sdsc_eea_majority_baseline")
TOP_TOOL_SETS = 10
SHOWN_TOOLS = 3


def tool_sets_by_frame(frame_label_df: pd.DataFrame) -> dict[str, frozenset[str]]:
    """Map every frame to its ground-truth tool set.

    Args:
        frame_label_df: One row per frame with ``frame_id`` and the JSON list
            ``truth_label``.

    Returns:
        Frame identifier to the set of tool names present in that frame.
    """
    frame_ids = frame_label_df["frame_id"]
    truth_labels = frame_label_df["truth_label"]
    return {
        frame_id: frozenset(json.loads(truth_label))
        for frame_id, truth_label in zip(frame_ids, truth_labels)
    }


def main() -> None:
    """Compute train and validation accuracy of the majority tool set."""
    print(f"Loading the dataset table from {DATASET_CSV}...")
    frame_label_df = pd.read_csv(DATASET_CSV)
    frame_label_train_df = frame_label_df[frame_label_df["split"] == "train"]
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"]

    train_tool_sets = tool_sets_by_frame(frame_label_train_df)
    val_tool_sets = tool_sets_by_frame(frame_label_val_df)
    train_count = len(train_tool_sets)
    val_count = len(val_tool_sets)
    print(f"Train frames: {train_count}")
    print(f"Val frames: {val_count}")

    print("\nCounting tool set frequencies...")
    train_counter = Counter(train_tool_sets.values())
    val_counter = Counter(val_tool_sets.values())

    most_common_set, most_common_count = train_counter.most_common(1)[0]

    print("\n" + "=" * 80)
    print("BASELINE ACCURACY ANALYSIS")
    print("=" * 80)

    print("\nMost common tool set in training data:")
    if most_common_set:
        print(f"  Tools: {sorted(most_common_set)}")
    else:
        print("  Tools: (empty set - no tools)")
    train_baseline_acc = most_common_count / train_count * 100
    val_baseline_acc = val_counter[most_common_set] / val_count * 100
    print(f"  Frequency: {most_common_count}/{train_count} ({train_baseline_acc:.2f}%)")

    print("\nBaseline accuracy (always predict most common set):")
    print(f"  Train: {train_baseline_acc:.2f}%")
    print(f"  Test:  {val_baseline_acc:.2f}%")

    top_tool_sets = train_counter.most_common(TOP_TOOL_SETS)
    print(f"\nTop {TOP_TOOL_SETS} most common tool sets in training data:")
    for rank, (tool_set, count) in enumerate(top_tool_sets, 1):
        percentage = 100 * count / train_count
        tools = sorted(tool_set) if tool_set else ["(empty)"]
        suffix = "..." if len(tools) > SHOWN_TOOLS else ""
        print(f"  {rank}. {tools[:SHOWN_TOOLS]}{suffix} - {count} ({percentage:.2f}%)")

    print("\nDataset diversity:")
    print(f"  Unique tool sets in train: {len(train_counter)}")
    print(f"  Unique tool sets in val: {len(val_counter)}")

    results = {
        "most_common_tool_set": sorted(most_common_set),
        "most_common_count_train": most_common_count,
        "train_baseline_accuracy": train_baseline_acc,
        "val_baseline_accuracy": val_baseline_acc,
        "unique_tool_sets_train": len(train_counter),
        "unique_tool_sets_val": len(val_counter),
        "top_10_train": [
            {
                "tools": sorted(tool_set),
                "count": count,
                "percentage": 100 * count / train_count,
            }
            for tool_set, count in top_tool_sets
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "baseline_accuracy.json"
    with output_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
