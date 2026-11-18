"""Extract PitVis-2023 frames and write the tool and step dataset.

Reads the raw PitVis-2023 release (25 endoscopic pituitary videos with
per-second step and instrument annotations) and writes one frame per
second plus a frame-level CSV. That CSV is the PitVis-2023 input for
the zero-shot, fine-tuning, and detector experiments in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from scaling_surgical_ai import paths

RANDOM_SEED = 42

SOURCE_DIR = paths.public_data_root() / "pitvis2023"
OUTPUT_DIR = paths.pitvis_root()
FRAMES_DIR = OUTPUT_DIR / "frames"

TRAIN_VIDEO_NUMS = {
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    20,
    22,
    23,
}
VAL_VIDEO_NUMS = {1, 12, 21, 24, 25}
SKIP_VIDEO_NUMS = {19}

PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def load_instrument_map() -> dict[int, str]:
    """Load the instrument code-to-name mapping.

    Returns:
        Mapping of instrument codes greater than zero to their names.
    """
    instrument_code_df = pd.read_csv(SOURCE_DIR / "misc" / "map_instruments.csv")
    return {
        int(row["int_instrument"]): row["str_instrument"].strip()
        for _, row in instrument_code_df.iterrows()
        if int(row["int_instrument"]) > 0
    }


def load_step_map() -> dict[int, str]:
    """Load the step code-to-name mapping.

    The first row wins when a step code is repeated.

    Returns:
        Mapping of step codes to step names.
    """
    step_code_df = pd.read_csv(SOURCE_DIR / "misc" / "map_steps.csv")
    step_map: dict[int, str] = {}
    for _, row in step_code_df.iterrows():
        code = int(row["int_step"])
        if code not in step_map:
            step_map[code] = row["str_step"].strip()
    return step_map


def extract_frames_at_1fps(
    video_path: Path,
    output_dir: Path,
    start_number: int = 0,
) -> int:
    """Extract one frame per second from video using OpenCV.

    Uses grab/retrieve pattern: grab() advances the decoder cheaply,
    retrieve() only decodes frames we actually want (1 per second).

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save extracted PNGs.
        start_number: Starting number for output filenames.

    Returns:
        Number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS ({fps}) for {video_path}")

    extracted = 0
    frame_count = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        current_sec = frame_count / fps
        target_sec = float(extracted)

        if current_sec >= target_sec:
            ret, frame = cap.retrieve()
            if ret:
                out_path = output_dir / f"{start_number + extracted:06d}.png"
                cv2.imwrite(str(out_path), frame, PNG_PARAMS)
                extracted += 1

        frame_count += 1

    cap.release()
    return extracted


def process_single_video(
    video_num: int,
    instrument_map: dict[int, str],
    step_map: dict[int, str],
) -> list[dict]:
    """Extract frames and build annotation rows for one video.

    Args:
        video_num: Video number (1-25).
        instrument_map: Code-to-name mapping for instruments.
        step_map: Code-to-name mapping for surgical steps.

    Returns:
        List of row dicts for dataset.csv.
    """
    video_name = f"video_{video_num:02d}"
    video_path = SOURCE_DIR / "videos" / f"{video_name}.mp4"
    frame_dir = FRAMES_DIR / video_name

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ann_path = SOURCE_DIR / "annotations" / f"annotations_{video_num:02d}.csv"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    split = "train" if video_num in TRAIN_VIDEO_NUMS else "val"

    second_annotation_df = pd.read_csv(ann_path)
    ann_by_time = {
        int(row["int_time"]): row for _, row in second_annotation_df.iterrows()
    }

    n_frames = extract_frames_at_1fps(video_path, frame_dir)

    rows = []
    matched = 0
    for t in sorted(ann_by_time.keys()):
        frame_path = frame_dir / f"{t:06d}.png"
        if not frame_path.exists():
            continue

        ann = ann_by_time[t]
        matched += 1

        instrument_codes = set()
        for col in ["int_instrument1", "int_instrument2"]:
            code = int(ann[col])
            if code > 0 and code in instrument_map:
                instrument_codes.add(code)

        tool_names = sorted(instrument_map[c] for c in instrument_codes)
        tool_ids = sorted(instrument_codes)

        step_code = int(ann["int_step"])
        step_name = step_map.get(step_code, "unknown")

        rows.append(
            {
                "frame_path": str(frame_path),
                "video_id": video_name,
                "frame_id": t,
                "split": split,
                "truth_label": json.dumps(tool_names),
                "instrument_ids": json.dumps(tool_ids),
                "step_label": step_name,
                "step_id": step_code,
            }
        )

    print(
        f"  {video_name}: {n_frames} frames extracted, "
        f"{len(ann_by_time)} annotations, {matched} matched"
    )
    return rows


def main() -> None:
    """Extract PitVis-2023 frames and write dataset.csv."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    instrument_map = load_instrument_map()
    step_map = load_step_map()

    print(f"Instruments ({len(instrument_map)}):")
    for code, name in sorted(instrument_map.items()):
        print(f"  {code}: {name}")

    print(f"\nSteps ({len(step_map)}):")
    for code, name in sorted(step_map.items()):
        print(f"  {code}: {name}")

    video_nums = sorted((TRAIN_VIDEO_NUMS | VAL_VIDEO_NUMS) - SKIP_VIDEO_NUMS)
    print(f"\nProcessing {len(video_nums)} videos (skipping {sorted(SKIP_VIDEO_NUMS)})")

    all_rows: list[dict] = []
    for video_num in tqdm(video_nums, desc="Videos"):
        rows = process_single_video(video_num, instrument_map, step_map)
        all_rows.extend(rows)

    frame_dataset_df = pd.DataFrame(all_rows)
    dataset_path = OUTPUT_DIR / "dataset.csv"
    frame_dataset_df.to_csv(dataset_path, index=False)

    n_train = len(frame_dataset_df[frame_dataset_df["split"] == "train"])
    n_val = len(frame_dataset_df[frame_dataset_df["split"] == "val"])

    print(f"\nDataset saved to {dataset_path}")
    print(f"Total frames: {len(frame_dataset_df)}")
    print(f"Train: {n_train}, Val: {n_val}")

    tool_counts: dict[str, int] = {}
    for tools_json in frame_dataset_df["truth_label"]:
        for tool in json.loads(tools_json):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        pct = count / len(frame_dataset_df) * 100
        print(f"  {tool}: {count} ({pct:.1f}%)")

    split_info = {
        "train_videos": sorted(
            f"video_{v:02d}" for v in TRAIN_VIDEO_NUMS - SKIP_VIDEO_NUMS
        ),
        "val_videos": sorted(f"video_{v:02d}" for v in VAL_VIDEO_NUMS),
        "skipped_videos": sorted(f"video_{v:02d}" for v in SKIP_VIDEO_NUMS),
        "instrument_names": {str(k): v for k, v in sorted(instrument_map.items())},
        "step_names": {str(k): v for k, v in sorted(step_map.items())},
        "total_frames": len(frame_dataset_df),
        "train_frames": n_train,
        "val_frames": n_val,
    }
    with (OUTPUT_DIR / "split_info.json").open("w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Split info saved to {OUTPUT_DIR / 'split_info.json'}")


if __name__ == "__main__":
    main()
