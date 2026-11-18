"""Extract cropped SurgVU frames and write the tool and task dataset.

SurgVU has 155 robotic surgery cases with tool install and uninstall
events and surgical task segments. This script converts those events
to per-frame multi-label tool presence, crops the burnt-in HUD, and
writes the frame CSV used by the SurgVU experiments in the paper.
The directory of SurgVU videos is the first command-line argument.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from scaling_surgical_ai import paths

RANDOM_SEED = 42

STRIDE_SECONDS = int(os.environ.get("STRIDE_SECONDS", "30"))
CROP_TOP_FRAC = float(os.environ.get("CROP_TOP_FRAC", "0.10"))
CROP_BOTTOM_FRAC = float(os.environ.get("CROP_BOTTOM_FRAC", "0.10"))

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python experiments/surgvu/create_dataset.py <directory of SurgVU videos>"
    )
VIDEO_DIR = Path(sys.argv[1])
LABELS_DIR = paths.env_path("SURGVU_LABELS_DIR")
OUTPUT_DIR = paths.surgvu_root()
FRAMES_DIR = OUTPUT_DIR / "frames"

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
NAME_TO_ID = {v: k for k, v in INSTRUMENT_NAMES.items()}
VALID_TOOL_NAMES = set(INSTRUMENT_NAMES.values())

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95]


def parse_hhmmss_to_seconds(time_str: str) -> float:
    """Parse HH:MM:SS[.microseconds] string to total seconds."""
    parts = time_str.strip().split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def load_tool_intervals(
    label_dir: Path,
) -> tuple[list[dict], dict[str, int]]:
    """Load tool install/uninstall intervals from tools.csv.

    Filters out unknown instruments, camera entries, and
    intervals where uninstall <= install within the same part.

    Args:
        label_dir: Directory containing tools.csv.

    Returns:
        Tuple of (intervals, drop_stats) where drop_stats counts
        each reason an annotation row was dropped. Each interval
        dict has keys: install_part, install_sec, uninstall_part,
        uninstall_sec, tool_name.
    """
    tools_path = label_dir / "tools.csv"
    if not tools_path.exists():
        raise FileNotFoundError(f"tools.csv not found: {tools_path}")

    tool_interval_df = pd.read_csv(tools_path)
    intervals: list[dict] = []
    drop_stats = {
        "empty_or_nan": 0,
        "unknown_tool": 0,
        "wrap_around": 0,
    }

    for _, row in tool_interval_df.iterrows():
        raw_name = str(row.get("groundtruth_toolname", "")).strip()

        if (
            not raw_name
            or raw_name == "nan"
            or "nan" in raw_name.lower()
            or '"' in raw_name
        ):
            drop_stats["empty_or_nan"] += 1
            continue

        if raw_name not in VALID_TOOL_NAMES:
            drop_stats["unknown_tool"] += 1
            continue

        install_part = int(float(row["install_case_part"]))
        uninstall_part = int(float(row["uninstall_case_part"]))
        install_sec = parse_hhmmss_to_seconds(str(row["install_case_time"]))
        uninstall_sec = parse_hhmmss_to_seconds(str(row["uninstall_case_time"]))

        if install_part == uninstall_part and uninstall_sec <= install_sec:
            drop_stats["wrap_around"] += 1
            continue

        intervals.append(
            {
                "install_part": install_part,
                "install_sec": install_sec,
                "uninstall_part": uninstall_part,
                "uninstall_sec": uninstall_sec,
                "tool_name": raw_name,
            }
        )

    return intervals, drop_stats


def load_task_segments(label_dir: Path) -> list[dict]:
    """Load task temporal segments from tasks.csv.

    Args:
        label_dir: Directory containing tasks.csv.

    Returns:
        List of segment dicts with keys: start_part,
        start_sec, stop_part, stop_sec, task_name.
    """
    tasks_path = label_dir / "tasks.csv"
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks.csv not found: {tasks_path}")

    task_segment_df = pd.read_csv(tasks_path)
    segments: list[dict] = []

    for _, row in task_segment_df.iterrows():
        task_name = str(row["groundtruth_taskname"]).strip()
        if not task_name or task_name == "nan":
            continue

        segments.append(
            {
                "start_part": int(row["start_part"]),
                "start_sec": float(row["start_time"]),
                "stop_part": int(row["stop_part"]),
                "stop_sec": float(row["stop_time"]),
                "task_name": task_name,
            }
        )

    return segments


def get_tools_at_time(
    intervals: list[dict],
    part: int,
    time_sec: float,
) -> list[str]:
    """Get sorted tool names active at (part, time_sec).

    Handles intervals that span across video parts.
    Uses half-open interval: [install_sec, uninstall_sec).
    """
    tools: set[str] = set()
    for iv in intervals:
        ip = iv["install_part"]
        up = iv["uninstall_part"]

        if ip == up == part:
            if iv["install_sec"] <= time_sec < iv["uninstall_sec"]:
                tools.add(iv["tool_name"])
        elif ip == part and up > part:
            if time_sec >= iv["install_sec"]:
                tools.add(iv["tool_name"])
        elif up == part and ip < part:
            if time_sec < iv["uninstall_sec"]:
                tools.add(iv["tool_name"])

    return sorted(tools)


def get_task_at_time(
    segments: list[dict],
    part: int,
    time_sec: float,
) -> str:
    """Get task name at (part, time_sec), or "" if none.

    Handles segments that span across video parts.
    Uses half-open interval: [start_sec, stop_sec).
    """
    for seg in segments:
        sp = seg["start_part"]
        ep = seg["stop_part"]

        if sp == ep == part:
            if seg["start_sec"] <= time_sec < seg["stop_sec"]:
                return seg["task_name"]
        elif sp == part and ep > part:
            if time_sec >= seg["start_sec"]:
                return seg["task_name"]
        elif ep == part and sp < part:
            if time_sec < seg["stop_sec"]:
                return seg["task_name"]

    return ""


class VideoOpenError(RuntimeError):
    """Raised when a video file cannot be opened or decoded.

    Distinguished from generic RuntimeError so the caller can choose
    to skip the offending part with a loud warning rather than abort
    the entire multi-hour extraction.
    """


def crop_hud(frame: np.ndarray) -> np.ndarray:
    """Crop top and bottom margins of a frame.

    SurgVU videos burn a "TRAINING INSTRUMENT" warning into the top
    and a four-arm tool HUD ("1 PROGRASP FORCEPS …") into the bottom.
    Both are direct label leakage so we slice them off before saving.
    Defaults remove 10% from each edge (configurable via
    ``CROP_TOP_FRAC`` / ``CROP_BOTTOM_FRAC`` env vars).
    """
    h = frame.shape[0]
    top = int(round(h * CROP_TOP_FRAC))
    bottom_drop = int(round(h * CROP_BOTTOM_FRAC))
    if top + bottom_drop >= h:
        raise ValueError(
            f"Crop fractions ({CROP_TOP_FRAC}, "
            f"{CROP_BOTTOM_FRAC}) leave no rows for height={h}"
        )
    return frame[top : h - bottom_drop, :, :]


def extract_frames_at_stride(
    video_path: Path,
    output_dir: Path,
    start_second: int = 0,
) -> tuple[int, int]:
    """Extract one cropped frame every ``STRIDE_SECONDS`` seconds.

    Uses grab/retrieve pattern: grab() advances the decoder cheaply,
    retrieve() only decodes frames we actually want. Every extracted
    frame is cropped via :func:`crop_hud` before being written, so
    the on-disk JPEGs no longer contain the burnt-in HUD labels.

    Filenames encode the global second-of-case-timeline:
    ``{start_second + i * STRIDE_SECONDS:06d}.jpg`` for the i-th
    extracted frame. This keeps ``frame_id % STRIDE_SECONDS == 0``
    true for every row in dataset.csv so existing FRAME_STRIDE
    filters in downstream training scripts remain a no-op.

    Note: ALWAYS decodes the input video and writes (or overwrites)
    frames at the deterministic stride positions. The caller is
    responsible for wiping output_dir before extraction if a fresh
    re-run is desired.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save extracted JPEGs.
        start_second: Global second offset of this part within the
            full case timeline.

    Returns:
        Tuple ``(n_extracted, total_seconds)``: number of frames
        actually written by this call, and the duration of the
        decoded video in whole seconds (used by the caller to
        compute the next part's ``start_second``).

    Raises:
        VideoOpenError: if the video cannot be opened or its FPS is
            invalid (e.g. corrupted source file).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise VideoOpenError(f"Invalid FPS ({fps}) for {video_path}")

    extracted = 0
    frame_count = 0
    write_failures = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        current_sec = frame_count / fps
        target_sec = float(extracted * STRIDE_SECONDS)

        if current_sec >= target_sec:
            ret, frame = cap.retrieve()
            if ret:
                cropped = crop_hud(frame)
                global_sec = start_second + extracted * STRIDE_SECONDS
                out_path = output_dir / f"{global_sec:06d}.jpg"
                ok = cv2.imwrite(str(out_path), cropped, JPEG_PARAMS)
                if not ok:
                    write_failures += 1
                    print(
                        f"cv2.imwrite failed ({write_failures}): {out_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if write_failures >= 3:
                        cap.release()
                        raise RuntimeError(
                            f"cv2.imwrite failed {write_failures} "
                            f"times. Last path: {out_path}"
                        )
                extracted += 1

        frame_count += 1

    cap.release()
    total_seconds = int(frame_count / fps) if fps > 0 else 0
    return extracted, total_seconds


def compute_part_metadata(
    video_path: Path,
) -> tuple[int, int]:
    """Predict ``(n_extracted, total_seconds)`` for a video part.

    Reads only the container metadata (total frame count + fps) so
    it is essentially free compared to a full decode. The formula
    mirrors the grab/retrieve loop in
    :func:`extract_frames_at_stride`: the loop fires at integer
    seconds-of-video, so for stride ``S`` the number of extracted
    frames equals ``floor(T / S) + 1`` where ``T`` is the duration
    of the last decoded frame in seconds (``(total_frames - 1) /
    fps``).

    Raises:
        VideoOpenError: if the file cannot be opened or its metadata
            is unusable.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if total_frames <= 0 or fps <= 0:
        raise VideoOpenError(
            f"Invalid metadata (frames={total_frames}, fps={fps}) for {video_path}"
        )
    last_frame_sec = (total_frames - 1) / fps
    n_extracted = int(last_frame_sec / STRIDE_SECONDS) + 1
    total_seconds = int(total_frames / fps)
    return n_extracted, total_seconds


def salvage_existing_cases(case_ids: list[str]) -> int:
    """Write ``.part_info.json`` for cases whose jpgs already exist.

    Iterates cases with non-empty ``frame_dir`` but no sentinel; for
    each one, recomputes the per-part frame counts from video
    metadata under the current ``STRIDE_SECONDS``, verifies the
    on-disk jpg count matches the stride-aligned expected count
    AND that the first jpg's pixel height matches the cropped
    dimensions, then writes the sentinel so
    :func:`process_single_case` can resume without re-decoding.

    Cases extracted with a different stride or without crop will
    fail validation and be re-extracted from scratch.

    Returns the number of cases successfully salvaged.
    """
    salvaged = 0
    for case_id in case_ids:
        frame_dir = FRAMES_DIR / case_id
        sentinel = frame_dir / ".part_info.json"
        if sentinel.exists() or not frame_dir.exists():
            continue
        existing_jpgs = sorted(frame_dir.glob("*.jpg"))
        if not existing_jpgs:
            continue

        video_dir = VIDEO_DIR / case_id
        video_parts = sorted(video_dir.glob("*.mp4"))
        if not video_parts:
            print(
                f"  [salvage-skip] {case_id}: no source videos",
                file=sys.stderr,
                flush=True,
            )
            continue

        part_info: list[dict] = []
        global_offset_sec = 0
        salvage_failed = False
        for vp in video_parts:
            try:
                part_num = parse_part_num(vp)
                n_frames, total_seconds = compute_part_metadata(vp)
            except (ValueError, VideoOpenError) as exc:
                print(
                    f"  [salvage-skip] {case_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                salvage_failed = True
                break
            part_info.append(
                {
                    "part_num": part_num,
                    "start_second": global_offset_sec,
                    "n_frames": n_frames,
                }
            )
            global_offset_sec += total_seconds

        if salvage_failed:
            continue

        expected_total = sum(p["n_frames"] for p in part_info)
        if expected_total != len(existing_jpgs):
            print(
                f"  [salvage-skip] {case_id}: jpg count mismatch "
                f"(disk={len(existing_jpgs)}, "
                f"expected={expected_total} "
                f"@ stride={STRIDE_SECONDS}s)",
                file=sys.stderr,
                flush=True,
            )
            continue

        first_pi = part_info[0]
        first_jpg = frame_dir / f"{first_pi['start_second']:06d}.jpg"
        last_pi = part_info[-1]
        last_jpg_sec = (
            last_pi["start_second"] + (last_pi["n_frames"] - 1) * STRIDE_SECONDS
        )
        last_jpg = frame_dir / f"{last_jpg_sec:06d}.jpg"
        if not (first_jpg.exists() and last_jpg.exists()):
            print(
                f"  [salvage-skip] {case_id}: stride-aligned boundary jpgs missing",
                file=sys.stderr,
                flush=True,
            )
            continue

        probe = cv2.imread(str(first_jpg))
        if probe is None:
            print(
                f"  [salvage-skip] {case_id}: cannot read first jpg",
                file=sys.stderr,
                flush=True,
            )
            continue
        probe_h = probe.shape[0]
        ratio = probe_h / max(probe.shape[1], 1)
        if not 0.30 < ratio < 0.85:
            print(
                f"  [salvage-skip] {case_id}: first jpg shape "
                f"{probe.shape} not consistent with crop "
                f"({CROP_TOP_FRAC},{CROP_BOTTOM_FRAC})",
                file=sys.stderr,
                flush=True,
            )
            continue

        sentinel.write_text(json.dumps(part_info))
        salvaged += 1

    return salvaged


def parse_part_num(video_path: Path) -> int:
    """Extract the case-part integer from a SurgVU video filename.

    Expected stem ends in ``_part_{n}`` (e.g. ``case_087_video_part_2``).
    Raises ValueError for any other naming convention so silent
    misclassification is impossible.
    """
    stem = video_path.stem
    sep = "_part_"
    if sep not in stem:
        raise ValueError(f"Video filename does not contain '_part_': {video_path}")
    suffix = stem.rsplit(sep, 1)[1]
    if not suffix.isdigit():
        raise ValueError(
            f"Cannot parse part number from {video_path} (suffix={suffix!r})"
        )
    return int(suffix)


def process_single_case(
    case_id: str,
    split: str,
) -> list[dict]:
    """Extract frames and build annotation rows for one case.

    Resumable: if ``frame_dir / .part_info.json`` exists from a prior
    successful extraction and every referenced jpg is on disk, we
    skip re-decoding the videos and rebuild rows from existing files.
    Otherwise the per-case frame directory is wiped and frames are
    re-extracted from scratch.

    Tolerates corrupt source videos: if a single ``_part_N.mp4``
    raises :class:`VideoOpenError`, we print a loud warning to stderr
    and skip that part. The case is still saved as long as at least
    one part was successfully extracted; otherwise the case is
    skipped entirely with a warning.

    Args:
        case_id: Case identifier (e.g., "case_000").
        split: Train/val split assignment.

    Returns:
        List of row dicts for dataset.csv.
    """
    label_dir = LABELS_DIR / case_id
    video_dir = VIDEO_DIR / case_id
    frame_dir = FRAMES_DIR / case_id
    sentinel = frame_dir / ".part_info.json"

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    tool_intervals, drop_stats = load_tool_intervals(label_dir)
    task_segments = load_task_segments(label_dir)

    video_parts = sorted(video_dir.glob("*.mp4"))
    if not video_parts:
        print(
            f"  [warn] {case_id}: no mp4 files in {video_dir}; "
            "skipping case (evicted from temporary storage or missing "
            "in the source release)",
            file=sys.stderr,
            flush=True,
        )
        return []

    part_info: list[dict] = []
    part_nums_in_videos: set[int] = set()
    skipped_parts: list[str] = []
    resumed = False

    if sentinel.exists():
        cached = json.loads(sentinel.read_text())
        first_jpg = frame_dir / f"{cached[0]['start_second']:06d}.jpg"
        last_pi = cached[-1]
        last_jpg_sec = (
            last_pi["start_second"] + (last_pi["n_frames"] - 1) * STRIDE_SECONDS
        )
        last_jpg = frame_dir / f"{last_jpg_sec:06d}.jpg"
        if first_jpg.exists() and last_jpg.exists():
            part_info = cached
            part_nums_in_videos = {p["part_num"] for p in cached}
            resumed = True
        else:
            print(
                f"  [warn] {case_id}: sentinel present but jpgs "
                f"missing; re-extracting from scratch",
                file=sys.stderr,
                flush=True,
            )

    if not resumed:
        if frame_dir.exists():
            for old in frame_dir.glob("*.jpg"):
                old.unlink()
            if sentinel.exists():
                sentinel.unlink()

        global_offset_sec = 0
        for vp in video_parts:
            part_num = parse_part_num(vp)
            part_nums_in_videos.add(part_num)
            try:
                n_frames, total_seconds = extract_frames_at_stride(
                    vp,
                    frame_dir,
                    start_second=global_offset_sec,
                )
            except VideoOpenError as exc:
                print(
                    f"  [warn] {case_id}: SKIPPING corrupt part {vp.name}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                skipped_parts.append(vp.name)
                continue
            if n_frames <= 0:
                raise RuntimeError(f"Extracted 0 frames from {vp}")
            part_info.append(
                {
                    "part_num": part_num,
                    "start_second": global_offset_sec,
                    "n_frames": n_frames,
                }
            )
            global_offset_sec += total_seconds

        if not part_info:
            print(
                f"  [warn] {case_id}: no parts extracted "
                f"(skipped: {skipped_parts}); skipping case",
                file=sys.stderr,
                flush=True,
            )
            return []

        sentinel.write_text(json.dumps(part_info))

    parts_in_labels = {iv["install_part"] for iv in tool_intervals} | {
        iv["uninstall_part"] for iv in tool_intervals
    }
    missing_parts = parts_in_labels - part_nums_in_videos
    if missing_parts:
        print(
            f"  [warn] {case_id}: parts referenced in labels "
            f"but not in videos: {sorted(missing_parts)}",
            flush=True,
        )

    total_frames = sum(p["n_frames"] for p in part_info)

    rows: list[dict] = []
    for pi in part_info:
        for i in range(pi["n_frames"]):
            local_t_sec = i * STRIDE_SECONDS
            global_frame_id = pi["start_second"] + local_t_sec
            frame_path = frame_dir / f"{global_frame_id:06d}.jpg"

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Expected frame missing after extraction: {frame_path}"
                )

            tools = get_tools_at_time(
                tool_intervals,
                pi["part_num"],
                float(local_t_sec),
            )
            tool_ids = sorted(NAME_TO_ID[t] for t in tools)
            task = get_task_at_time(
                task_segments,
                pi["part_num"],
                float(local_t_sec),
            )

            rows.append(
                {
                    "frame_path": str(frame_path),
                    "video_id": case_id,
                    "frame_id": global_frame_id,
                    "split": split,
                    "truth_label": json.dumps(tools),
                    "instrument_ids": json.dumps(tool_ids),
                    "task_label": task,
                }
            )

    n_labeled = sum(1 for r in rows if r["truth_label"] != "[]")
    status = "RESUMED" if resumed else "extracted"
    skip_note = f", SKIPPED parts: {skipped_parts}" if skipped_parts else ""
    print(
        f"  {case_id}: {total_frames} frames {status} "
        f"({n_labeled} non-empty labels), "
        f"{len(tool_intervals)} kept intervals, "
        f"dropped {drop_stats}, "
        f"{len(task_segments)} task segments, "
        f"{len(video_parts)} parts ({sorted(part_nums_in_videos)})"
        f"{skip_note}",
        flush=True,
    )
    return rows


def main() -> None:
    """Extract SurgVU frames and create dataset.csv."""
    print(f"Video source: {VIDEO_DIR}")
    print(f"Labels source: {LABELS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"STRIDE_SECONDS={STRIDE_SECONDS}  "
        f"CROP_TOP_FRAC={CROP_TOP_FRAC}  "
        f"CROP_BOTTOM_FRAC={CROP_BOTTOM_FRAC}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nInstruments ({len(INSTRUMENT_NAMES)}):")
    for code, name in sorted(INSTRUMENT_NAMES.items()):
        print(f"  {code}: {name}")

    case_ids = sorted(
        d.name
        for d in LABELS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("case_")
    )
    print(f"\nFound {len(case_ids)} cases (from labels)")

    available_videos = sorted(
        d.name for d in VIDEO_DIR.iterdir() if d.is_dir() and d.name.startswith("case_")
    )
    print(f"Available video dirs: {len(available_videos)}")

    case_ids = [c for c in case_ids if c in set(available_videos)]
    print(f"Cases with both labels and videos: {len(case_ids)}")

    random.seed(RANDOM_SEED)
    shuffled_ids = case_ids.copy()
    random.shuffle(shuffled_ids)
    split_idx = int(len(shuffled_ids) * 0.8)
    train_cases = set(shuffled_ids[:split_idx])
    val_cases = set(shuffled_ids[split_idx:])

    print(f"Train: {len(train_cases)} cases, Val: {len(val_cases)} cases")

    case_split_map = {
        cid: ("train" if cid in train_cases else "val") for cid in case_ids
    }

    print("\n=== Salvage: writing sentinels for cached cases ===")
    n_salvaged = salvage_existing_cases(case_ids)
    print(f"Salvaged {n_salvaged} cases (will skip re-extraction)")

    all_rows: list[dict] = []
    for case_id in tqdm(case_ids, desc="Cases"):
        rows = process_single_case(case_id, case_split_map[case_id])
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

    print("\nTool distribution (frames):")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        pct = count / len(frame_dataset_df) * 100
        print(f"  {tool}: {count} ({pct:.1f}%)")

    cases_per_tool: dict[str, set[str]] = {
        name: set() for name in INSTRUMENT_NAMES.values()
    }
    for case_id, case_frame_label_df in frame_dataset_df.groupby("video_id"):
        for tools_json in case_frame_label_df["truth_label"]:
            for tool in json.loads(tools_json):
                cases_per_tool[tool].add(str(case_id))
    print("\nTool case-coverage (cases with >=1 positive frame):")
    for tool in sorted(cases_per_tool, key=lambda t: -len(cases_per_tool[t])):
        n_cases = len(cases_per_tool[tool])
        print(f"  {tool}: {n_cases}/{len(case_ids)} cases")

    no_tools = len(frame_dataset_df[frame_dataset_df["truth_label"] == "[]"])
    pct_no = no_tools / len(frame_dataset_df) * 100
    print(f"\nFrames with no tools: {no_tools} ({pct_no:.1f}%)")

    task_counts = frame_dataset_df["task_label"].value_counts()
    print("\nTask distribution:")
    for task, count in task_counts.items():
        pct = count / len(frame_dataset_df) * 100
        print(f"  {task or '(none)'}: {count} ({pct:.1f}%)")

    split_info = {
        "train_cases": sorted(train_cases),
        "val_cases": sorted(val_cases),
        "instrument_names": {str(k): v for k, v in sorted(INSTRUMENT_NAMES.items())},
        "total_frames": len(frame_dataset_df),
        "train_frames": n_train,
        "val_frames": n_val,
    }
    with (OUTPUT_DIR / "split_info.json").open("w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Split info saved to {OUTPUT_DIR / 'split_info.json'}")


if __name__ == "__main__":
    main()
