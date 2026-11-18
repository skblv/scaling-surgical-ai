"""Zero-shot tool detection with Llama-3.2-90B-Vision on SDSC-EEA.

Scores every labeled frame with ``meta-llama/Llama-3.2-90B-Vision-Instruct``
sharded across the visible GPUs. Exact match is the Llama-90B zero-shot
result. A run can resume from ``results.csv`` and can take one split of the
remaining frames. An unparseable answer is an empty prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import List, Literal, Optional

import torch
from PIL import Image
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration

from scaling_surgical_ai import paths

RANDOM_SEED = 42
LABELS_ROOT = paths.eea_labels_root()
OUTPUT_BASE_DIR = paths.experiment_output_dir("sdsc_eea_zero_shot_llama_vision")
LOGS_DIR = paths.logs_dir()
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()

MODEL_ID = "meta-llama/Llama-3.2-90B-Vision-Instruct"

MAX_NEW_TOKENS = 1024
TOP_P = 1.0
NUM_EXAMPLES_PER_CLASS = 25


Tool = Literal[
    "Suction Coagulator",
    "Straight Curette",
    "Surgical Knife",
    "Suction microdebrider",
    "Local Anesthesia Needle",
    "Suction",
    "Grasper",
    "Scissor",
    "Rongeur",
    "Rhoton Dissector",
    "Curette",
    "Irrigation",
    "Doppler",
    "Straight Forceps",
    "Drill",
    "Sonopet- pineapple tip",
    "unknown",
    "Tissue shaver",
    "Bipolar Forceps",
    "Backbiting rongeur",
    "Cotton Patty",
    "Aspirating dissector",
    "Through cutting forceps",
    "Floseal Applicator",
    "Floseal Gel",
    "Surgicel",
    "Tisseel Applicator",
    "Cottle Elevator",
    "Monopolar Electrocautery",
    "Collagen Matrix",
    "Needle",
]


class ToolDetectionResult(BaseModel):
    """Structured output for tool detection."""

    detected_tools: List[Tool] = Field(
        description="List of surgical tool names visible in the frame"
    )


def bootstrap_ci(
    correct_array: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for a 0/1 vector."""
    if len(correct_array) == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(correct_array)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(correct_array) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)
    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    return bootstrap_means[lower_idx], bootstrap_means[upper_idx]


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tool detection accuracy evaluation using Llama-3.2-90B-Vision"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 for greedy)",
    )
    parser.add_argument(
        "--split-id",
        type=int,
        default=None,
        help="Which split to process (0-indexed). If None, process all.",
    )
    parser.add_argument(
        "--num-splits", type=int, default=1, help="Total number of splits"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3'). If None, use all.",
    )
    return parser.parse_args()


def load_yolo_mapping() -> dict[int, str]:
    """Load the YOLO class id to tool name mapping."""
    with open(YOLO_MAPPING_FILE, "r") as f:
        data = json.load(f)
    mapping = {}
    for class_id_str, tool_name in data["mapping"].items():
        mapping[int(class_id_str)] = tool_name
    return mapping


def get_valid_tool_names(yolo_mapping: dict[int, str]) -> List[str]:
    """Return the sorted unique tool names in a YOLO mapping."""
    return sorted(set(yolo_mapping.values()))


def collect_all_frames() -> List[Path]:
    """Collect PNG frames from each procedure's images directory."""
    all_frames = []
    for images_dir in LABELS_ROOT.glob("*/images"):
        for png_file in images_dir.glob("*.png"):
            all_frames.append(png_file)
    return all_frames


def parse_yolo_label(label_path: Path) -> List[int]:
    """Return the sorted unique class ids in a YOLO label file."""
    class_ids = []
    if not label_path.exists():
        return class_ids
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                class_id = int(parts[0])
                if class_id not in class_ids:
                    class_ids.append(class_id)
    return sorted(class_ids)


def get_ground_truth_tools(
    frame_path: Path, yolo_mapping: dict[int, str]
) -> Optional[List[str]]:
    """Return sorted tool names for a frame, or None when the label is missing."""
    label_path = frame_path.parent.parent / "labels" / f"{frame_path.stem}.txt"
    if not label_path.exists():
        return None
    class_ids = parse_yolo_label(label_path)
    tool_names = []
    for class_id in class_ids:
        if class_id in yolo_mapping:
            tool_names.append(yolo_mapping[class_id])
    return sorted(tool_names)


def create_prompt(valid_tools: List[str]) -> str:
    """Build the JSON tool-detection prompt."""
    tools_list = "\n".join([f"- {tool}" for tool in valid_tools])
    prompt = f"""Identify all surgical tools visible in this surgical video frame.

Valid tool names:
{tools_list}

Return your answer as a JSON object with this exact format:
{{"detected_tools": ["Tool Name 1", "Tool Name 2"]}}

Use ONLY the exact tool names from the list above.
If a tool appears multiple times, list it only once.
If no tools are visible, return: {{"detected_tools": []}}

Return ONLY valid JSON, nothing else."""
    return prompt


def load_processed_frames(output_csv: Path) -> set:
    """Load set of already processed frame paths from existing results."""
    processed = set()
    if output_csv.exists():
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["frame_path"])
    return processed


def generate_summary(
    output_dir: Path,
    results: List[dict],
    total_frames: int,
    skipped_no_label: int,
    skipped_image_error: int,
    skipped_gen_error: int,
    parse_errors: int,
) -> None:
    """Generate summary statistics from results."""
    total_processed = len(results)
    total_exact_matches = sum(1 for r in results if r["exact_match"])
    exact_match_accuracy = (
        (total_exact_matches / total_processed * 100) if total_processed > 0 else 0.0
    )

    exact_match_results = [1 if r["exact_match"] else 0 for r in results]
    ci_low, ci_high = bootstrap_ci(exact_match_results)

    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "total_frames",
                "total_processed",
                "skipped_no_label",
                "skipped_image_error",
                "skipped_gen_error",
                "parse_errors",
                "exact_matches",
                "exact_match_accuracy_pct",
                "exact_match_ci_low_pct",
                "exact_match_ci_high_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": MODEL_ID,
                "total_frames": total_frames,
                "total_processed": total_processed,
                "skipped_no_label": skipped_no_label,
                "skipped_image_error": skipped_image_error,
                "skipped_gen_error": skipped_gen_error,
                "parse_errors": parse_errors,
                "exact_matches": total_exact_matches,
                "exact_match_accuracy_pct": f"{exact_match_accuracy:.2f}",
                "exact_match_ci_low_pct": f"{ci_low * 100:.2f}",
                "exact_match_ci_high_pct": f"{ci_high * 100:.2f}",
            }
        )

    print(f"\n{'=' * 60}")
    print("RESULTS - Llama-3.2-90B-Vision")
    print(f"{'=' * 60}")
    print(f"Total frames: {total_frames}")
    print(f"Total processed: {total_processed}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (image error): {skipped_image_error}")
    print(f"  Skipped (gen error): {skipped_gen_error}")
    print(f"  Parse errors: {parse_errors}")
    print(f"Exact matches: {total_exact_matches}/{total_processed}")
    print(
        f"Exact match accuracy: {exact_match_accuracy:.2f}% (95% CI: {ci_low * 100:.2f}% - {ci_high * 100:.2f}%)"
    )
    print(f"Output: {output_dir / 'results.csv'}")
    print(f"{'=' * 60}")


def main() -> None:
    """Score every frame and write results, summary, and the resume cache."""
    args = parse_args()
    temperature = args.temperature

    temp_str = f"temp{temperature:.1f}".replace(".", "_")
    output_dir = OUTPUT_BASE_DIR / f"llama-3.2-90b-vision_{temp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Llama-3.2-90B-Vision tool detection")
    print(f"Model: {MODEL_ID}")
    print(f"Temperature: {temperature}")
    print(f"Output directory: {output_dir}")
    print("Model sharded across the visible GPUs with device_map='auto'")

    yolo_mapping = load_yolo_mapping()
    valid_tools = get_valid_tool_names(yolo_mapping)
    print(f"Loaded {len(valid_tools)} valid tool names")

    all_frames = collect_all_frames()
    print(f"Found {len(all_frames)} total frames")

    output_csv = output_dir / "results.csv"
    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "num_ground_truth",
        "num_predicted",
    ]

    processed_frames = load_processed_frames(output_csv)
    if processed_frames:
        print(f"RESUMING: Found {len(processed_frames)} already processed frames")
        frames_to_process = [f for f in all_frames if str(f) not in processed_frames]
        print(f"Remaining frames to process: {len(frames_to_process)}")
    else:
        frames_to_process = all_frames
        print(f"Starting fresh - processing ALL {len(frames_to_process)} frames")
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    split_id = args.split_id
    num_splits = args.num_splits
    if split_id is not None and num_splits > 1:
        # Sort frames for deterministic splitting
        frames_to_process = sorted(frames_to_process, key=lambda x: str(x))
        total_remaining = len(frames_to_process)
        split_size = total_remaining // num_splits
        start_idx = split_id * split_size
        if split_id == num_splits - 1:
            # Last split gets the remainder
            end_idx = total_remaining
        else:
            end_idx = start_idx + split_size
        frames_to_process = frames_to_process[start_idx:end_idx]
        print(
            f"SPLIT {split_id}/{num_splits}: Processing frames "
            f"{start_idx} to {end_idx} ({len(frames_to_process)} frames)"
        )

    if len(frames_to_process) == 0:
        print("All frames already processed! Generating summary...")
        results = []
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["exact_match"] = row["exact_match"] == "True"
                results.append(row)
        generate_summary(output_dir, results, len(all_frames), 0, 0, 0, 0)
        return

    prompt = create_prompt(valid_tools)

    gpu_ids = args.gpu_ids
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        num_gpus = len(gpu_ids.split(","))
        print(
            f"\nLoading {MODEL_ID} across {num_gpus} GPUs (CUDA_VISIBLE_DEVICES={gpu_ids})..."
        )
    else:
        num_gpus = torch.cuda.device_count()
        print(f"\nLoading {MODEL_ID} across {num_gpus} GPUs...")

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model loaded successfully!")

    results = []
    skipped_no_label = 0
    skipped_image_error = 0
    skipped_gen_error = 0
    parse_errors = 0

    print(f"\nProcessing {len(frames_to_process)} frames...")

    for frame_path in tqdm(frames_to_process, desc="Evaluating frames"):
        ground_truth_tools = get_ground_truth_tools(frame_path, yolo_mapping)

        if ground_truth_tools is None:
            skipped_no_label += 1
            continue

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"Image load error {frame_path}: {e}", flush=True)
            skipped_image_error += 1
            continue

        # Llama 3.2 Vision message format
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(images=image, text=input_text, return_tensors="pt").to(
            model.device
        )

        try:
            do_sample = temperature > 0
            gen_temperature = temperature if do_sample else 1.0

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=do_sample,
                    temperature=gen_temperature,
                    top_p=TOP_P if do_sample else 1.0,
                )

            decoded_text = processor.decode(
                output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            predicted_tools = []
            try:
                json_start = decoded_text.find("{")
                json_end = decoded_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_str = decoded_text[json_start:json_end]
                    result = ToolDetectionResult.model_validate_json(json_str)
                    predicted_tools = sorted(result.detected_tools)
            except Exception as e:
                print(
                    f"JSON parse error: {e} | Response: {decoded_text[:200]}",
                    flush=True,
                )
                parse_errors += 1

            exact_match = set(predicted_tools) == set(ground_truth_tools)

            row = {
                "frame_path": str(frame_path),
                "ground_truth_tools": json.dumps(ground_truth_tools),
                "predicted_tools": json.dumps(predicted_tools),
                "exact_match": exact_match,
                "num_ground_truth": len(ground_truth_tools),
                "num_predicted": len(predicted_tools),
            }
            results.append(row)

            with open(output_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

        except Exception as e:
            print(f"Generation error {frame_path}: {e}", flush=True)
            skipped_gen_error += 1
            continue

    all_results = []
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["exact_match"] = row["exact_match"] == "True"
            all_results.append(row)

    print(f"\nThis run processed {len(results)} frames")
    print(f"Total in cache: {len(all_results)} frames")

    generate_summary(
        output_dir,
        all_results,
        len(all_frames),
        skipped_no_label,
        skipped_image_error,
        skipped_gen_error,
        parse_errors,
    )


if __name__ == "__main__":
    main()
