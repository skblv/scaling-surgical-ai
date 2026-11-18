"""Zero-shot tool detection with MedGemma 27B on SDSC-EEA.

Scores every labeled frame with ``google/medgemma-27b-it`` in bfloat16.
Each worker shards the model across two GPUs. Exact match is the MedGemma
zero-shot result. An unparseable answer is an empty prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Optional

import torch
from PIL import Image
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from scaling_surgical_ai import paths

RANDOM_SEED = 42
LABELS_ROOT = paths.eea_labels_root()
OUTPUT_BASE_DIR = paths.experiment_output_dir("sdsc_eea_zero_shot_medgemma")
LOGS_DIR = paths.logs_dir()
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()

MODEL_PATH = paths.models_root() / "medgemma-27b-it"

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
        description="Tool detection accuracy evaluation using MedGemma 27B"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 for greedy)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (each uses 2 GPUs)",
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


def process_frame_batch(args):
    """Process a batch of frames on 2 GPUs (sharded)."""
    gpu_id, frame_paths, prompt, yolo_mapping, temperature, output_file = args

    gpu_ids = [gpu_id * 2, gpu_id * 2 + 1]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids[0]},{gpu_ids[1]}"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    results = []

    for frame_path in tqdm(frame_paths, desc=f"GPU {gpu_id}", position=gpu_id):
        frame_path = Path(frame_path)
        ground_truth_tools = get_ground_truth_tools(frame_path, yolo_mapping)

        if ground_truth_tools is None:
            continue

        try:
            image = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"[Worker {gpu_id}] Image load error {frame_path}: {e}", flush=True)
            continue

        # Gemma3 message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        inputs = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

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
                    top_k=None,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )

            trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], output_ids)]
            decoded_text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

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
                    f"[Worker {gpu_id}] JSON parse error: {e} | Response: {decoded_text[:200]}",
                    flush=True,
                )

            exact_match = set(predicted_tools) == set(ground_truth_tools)

            results.append(
                {
                    "frame_path": str(frame_path),
                    "ground_truth_tools": json.dumps(ground_truth_tools),
                    "predicted_tools": json.dumps(predicted_tools),
                    "exact_match": exact_match,
                    "num_ground_truth": len(ground_truth_tools),
                    "num_predicted": len(predicted_tools),
                }
            )

        except Exception as e:
            print(f"[Worker {gpu_id}] Generation error {frame_path}: {e}", flush=True)
            continue

    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "frame_path",
            "ground_truth_tools",
            "predicted_tools",
            "exact_match",
            "num_ground_truth",
            "num_predicted",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return len(results)


def main() -> None:
    """Score every frame in parallel and write the merged results."""
    args = parse_args()
    temperature = args.temperature
    num_workers = args.num_workers

    temp_str = f"temp{temperature:.1f}".replace(".", "_")
    output_dir = OUTPUT_BASE_DIR / f"medgemma-27b-it_{temp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting MedGemma tool detection")
    print("Model: google/medgemma-27b-it (full bf16, no quantization)")
    print(f"Model path: {MODEL_PATH}")
    print(f"Temperature: {temperature}")
    print(f"Number of workers: {num_workers} (using {num_workers * 2} GPUs)")
    print(f"Output directory: {output_dir}")

    yolo_mapping = load_yolo_mapping()
    valid_tools = get_valid_tool_names(yolo_mapping)
    print(f"Loaded {len(valid_tools)} valid tool names")

    all_frames = collect_all_frames()
    print(f"Found {len(all_frames)} total frames")
    print(f"Using ALL {len(all_frames)} frames")

    prompt = create_prompt(valid_tools)

    frames_per_worker = len(all_frames) // num_workers
    worker_tasks = []
    for i in range(num_workers):
        start_idx = i * frames_per_worker
        end_idx = (
            start_idx + frames_per_worker if i < num_workers - 1 else len(all_frames)
        )
        worker_frames = [str(f) for f in all_frames[start_idx:end_idx]]
        output_file = output_dir / f"results_worker{i}.csv"
        worker_tasks.append(
            (i, worker_frames, prompt, yolo_mapping, temperature, str(output_file))
        )

    print(f"\nSplitting {len(all_frames)} frames across {num_workers} workers")
    for i, (worker_id, frames, _, _, _, _) in enumerate(worker_tasks):
        print(
            f"  Worker {worker_id} "
            f"(GPUs {worker_id * 2},{worker_id * 2 + 1}): {len(frames)} frames"
        )

    print("\nStarting parallel processing...")
    with Pool(num_workers) as pool:
        counts = pool.map(process_frame_batch, worker_tasks)

    total_processed = sum(counts)
    print(f"\nTotal processed: {total_processed}")

    print("\nMerging results from all workers...")
    all_results = []
    for i in range(num_workers):
        worker_file = output_dir / f"results_worker{i}.csv"
        if worker_file.exists():
            with open(worker_file, "r") as f:
                reader = csv.DictReader(f)
                all_results.extend(list(reader))

    output_csv = output_dir / "results.csv"
    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "num_ground_truth",
        "num_predicted",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    total_exact_matches = sum(1 for r in all_results if r["exact_match"] == "True")
    exact_match_accuracy = (
        (total_exact_matches / len(all_results) * 100) if all_results else 0.0
    )

    exact_match_results = [1 if r["exact_match"] == "True" else 0 for r in all_results]
    ci_low, ci_high = bootstrap_ci(exact_match_results)

    summary_csv = output_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "total_frames",
                "total_processed",
                "exact_matches",
                "exact_match_accuracy_pct",
                "exact_match_ci_low_pct",
                "exact_match_ci_high_pct",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": "google/medgemma-27b-it",
                "total_frames": len(all_frames),
                "total_processed": len(all_results),
                "exact_matches": total_exact_matches,
                "exact_match_accuracy_pct": f"{exact_match_accuracy:.2f}",
                "exact_match_ci_low_pct": f"{ci_low * 100:.2f}",
                "exact_match_ci_high_pct": f"{ci_high * 100:.2f}",
            }
        )

    print(f"\n{'=' * 60}")
    print("RESULTS - MedGemma 27B (full bf16)")
    print(f"{'=' * 60}")
    print(f"Total frames: {len(all_frames)}")
    print(f"Total processed: {len(all_results)}")
    print(f"Exact matches: {total_exact_matches}/{len(all_results)}")
    print(
        f"Exact match accuracy: {exact_match_accuracy:.2f}% (95% CI: {ci_low * 100:.2f}% - {ci_high * 100:.2f}%)"
    )
    print(f"Output: {output_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
