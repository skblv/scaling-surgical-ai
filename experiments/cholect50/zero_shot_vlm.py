"""Evaluate zero-shot CholecT50 instrument detection with Gemma 3 27B.

Runs the validation split in parallel workers and writes exact-match accuracy
with a bootstrap confidence interval. Unparseable answers count as empty
predictions. Frames that fail to load are excluded from the total.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import torch
import transformers
from PIL import Image
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoProcessor

from scaling_surgical_ai import paths

RANDOM_SEED = 42
CHOLECT50_ROOT = paths.cholect50_root()
DATASET_CSV = paths.experiment_output_dir("cholect50_dataset") / "dataset.csv"
OUTPUT_BASE_DIR = paths.experiment_output_dir("cholect50_zero_shot_vlm")
LOGS_DIR = paths.logs_dir()

INSTRUMENT_NAMES = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]

MODEL_CONFIGS = {
    "gemma-3-27b-it": {
        "path": paths.models_root() / "gemma-3-27b-it",
        "model_class": "Gemma3ForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
}

MAX_NEW_TOKENS = 1024
TOP_P = 1.0
NUM_EXAMPLES_PER_CLASS = 25
NUM_WORKERS = 8


Tool = Literal["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]


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
    """Compute bootstrap confidence interval for accuracy.

    Since frames are temporally correlated (consecutive frames from same procedure
    may be only seconds apart), standard i.i.d. assumptions are violated.
    Bootstrap resampling provides a non-parametric CI that doesn't assume independence.

    Args:
        correct_array: List of 0/1 values indicating correct predictions
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        (ci_lower, ci_upper) as proportions (0-1)
    """
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tool detection on CholecT50 using Gemma 3 27B"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="gemma-3-27b-it",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation (0.0 for greedy, 1.0 for sampling)",
    )
    parser.add_argument(
        "--split",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Which quarter of data to process: 1-4, None=all",
    )
    return parser.parse_args()


def load_dataset() -> pd.DataFrame:
    """Load CholecT50 dataset from CSV."""
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {DATASET_CSV}\n"
            "Run experiments/cholect50/create_dataset.py first to create it."
        )
    return pd.read_csv(DATASET_CSV)


def get_valid_tool_names() -> List[str]:
    """Get list of valid tool names for CholecT50."""
    return INSTRUMENT_NAMES


def collect_frames_from_dataset(
    frame_label_df: pd.DataFrame, split: str = "val"
) -> List[Path]:
    """Collect frames from dataset CSV for the specified split."""
    frame_label_split_df = frame_label_df[frame_label_df["split"] == split]
    return [Path(p) for p in frame_label_split_df["frame_path"].tolist()]


def get_ground_truth_tools_from_dataset(
    frame_path: Path, frame_to_label: dict
) -> Optional[List[str]]:
    """Get ground truth tool names for a frame from dataset."""
    frame_path_str = str(frame_path)
    if frame_path_str in frame_to_label:
        return frame_to_label[frame_path_str]
    return None


def create_prompt(valid_tools: List[str]) -> str:
    """Create simplified prompt for tool detection."""
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


def log_debug_response(
    debug_log_path: Path,
    frame_path: Path,
    raw_response: str,
    parse_success: bool,
    error: str = "",
):
    """Log model's raw response for debugging."""
    log_entry = {
        "timestamp": time.time(),
        "frame_path": str(frame_path),
        "raw_response": raw_response,
        "parse_success": parse_success,
        "error": error,
    }

    try:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Failed to write debug log {debug_log_path}: {exc}")


def call_vlm_with_structured_output(
    model,
    processor,
    prompt: str,
    image_path: Path,
    debug_log_path: Path,
    model_class_name: str,
    temperature: float = 1.0,
) -> tuple[Optional[ToolDetectionResult], str]:
    """Call VLM with structured output (JSON parsing from prompt-guided output).

    Returns:
        (result, status) where status is one of:
        - "success": valid result returned
        - "parse_error": model ran but JSON parsing failed (count as failure)
        - "skip": technical error like OOM or bad image (exclude from calculation)
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        log_debug_response(
            debug_log_path, image_path, "", False, f"Image load error: {repr(e)}"
        )
        return None, "skip"

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

    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    try:
        do_sample = temperature > 0
        gen_temperature = temperature if do_sample else 1.0

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=gen_temperature,
                top_p=TOP_P if do_sample else 1.0,
                top_k=None,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out_ids)]
        decoded_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        try:
            json_start = decoded_text.find("{")
            json_end = decoded_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = decoded_text[json_start:json_end]
                result = ToolDetectionResult.model_validate_json(json_str)
            else:
                result = ToolDetectionResult.model_validate_json(decoded_text)
            log_debug_response(debug_log_path, image_path, decoded_text, True)
            return result, "success"
        except Exception as e:
            log_debug_response(
                debug_log_path,
                image_path,
                decoded_text,
                False,
                f"Parse error: {repr(e)}",
            )
            return None, "parse_error"

    except Exception as e:
        log_debug_response(
            debug_log_path, image_path, "", False, f"Generation error: {repr(e)}"
        )
        return None, "skip"


def calculate_accuracy(
    predicted_tools: List[str], ground_truth_tools: List[str]
) -> bool:
    """Calculate if prediction exactly matches ground truth.

    Returns:
        True if sets match exactly, False otherwise
    """
    return set(predicted_tools) == set(ground_truth_tools)


def worker_process(
    worker_id: int,
    gpu_id: int,
    frames_chunk: List[Path],
    frame_to_label: dict,
    model_name: str,
    temperature: float,
    output_dir: Path,
    valid_tools: List[str],
) -> dict:
    """Worker process that loads model on specific GPU and processes assigned frames.

    Returns dict with results for aggregation.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    debug_log = output_dir / f"debug_responses_worker{worker_id}.jsonl"
    results_file = output_dir / f"results_worker{worker_id}.csv"

    print(
        f"[Worker {worker_id}] Starting on GPU {gpu_id}, processing {len(frames_chunk)} frames"
    )

    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    model_class_name = config["model_class"]
    dtype = config["dtype"]

    print(f"[Worker {worker_id}] Loading model on GPU {gpu_id}...")

    if model_class_name == "Gemma3ForConditionalGeneration":
        model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="cuda:0",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    print(f"[Worker {worker_id}] Model loaded")

    prompt = create_prompt(valid_tools)

    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "num_ground_truth",
        "num_predicted",
    ]

    worker_results = {
        "total_processed": 0,
        "total_exact_matches": 0,
        "skipped_technical": 0,
        "parse_failures": 0,
        "exact_match_results": [],
        "tool_stats": {
            tool: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt": 0, "pred": 0}
            for tool in valid_tools
        },
        "correct_examples": [],
        "incorrect_examples": [],
    }

    with results_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame_path in tqdm(
            frames_chunk, desc=f"Worker {worker_id}", position=worker_id
        ):
            ground_truth_tools = get_ground_truth_tools_from_dataset(
                frame_path, frame_to_label
            )

            if ground_truth_tools is None:
                worker_results["skipped_technical"] += 1
                continue

            result, status = call_vlm_with_structured_output(
                model,
                processor,
                prompt,
                frame_path,
                debug_log,
                model_class_name,
                temperature,
            )

            if status == "skip":
                worker_results["skipped_technical"] += 1
                continue

            if status == "parse_error":
                worker_results["parse_failures"] += 1
                predicted_tools = []
            else:
                predicted_tools = sorted(result.detected_tools)

            exact_match = calculate_accuracy(predicted_tools, ground_truth_tools)
            worker_results["exact_match_results"].append(1 if exact_match else 0)

            if exact_match:
                worker_results["total_exact_matches"] += 1

            row = {
                "frame_path": str(frame_path),
                "ground_truth_tools": json.dumps(ground_truth_tools),
                "predicted_tools": json.dumps(predicted_tools),
                "exact_match": exact_match,
                "num_ground_truth": len(ground_truth_tools),
                "num_predicted": len(predicted_tools),
            }
            writer.writerow(row)
            f.flush()

            worker_results["total_processed"] += 1

            gt_set = set(ground_truth_tools)
            pred_set = set(predicted_tools)

            for tool in valid_tools:
                in_gt = tool in gt_set
                in_pred = tool in pred_set

                if in_gt:
                    worker_results["tool_stats"][tool]["gt"] += 1
                if in_pred:
                    worker_results["tool_stats"][tool]["pred"] += 1

                if in_gt and in_pred:
                    worker_results["tool_stats"][tool]["tp"] += 1
                elif in_pred and not in_gt:
                    worker_results["tool_stats"][tool]["fp"] += 1
                elif in_gt and not in_pred:
                    worker_results["tool_stats"][tool]["fn"] += 1
                else:
                    worker_results["tool_stats"][tool]["tn"] += 1

            example = {
                "frame_path": str(frame_path),
                "ground_truth_tools": ground_truth_tools,
                "predicted_tools": predicted_tools,
            }
            if (
                exact_match
                and len(worker_results["correct_examples"]) < NUM_EXAMPLES_PER_CLASS
            ):
                worker_results["correct_examples"].append(example)
            elif (
                not exact_match
                and len(worker_results["incorrect_examples"]) < NUM_EXAMPLES_PER_CLASS
            ):
                worker_results["incorrect_examples"].append(example)

    print(
        f"[Worker {worker_id}] Done. Processed {worker_results['total_processed']} frames"
    )
    return worker_results


def main() -> None:
    """Run tool detection accuracy evaluation on CholecT50."""
    args = parse_args()
    model_name = args.model
    temperature = args.temperature
    split = args.split

    temp_str = f"temp{temperature:.1f}".replace(".", "_")
    split_str = f"_split{split}" if split else ""
    output_dir = OUTPUT_BASE_DIR / f"{model_name}_{temp_str}{split_str}"

    print("Starting CholecT50 Tool detection with Gemma 3 27B")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Split: {split if split else 'ALL'}")
    print(f"Output directory: {output_dir}")

    print(f"\nLoading CholecT50 dataset from {DATASET_CSV}")
    frame_label_df = load_dataset()
    valid_tools = get_valid_tool_names()
    print(f"Tool classes ({len(valid_tools)}): {valid_tools}")

    frame_to_label = {
        row["frame_path"]: json.loads(row["truth_label"])
        for _, row in frame_label_df.iterrows()
    }

    print("\nCollecting validation frames...")
    all_frames = collect_frames_from_dataset(frame_label_df, split="val")
    all_frames = sorted(all_frames)
    print(f"Found {len(all_frames)} validation frames")

    random.seed(RANDOM_SEED)
    shuffled_frames = random.sample(all_frames, len(all_frames))

    n_total = len(shuffled_frames)
    quarter = n_total // 4

    if split == 1:
        sampled_frames = shuffled_frames[0:quarter]
        print(f"Split 1/4: frames 0-{quarter - 1} = {len(sampled_frames)} frames")
    elif split == 2:
        sampled_frames = shuffled_frames[quarter : 2 * quarter]
        print(
            f"Split 2/4: frames {quarter}-{2 * quarter - 1} = {len(sampled_frames)} frames"
        )
    elif split == 3:
        sampled_frames = shuffled_frames[2 * quarter : 3 * quarter]
        print(
            f"Split 3/4: frames {2 * quarter}-{3 * quarter - 1} = {len(sampled_frames)} frames"
        )
    elif split == 4:
        sampled_frames = shuffled_frames[3 * quarter :]
        print(
            f"Split 4/4: frames {3 * quarter}-{n_total - 1} = {len(sampled_frames)} frames"
        )
    else:
        sampled_frames = shuffled_frames
        print(f"Using ALL frames: {len(sampled_frames)} frames")

    print(f"Processing {len(sampled_frames)} frames")

    output_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    sampled_frames_file = output_dir / "sampled_frames.txt"
    with sampled_frames_file.open("w", encoding="utf-8") as f:
        for frame_path in sampled_frames:
            f.write(f"{frame_path}\n")
    print(f"Saved sampled frames list to: {sampled_frames_file}")

    num_gpus = torch.cuda.device_count()
    num_workers = min(NUM_WORKERS, num_gpus, len(sampled_frames))
    print(f"\nParallel processing with {num_workers} workers on {num_gpus} GPUs")
    print(f"Each worker processes ~{len(sampled_frames) // num_workers} frames")

    chunks = [[] for _ in range(num_workers)]
    for i, frame in enumerate(sampled_frames):
        chunks[i % num_workers].append(frame)

    print(f"Frame distribution: {[len(c) for c in chunks]}")

    mp.set_start_method("spawn", force=True)

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for worker_id in range(num_workers):
            gpu_id = worker_id % num_gpus
            future = executor.submit(
                worker_process,
                worker_id,
                gpu_id,
                chunks[worker_id],
                frame_to_label,
                model_name,
                temperature,
                output_dir,
                valid_tools,
            )
            futures[future] = worker_id

        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"[Main] Worker {worker_id} completed")
            except Exception as e:
                print(f"[Main] Worker {worker_id} failed: {e}")

    print("\nAggregating results from all workers...")

    total_processed = sum(r["total_processed"] for r in all_results)
    total_exact_matches = sum(r["total_exact_matches"] for r in all_results)
    skipped_technical = sum(r["skipped_technical"] for r in all_results)
    parse_failures = sum(r["parse_failures"] for r in all_results)
    exact_match_results = []
    for r in all_results:
        exact_match_results.extend(r["exact_match_results"])

    tool_true_positives = {
        tool: sum(r["tool_stats"][tool]["tp"] for r in all_results)
        for tool in valid_tools
    }
    tool_false_positives = {
        tool: sum(r["tool_stats"][tool]["fp"] for r in all_results)
        for tool in valid_tools
    }
    tool_false_negatives = {
        tool: sum(r["tool_stats"][tool]["fn"] for r in all_results)
        for tool in valid_tools
    }
    tool_true_negatives = {
        tool: sum(r["tool_stats"][tool]["tn"] for r in all_results)
        for tool in valid_tools
    }
    tool_ground_truth_counts = {
        tool: sum(r["tool_stats"][tool]["gt"] for r in all_results)
        for tool in valid_tools
    }
    tool_predicted_counts = {
        tool: sum(r["tool_stats"][tool]["pred"] for r in all_results)
        for tool in valid_tools
    }

    correct_examples = []
    incorrect_examples = []
    for r in all_results:
        correct_examples.extend(
            r["correct_examples"][: NUM_EXAMPLES_PER_CLASS - len(correct_examples)]
        )
        incorrect_examples.extend(
            r["incorrect_examples"][: NUM_EXAMPLES_PER_CLASS - len(incorrect_examples)]
        )

    output_csv = output_dir / "results.csv"
    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "num_ground_truth",
        "num_predicted",
    ]

    print("Merging worker result files...")
    with output_csv.open("w", encoding="utf-8", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for worker_id in range(num_workers):
            worker_file = output_dir / f"results_worker{worker_id}.csv"
            if worker_file.exists():
                with worker_file.open("r", encoding="utf-8") as inf:
                    reader = csv.DictReader(inf)
                    for row in reader:
                        writer.writerow(row)

    exact_match_accuracy = (
        (total_exact_matches / total_processed * 100) if total_processed > 0 else 0.0
    )

    tool_metrics_csv = output_dir / "tool_metrics.csv"
    tool_fieldnames = [
        "tool_name",
        "ground_truth_count",
        "predicted_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    with tool_metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tool_fieldnames)
        writer.writeheader()

        for tool in sorted(valid_tools):
            tp = tool_true_positives[tool]
            fp = tool_false_positives[tool]
            fn = tool_false_negatives[tool]
            tn = tool_true_negatives[tool]

            total_tool = tp + fp + fn + tn
            accuracy = (tp + tn) / total_tool if total_tool > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            writer.writerow(
                {
                    "tool_name": tool,
                    "ground_truth_count": tool_ground_truth_counts[tool],
                    "predicted_count": tool_predicted_counts[tool],
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn,
                    "accuracy": f"{accuracy:.4f}",
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}",
                    "f1_score": f"{f1:.4f}",
                }
            )

    # Bootstrap CI for exact match (handles temporal correlation between frames)
    ci_low, ci_high = bootstrap_ci(exact_match_results, n_bootstrap=1000)

    summary_csv = output_dir / "summary.csv"
    summary_fieldnames = [
        "model_name",
        "total_sampled",
        "total_processed",
        "skipped_technical",
        "parse_failures",
        "exact_matches",
        "exact_match_accuracy_pct",
        "exact_match_ci_low_pct",
        "exact_match_ci_high_pct",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model_name": model_name,
                "total_sampled": len(sampled_frames),
                "total_processed": total_processed,
                "skipped_technical": skipped_technical,
                "parse_failures": parse_failures,
                "exact_matches": total_exact_matches,
                "exact_match_accuracy_pct": f"{exact_match_accuracy:.2f}",
                "exact_match_ci_low_pct": f"{ci_low * 100:.2f}",
                "exact_match_ci_high_pct": f"{ci_high * 100:.2f}",
            }
        )

    examples_dir = output_dir / "example_frames"
    examples_dir.mkdir(parents=True, exist_ok=True)

    correct_frames_dir = examples_dir / "correct"
    incorrect_frames_dir = examples_dir / "incorrect"
    correct_frames_dir.mkdir(exist_ok=True)
    incorrect_frames_dir.mkdir(exist_ok=True)

    examples_csv = output_dir / "examples.csv"
    examples_fieldnames = [
        "example_type",
        "frame_path",
        "saved_frame_name",
        "ground_truth_tools",
        "predicted_tools",
    ]

    print(f"\nCopying example frames to {examples_dir}")

    with examples_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=examples_fieldnames)
        writer.writeheader()

        for idx, example in enumerate(correct_examples):
            frame_path = Path(example["frame_path"])
            saved_name = f"correct_{idx:03d}_{frame_path.name}"
            dest_path = correct_frames_dir / saved_name

            try:
                shutil.copy2(frame_path, dest_path)
            except Exception as e:
                print(f"Warning: Could not copy {frame_path}: {e}")
                saved_name = "COPY_FAILED"

            writer.writerow(
                {
                    "example_type": "correct",
                    "frame_path": example["frame_path"],
                    "saved_frame_name": saved_name,
                    "ground_truth_tools": json.dumps(example["ground_truth_tools"]),
                    "predicted_tools": json.dumps(example["predicted_tools"]),
                }
            )

        for idx, example in enumerate(incorrect_examples):
            frame_path = Path(example["frame_path"])
            saved_name = f"incorrect_{idx:03d}_{frame_path.name}"
            dest_path = incorrect_frames_dir / saved_name

            try:
                shutil.copy2(frame_path, dest_path)
            except Exception as e:
                print(f"Warning: Could not copy {frame_path}: {e}")
                saved_name = "COPY_FAILED"

            writer.writerow(
                {
                    "example_type": "incorrect",
                    "frame_path": example["frame_path"],
                    "saved_frame_name": saved_name,
                    "ground_truth_tools": json.dumps(example["ground_truth_tools"]),
                    "predicted_tools": json.dumps(example["predicted_tools"]),
                }
            )

    print(f"\n{'=' * 60}")
    print(f"RESULTS - {model_name}")
    print(f"{'=' * 60}")
    print(f"Total sampled: {len(sampled_frames)}")
    print(f"Total processed (included in accuracy): {total_processed}")
    print(f"  - Parse failures (counted as wrong): {parse_failures}")
    print(f"Skipped technical (OOM/bad image, excluded): {skipped_technical}")
    print(f"Exact matches: {total_exact_matches}/{total_processed}")
    print(
        f"Exact match accuracy: {exact_match_accuracy:.2f}% (95% CI: {ci_low * 100:.2f}% - {ci_high * 100:.2f}%)"
    )
    print("\nOutput files:")
    print(f"  - Results: {output_csv}")
    print(f"  - Summary: {summary_csv}")
    print(f"  - Tool metrics: {tool_metrics_csv}")
    print(f"  - Examples: {examples_csv}")
    print(f"  - Correct examples: {len(correct_examples)} in {correct_frames_dir}")
    print(
        f"  - Incorrect examples: {len(incorrect_examples)} in {incorrect_frames_dir}"
    )
    print(f"{'=' * 60}")

    if total_processed == 0:
        raise RuntimeError("No frames were successfully processed")


if __name__ == "__main__":
    main()
