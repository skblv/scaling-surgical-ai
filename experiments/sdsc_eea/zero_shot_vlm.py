"""Zero-shot surgical-tool detection with open-weight VLMs on SDSC-EEA.

Scores every labeled frame (or one quarter, when ``--split`` is set) with a
prompt that asks for a JSON tool list. Exact match and per-tool precision,
recall, and F1 are the zero-shot numbers in the paper. A bootstrap interval
is reported because consecutive frames are temporally correlated. An
unparseable answer is an empty prediction and counts as a miss; a failure to
load the image or to run the model is excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path
from typing import List, Literal, Optional

import torch
import transformers
from PIL import Image
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from transformers import AutoProcessor

from scaling_surgical_ai import paths

RANDOM_SEED = 42
LABELS_ROOT = paths.eea_labels_root()
OUTPUT_BASE_DIR = paths.experiment_output_dir("sdsc_eea_zero_shot_vlm")
LOGS_DIR = paths.logs_dir()
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()

MODEL_CONFIGS = {
    "qwen3-vl-30b": {
        "path": paths.models_root() / "Qwen3-VL-30B-A3B-Instruct",
        "model_class": "Qwen3VLMoeForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
    "qwen3-vl-235b": {
        "path": paths.models_root() / "Qwen3-VL-235B-A22B-Thinking",
        "model_class": "Qwen3VLMoeForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
    "gemma-3-27b-it": {
        "path": paths.models_root() / "gemma-3-27b-it",
        "model_class": "Gemma3ForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
    "medgemma-27b-it": {
        "path": paths.models_root() / "medgemma-27b-it",
        "model_class": "Gemma3ForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
    "gemma-4-31b-it": {
        "path": "google/gemma-4-31B-it",
        "model_class": "Gemma4ForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
}

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
        description="Tool detection accuracy evaluation using VLMs"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="qwen3-vl-30b",
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


def load_model(model_name: str):
    """Load the specified model and processor."""
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    model_class_name = config["model_class"]
    dtype = config["dtype"]

    print(f"Loading model: {model_name}")
    print(f"Model path: {model_path}")

    if model_class_name == "Qwen3VLMoeForConditionalGeneration":
        model = transformers.Qwen3VLMoeForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    elif model_class_name == "Gemma3ForConditionalGeneration":
        model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    elif model_class_name == "Gemma4ForConditionalGeneration":
        model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    return model, processor, config


def load_yolo_mapping() -> dict[int, str]:
    """Load YOLO class ID to tool name mapping."""
    with open(YOLO_MAPPING_FILE, "r") as f:
        data = json.load(f)

    mapping = {}
    for class_id_str, tool_name in data["mapping"].items():
        mapping[int(class_id_str)] = tool_name

    return mapping


def get_valid_tool_names(yolo_mapping: dict[int, str]) -> List[str]:
    """Get list of valid tool names from YOLO mapping."""
    return sorted(set(yolo_mapping.values()))


def collect_all_frames() -> List[Path]:
    """Collect all PNG frames from YOLO labels images folders."""
    all_frames = []

    for images_dir in LABELS_ROOT.glob("*/images"):
        for png_file in images_dir.glob("*.png"):
            all_frames.append(png_file)

    return all_frames


def parse_yolo_label(label_path: Path) -> List[int]:
    """Parse YOLO label file and extract class IDs.

    Returns:
        List of class IDs present in the frame
    """
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
    """Get ground truth tool names for a frame.

    The label is the ``labels`` sibling of the frame's ``images`` directory.

    Returns:
        Sorted list of tool names, or None if label file not found
    """
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

    is_gemma4 = model_class_name == "Gemma4ForConditionalGeneration"

    if is_gemma4:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
    else:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
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


def main() -> None:
    """Run tool detection accuracy evaluation."""
    args = parse_args()
    model_name = args.model
    temperature = args.temperature
    split = args.split

    temp_str = f"temp{temperature:.1f}".replace(".", "_")
    split_str = f"_split{split}" if split else ""
    output_dir = OUTPUT_BASE_DIR / f"{model_name}_{temp_str}{split_str}"
    debug_log = output_dir / "debug_responses.jsonl"

    print("Starting zero-shot VLM tool detection")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Split: {split if split else 'ALL'}")
    print("Split mode: processing exact half of dataset")
    print(f"Output directory: {output_dir}")

    if not LABELS_ROOT.exists():
        print(f"Labels directory not found: {LABELS_ROOT}")
        return

    print(f"\nLoading YOLO class mapping from {YOLO_MAPPING_FILE}")
    yolo_mapping = load_yolo_mapping()
    valid_tools = get_valid_tool_names(yolo_mapping)
    print(f"Loaded {len(valid_tools)} valid tool names")

    print(f"\nCollecting all frames from {LABELS_ROOT}")
    all_frames = collect_all_frames()
    all_frames = sorted(all_frames)
    print(f"Found {len(all_frames)} total frames")

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

    model, processor, config = load_model(model_name)
    model_class_name = config["model_class"]
    print("Model loaded successfully")

    prompt = create_prompt(valid_tools)

    output_csv = output_dir / "results.csv"
    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "num_ground_truth",
        "num_predicted",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"\nProcessing {len(sampled_frames)} frames...")

    total_processed = 0
    total_exact_matches = 0
    skipped_technical = 0
    parse_failures = 0

    tool_true_positives = {tool: 0 for tool in valid_tools}
    tool_false_positives = {tool: 0 for tool in valid_tools}
    tool_false_negatives = {tool: 0 for tool in valid_tools}
    tool_true_negatives = {tool: 0 for tool in valid_tools}
    tool_ground_truth_counts = {tool: 0 for tool in valid_tools}
    tool_predicted_counts = {tool: 0 for tool in valid_tools}

    exact_match_results: List[int] = []

    correct_examples = []
    incorrect_examples = []

    with output_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for frame_path in tqdm(sampled_frames, desc="Evaluating frames", unit="frame"):
            ground_truth_tools = get_ground_truth_tools(frame_path, yolo_mapping)

            if ground_truth_tools is None:
                skipped_technical += 1
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
                skipped_technical += 1
                continue

            if status == "parse_error":
                parse_failures += 1
                predicted_tools = []
            else:
                predicted_tools = sorted(result.detected_tools)

            exact_match = calculate_accuracy(predicted_tools, ground_truth_tools)
            exact_match_results.append(1 if exact_match else 0)

            if exact_match:
                total_exact_matches += 1

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

            total_processed += 1

            ground_truth_set = set(ground_truth_tools)
            predicted_set = set(predicted_tools)

            for tool in ground_truth_set:
                tool_ground_truth_counts[tool] += 1

            for tool in predicted_set:
                tool_predicted_counts[tool] += 1

            for tool in valid_tools:
                in_ground_truth = tool in ground_truth_set
                in_predicted = tool in predicted_set

                if in_ground_truth and in_predicted:
                    tool_true_positives[tool] += 1
                elif not in_ground_truth and in_predicted:
                    tool_false_positives[tool] += 1
                elif in_ground_truth and not in_predicted:
                    tool_false_negatives[tool] += 1
                else:
                    tool_true_negatives[tool] += 1

            example_data = {
                "frame_path": str(frame_path),
                "ground_truth_tools": ground_truth_tools,
                "predicted_tools": predicted_tools,
            }

            if exact_match and len(correct_examples) < NUM_EXAMPLES_PER_CLASS:
                correct_examples.append(example_data)
            elif not exact_match and len(incorrect_examples) < NUM_EXAMPLES_PER_CLASS:
                incorrect_examples.append(example_data)

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
