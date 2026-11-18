"""Evaluate JSON-head LoRA adapters on the SDSC-EEA validation set.

Loads the adapters trained by ``experiments/sdsc_eea/lora_json_head.py`` and
writes per-frame predictions, tool metrics, and a bootstrap confidence
interval for exact-match accuracy. The clustered-bootstrap loaders read this
run. The output directory is ``sdsc_eea_lora_json_head_eval`` with the job id
appended.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from tqdm.auto import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from scaling_surgical_ai import paths

RANDOM_SEED = 42
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()
BASE_MODEL_PATH = paths.models_root() / "gemma-3-27b-it"
DEFAULT_LORA_PATH = (
    paths.experiment_output_dir("sdsc_eea_lora_json_head", "44968")
    / "final_model"
    / "lora_adapters"
)

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = paths.experiment_output_dir("sdsc_eea_lora_json_head_eval", SLURM_JOB_ID)

MAX_NEW_TOKENS = 256
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA fine-tuned Gemma on validation set"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=str(DEFAULT_LORA_PATH),
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (0.0 for greedy)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of random samples to evaluate (default: all)",
    )
    return parser.parse_args()


def bootstrap_ci(
    correct_array: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for accuracy.

    Consecutive frames from the same procedure are only seconds apart, so an
    i.i.d. interval would be misleading. Bootstrap resampling does not assume
    independence.

    Args:
        correct_array: List of 0/1 values indicating correct predictions.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level. 0.95 is a 95% interval.
        seed: Random seed for reproducibility.

    Returns:
        Lower and upper bounds as proportions in ``[0, 1]``.
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


def load_yolo_mapping() -> Dict[int, str]:
    """Load the YOLO class id to tool name mapping.

    Returns:
        Mapping from class id to tool name.
    """
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    return {int(k): v for k, v in mapping.items()}


def get_valid_tool_names(yolo_mapping: Dict[int, str]) -> List[str]:
    """Get sorted list of valid tool names."""
    return sorted(set(yolo_mapping.values()))


def load_validation_set() -> pd.DataFrame:
    """Load the validation split written by create_dataset.py.

    Returns:
        One row per validation frame, with the dataset columns unchanged.
    """
    print(f"Loading dataset from {DATASET_CSV}")
    frame_label_df = pd.read_csv(DATASET_CSV)
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"].copy()
    print(f"Loaded {len(frame_label_val_df)} validation frames")
    return frame_label_val_df


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


def get_ground_truth_tools(frame_path: Path, yolo_mapping: Dict[int, str]) -> List[str]:
    """Get ground truth tool names for a frame."""
    label_path = frame_path.parent.parent / "labels" / f"{frame_path.stem}.txt"
    class_ids = parse_yolo_label(label_path)
    tools = [yolo_mapping[cid] for cid in class_ids if cid in yolo_mapping]
    return sorted(set(tools))


def create_prompt(valid_tools: List[str]) -> str:
    """Create prompt for tool detection."""
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


def parse_model_output(output_text: str) -> Optional[ToolDetectionResult]:
    """Parse model output to extract ToolDetectionResult."""
    if not output_text:
        return None

    # Strategy 1: Direct parse
    try:
        return ToolDetectionResult.model_validate_json(output_text.strip())
    except (ValidationError, json.JSONDecodeError):
        pass

    # Strategy 2: extract a flat {"detected_tools": [...]} object.
    # [^{}]* stops at a nested brace, so this does not parse nested objects.
    json_pattern = r'\{[^{}]*"detected_tools"\s*:\s*\[[^\]]*\][^{}]*\}'
    matches = re.findall(json_pattern, output_text, re.DOTALL)
    for match in matches:
        try:
            return ToolDetectionResult.model_validate_json(match.strip())
        except (ValidationError, json.JSONDecodeError):
            continue

    # Strategy 3: Find first { and last }
    json_start = output_text.find("{")
    json_end = output_text.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end].strip()
        try:
            return ToolDetectionResult.model_validate_json(json_str)
        except (ValidationError, json.JSONDecodeError):
            pass

    return None


def predict_tools(
    model,
    processor,
    frame_path: Path,
    prompt: str,
    temperature: float = 0.0,
) -> Optional[ToolDetectionResult]:
    """Generate tool prediction for a single frame."""
    image = Image.open(frame_path).convert("RGB")

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
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    do_sample = temperature > 0
    gen_temperature = temperature if do_sample else 1.0

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=gen_temperature,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    trimmed = [o[len(i) :] for i, o in zip(inputs["input_ids"], out_ids)]
    decoded_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return parse_model_output(decoded_text)


def main() -> None:
    """Run evaluation on validation set."""
    args = parse_args()
    lora_path = Path(args.lora_path)
    temperature = args.temperature

    print("=" * 80)
    print("Evaluate JSON-head LoRA Gemma on the validation set")
    print("=" * 80)
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"LoRA adapters: {lora_path}")
    print(f"Temperature: {temperature}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading YOLO mapping from {YOLO_MAPPING_FILE}")
    yolo_mapping = load_yolo_mapping()
    valid_tools = get_valid_tool_names(yolo_mapping)
    print(f"Loaded {len(valid_tools)} tool classes")

    frame_label_val_df = load_validation_set()
    val_frames = [Path(p) for p in frame_label_val_df["frame_path"].tolist()]

    random.seed(RANDOM_SEED)
    if args.num_samples is not None and args.num_samples < len(val_frames):
        val_frames = random.sample(val_frames, args.num_samples)
        print(f"Sampled {len(val_frames)} random frames for evaluation")

    print(f"\nLoading base model from {BASE_MODEL_PATH}")
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapters from {lora_path}")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()
    print("Model loaded successfully")

    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL_PATH), trust_remote_code=True
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    prompt = create_prompt(valid_tools)

    results_csv = OUTPUT_DIR / "results.csv"
    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "jaccard",
        "num_ground_truth",
        "num_predicted",
    ]

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    tool_true_positives = {tool: 0 for tool in valid_tools}
    tool_false_positives = {tool: 0 for tool in valid_tools}
    tool_false_negatives = {tool: 0 for tool in valid_tools}
    tool_true_negatives = {tool: 0 for tool in valid_tools}
    tool_ground_truth_counts = {tool: 0 for tool in valid_tools}
    tool_predicted_counts = {tool: 0 for tool in valid_tools}

    exact_match_results: List[int] = []

    total_processed = 0
    total_exact_matches = 0
    total_jaccard_sum = 0.0
    skipped = 0

    correct_examples = []
    incorrect_examples = []

    print(f"\nEvaluating {len(val_frames)} validation frames...")

    with results_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for frame_path in tqdm(val_frames, desc="Evaluating", unit="frame"):
            ground_truth_tools = get_ground_truth_tools(frame_path, yolo_mapping)

            torch.cuda.empty_cache()
            result = predict_tools(model, processor, frame_path, prompt, temperature)

            if result is None:
                skipped += 1
                continue

            predicted_tools = sorted(result.detected_tools)
            pred_set = set(predicted_tools)
            gt_set = set(ground_truth_tools)

            exact_match = pred_set == gt_set
            exact_match_results.append(1 if exact_match else 0)
            if exact_match:
                total_exact_matches += 1

            union = pred_set | gt_set
            jaccard = len(pred_set & gt_set) / len(union) if union else 1.0
            total_jaccard_sum += jaccard

            row = {
                "frame_path": str(frame_path),
                "ground_truth_tools": json.dumps(ground_truth_tools),
                "predicted_tools": json.dumps(predicted_tools),
                "exact_match": exact_match,
                "jaccard": f"{jaccard:.4f}",
                "num_ground_truth": len(ground_truth_tools),
                "num_predicted": len(predicted_tools),
            }
            writer.writerow(row)
            f.flush()

            total_processed += 1

            for tool in gt_set:
                if tool in tool_ground_truth_counts:
                    tool_ground_truth_counts[tool] += 1

            for tool in pred_set:
                if tool in tool_predicted_counts:
                    tool_predicted_counts[tool] += 1

            for tool in valid_tools:
                in_gt = tool in gt_set
                in_pred = tool in pred_set

                if in_gt and in_pred:
                    tool_true_positives[tool] += 1
                elif not in_gt and in_pred:
                    tool_false_positives[tool] += 1
                elif in_gt and not in_pred:
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
    jaccard_accuracy = (
        (total_jaccard_sum / total_processed * 100) if total_processed > 0 else 0.0
    )

    # Bootstrap CI for exact match (handles temporal correlation between frames)
    ci_low, ci_high = bootstrap_ci(exact_match_results, n_bootstrap=1000)

    tool_metrics_csv = OUTPUT_DIR / "tool_metrics.csv"
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

    summary = {
        "lora_path": str(lora_path),
        "base_model": str(BASE_MODEL_PATH),
        "temperature": temperature,
        "total_val_frames": len(val_frames),
        "total_processed": total_processed,
        "skipped": skipped,
        "exact_matches": total_exact_matches,
        "exact_match_accuracy_pct": exact_match_accuracy,
        "exact_match_ci_low_pct": ci_low * 100,
        "exact_match_ci_high_pct": ci_high * 100,
        "jaccard_similarity_pct": jaccard_accuracy,
    }

    summary_file = OUTPUT_DIR / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    examples_dir = OUTPUT_DIR / "example_frames"
    correct_dir = examples_dir / "correct"
    incorrect_dir = examples_dir / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    examples_csv = OUTPUT_DIR / "examples.csv"
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
            dest_path = correct_dir / saved_name

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
            dest_path = incorrect_dir / saved_name

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

    print(f"\n{'=' * 80}")
    print("RESULTS - LoRA Fine-tuned Gemma on Validation Set")
    print(f"{'=' * 80}")
    print(f"Total validation frames: {len(val_frames)}")
    print(f"Successfully processed: {total_processed}")
    print(f"Skipped (parse errors): {skipped}")
    print(f"\nExact matches: {total_exact_matches}/{total_processed}")
    print(
        f"Exact match accuracy: {exact_match_accuracy:.2f}% (95% CI: [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%])"
    )
    print(f"Jaccard similarity: {jaccard_accuracy:.2f}%")
    print("\nOutput files:")
    print(f"  - Results: {results_csv}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Tool metrics: {tool_metrics_csv}")
    print(f"  - Examples: {examples_csv}")
    print(f"  - Correct examples: {len(correct_examples)} in {correct_dir}")
    print(f"  - Incorrect examples: {len(incorrect_examples)} in {incorrect_dir}")
    print(f"{'=' * 80}")

    if total_processed == 0:
        raise RuntimeError("No frames were successfully processed")


if __name__ == "__main__":
    main()
