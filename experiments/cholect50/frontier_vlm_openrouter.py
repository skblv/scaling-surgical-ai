"""Evaluate frontier VLMs on CholecT50 tool detection through OpenRouter.

Scores the proprietary models on the CholecT50 validation frames with the
same prompt and exact-match protocol as the open-weight zero-shot evaluation
in ``experiments/cholect50/zero_shot_vlm.py``. Reports exact-match accuracy,
a bootstrap confidence interval, and per-tool precision, recall, and F1.
Unparseable answers are counted as empty predictions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from scaling_surgical_ai.paths import (
    cholect50_root,
    experiment_output_dir,
    public_data_root,
)

RANDOM_SEED = 42
DATASET_CSV = experiment_output_dir("cholect50_dataset") / "dataset.csv"
CHOLECT50_ROOT = cholect50_root()
CHOLECT50_OLD_ROOT = str(public_data_root() / "CholecT50")
OUTPUT_BASE_DIR = experiment_output_dir("cholect50_frontier_vlm_openrouter")
CACHE_LOCK = threading.Lock()

INSTRUMENT_NAMES = [
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
]

OPENROUTER_MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3-flash": "google/gemini-3-flash-preview",
    "claude-opus-4.6": "anthropic/claude-opus-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "claude-fable-5": "anthropic/claude-fable-5",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
TEMPERATURE = 1.0
NUM_WORKERS = 10

Tool = Literal[
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
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

    Args:
        correct_array: List of 0/1 values indicating correct predictions.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        (ci_lower, ci_upper) as proportions (0-1).
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


def create_prompt(valid_tools: List[str]) -> str:
    """Create the tool detection prompt.

    Args:
        valid_tools: Tool names the model is allowed to return.

    Returns:
        The prompt text.
    """
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


def encode_image_base64(image_path: Path) -> str:
    """Read an image file and return a base64-encoded string.

    Args:
        image_path: Path to the frame image.

    Returns:
        Base64 encoding of the file contents.
    """
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_cache(cache_path: Path) -> dict[str, dict]:
    """Load a JSONL cache into a dict keyed by frame path.

    Args:
        cache_path: JSONL file of cached frame results.

    Returns:
        Mapping of frame path to the cached entry.
    """
    cache = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[entry["frame_path"]] = entry

    return cache


def append_cache(cache_path: Path, entry: dict) -> None:
    """Append one entry to the JSONL cache file.

    Args:
        cache_path: JSONL file to append to.
        entry: Result record for one frame.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_LOCK:
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_openrouter_vlm(
    client: OpenAI,
    model_id: str,
    prompt: str,
    image_path: Path,
) -> tuple[Optional[ToolDetectionResult], str, str]:
    """Call an OpenRouter VLM with one frame.

    Args:
        client: OpenAI-compatible OpenRouter client.
        model_id: OpenRouter model identifier.
        prompt: Tool detection prompt.
        image_path: Path to the frame image.

    Returns:
        Result, status, and raw response. Status is ``success`` for a valid
        structured result, ``parse_error`` when JSON parsing failed, or
        ``skip`` after a technical failure.
    """
    try:
        img_b64 = encode_image_base64(image_path)
    except Exception as e:
        return None, "skip", f"Image load error: {repr(e)}"

    suffix = image_path.suffix.lower().lstrip(".")
    mime_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "gif": "gif",
        "webp": "webp",
    }
    mime_subtype = mime_map.get(suffix, "jpeg")
    data_url = f"data:image/{mime_subtype};base64,{img_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"{repr(e)}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                return (
                    None,
                    "skip",
                    f"API error after {MAX_RETRIES} retries: {repr(e)}",
                )

    try:
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = raw_text[json_start:json_end]
            result = ToolDetectionResult.model_validate_json(json_str)
        else:
            result = ToolDetectionResult.model_validate_json(raw_text)
        return result, "success", raw_text
    except Exception as exc:
        print(f"Parse error: {exc!r}")
        return None, "parse_error", raw_text


def calculate_accuracy(
    predicted_tools: List[str],
    ground_truth_tools: List[str],
) -> bool:
    """Check whether the prediction equals the ground truth as a set.

    Args:
        predicted_tools: Tool names returned for the frame.
        ground_truth_tools: Tool names labeled on the frame.

    Returns:
        True when the two sets are equal.
    """
    return set(predicted_tools) == set(ground_truth_tools)


def compute_and_save_metrics(
    results: List[dict],
    model_label: str,
    valid_tools: List[str],
    output_dir: Path,
    total_sampled: int,
) -> None:
    """Compute metrics and write the results, summary, and per-tool CSVs.

    Args:
        results: Per-frame result records.
        model_label: Short model name used in the summary.
        valid_tools: Tool classes scored in the per-tool metrics.
        output_dir: Directory for ``results.csv``, ``summary.csv``, and
            ``tool_metrics.csv``.
        total_sampled: Number of frames sampled for this model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_exact_matches = 0
    skipped_technical = 0
    parse_failures = 0
    exact_match_results = []

    tool_stats = {
        tool: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt": 0, "pred": 0}
        for tool in valid_tools
    }

    results_csv = output_dir / "results.csv"
    results_fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "num_ground_truth",
        "num_predicted",
    ]

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_fieldnames)
        writer.writeheader()

        for r in results:
            status = r["status"]
            if status == "skip":
                skipped_technical += 1
                continue

            gt_tools = r["ground_truth_tools"]
            if status == "parse_error":
                parse_failures += 1
                pred_tools = []
            else:
                pred_tools = r["predicted_tools"]

            exact_match = calculate_accuracy(pred_tools, gt_tools)
            exact_match_results.append(1 if exact_match else 0)
            if exact_match:
                total_exact_matches += 1
            total_processed += 1

            gt_set = set(gt_tools)
            pred_set = set(pred_tools)

            for tool in valid_tools:
                in_gt = tool in gt_set
                in_pred = tool in pred_set
                if in_gt:
                    tool_stats[tool]["gt"] += 1
                if in_pred:
                    tool_stats[tool]["pred"] += 1
                if in_gt and in_pred:
                    tool_stats[tool]["tp"] += 1
                elif in_pred and not in_gt:
                    tool_stats[tool]["fp"] += 1
                elif in_gt and not in_pred:
                    tool_stats[tool]["fn"] += 1
                else:
                    tool_stats[tool]["tn"] += 1

            writer.writerow(
                {
                    "frame_path": r["frame_path"],
                    "ground_truth_tools": json.dumps(gt_tools),
                    "predicted_tools": json.dumps(pred_tools),
                    "exact_match": exact_match,
                    "num_ground_truth": len(gt_tools),
                    "num_predicted": len(pred_tools),
                }
            )

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
            tp = tool_stats[tool]["tp"]
            fp = tool_stats[tool]["fp"]
            fn = tool_stats[tool]["fn"]
            tn = tool_stats[tool]["tn"]

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
                    "ground_truth_count": tool_stats[tool]["gt"],
                    "predicted_count": tool_stats[tool]["pred"],
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
                "model_name": model_label,
                "total_sampled": total_sampled,
                "total_processed": total_processed,
                "skipped_technical": skipped_technical,
                "parse_failures": parse_failures,
                "exact_matches": total_exact_matches,
                "exact_match_accuracy_pct": f"{exact_match_accuracy:.2f}",
                "exact_match_ci_low_pct": f"{ci_low * 100:.2f}",
                "exact_match_ci_high_pct": f"{ci_high * 100:.2f}",
            }
        )

    print(f"\n{'=' * 60}")
    print(f"RESULTS - {model_label}")
    print(f"{'=' * 60}")
    print(f"Total sampled: {total_sampled}")
    print(f"Total processed (included in accuracy): {total_processed}")
    print(f"  - Parse failures (counted as wrong): {parse_failures}")
    print(f"Skipped technical (API/image errors, excluded): {skipped_technical}")
    print(f"Exact matches: {total_exact_matches}/{total_processed}")
    print(
        f"Exact match accuracy: {exact_match_accuracy:.2f}% "
        f"(95% CI: {ci_low * 100:.2f}% - {ci_high * 100:.2f}%)"
    )

    print("\nPer-tool metrics:")
    for tool in sorted(valid_tools):
        tp = tool_stats[tool]["tp"]
        fp = tool_stats[tool]["fp"]
        fn = tool_stats[tool]["fn"]
        total_tool = tp + fp + fn + tool_stats[tool]["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(
            f"  {tool:12s}: P={prec:.3f} R={rec:.3f} "
            f"F1={f1_val:.3f} (n={tool_stats[tool]['gt']})"
        )

    print("\nOutput files:")
    print(f"  - Results: {results_csv}")
    print(f"  - Summary: {summary_csv}")
    print(f"  - Tool metrics: {tool_metrics_csv}")
    print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="CholecT50 tool detection via OpenRouter VLMs"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(OPENROUTER_MODELS.keys()),
        default=list(OPENROUTER_MODELS.keys()),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=None,
        help="Number of observations to process (None=all)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process 100 observations per model",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of concurrent API workers (default {NUM_WORKERS})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CholecT50 tool detection evaluation through OpenRouter."""
    args = parse_args()

    if args.test:
        n_obs = 100
        print("TEST MODE: Processing 100 observations per model")
    else:
        n_obs = args.n_obs

    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    print(f"Loading CholecT50 dataset from {DATASET_CSV}")
    frame_label_df = pd.read_csv(DATASET_CSV)
    frame_label_df["frame_path"] = frame_label_df["frame_path"].str.replace(
        CHOLECT50_OLD_ROOT,
        str(CHOLECT50_ROOT),
        regex=False,
    )
    valid_tools = INSTRUMENT_NAMES
    print(f"Tool classes ({len(valid_tools)}): {valid_tools}")

    frame_to_label = {
        row["frame_path"]: json.loads(row["truth_label"])
        for _, row in frame_label_df.iterrows()
    }

    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"]
    all_val_frames = sorted(Path(p) for p in frame_label_val_df["frame_path"].tolist())
    random.seed(RANDOM_SEED)
    shuffled_frames = random.sample(all_val_frames, len(all_val_frames))

    if n_obs is not None:
        sampled_frames = shuffled_frames[:n_obs]
    else:
        sampled_frames = shuffled_frames

    num_workers = args.workers
    print(f"Total validation frames: {len(all_val_frames)}")
    print(f"Processing: {len(sampled_frames)} frames per model")
    print(f"Concurrent workers: {num_workers}")

    prompt = create_prompt(valid_tools)
    models_to_eval = args.models

    for model_label in models_to_eval:
        model_id = OPENROUTER_MODELS[model_label]
        model_dir_name = model_label.replace("/", "_").replace(".", "_")
        model_output_dir = OUTPUT_BASE_DIR / model_dir_name
        cache_path = OUTPUT_BASE_DIR / f"cache_{model_dir_name}.jsonl"

        print(f"\n{'#' * 60}")
        print(f"Evaluating: {model_label} ({model_id})")
        print(f"Cache: {cache_path}")
        print(f"Output: {model_output_dir}")
        print(f"Workers: {num_workers}")
        print(f"{'#' * 60}")

        cache = load_cache(cache_path)

        frames_to_process = []
        all_results_dict = {}

        for frame_path in sampled_frames:
            fp_str = str(frame_path)
            gt_tools = frame_to_label.get(fp_str)

            if gt_tools is None:
                all_results_dict[fp_str] = {
                    "frame_path": fp_str,
                    "ground_truth_tools": [],
                    "predicted_tools": [],
                    "status": "skip",
                    "raw_response": "No ground truth label",
                }
                continue

            if fp_str in cache:
                all_results_dict[fp_str] = cache[fp_str]
                continue

            frames_to_process.append((frame_path, fp_str, gt_tools))

        cached_count = len(sampled_frames) - len(frames_to_process)
        print(
            f"Cached: {cached_count}/{len(sampled_frames)}, "
            f"to process: {len(frames_to_process)}"
        )

        def _process_frame(
            frame_path: Path,
            fp_str: str,
            gt_tools: List[str],
        ) -> dict:
            """Score one frame and append the result to the cache.

            Args:
                frame_path: Path to the frame image.
                fp_str: Frame path stored in the cache and result rows.
                gt_tools: Ground-truth tool names for the frame.

            Returns:
                The cached result record.
            """
            result, status, raw_response = call_openrouter_vlm(
                client,
                model_id,
                prompt,
                frame_path,
            )
            pred_tools = sorted(result.detected_tools) if result is not None else []
            entry = {
                "frame_path": fp_str,
                "ground_truth_tools": gt_tools,
                "predicted_tools": pred_tools,
                "status": status,
                "raw_response": raw_response,
                "model_id": model_id,
                "timestamp": time.time(),
            }
            append_cache(cache_path, entry)
            return entry

        new_calls = 0
        if frames_to_process:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_fp = {
                    executor.submit(_process_frame, fp, fp_str, gt): fp_str
                    for fp, fp_str, gt in frames_to_process
                }
                for future in tqdm(
                    as_completed(future_to_fp),
                    total=len(future_to_fp),
                    desc=f"{model_label}",
                ):
                    fp_str = future_to_fp[future]
                    entry = future.result()
                    all_results_dict[fp_str] = entry
                    new_calls += 1

        all_results = [
            all_results_dict[str(fp)]
            for fp in sampled_frames
            if str(fp) in all_results_dict
        ]

        print(f"\n{model_label}: {new_calls} new API calls, {cached_count} from cache")

        compute_and_save_metrics(
            results=all_results,
            model_label=model_label,
            valid_tools=valid_tools,
            output_dir=model_output_dir,
            total_sampled=len(sampled_frames),
        )

    if len(models_to_eval) > 1:
        print(f"\n\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        for model_label in models_to_eval:
            model_dir_name = model_label.replace("/", "_").replace(".", "_")
            summary_path = OUTPUT_BASE_DIR / model_dir_name / "summary.csv"
            if summary_path.exists():
                model_summary_df = pd.read_csv(summary_path)
                row = model_summary_df.iloc[0]
                print(
                    f"  {model_label:25s}: "
                    f"{row['exact_match_accuracy_pct']:6.2f}% "
                    f"(95% CI: {row['exact_match_ci_low_pct']:.2f}% - "
                    f"{row['exact_match_ci_high_pct']:.2f}%) "
                    f"[n={int(row['total_processed'])}, "
                    f"parse_err={int(row['parse_failures'])}, "
                    f"skip={int(row['skipped_technical'])}]"
                )
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
