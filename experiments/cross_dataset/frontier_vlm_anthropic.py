"""Evaluate Claude Fable 5 on tool detection through the Anthropic API.

Runs the same tool-detection protocol as the OpenRouter evaluations on
CholecT50, PitVis-2023, and SurgVU, calling the Anthropic API directly.
Frames are the first ``--n-obs`` items of the seed-42 shuffle of each
validation set. Safety-classifier refusals (HTTP 200 with
``stop_reason`` of ``refusal``) are recorded with their category and
explanation and counted as empty predictions. No fallback model is used.
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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from anthropic import Anthropic, APIError
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from scaling_surgical_ai.paths import (
    cholect50_root,
    experiment_output_dir,
    pitvis_recorded_root,
    pitvis_root,
    surgvu_root,
)

RANDOM_SEED = 42
MODEL_ID = "claude-fable-5"
OUTPUT_BASE_DIR = experiment_output_dir("frontier_vlm_anthropic")
CACHE_LOCK = threading.Lock()

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
TEMPERATURE = 1.0
MAX_TOKENS = 1024
NUM_WORKERS = 4
DEFAULT_N_OBS = 100

PRICE_INPUT_PER_MTOK = 10.0
PRICE_OUTPUT_PER_MTOK = 50.0

MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

CHOLECT50_ROOT = cholect50_root()
PITVIS_ROOT = pitvis_root()
SURGVU_ROOT = surgvu_root()
PITVIS_RECORDED_ROOT = pitvis_recorded_root()


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for one benchmark dataset.

    Attributes:
        dataset_csv: Path to the dataset.csv with frame paths and labels.
        old_root: Stale path prefix in dataset.csv to rewrite (or None).
        new_root: Replacement prefix for old_root (or None).
        instrument_names: Valid tool classes for the prompt and validation.
    """

    dataset_csv: Path
    old_root: Optional[str]
    new_root: Optional[str]
    instrument_names: List[str]


DATASET_CONFIGS = {
    "cholect50": DatasetConfig(
        dataset_csv=(experiment_output_dir("cholect50_dataset") / "dataset.csv"),
        old_root=str(CHOLECT50_ROOT),
        new_root=str(CHOLECT50_ROOT),
        instrument_names=[
            "grasper",
            "bipolar",
            "hook",
            "scissors",
            "clipper",
            "irrigator",
        ],
    ),
    "pitvis": DatasetConfig(
        dataset_csv=PITVIS_ROOT / "dataset.csv",
        old_root=str(PITVIS_RECORDED_ROOT),
        new_root=str(PITVIS_ROOT),
        instrument_names=[
            "bipolar_forceps",
            "cottle",
            "cup_forceps",
            "dural_scissors",
            "freer_elevator",
            "haemostatic_foam",
            "irrigation_syringe",
            "kerrisons",
            "micro_doppler",
            "nasal_cutting_forceps",
            "pituitary_rongeurs",
            "retractable_knife",
            "ring_curette",
            "spatula_dissector",
            "stealth_pointer",
            "suction",
            "surgical_drill",
            "tissue_glue",
        ],
    ),
    "surgvu": DatasetConfig(
        dataset_csv=SURGVU_ROOT / "dataset.csv",
        old_root=None,
        new_root=None,
        instrument_names=[
            "bipolar dissector",
            "bipolar forceps",
            "cadiere forceps",
            "clip applier",
            "force bipolar",
            "grasping retractor",
            "monopolar curved scissors",
            "needle driver",
            "permanent cautery hook/spatula",
            "potts scissors",
            "prograsp forceps",
            "stapler",
            "suction irrigator",
            "synchroseal",
            "tenaculum forceps",
            "tip-up fenestrated grasper",
            "vessel sealer",
        ],
    ),
}


class ToolDetectionResult(BaseModel):
    """Structured output for tool detection."""

    detected_tools: List[str] = Field(
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


def call_fable(
    client: Anthropic,
    prompt: str,
    image_path: Path,
    valid_tools: List[str],
) -> dict:
    """Call Claude Fable 5 with one frame and classify the outcome.

    Args:
        client: Anthropic client.
        prompt: Tool detection prompt.
        image_path: Path to the frame image.
        valid_tools: Allowed tool names for output validation.

    Returns:
        Dict with predicted_tools, status, raw_response, stop_reason,
        refusal_category, refusal_explanation, and usage. Status is one of
        "success", "parse_error", "refusal", "skip".
    """
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    media_type = MEDIA_TYPES[image_path.suffix.lower()]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_b64,
                    },
                },
            ],
        }
    ]

    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=messages,
            )
            break
        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"{repr(e)}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                return {
                    "predicted_tools": [],
                    "status": "skip",
                    "raw_response": (
                        f"API error after {MAX_RETRIES} retries: {repr(e)}"
                    ),
                    "stop_reason": None,
                    "refusal_category": None,
                    "refusal_explanation": None,
                    "usage": {},
                }

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    if response.stop_reason == "refusal":
        stop_details = getattr(response, "stop_details", None)
        return {
            "predicted_tools": [],
            "status": "refusal",
            "raw_response": "",
            "stop_reason": response.stop_reason,
            "refusal_category": (
                getattr(stop_details, "category", None)
                if stop_details is not None
                else None
            ),
            "refusal_explanation": (
                getattr(stop_details, "explanation", None)
                if stop_details is not None
                else None
            ),
            "usage": usage,
        }

    raw_text = "".join(block.text for block in response.content if block.type == "text")

    try:
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            result = ToolDetectionResult.model_validate_json(
                raw_text[json_start:json_end]
            )
        else:
            result = ToolDetectionResult.model_validate_json(raw_text)
        invalid = set(result.detected_tools) - set(valid_tools)
        if invalid:
            raise ValueError(f"Invalid tool names: {sorted(invalid)}")
        predicted_tools = sorted(set(result.detected_tools))
        status = "success"
    except Exception as exc:
        print(f"Parse error: {exc!r}")
        predicted_tools = []
        status = "parse_error"

    return {
        "predicted_tools": predicted_tools,
        "status": status,
        "raw_response": raw_text,
        "stop_reason": response.stop_reason,
        "refusal_category": None,
        "refusal_explanation": None,
        "usage": usage,
    }


def compute_and_save_metrics(
    results: List[dict],
    dataset_name: str,
    valid_tools: List[str],
    output_dir: Path,
    total_sampled: int,
) -> None:
    """Compute metrics and write CSV outputs (refusals reported separately).

    Refusals are counted as wrong predictions (empty set) in the headline
    exact-match accuracy; a second accuracy excluding refusals is also
    reported. Technical skips are excluded entirely.

    Args:
        results: Per-frame result dicts.
        dataset_name: Dataset label for the summary.
        valid_tools: Tool classes.
        output_dir: Directory for results.csv, summary.csv, tool_metrics.csv.
        total_sampled: Number of frames sampled.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped_technical = 0
    parse_failures = 0
    refusals = 0
    exact_match_results = []
    exact_match_results_no_refusal = []
    total_input_tokens = 0
    total_output_tokens = 0

    tool_stats = {
        tool: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "gt": 0, "pred": 0}
        for tool in valid_tools
    }

    results_csv = output_dir / "results.csv"
    results_fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "status",
        "exact_match",
        "refusal_category",
        "refusal_explanation",
    ]

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_fieldnames)
        writer.writeheader()

        for r in results:
            usage = r.get("usage") or {}
            total_input_tokens += usage.get("input_tokens", 0) or 0
            total_output_tokens += usage.get("output_tokens", 0) or 0

            status = r["status"]
            if status == "skip":
                skipped_technical += 1
                continue

            gt_tools = r["ground_truth_tools"]
            pred_tools = r["predicted_tools"]
            if status == "parse_error":
                parse_failures += 1
            if status == "refusal":
                refusals += 1

            exact_match = set(pred_tools) == set(gt_tools)
            exact_match_results.append(1 if exact_match else 0)
            if status != "refusal":
                exact_match_results_no_refusal.append(1 if exact_match else 0)

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
                    "status": status,
                    "exact_match": exact_match,
                    "refusal_category": r.get("refusal_category") or "",
                    "refusal_explanation": r.get("refusal_explanation") or "",
                }
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
                    "true_negatives": tool_stats[tool]["tn"],
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}",
                    "f1_score": f"{f1:.4f}",
                }
            )

    n_scored = len(exact_match_results)
    n_scored_nr = len(exact_match_results_no_refusal)
    em_pct = sum(exact_match_results) / n_scored * 100 if n_scored > 0 else 0.0
    em_nr_pct = (
        sum(exact_match_results_no_refusal) / n_scored_nr * 100
        if n_scored_nr > 0
        else 0.0
    )
    ci_low, ci_high = bootstrap_ci(exact_match_results)
    cost_usd = (
        total_input_tokens / 1e6 * PRICE_INPUT_PER_MTOK
        + total_output_tokens / 1e6 * PRICE_OUTPUT_PER_MTOK
    )

    summary_csv = output_dir / "summary.csv"
    summary_fieldnames = [
        "dataset",
        "model_name",
        "total_sampled",
        "total_scored",
        "skipped_technical",
        "parse_failures",
        "refusals",
        "exact_match_accuracy_pct",
        "exact_match_ci_low_pct",
        "exact_match_ci_high_pct",
        "exact_match_excl_refusals_pct",
        "total_input_tokens",
        "total_output_tokens",
        "cost_usd",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": dataset_name,
                "model_name": MODEL_ID,
                "total_sampled": total_sampled,
                "total_scored": n_scored,
                "skipped_technical": skipped_technical,
                "parse_failures": parse_failures,
                "refusals": refusals,
                "exact_match_accuracy_pct": f"{em_pct:.2f}",
                "exact_match_ci_low_pct": f"{ci_low * 100:.2f}",
                "exact_match_ci_high_pct": f"{ci_high * 100:.2f}",
                "exact_match_excl_refusals_pct": f"{em_nr_pct:.2f}",
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "cost_usd": f"{cost_usd:.4f}",
            }
        )

    print(f"\n{'=' * 60}")
    print(f"RESULTS - {dataset_name} - {MODEL_ID}")
    print(f"{'=' * 60}")
    print(f"Total sampled: {total_sampled}")
    print(f"Total scored (refusals counted as wrong): {n_scored}")
    print(f"  - Parse failures (counted as wrong): {parse_failures}")
    print(f"  - Refusals (counted as wrong): {refusals}")
    print(f"Skipped technical (excluded): {skipped_technical}")
    print(
        f"Exact match: {em_pct:.2f}% "
        f"(95% CI: {ci_low * 100:.2f}% - {ci_high * 100:.2f}%)"
    )
    print(f"Exact match excluding refusals: {em_nr_pct:.2f}% (n={n_scored_nr})")
    print(
        f"Tokens: input={total_input_tokens}, output={total_output_tokens}, "
        f"cost=${cost_usd:.2f}"
    )
    print(f"Output files in: {output_dir}")
    print(f"{'=' * 60}")


def run_dataset(
    client: Anthropic,
    dataset_name: str,
    config: DatasetConfig,
    n_obs: int,
    num_workers: int,
) -> None:
    """Run the eval for one dataset.

    Args:
        client: Anthropic client.
        dataset_name: Key from DATASET_CONFIGS.
        config: Dataset configuration.
        n_obs: Number of validation frames to evaluate.
        num_workers: Concurrent API workers.
    """
    print(f"\n{'#' * 60}")
    print(f"Dataset: {dataset_name} ({config.dataset_csv})")
    print(f"{'#' * 60}")

    frame_label_df = pd.read_csv(config.dataset_csv)
    if config.old_root is not None:
        frame_label_df["frame_path"] = frame_label_df["frame_path"].str.replace(
            config.old_root,
            config.new_root,
            regex=False,
        )

    frame_to_label = {
        row["frame_path"]: json.loads(row["truth_label"])
        for _, row in frame_label_df.iterrows()
    }

    val_mask = frame_label_df["split"] == "val"
    all_val_frames = sorted(
        Path(p) for p in frame_label_df[val_mask]["frame_path"].tolist()
    )
    random.seed(RANDOM_SEED)
    shuffled_frames = random.sample(all_val_frames, len(all_val_frames))
    sampled_frames = shuffled_frames[:n_obs]

    print(f"Total validation frames: {len(all_val_frames)}")
    print(f"Processing: {len(sampled_frames)} frames")

    prompt = create_prompt(config.instrument_names)
    output_dir = OUTPUT_BASE_DIR / dataset_name
    cache_path = OUTPUT_BASE_DIR / f"cache_{dataset_name}.jsonl"
    cache = load_cache(cache_path)

    frames_to_process = []
    all_results_dict = {}
    for frame_path in sampled_frames:
        fp_str = str(frame_path)
        if fp_str in cache:
            all_results_dict[fp_str] = cache[fp_str]
        else:
            frames_to_process.append((frame_path, fp_str))

    print(
        f"Cached: {len(all_results_dict)}/{len(sampled_frames)}, "
        f"to process: {len(frames_to_process)}"
    )

    def _process_frame(frame_path: Path, fp_str: str) -> dict:
        """Score one frame and append the result to the cache.

        Args:
            frame_path: Path to the frame image.
            fp_str: Frame path stored in the cache and result rows.

        Returns:
            The cached result record.
        """
        outcome = call_fable(
            client,
            prompt,
            frame_path,
            config.instrument_names,
        )
        entry = {
            "frame_path": fp_str,
            "ground_truth_tools": frame_to_label[fp_str],
            "model_id": MODEL_ID,
            "timestamp": time.time(),
            **outcome,
        }
        append_cache(cache_path, entry)
        return entry

    if frames_to_process:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_fp = {
                executor.submit(_process_frame, fp, fp_str): fp_str
                for fp, fp_str in frames_to_process
            }
            for future in tqdm(
                as_completed(future_to_fp),
                total=len(future_to_fp),
                desc=dataset_name,
            ):
                fp_str = future_to_fp[future]
                all_results_dict[fp_str] = future.result()

    ordered_results = [all_results_dict[str(fp)] for fp in sampled_frames]
    compute_and_save_metrics(
        results=ordered_results,
        dataset_name=dataset_name,
        valid_tools=config.instrument_names,
        output_dir=output_dir,
        total_sampled=len(sampled_frames),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Claude Fable 5 tool detection via the Anthropic API"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=DEFAULT_N_OBS,
        help=f"Validation frames per dataset (default {DEFAULT_N_OBS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Concurrent API workers (default {NUM_WORKERS})",
    )
    return parser.parse_args()


def main() -> None:
    """Run Claude Fable 5 evals on the selected datasets."""
    args = parse_args()
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = Anthropic(api_key=api_key)

    for dataset_name in args.datasets:
        run_dataset(
            client=client,
            dataset_name=dataset_name,
            config=DATASET_CONFIGS[dataset_name],
            n_obs=args.n_obs,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()
