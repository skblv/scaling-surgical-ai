"""Zero-shot Gemma 3 27B exact match across temperatures on SDSC-EEA.

Evaluates ``google/gemma-3-27b-it`` with no fine-tuning on the validation
split at temperatures 0.0, 0.1, ..., 2.0. Exact match and Jaccard are the
temperature-sweep result. A previous job directory can be resumed.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from tqdm.auto import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from scaling_surgical_ai import paths

matplotlib.use("Agg")

RANDOM_SEED = 42
MODEL_PATH = "google/gemma-3-27b-it"
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = paths.experiment_output_dir(
    "sdsc_eea_zero_shot_temperature_sweep",
    SLURM_JOB_ID,
)

MAX_NEW_TOKENS = 256
ALL_TEMPERATURES = [round(t * 0.1, 1) for t in range(21)]
RESUME_FROM_DIR = paths.experiment_output_dir(
    "sdsc_eea_zero_shot_temperature_sweep",
    "77689",
)

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yolo_mapping() -> Dict[int, str]:
    """Load the YOLO class id to tool name mapping."""
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    return {int(k): v for k, v in mapping.items()}


def get_valid_tool_names(yolo_mapping: Dict[int, str]) -> List[str]:
    """Return the sorted unique tool names in a YOLO mapping."""
    return sorted(set(yolo_mapping.values()))


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dataset and return the train and validation tables.

    Returns:
        Train rows and validation rows of the SDSC-EEA dataset.
    """
    logger.info(f"Loading dataset from {DATASET_CSV}")
    frame_split_df = pd.read_csv(DATASET_CSV)
    frame_split_train_df = frame_split_df[frame_split_df["split"] == "train"].copy()
    frame_split_val_df = frame_split_df[frame_split_df["split"] == "val"].copy()
    logger.info(
        f"Loaded {len(frame_split_train_df)} train frames, "
        f"{len(frame_split_val_df)} val frames"
    )
    return frame_split_train_df, frame_split_val_df


def load_labels_from_dataset(
    frame_split_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Map each frame path to its ground-truth tool list.

    Args:
        frame_split_df: Dataset rows with ``frame_path`` and ``truth_label``.

    Returns:
        Frame path to the JSON tool list.
    """
    frame_to_tools = {}
    for _, row in frame_split_df.iterrows():
        frame_path = row["frame_path"]
        tools = json.loads(row["truth_label"])
        frame_to_tools[frame_path] = tools
    return frame_to_tools


def create_prompt(valid_tools: List[str]) -> str:
    """Build the JSON tool-detection prompt."""
    tools_list = "\n".join([f"- {tool}" for tool in valid_tools])
    prompt = (
        "Identify all surgical tools visible in this surgical video frame.\n\n"
        f"Valid tool names:\n{tools_list}\n\n"
        "Return your answer as a JSON object with this exact format:\n"
        '{"detected_tools": ["Tool Name 1", "Tool Name 2"]}\n\n'
        "Use ONLY the exact tool names from the list above.\n"
        "If a tool appears multiple times, list it only once.\n"
        'If no tools are visible, return: {"detected_tools": []}\n\n'
        "Return ONLY valid JSON, nothing else."
    )
    return prompt


def parse_model_output(
    output_text: str,
) -> Optional[ToolDetectionResult]:
    """Parse model output, trying multiple extraction strategies."""
    if not output_text:
        return None

    try:
        return ToolDetectionResult.model_validate_json(output_text.strip())
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.debug("Direct JSON parse failed: %s", exc)

    json_pattern = r'\{[^{}]*"detected_tools"\s*:\s*\[[^\]]*\][^{}]*\}'
    matches = re.findall(json_pattern, output_text, re.DOTALL)
    for match in matches:
        try:
            return ToolDetectionResult.model_validate_json(match.strip())
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.debug("Regex JSON parse failed: %s", exc)
            continue

    json_start = output_text.find("{")
    json_end = output_text.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end].strip()
        try:
            return ToolDetectionResult.model_validate_json(json_str)
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.debug("Bracket JSON parse failed: %s", exc)

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

    inputs = {
        k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
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


def evaluate_at_temperature(
    model,
    processor,
    val_frames: List[Path],
    frame_to_tools: Dict[str, List[str]],
    prompt: str,
    temperature: float,
    output_dir: Path,
) -> Dict:
    """Evaluate all val frames at a single temperature.

    Returns dict with temperature, exact_match_accuracy, jaccard, counts.
    """
    model.eval()

    temp_label = f"temp_{temperature:.1f}"
    predictions_file = output_dir / f"{temp_label}_predictions.jsonl"

    total_processed = 0
    total_exact_matches = 0
    total_jaccard_sum = 0.0
    skipped = 0

    with predictions_file.open("w", encoding="utf-8") as pf:
        for frame_path in tqdm(
            val_frames,
            desc=f"T={temperature:.1f}",
        ):
            gt_tools = frame_to_tools.get(str(frame_path), [])

            torch.cuda.empty_cache()
            result = predict_tools(
                model,
                processor,
                frame_path,
                prompt,
                temperature=temperature,
            )

            if result is None:
                skipped += 1
                pf.write(
                    json.dumps(
                        {
                            "frame_path": str(frame_path),
                            "ground_truth": gt_tools,
                            "predicted": None,
                            "parse_error": True,
                        }
                    )
                    + "\n"
                )
                continue

            predicted_tools = sorted(result.detected_tools)
            pred_set = set(predicted_tools)
            gt_set = set(gt_tools)

            exact_match = pred_set == gt_set
            if exact_match:
                total_exact_matches += 1

            union = pred_set | gt_set
            jaccard = len(pred_set & gt_set) / len(union) if union else 1.0
            total_jaccard_sum += jaccard

            pf.write(
                json.dumps(
                    {
                        "frame_path": str(frame_path),
                        "ground_truth": gt_tools,
                        "predicted": predicted_tools,
                        "exact_match": exact_match,
                        "jaccard": round(jaccard, 4),
                    }
                )
                + "\n"
            )
            pf.flush()

            total_processed += 1

    exact_match_acc = (
        (total_exact_matches / total_processed * 100) if total_processed > 0 else 0.0
    )
    jaccard_avg = (
        (total_jaccard_sum / total_processed * 100) if total_processed > 0 else 0.0
    )

    metrics = {
        "temperature": temperature,
        "total_processed": total_processed,
        "skipped": skipped,
        "exact_matches": total_exact_matches,
        "exact_match_accuracy": round(exact_match_acc, 2),
        "jaccard_accuracy": round(jaccard_avg, 2),
    }

    logger.info(
        f"T={temperature:.1f}: "
        f"exact_match={exact_match_acc:.2f}% "
        f"({total_exact_matches}/{total_processed}), "
        f"skipped={skipped}, "
        f"jaccard={jaccard_avg:.2f}%"
    )

    return metrics


def plot_accuracy_vs_temperature(
    temperature_metric_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot exact match and Jaccard against temperature.

    Args:
        temperature_metric_df: One row per temperature.
        output_dir: Directory for the PNG and SVG.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        temperature_metric_df["temperature"],
        temperature_metric_df["exact_match_accuracy"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2196F3",
        label="Exact Match Accuracy",
    )
    ax.plot(
        temperature_metric_df["temperature"],
        temperature_metric_df["jaccard_accuracy"],
        marker="s",
        linewidth=2,
        markersize=5,
        color="#FF9800",
        label="Jaccard Similarity",
    )

    ax.set_xlabel("Temperature", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(
        "Gemma-3-27b-it (Zero-Shot) — Accuracy vs Temperature",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, 2.1, 0.2))

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"accuracy_vs_temperature.{ext}", dpi=200)
    plt.close(fig)
    logger.info("Saved accuracy vs temperature plots")


def main() -> None:
    """Run the temperature sweep and write the accuracy plot."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    yolo_mapping = load_yolo_mapping()
    valid_tools = get_valid_tool_names(yolo_mapping)
    prompt = create_prompt(valid_tools)

    _, frame_split_val_df = load_dataset()
    frame_to_tools = load_labels_from_dataset(frame_split_val_df)
    val_frames = [Path(fp) for fp in frame_split_val_df["frame_path"].tolist()]
    logger.info(f"Validation frames: {len(val_frames)}")

    all_metrics = []
    completed_temps = set()
    prev_csv = RESUME_FROM_DIR / "temperature_sweep_results.csv"
    if prev_csv.exists():
        temperature_metric_prev_df = pd.read_csv(prev_csv)
        all_metrics = temperature_metric_prev_df.to_dict("records")
        completed_temps = set(temperature_metric_prev_df["temperature"].tolist())
        logger.info(
            f"Resumed {len(all_metrics)} temperatures from {prev_csv}: "
            f"{sorted(completed_temps)}"
        )

    remaining_temps = [t for t in ALL_TEMPERATURES if t not in completed_temps]
    if not remaining_temps:
        logger.info("All temperatures already completed!")
    else:
        logger.info(f"Remaining temperatures: {remaining_temps}")

    logger.info(f"Loading model: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded and distributed across GPUs")

    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {alloc:.1f}GB / {total:.1f}GB")

    for temperature in remaining_temps:
        metrics = evaluate_at_temperature(
            model=model,
            processor=processor,
            val_frames=val_frames,
            frame_to_tools=frame_to_tools,
            prompt=prompt,
            temperature=temperature,
            output_dir=OUTPUT_DIR,
        )
        all_metrics.append(metrics)

        temperature_metric_df = pd.DataFrame(all_metrics)
        temperature_metric_df.to_csv(
            OUTPUT_DIR / "temperature_sweep_results.csv", index=False
        )
        logger.info(
            f"Saved intermediate results "
            f"({len(all_metrics)}/{len(ALL_TEMPERATURES)} temps)"
        )

    temperature_metric_df = pd.DataFrame(all_metrics)
    temperature_metric_df.to_csv(
        OUTPUT_DIR / "temperature_sweep_results.csv", index=False
    )

    plot_accuracy_vs_temperature(temperature_metric_df, OUTPUT_DIR)

    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(
            {
                "model": MODEL_PATH,
                "num_val_frames": len(val_frames),
                "temperatures": ALL_TEMPERATURES,
                "results": all_metrics,
            },
            f,
            indent=2,
        )

    logger.info("=" * 60)
    logger.info("TEMPERATURE SWEEP COMPLETE")
    logger.info("=" * 60)
    for m in all_metrics:
        logger.info(
            f"  T={m['temperature']:.1f}: "
            f"exact_match={m['exact_match_accuracy']:.2f}%, "
            f"jaccard={m['jaccard_accuracy']:.2f}%"
        )
    logger.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
