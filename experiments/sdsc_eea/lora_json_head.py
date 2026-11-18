"""Fine-tune Gemma 3 27B with LoRA to emit tool detections as JSON.

The SDSC-EEA split comes from ``experiments/sdsc_eea/create_dataset.py``.
Procedure-level splits keep a surgery out of both train and validation. The
run reports exact-match accuracy and per-tool metrics, and writes loss and
accuracy curves. The output directory is ``sdsc_eea_lora_json_head_<job id>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from scaling_surgical_ai import paths

matplotlib.use("Agg")

RANDOM_SEED = 42
MODEL_PATH = "google/gemma-3-27b-it"
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = paths.experiment_output_dir("sdsc_eea_lora_json_head", SLURM_JOB_ID)

LORA_R = 1024
LORA_ALPHA = 2048
LORA_DROPOUT = 0.05
# q_proj, k_proj, and v_proj occur in both the language model and the vision
# encoder. o_proj is the language-model output projection; out_proj is the
# vision-encoder output projection.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]

NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_NEW_TOKENS = 256
NUM_ACCURACY_SAMPLES = 100
NUM_EVALS_DURING_TRAINING = 100

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
    """Structured JSON object returned by the tool detector."""

    detected_tools: List[Tool] = Field(
        description="List of surgical tool names visible in the frame"
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def log_gpu_memory(prefix: str = "") -> None:
    """Log GPU memory usage for debugging."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(
                f"[GPU_MEM] {prefix} GPU {i}: "
                f"{allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total"
            )


def is_main_process() -> bool:
    """Return True on global rank 0 (or single-process runs)."""
    return int(os.environ.get("RANK", "0")) == 0


def load_yolo_mapping() -> Dict[int, str]:
    """Load YOLO class ID to tool name mapping."""
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    return {int(k): v for k, v in mapping.items()}


def get_valid_tool_names(yolo_mapping: Dict[int, str]) -> List[str]:
    """Get sorted list of valid tool names from YOLO mapping."""
    return sorted(set(yolo_mapping.values()))


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load the split written by create_dataset.py.

    Returns:
        Training rows, validation rows, and frame path to procedure id.
    """
    logger.info(f"Loading dataset from {DATASET_CSV}")
    frame_label_df = pd.read_csv(DATASET_CSV)

    frame_label_train_df = frame_label_df[frame_label_df["split"] == "train"].copy()
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"].copy()

    frame_to_procedure = dict(
        zip(
            frame_label_df["frame_path"],
            frame_label_df["procedure_id"].astype(str),
        )
    )

    logger.info(
        f"Loaded {len(frame_label_train_df)} train frames, "
        f"{len(frame_label_val_df)} val frames"
    )
    return frame_label_train_df, frame_label_val_df, frame_to_procedure


def load_labels_from_dataset(
    frame_label_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Load labels from dataset CSV.

    Returns mapping from frame_path -> list of tool names.
    """
    frame_to_tools = {}
    for _, row in tqdm(
        frame_label_df.iterrows(),
        total=len(frame_label_df),
        desc="Loading labels",
    ):
        frame_path = row["frame_path"]
        tools = json.loads(row["truth_label"])
        frame_to_tools[frame_path] = tools
    return frame_to_tools


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
    """Create the JSON tool-detection prompt.

    Args:
        valid_tools: Tool names the model is allowed to emit.

    Returns:
        The prompt string.
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


def create_target_json(tools: List[str]) -> str:
    """Create target JSON string for training."""
    return json.dumps({"detected_tools": sorted(tools)}, ensure_ascii=False)


def parse_model_output(output_text: str) -> Optional[ToolDetectionResult]:
    """Parse model output to extract ToolDetectionResult.

    Handles cases where model outputs text surrounding the JSON.
    Tries multiple extraction strategies.
    """
    if not output_text:
        return None

    # Strategy 1: Try direct parse (if output is clean JSON)
    try:
        return ToolDetectionResult.model_validate_json(output_text.strip())
    except (ValidationError, json.JSONDecodeError):
        pass

    # Strategy 2: match {"detected_tools": [...]} without nested braces.
    # [^{}] and [^\]] stop at the first brace or bracket.
    json_pattern = r'\{[^{}]*"detected_tools"\s*:\s*\[[^\]]*\][^{}]*\}'
    matches = re.findall(json_pattern, output_text, re.DOTALL)
    for match in matches:
        try:
            return ToolDetectionResult.model_validate_json(match.strip())
        except (ValidationError, json.JSONDecodeError):
            continue

    # Strategy 3: Find first { and last } (fallback)
    json_start = output_text.find("{")
    json_end = output_text.rfind("}") + 1
    if json_start != -1 and json_end > json_start:
        json_str = output_text[json_start:json_end].strip()
        try:
            return ToolDetectionResult.model_validate_json(json_str)
        except (ValidationError, json.JSONDecodeError):
            pass

    return None


class SurgicalToolSFTDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning on tool detection with JSON output."""

    def __init__(
        self,
        frames: List[Path],
        frame_path_to_tools: Dict[str, List[str]],
        prompt_text: str,
        processor,
        max_length: int = 2048,
    ):
        """Store frames, labels, and the processor.

        Args:
            frames: Frame paths in this split.
            frame_path_to_tools: Frame path string to ground-truth tool names.
            prompt_text: Prompt sent with every frame.
            processor: Gemma processor.
            max_length: Maximum token length of the packed example.
        """
        self.frames = frames
        self.frame_path_to_tools = frame_path_to_tools
        self.prompt_text = prompt_text
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Build one supervised example.

        Args:
            idx: Position in ``frames``.

        Returns:
            Token ids, masks, labels with the prompt masked, and pixel values.
        """
        frame_path = self.frames[idx]
        image = Image.open(frame_path).convert("RGB")

        ground_truth_tools = self.frame_path_to_tools.get(str(frame_path), [])
        target_json = create_target_json(ground_truth_tools)

        # Full conversation (user + assistant)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt_text},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_json}],
            },
        ]

        # Get prompt-only text (for masking) - just user message + generation prompt
        prompt_only_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_only_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize prompt to get its length
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_length = prompt_inputs["input_ids"].shape[1]

        # Full text with response
        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")

        # Mask labels: -100 for prompt tokens, only compute loss on response.
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0)

        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids.squeeze(0)

        return result


class SFTCollator:
    """Collator for SFT dataset."""

    def __init__(self, pad_token_id: int):
        """Store the padding token id.

        Args:
            pad_token_id: Token id used to pad text.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Stack already-padded dataset rows into one batch.

        Args:
            examples: Rows from ``SurgicalToolSFTDataset``.

        Returns:
            A batch dict accepted by the Gemma trainer.
        """
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        attention_mask = torch.stack([ex["attention_mask"] for ex in examples])
        labels = torch.stack([ex["labels"] for ex in examples])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in examples[0]:
            batch["pixel_values"] = torch.stack([ex["pixel_values"] for ex in examples])

        if "token_type_ids" in examples[0]:
            batch["token_type_ids"] = torch.stack(
                [ex["token_type_ids"] for ex in examples]
            )

        return batch


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


def evaluate_model(
    model,
    processor,
    frames: List[Path],
    yolo_mapping: Dict[int, str],
    prompt: str,
    valid_tools: List[str],
    output_dir: Path,
    split_name: str,
) -> Dict:
    """Evaluate model on a set of frames."""
    model.eval()

    results_file = output_dir / f"{split_name}_results.csv"
    predictions_file = output_dir / f"{split_name}_predictions.jsonl"

    fieldnames = [
        "frame_path",
        "ground_truth_tools",
        "predicted_tools",
        "exact_match",
        "jaccard",
        "num_ground_truth",
        "num_predicted",
    ]

    with results_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_processed = 0
    total_exact_matches = 0
    total_jaccard_sum = 0.0
    skipped = 0

    tool_true_positives = {tool: 0 for tool in valid_tools}
    tool_false_positives = {tool: 0 for tool in valid_tools}
    tool_false_negatives = {tool: 0 for tool in valid_tools}
    tool_true_negatives = {tool: 0 for tool in valid_tools}
    tool_ground_truth_counts = {tool: 0 for tool in valid_tools}
    tool_predicted_counts = {tool: 0 for tool in valid_tools}

    with (
        results_file.open("a", encoding="utf-8", newline="") as rf,
        predictions_file.open("w", encoding="utf-8") as pf,
    ):
        writer = csv.DictWriter(rf, fieldnames=fieldnames)

        for frame_path in tqdm(frames, desc=f"Evaluating {split_name}"):
            ground_truth_tools = get_ground_truth_tools(frame_path, yolo_mapping)

            torch.cuda.empty_cache()
            result = predict_tools(model, processor, frame_path, prompt)

            if result is None:
                skipped += 1
                pf.write(
                    json.dumps(
                        {
                            "frame_path": str(frame_path),
                            "ground_truth": ground_truth_tools,
                            "predicted": None,
                            "parse_error": True,
                        }
                    )
                    + "\n"
                )
                continue

            predicted_tools = sorted(result.detected_tools)
            pred_set = set(predicted_tools)
            gt_set = set(ground_truth_tools)

            exact_match = pred_set == gt_set
            if exact_match:
                total_exact_matches += 1

            # Jaccard similarity
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
            rf.flush()

            pf.write(
                json.dumps(
                    {
                        "frame_path": str(frame_path),
                        "ground_truth": ground_truth_tools,
                        "predicted": predicted_tools,
                        "exact_match": exact_match,
                    }
                )
                + "\n"
            )
            pf.flush()

            total_processed += 1

            ground_truth_set = set(ground_truth_tools)
            predicted_set = set(predicted_tools)

            for tool in ground_truth_set:
                if tool in tool_ground_truth_counts:
                    tool_ground_truth_counts[tool] += 1

            for tool in predicted_set:
                if tool in tool_predicted_counts:
                    tool_predicted_counts[tool] += 1

            for tool in valid_tools:
                in_gt = tool in ground_truth_set
                in_pred = tool in predicted_set

                if in_gt and in_pred:
                    tool_true_positives[tool] += 1
                elif not in_gt and in_pred:
                    tool_false_positives[tool] += 1
                elif in_gt and not in_pred:
                    tool_false_negatives[tool] += 1
                else:
                    tool_true_negatives[tool] += 1

    exact_match_accuracy = (
        (total_exact_matches / total_processed * 100) if total_processed > 0 else 0.0
    )
    jaccard_accuracy = (
        (total_jaccard_sum / total_processed * 100) if total_processed > 0 else 0.0
    )

    tool_metrics_csv = output_dir / f"{split_name}_tool_metrics.csv"
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

    return {
        "split": split_name,
        "total_processed": total_processed,
        "skipped": skipped,
        "exact_matches": total_exact_matches,
        "exact_match_accuracy": exact_match_accuracy,
        "jaccard_accuracy": jaccard_accuracy,
    }


class AccuracyEvaluationCallback(TrainerCallback):
    """Callback to evaluate accuracy, validation loss, and save plots during training."""

    def __init__(
        self,
        output_dir: Path,
        train_frames: List[Path],
        val_frames: List[Path],
        processor,
        yolo_mapping: Dict,
        valid_tools: List[str],
        eval_every_n_steps: int,
        num_samples: int = 100,
        frame_path_to_tools: Optional[Dict[str, List[str]]] = None,
    ):
        """Sample fixed train and validation frames for in-training plots.

        Args:
            output_dir: Directory for plots and CSV histories.
            train_frames: Training frame paths.
            val_frames: Validation frame paths.
            processor: Gemma processor.
            yolo_mapping: Class id to tool name.
            valid_tools: Tool names in a fixed order.
            eval_every_n_steps: Steps between accuracy evaluations.
            num_samples: Frames drawn from each split.
            frame_path_to_tools: Frame path string to ground-truth tool names.
        """
        self.output_dir = output_dir
        self.train_frames = train_frames
        self.val_frames = val_frames
        self.processor = processor
        self.yolo_mapping = yolo_mapping
        self.valid_tools = valid_tools
        self.eval_every_n_steps = eval_every_n_steps
        self.num_samples = num_samples
        self.frame_path_to_tools = frame_path_to_tools or {}
        self.last_eval_step = -1

        self.train_accuracy_history = []
        self.val_accuracy_history = []
        self.train_jaccard_history = []
        self.val_jaccard_history = []
        self.val_loss_history = []

        random.seed(RANDOM_SEED)
        self.train_sample_indices = random.sample(
            range(len(train_frames)), min(num_samples, len(train_frames))
        )
        self.val_sample_indices = random.sample(
            range(len(val_frames)), min(num_samples, len(val_frames))
        )

        self.train_sample_frames = [train_frames[i] for i in self.train_sample_indices]
        self.val_sample_frames = [val_frames[i] for i in self.val_sample_indices]

        logger.info(
            f"AccuracyCallback: will evaluate on {len(self.train_sample_frames)} train "
            f"and {len(self.val_sample_frames)} val samples every {eval_every_n_steps} steps"
        )

    def _compute_accuracy_on_samples(
        self, model, frames: List[Path], desc: str
    ) -> Tuple[float, float]:
        """Generate predictions and compute exact match accuracy and Jaccard on sample frames."""
        model.eval()
        device = next(model.parameters()).device

        exact_matches = 0
        jaccard_sum = 0.0
        total = 0

        prompt_text = create_prompt(self.valid_tools)

        with torch.inference_mode():
            for frame_path in tqdm(frames, desc=desc, leave=False):
                ground_truth = get_ground_truth_tools(frame_path, self.yolo_mapping)

                image = Image.open(frame_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image", "image": image},
                        ],
                    }
                ]

                input_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[input_text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
                prompt_len = inputs["input_ids"].shape[1]
                output_text = self.processor.decode(
                    generated_ids[0][prompt_len:], skip_special_tokens=True
                )

                parsed = parse_model_output(output_text)
                predicted_tools = parsed.detected_tools if parsed else []

                pred_set = set(predicted_tools)
                gt_set = set(ground_truth)

                if pred_set == gt_set:
                    exact_matches += 1

                # Jaccard similarity
                union = pred_set | gt_set
                jaccard = len(pred_set & gt_set) / len(union) if union else 1.0
                jaccard_sum += jaccard
                total += 1

        model.train()
        exact_acc = (exact_matches / total * 100) if total > 0 else 0.0
        jaccard_acc = (jaccard_sum / total * 100) if total > 0 else 0.0
        return exact_acc, jaccard_acc

    def _compute_val_loss_on_samples(self, model, frames: List[Path]) -> float:
        """Compute average validation loss on sampled frames."""
        model.eval()
        device = next(model.parameters()).device
        prompt_text = create_prompt(self.valid_tools)

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for frame_path in frames:
                ground_truth_tools = self.frame_path_to_tools.get(str(frame_path), [])
                target_json = create_target_json(ground_truth_tools)
                image = Image.open(frame_path).convert("RGB")

                # Full conversation with response
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image", "image": image},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target_json}],
                    },
                ]

                # Prompt only for masking
                prompt_only = [messages[0]]
                prompt_text_only = self.processor.apply_chat_template(
                    prompt_only, tokenize=False, add_generation_prompt=True
                )
                prompt_inputs = self.processor(
                    text=[prompt_text_only],
                    images=[image],
                    return_tensors="pt",
                    padding=False,
                )
                prompt_length = prompt_inputs["input_ids"].shape[1]

                # Full text
                full_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = self.processor(
                    text=[full_text], images=[image], return_tensors="pt", padding=True
                )

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device)

                # Create labels with prompt masked
                labels = input_ids.clone()
                labels[:, :prompt_length] = -100
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                count += 1

        model.train()
        return total_loss / count if count > 0 else 0.0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called after each training step."""
        if not is_main_process():
            return

        current_step = state.global_step

        # Evaluate at step 1 (initial) and then every eval_every_n_steps
        is_first_step = current_step == 1 and self.last_eval_step == -1
        is_eval_step = current_step - self.last_eval_step >= self.eval_every_n_steps

        if not (is_first_step or is_eval_step):
            return

        self.last_eval_step = current_step

        logger.info(
            f"\n[Step {current_step}] Computing val loss and accuracy on {self.num_samples} samples..."
        )

        # Compute validation loss on sampled frames
        val_loss = self._compute_val_loss_on_samples(model, self.val_sample_frames)
        self.val_loss_history.append({"step": current_step, "loss": val_loss})

        train_exact, train_jaccard = self._compute_accuracy_on_samples(
            model, self.train_sample_frames, f"Train accuracy (step {current_step})"
        )
        val_exact, val_jaccard = self._compute_accuracy_on_samples(
            model, self.val_sample_frames, f"Val accuracy (step {current_step})"
        )

        self.train_accuracy_history.append(
            {"step": current_step, "accuracy": train_exact}
        )
        self.val_accuracy_history.append({"step": current_step, "accuracy": val_exact})
        self.train_jaccard_history.append(
            {"step": current_step, "jaccard": train_jaccard}
        )
        self.val_jaccard_history.append({"step": current_step, "jaccard": val_jaccard})

        logger.info(
            f"[Step {current_step}] Val Loss={val_loss:.4f} | "
            f"Train: Exact={train_exact:.2f}%, Jaccard={train_jaccard:.2f}% | "
            f"Val: Exact={val_exact:.2f}%, Jaccard={val_jaccard:.2f}%"
        )

        self._save_plots(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics - update loss plot every 100 steps."""
        if not is_main_process():
            return

        # Only save plots every 100 steps to reduce overhead
        if state.global_step % 100 == 0:
            self._save_plots(state)

    def _save_plots(self, state):
        """Save both loss and accuracy plots."""
        log_history = state.log_history
        current_step = state.global_step

        train_logs = [
            entry
            for entry in log_history
            if "loss" in entry and "eval_loss" not in entry
        ]
        # Divide by GRADIENT_ACCUMULATION_STEPS to match eval loss scale
        train_loss_points = [
            {
                "step": entry.get("step", idx),
                "loss": entry["loss"] / GRADIENT_ACCUMULATION_STEPS,
            }
            for idx, entry in enumerate(train_logs)
            if "loss" in entry
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss plot (train only, log scale)
        ax1 = axes[0]
        if train_loss_points:
            train_steps = [p["step"] for p in train_loss_points]
            train_losses = [p["loss"] for p in train_loss_points]
            ax1.plot(
                train_steps,
                train_losses,
                linewidth=1.5,
                color="#1f77b4",
                label="Train Loss",
            )

        ax1.set_xlabel("Step", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_yscale("log")
        ax1.set_title("Training Loss", fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Exact match accuracy plot
        ax2 = axes[1]
        if self.train_accuracy_history:
            acc_steps = [p["step"] for p in self.train_accuracy_history]
            acc_vals = [p["accuracy"] for p in self.train_accuracy_history]
            ax2.plot(
                acc_steps,
                acc_vals,
                linewidth=1.5,
                color="#1f77b4",
                marker="s",
                markersize=4,
                label="Train Exact Match",
            )

        if self.val_accuracy_history:
            acc_steps = [p["step"] for p in self.val_accuracy_history]
            acc_vals = [p["accuracy"] for p in self.val_accuracy_history]
            ax2.plot(
                acc_steps,
                acc_vals,
                linewidth=1.5,
                color="#ff7f0e",
                linestyle="--",
                marker="o",
                markersize=4,
                label="Val Exact Match",
            )

        ax2.set_xlabel("Step", fontsize=11)
        ax2.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
        ax2.set_title("Exact Match Accuracy", fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Jaccard similarity plot
        ax3 = axes[2]
        if self.train_jaccard_history:
            jac_steps = [p["step"] for p in self.train_jaccard_history]
            jac_vals = [p["jaccard"] for p in self.train_jaccard_history]
            ax3.plot(
                jac_steps,
                jac_vals,
                linewidth=1.5,
                color="#1f77b4",
                marker="s",
                markersize=4,
                label="Train Jaccard",
            )

        if self.val_jaccard_history:
            jac_steps = [p["step"] for p in self.val_jaccard_history]
            jac_vals = [p["jaccard"] for p in self.val_jaccard_history]
            ax3.plot(
                jac_steps,
                jac_vals,
                linewidth=1.5,
                color="#ff7f0e",
                linestyle="--",
                marker="o",
                markersize=4,
                label="Val Jaccard",
            )

        ax3.set_xlabel("Step", fontsize=11)
        ax3.set_ylabel("Jaccard Similarity (%)", fontsize=11)
        ax3.set_title("Jaccard Similarity", fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plot_path_png = self.output_dir / "training_loss.png"
        plot_path_svg = self.output_dir / "training_loss.svg"
        plt.savefig(str(plot_path_png), dpi=150, bbox_inches="tight", facecolor="white")
        plt.savefig(
            str(plot_path_svg), format="svg", bbox_inches="tight", facecolor="white"
        )
        plt.close()
        logger.info(
            f"Saved plots to {plot_path_png} and {plot_path_svg} (step {current_step})"
        )

        # Save underlying data as CSV
        if train_loss_points:
            step_train_loss_df = pd.DataFrame(train_loss_points)
            step_train_loss_df.to_csv(
                self.output_dir / "train_loss_data.csv", index=False
            )
        if self.val_loss_history:
            step_val_loss_df = pd.DataFrame(self.val_loss_history)
            step_val_loss_df.to_csv(self.output_dir / "val_loss_data.csv", index=False)
        if self.train_accuracy_history:
            step_train_accuracy_df = pd.DataFrame(self.train_accuracy_history)
            step_train_accuracy_df.to_csv(
                self.output_dir / "train_accuracy_data.csv", index=False
            )
        if self.val_accuracy_history:
            step_val_accuracy_df = pd.DataFrame(self.val_accuracy_history)
            step_val_accuracy_df.to_csv(
                self.output_dir / "val_accuracy_data.csv", index=False
            )
        if self.train_jaccard_history:
            step_train_jaccard_df = pd.DataFrame(self.train_jaccard_history)
            step_train_jaccard_df.to_csv(
                self.output_dir / "train_jaccard_data.csv", index=False
            )
        if self.val_jaccard_history:
            step_val_jaccard_df = pd.DataFrame(self.val_jaccard_history)
            step_val_jaccard_df.to_csv(
                self.output_dir / "val_jaccard_data.csv", index=False
            )


def compute_metrics(eval_pred):
    """Compute shifted CE metric (same as training loss).

    NOTE: We return a different key to avoid overwriting HF's eval_loss.
    HF Trainer already computes eval_loss from model.forward().
    """
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Shifted CE (same as training loss computation)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    # Return with different key to not overwrite HF's eval_loss
    return {"shifted_ce": ce.item()}


def main() -> None:
    """Run LoRA fine-tuning for Gemma tool detection."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of Gemma for tool detection"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only run evaluation on existing checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation (default: use final model)",
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    logger.info("=" * 80)
    logger.info("LoRA fine-tuning Gemma for JSON tool detection")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nLoading YOLO class mapping from {YOLO_MAPPING_FILE}")
    yolo_mapping = load_yolo_mapping()
    valid_tools = get_valid_tool_names(yolo_mapping)
    logger.info(f"Loaded {len(valid_tools)} tool classes")

    frame_label_train_df, frame_label_val_df, frame_to_procedure = load_dataset()

    train_frames = [Path(p) for p in frame_label_train_df["frame_path"].tolist()]
    val_frames = [Path(p) for p in frame_label_val_df["frame_path"].tolist()]

    # Load ground truth labels from dataset
    frame_label_df = pd.concat(
        [frame_label_train_df, frame_label_val_df], ignore_index=True
    )
    frame_path_to_tools = load_labels_from_dataset(frame_label_df)

    logger.info(f"Training frames: {len(train_frames)}")
    logger.info(f"Validation frames: {len(val_frames)}")

    # Verify no procedure leakage
    train_procedures = set(frame_label_train_df["procedure_id"].astype(str).unique())
    val_procedures = set(frame_label_val_df["procedure_id"].astype(str).unique())
    overlap = train_procedures & val_procedures

    if overlap:
        raise RuntimeError(
            f"PROCEDURE LEAKAGE DETECTED! {len(overlap)} procedures in both splits"
        )

    logger.info(
        f"✓ No procedure overlap between train ({len(train_procedures)}) and val ({len(val_procedures)})"
    )

    prompt = create_prompt(valid_tools)

    logger.info(f"\nLoading processor from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    if not args.eval_only:
        logger.info("\n" + "=" * 80)
        logger.info("CREATING DATASETS")
        logger.info("=" * 80)

        train_dataset = SurgicalToolSFTDataset(
            frames=train_frames,
            frame_path_to_tools=frame_path_to_tools,
            prompt_text=prompt,
            processor=processor,
        )

        val_dataset = SurgicalToolSFTDataset(
            frames=val_frames,
            frame_path_to_tools=frame_path_to_tools,
            prompt_text=prompt,
            processor=processor,
        )

        logger.info(f"Train dataset: {len(train_dataset)} examples")
        logger.info(f"Val dataset: {len(val_dataset)} examples")

        logger.info(f"\nLoading base model from {MODEL_PATH}")
        log_gpu_memory("Before loading base model")

        # Load model without device_map for DDP compatibility
        # Each DDP process will load the model to its assigned GPU
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        logger.info("Base model loaded successfully")
        log_gpu_memory("After loading base model")

        if getattr(model.config, "use_cache", False):
            logger.info("Disabling KV cache for training")
            model.config.use_cache = False

        logger.info(
            "Enabling gradient checkpointing (non-reentrant for DDP compatibility)"
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Log trainable modules to verify vision encoder is included
        trainable_modules = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        vision_modules = [m for m in trainable_modules if "vision" in m.lower()]
        language_modules = [m for m in trainable_modules if "vision" not in m.lower()]
        logger.info(f"Total trainable modules: {len(trainable_modules)}")
        logger.info(f"Vision encoder modules: {len(vision_modules)}")
        logger.info(f"Language model modules: {len(language_modules)}")
        if vision_modules:
            logger.info(f"Sample vision modules: {vision_modules[:5]}")
        else:
            logger.warning("NO VISION ENCODER MODULES ARE TRAINABLE!")

        logger.info("LoRA applied successfully")
        log_gpu_memory("After applying LoRA")

        collator = SFTCollator(pad_token_id=processor.tokenizer.pad_token_id)

        # DDP: samples are split across GPUs
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        samples_per_gpu = len(train_dataset) // world_size
        steps_per_epoch = samples_per_gpu // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        total_steps = steps_per_epoch * NUM_EPOCHS

        # Evaluate ~100 times during training
        eval_every = max(1, total_steps // NUM_EVALS_DURING_TRAINING)

        logger.info(f"World size: {world_size}, samples/GPU: {samples_per_gpu}")
        logger.info(f"Total training steps: ~{total_steps}")
        logger.info(
            f"Eval/accuracy every {eval_every} steps (~{NUM_EVALS_DURING_TRAINING} times total)"
        )

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            bf16=True,
            logging_steps=1,
            # Built-in eval would score the full validation set on every check.
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            # Vision-encoder LoRA leaves out_proj unused on some steps.
            ddp_find_unused_parameters=True,
        )

        accuracy_callback = AccuracyEvaluationCallback(
            output_dir=OUTPUT_DIR,
            train_frames=train_frames,
            val_frames=val_frames,
            processor=processor,
            yolo_mapping=yolo_mapping,
            valid_tools=valid_tools,
            eval_every_n_steps=eval_every,
            num_samples=NUM_ACCURACY_SAMPLES,
            frame_path_to_tools=frame_path_to_tools,
        )

        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            callbacks=[accuracy_callback],
        )

        log_gpu_memory("Before training")
        trainer.train()
        log_gpu_memory("After training")

        # Run FULL validation evaluation at the end of training (only on main process)
        if is_main_process():
            logger.info("\n" + "=" * 80)
            logger.info("FINAL FULL VALIDATION EVALUATION (20k frames)")
            logger.info("=" * 80)

            # Unwrap DDP for evaluation
            eval_model = model.module if hasattr(model, "module") else model
            final_val_results = evaluate_model(
                eval_model,
                processor,
                val_frames,
                yolo_mapping,
                prompt,
                valid_tools,
                OUTPUT_DIR,
                "final_validation",
            )

            logger.info(f"\nFinal Validation Results (FULL {len(val_frames)} frames):")
            logger.info(f"  Total processed: {final_val_results['total_processed']}")
            logger.info(f"  Skipped (parse errors): {final_val_results['skipped']}")
            logger.info(f"  Exact matches: {final_val_results['exact_matches']}")
            logger.info(
                f"  Exact match accuracy: {final_val_results['exact_match_accuracy']:.2f}%"
            )
            logger.info(
                f"  Jaccard similarity: {final_val_results['jaccard_accuracy']:.2f}%"
            )

            # Save final results to JSON
            with (OUTPUT_DIR / "final_validation_results.json").open("w") as f:
                json.dump(final_val_results, f, indent=2)

        if is_main_process():
            final_model_path = OUTPUT_DIR / "final_model"
            if final_model_path.exists():
                shutil.rmtree(final_model_path)
            final_model_path.mkdir(parents=True, exist_ok=True)

            # Unwrap DDP model before saving
            model_to_save = model
            # Trainer wraps the module when DistributedDataParallel is active.
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
            model_to_save.save_pretrained(final_model_path / "lora_adapters")

            model_config = {
                "base_model_path": str(MODEL_PATH),
                "lora_config": {
                    "r": LORA_R,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                    "target_modules": LORA_TARGET_MODULES,
                },
                "training_config": {
                    "num_epochs": NUM_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                    "num_train_samples": len(train_frames),
                    "num_val_samples": len(val_frames),
                    "random_seed": RANDOM_SEED,
                },
                "valid_tools": valid_tools,
                "saved_at": datetime.now().isoformat(),
            }

            with (final_model_path / "model_config.json").open("w") as f:
                json.dump(model_config, f, indent=2)

            logger.info(f"\nFinal model saved to: {final_model_path}")
    else:
        checkpoint_path = args.checkpoint or str(
            OUTPUT_DIR / "final_model" / "lora_adapters"
        )
        logger.info(f"\nLoading model from checkpoint: {checkpoint_path}")

        # For eval-only, load to single GPU (don't use device_map="auto" with torchrun)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info("Model loaded from checkpoint")

    if is_main_process():
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON VALIDATION SET")
        logger.info("=" * 80)

        val_results = evaluate_model(
            model,
            processor,
            val_frames,
            yolo_mapping,
            prompt,
            valid_tools,
            OUTPUT_DIR,
            "validation",
        )

        logger.info("\nValidation Results:")
        logger.info(f"  Total processed: {val_results['total_processed']}")
        logger.info(f"  Skipped (parse errors): {val_results['skipped']}")
        logger.info(f"  Exact matches: {val_results['exact_matches']}")
        logger.info(
            f"  Exact match accuracy: {val_results['exact_match_accuracy']:.2f}%"
        )
        logger.info(f"  Jaccard similarity: {val_results['jaccard_accuracy']:.2f}%")

        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON TRAINING SET (subset)")
        logger.info("=" * 80)

        train_eval_frames = train_frames[: min(200, len(train_frames))]
        train_results = evaluate_model(
            model,
            processor,
            train_eval_frames,
            yolo_mapping,
            prompt,
            valid_tools,
            OUTPUT_DIR,
            "train",
        )

        logger.info("\nTraining Results (subset):")
        logger.info(f"  Total processed: {train_results['total_processed']}")
        logger.info(f"  Skipped (parse errors): {train_results['skipped']}")
        logger.info(f"  Exact matches: {train_results['exact_matches']}")
        logger.info(
            f"  Exact match accuracy: {train_results['exact_match_accuracy']:.2f}%"
        )
        logger.info(f"  Jaccard similarity: {train_results['jaccard_accuracy']:.2f}%")

        # Plot both exact match and Jaccard
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Exact match plot
        ax1 = axes[0]
        splits = ["Train", "Validation"]
        exact_accs = [
            train_results["exact_match_accuracy"],
            val_results["exact_match_accuracy"],
        ]
        colors = ["#2ca02c", "#ff7f0e"]

        bars1 = ax1.bar(splits, exact_accs, color=colors)
        ax1.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
        ax1.set_title("Exact Match Accuracy", fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(axis="y", alpha=0.3)

        for bar, acc in zip(bars1, exact_accs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                fontsize=10,
            )

        # Jaccard plot
        ax2 = axes[1]
        jaccard_accs = [
            train_results["jaccard_accuracy"],
            val_results["jaccard_accuracy"],
        ]
        colors2 = ["#9467bd", "#e377c2"]

        bars2 = ax2.bar(splits, jaccard_accs, color=colors2)
        ax2.set_ylabel("Jaccard Similarity (%)", fontsize=11)
        ax2.set_title("Jaccard Similarity", fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(axis="y", alpha=0.3)

        for bar, acc in zip(bars2, jaccard_accs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1f}%",
                ha="center",
                fontsize=10,
            )

        plt.suptitle("Gemma LoRA JSON tool detection", fontsize=14)
        plt.tight_layout()
        accuracy_plot_path_png = OUTPUT_DIR / "final_accuracy.png"
        accuracy_plot_path_svg = OUTPUT_DIR / "final_accuracy.svg"
        plt.savefig(
            str(accuracy_plot_path_png), dpi=150, bbox_inches="tight", facecolor="white"
        )
        plt.savefig(
            str(accuracy_plot_path_svg),
            format="svg",
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()
        logger.info(
            f"Saved accuracy plot to {accuracy_plot_path_png} and {accuracy_plot_path_svg}"
        )

        # Save final accuracy data as CSV
        final_accuracy_data = {
            "split": ["train", "validation"],
            "exact_match_accuracy": [
                train_results["exact_match_accuracy"],
                val_results["exact_match_accuracy"],
            ],
            "jaccard_accuracy": [
                train_results["jaccard_accuracy"],
                val_results["jaccard_accuracy"],
            ],
        }
        pd.DataFrame(final_accuracy_data).to_csv(
            OUTPUT_DIR / "final_accuracy_data.csv", index=False
        )

        summary = {
            "train": train_results,
            "validation": val_results,
            "config": {
                "model": str(MODEL_PATH),
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "num_train_samples": len(train_frames),
                "num_val_samples": len(val_frames),
            },
        }

        summary_file = OUTPUT_DIR / "summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(
            "Train - Exact Match: "
            f"{train_results['exact_match_accuracy']:.2f}%, Jaccard: "
            f"{train_results['jaccard_accuracy']:.2f}%"
        )
        logger.info(
            "Val   - Exact Match: "
            f"{val_results['exact_match_accuracy']:.2f}%, Jaccard: "
            f"{val_results['jaccard_accuracy']:.2f}%"
        )
        logger.info("\nOutput files:")
        logger.info(f"  - {OUTPUT_DIR / 'training_loss.png'}")
        logger.info(f"  - {OUTPUT_DIR / 'final_accuracy.png'}")
        logger.info(f"  - {OUTPUT_DIR / 'validation_results.csv'}")
        logger.info(f"  - {OUTPUT_DIR / 'validation_tool_metrics.csv'}")
        logger.info(f"  - {OUTPUT_DIR / 'summary.json'}")

    logger.info("\nJSON-head LoRA fine-tuning complete!")


if __name__ == "__main__":
    main()
