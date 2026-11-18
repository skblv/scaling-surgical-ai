"""Fine-tune Gemma 3 27B with LoRA and a linear classification head.

This is the SDSC-EEA multi-label classifier. The train and validation split
comes from ``experiments/sdsc_eea/create_dataset.py``. The run writes
``sdsc_eea_lora_classification_head_<job id>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
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
LABELS_ROOT = paths.eea_labels_root()
MODEL_PATH = "google/gemma-3-27b-it"
YOLO_MAPPING_FILE = paths.eea_class_mapping_path()
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"

LORA_R = 1024
LORA_ALPHA = 2048
LORA_DROPOUT = 0.05

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = paths.experiment_output_dir(
    "sdsc_eea_lora_classification_head", SLURM_JOB_ID
)
NUM_EPOCHS = 10
LEARNING_RATE = 5e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
NUM_ACCURACY_SAMPLES = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def log_gpu_memory(prefix: str = ""):
    """Log GPU memory usage for debugging."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(
                (
                    f"[GPU_MEM] {prefix} GPU {i}: {allocated:.2f}GB allocated, "
                    f"{reserved:.2f}GB reserved, {total:.2f}GB total"
                ),
                flush=True,
            )


def pad_sequence_right(sequences, padding_value=0):
    """Pad a list of sequences to the same length (right padding)."""
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    trailing_dims = sequences[0].size()[1:] if len(sequences[0].size()) > 1 else ()
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        output.data[i, :length] = seq
    return output


def load_yolo_mapping() -> Dict[int, str]:
    """Load YOLO class ID to tool name mapping."""
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    return {int(k): v for k, v in mapping.items()}


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
    valid_tools: List[str],
) -> Dict[str, torch.Tensor]:
    """Load multi-hot labels from the dataset CSV.

    Args:
        frame_label_df: One row per frame, including ``truth_label``.
        valid_tools: Tool names in classifier order.

    Returns:
        Mapping from frame path to a multi-hot label tensor.
    """
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    num_tools = len(valid_tools)

    frame_to_label = {}
    for _, row in tqdm(
        frame_label_df.iterrows(),
        total=len(frame_label_df),
        desc="Loading labels",
    ):
        frame_path = row["frame_path"]
        tools = json.loads(row["truth_label"])

        label = torch.zeros(num_tools, dtype=torch.float32)
        for tool in tools:
            if tool in tool_to_idx:
                label[tool_to_idx[tool]] = 1.0

        frame_to_label[frame_path] = label

    return frame_to_label


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
    # Case A: .../proc/images/img.jpg -> .../proc/labels/img.txt
    if frame_path.parent.name == "images":
        label_dir = frame_path.parent.parent / "labels"
    else:
        # Case B: .../proc/img.jpg -> .../proc/labels/img.txt
        label_dir = frame_path.parent / "labels"

    label_path = label_dir / f"{frame_path.stem}.txt"
    class_ids = parse_yolo_label(label_path)
    tools = [yolo_mapping[cid] for cid in class_ids if cid in yolo_mapping]
    return sorted(set(tools))


def is_main_process() -> bool:
    """Return True on global rank 0 (or single-process runs)."""
    return int(os.environ.get("RANK", "0")) == 0


def create_prompt(valid_tools: List[str]) -> str:
    """Create simple prompt for tool detection."""
    tools_list = "\n".join([f"- {tool}" for tool in valid_tools])

    prompt = f"""Identify all surgical tools visible in this surgical video frame.

Valid tool names:
{tools_list}

Carefully examine the image and identify which tools are present."""

    return prompt


class SurgicalToolClassificationDataset(torch.utils.data.Dataset):
    """Dataset for multi-label classification training."""

    def __init__(
        self,
        frames: List[Path],
        frame_path_to_label: Dict[str, torch.Tensor],
        num_tools: int,
        prompt_text: str,
        processor,
    ):
        """Store frames, labels, and the processor.

        Args:
            frames: Frame paths in this split.
            frame_path_to_label: Frame path string to multi-hot label.
            num_tools: Number of tool classes.
            prompt_text: Prompt sent with every frame.
            processor: Gemma processor.
        """
        self.frames = frames
        self.frame_path_to_label = frame_path_to_label
        self.num_tools = num_tools
        self.prompt_text = prompt_text
        self.processor = processor

    def __len__(self):
        """Return the number of frames."""
        return len(self.frames)

    def __getitem__(self, idx):
        """Build one training example.

        Args:
            idx: Position in ``frames``.

        Returns:
            Token ids, masks, pixel values, and the multi-hot label.
        """
        frame_path = self.frames[idx]
        image = Image.open(frame_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_text},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Get multi-hot label from dataset (key is frame path string)
        label = self.frame_path_to_label.get(
            str(frame_path), torch.zeros(self.num_tools, dtype=torch.float32)
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": label,
        }

        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids.squeeze(0)

        return result


class ToolClassificationCollator:
    """Collator for variable-length sequences."""

    def __init__(self, pad_token_id: int, num_tools: int):
        """Store padding settings.

        Args:
            pad_token_id: Token id used to pad text.
            num_tools: Number of tool classes.
        """
        self.pad_token_id = pad_token_id
        self.num_tools = num_tools

    def __call__(self, examples):
        """Pad a list of dataset rows into one batch.

        Args:
            examples: Rows from ``SurgicalToolClassificationDataset``.

        Returns:
            A batch dict accepted by ``GemmaToolClassifier.forward``.
        """
        input_ids = [ex["input_ids"] for ex in examples]
        attention_masks = [ex["attention_mask"] for ex in examples]
        # labels has shape (B, num_tools).
        labels = torch.stack([ex["labels"] for ex in examples], dim=0)

        # Pad sequences
        input_ids_padded = pad_sequence_right(
            input_ids, padding_value=self.pad_token_id
        )
        attention_mask_padded = pad_sequence_right(attention_masks, padding_value=0)

        # Stack vision tensors
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "labels": labels,
        }

        if "token_type_ids" in examples[0]:
            token_type_ids = [ex["token_type_ids"] for ex in examples]
            batch["token_type_ids"] = pad_sequence_right(
                token_type_ids, padding_value=0
            )

        return batch


class GemmaToolClassifier(nn.Module):
    """Gemma-3 with classification head for multi-label tool detection."""

    def __init__(self, base_model, hidden_size: int, num_tools: int):
        """Attach a linear head to a Gemma backbone.

        Args:
            base_model: Gemma model with LoRA adapters applied.
            hidden_size: Width of the last hidden state.
            num_tools: Number of tool classes.
        """
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_tools)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels=None,
        token_type_ids=None,
        **kwargs,
    ):
        """Pool the last hidden state and return classification logits.

        Args:
            input_ids: Token ids.
            attention_mask: Padding mask.
            pixel_values: Vision input.
            labels: Multi-hot targets. Unused by the forward pass.
            token_type_ids: Optional token type ids.
            **kwargs: Extra batch fields ignored by this head.

        Returns:
            A dict with ``logits`` and, when labels are present, ``loss``.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        # hidden has shape (B, T, H).
        hidden = outputs.hidden_states[-1]

        # Use masked mean pooling over all non-padding tokens
        # mask has shape (B, T, 1) and pooled has shape (B, H).
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # logits has shape (B, num_tools).
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Return dict like Qwen version - HF Trainer expects this
        return {"loss": loss, "logits": logits}


def predict_tools(
    model: GemmaToolClassifier,
    processor,
    frame_path: Path,
    prompt: str,
    valid_tools: List[str],
    threshold: float = 0.5,
) -> List[str]:
    """Predict tools for a single frame using classifier."""
    image = Image.open(frame_path).convert("RGB")

    # Format for Gemma-3 vision model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Get device from model parameters (works with DDP/PEFT/device_map)
    device = next(model.parameters()).device

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        fwd_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
        }
        if "token_type_ids" in inputs:
            fwd_kwargs["token_type_ids"] = inputs["token_type_ids"]
        out = model(**fwd_kwargs)
        probs = torch.sigmoid(out["logits"]).squeeze(0)

    pred_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    return sorted([valid_tools[i] for i in pred_indices])


def evaluate_accuracy_from_predictions(
    model: GemmaToolClassifier,
    processor,
    frames: List[Path],
    yolo_mapping: Dict[int, str],
    prompt: str,
    valid_tools: List[str],
    predictions_file: Path,
    desc: str,
) -> Tuple[float, float, Dict[str, Dict]]:
    """Evaluate accuracy on a set of frames.

    Returns:
        (exact_match_accuracy, jaccard_accuracy, predictions_dict)
    """
    # Load cache if exists
    cache = {}
    if predictions_file.exists():
        logger.info(f"Loading predictions cache from {predictions_file.name}...")
        with predictions_file.open("r", encoding="utf-8") as f:
            for line in f:
                pred = json.loads(line)
                cache[pred["frame_path"]] = pred
        logger.info(f"Cache contains {len(cache)} predictions")

    predictions = {}
    frames_to_process = []

    for frame_path in frames:
        canonical_path = str(frame_path)

        if canonical_path in cache:
            cached = cache[canonical_path]
            # Compute jaccard if not in cache (backwards compat)
            pred_set = set(cached["predicted"])
            gt_set = set(cached["ground_truth"])
            union = pred_set | gt_set
            jaccard = cached.get(
                "jaccard", len(pred_set & gt_set) / len(union) if union else 1.0
            )
            predictions[canonical_path] = {
                "ground_truth": cached["ground_truth"],
                "predicted": cached["predicted"],
                "is_correct": cached["is_correct"],
                "jaccard": jaccard,
            }
        else:
            ground_truth = get_ground_truth_tools(frame_path, yolo_mapping)
            frames_to_process.append((frame_path, canonical_path, ground_truth))

    logger.info(f"Using {len(predictions)} cached predictions")
    logger.info(f"Running inference on {len(frames_to_process)} frames")

    # Run inference on remaining frames
    num_failures = 0
    with predictions_file.open("a", encoding="utf-8") as f:
        for frame_path, canonical_path, ground_truth in tqdm(
            frames_to_process, desc=desc
        ):
            torch.cuda.empty_cache()

            try:
                predicted = predict_tools(
                    model, processor, frame_path, prompt, valid_tools
                )
            except Exception as e:
                logger.warning(f"Inference failed for {frame_path}: {e}")
                predicted = []
                num_failures += 1

            pred_set = set(predicted)
            gt_set = set(ground_truth)
            is_correct = pred_set == gt_set

            # Jaccard similarity
            union = pred_set | gt_set
            jaccard = len(pred_set & gt_set) / len(union) if union else 1.0

            predictions[canonical_path] = {
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "jaccard": jaccard,
            }

            # Write to cache
            f.write(
                json.dumps(
                    {
                        "frame_path": canonical_path,
                        "ground_truth": ground_truth,
                        "predicted": predicted,
                        "is_correct": is_correct,
                        "jaccard": jaccard,
                    }
                )
                + "\n"
            )
            f.flush()

            torch.cuda.empty_cache()

    if num_failures > 0:
        logger.warning(
            f"{num_failures}/{len(frames_to_process)} frames failed inference"
        )

    if not predictions:
        raise RuntimeError("No valid predictions!")

    correct = sum(1 for pred in predictions.values() if pred["is_correct"])
    exact_accuracy = correct / len(predictions) * 100

    jaccard_sum = sum(pred["jaccard"] for pred in predictions.values())
    jaccard_accuracy = jaccard_sum / len(predictions) * 100

    # Compute per-tool metrics
    tool_true_positives = {tool: 0 for tool in valid_tools}
    tool_false_positives = {tool: 0 for tool in valid_tools}
    tool_false_negatives = {tool: 0 for tool in valid_tools}
    tool_true_negatives = {tool: 0 for tool in valid_tools}
    tool_ground_truth_counts = {tool: 0 for tool in valid_tools}
    tool_predicted_counts = {tool: 0 for tool in valid_tools}

    for pred_data in predictions.values():
        gt_set = set(pred_data["ground_truth"])
        pred_set = set(pred_data["predicted"])

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

    # Write per-tool metrics CSV
    tool_metrics_csv = (
        predictions_file.parent / f"{predictions_file.stem}_tool_metrics.csv"
    )
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

    logger.info(f"Saved per-tool metrics to {tool_metrics_csv}")

    return exact_accuracy, jaccard_accuracy, predictions


def compute_metrics(eval_pred):
    """Compute accuracy and Jaccard for multi-label classification."""
    logits, labels = eval_pred
    # logits: (B, num_tools)
    # labels: (B, num_tools)

    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Threshold at 0.5
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    # Exact match accuracy (all labels for a sample must match)
    matches = np.all(preds == labels, axis=1)
    exact_accuracy = matches.mean()

    # Jaccard similarity per sample
    jaccard_scores = []
    for pred_row, label_row in zip(preds, labels):
        pred_set = set(np.where(pred_row == 1)[0])
        label_set = set(np.where(label_row == 1)[0])
        union = pred_set | label_set
        jaccard = len(pred_set & label_set) / len(union) if union else 1.0
        jaccard_scores.append(jaccard)
    jaccard_accuracy = np.mean(jaccard_scores)

    return {"accuracy": exact_accuracy * 100.0, "jaccard": jaccard_accuracy * 100.0}


def main() -> None:
    """Run multi-label classification training."""
    parser = argparse.ArgumentParser(
        description="Multi-label classification for Gemma-3 27B on SDSC-EEA"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Load existing checkpoint for evaluation (path to checkpoint dir)",
    )
    args = parser.parse_args()

    logger.info("Starting multi-label classification for Gemma-3")
    logger.info(f"Load checkpoint: {args.load_checkpoint}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nLoading YOLO class mapping from {YOLO_MAPPING_FILE}")
    yolo_mapping = load_yolo_mapping()
    valid_tools = sorted(set(yolo_mapping.values()))
    num_tools = len(valid_tools)
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    logger.info(f"Loaded {num_tools} tool classes")

    frame_label_train_df, frame_label_val_df, frame_to_procedure = load_dataset()

    train_frames = [Path(p) for p in frame_label_train_df["frame_path"].tolist()]
    val_frames = [Path(p) for p in frame_label_val_df["frame_path"].tolist()]

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

    logger.info("\n" + "=" * 80)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 80)
    logger.info(f"MODEL_PATH        = {MODEL_PATH}")
    logger.info(f"YOLO_MAPPING_FILE = {YOLO_MAPPING_FILE}")
    logger.info(f"num_tools         = {num_tools}")
    logger.info(f"valid_tools[:10]  = {valid_tools[:10]}")
    logger.info(f"tool_to_idx (first 5): {dict(list(tool_to_idx.items())[:5])}")
    logger.info(f"hash(valid_tools) = {hash(tuple(valid_tools))}")
    logger.info("=" * 80)

    prompt = create_prompt(valid_tools)

    # Load labels from dataset (truth_label column has YOLO-derived labels)
    logger.info("\nLoading labels from dataset...")
    frame_label_df = pd.concat(
        [frame_label_train_df, frame_label_val_df], ignore_index=True
    )
    frame_path_to_label = load_labels_from_dataset(frame_label_df, valid_tools)
    logger.info(f"Loaded labels for {len(frame_path_to_label)} frames")

    # Verify labels are populated
    sample_labels = list(frame_path_to_label.values())[:100]
    avg_tools = sum(label.sum().item() for label in sample_labels) / len(sample_labels)
    logger.info(f"Average tools per frame (sample): {avg_tools:.2f}")

    # Load processor
    logger.info(f"\nLoading processor from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    # Training path
    logger.info("\n" + "=" * 80)
    logger.info("CREATING DATASETS")
    logger.info("=" * 80)

    train_dataset = SurgicalToolClassificationDataset(
        frames=train_frames,
        frame_path_to_label=frame_path_to_label,
        num_tools=num_tools,
        prompt_text=prompt,
        processor=processor,
    )

    val_dataset = SurgicalToolClassificationDataset(
        frames=val_frames,
        frame_path_to_label=frame_path_to_label,
        num_tools=num_tools,
        prompt_text=prompt,
        processor=processor,
    )

    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Val dataset: {len(val_dataset)} examples")

    # Load base model
    logger.info(f"\nLoading base model from {MODEL_PATH}")

    # Use bfloat16 - float16 causes NaN in Gemma hidden states!
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        str(MODEL_PATH), torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    log_gpu_memory("After loading base model")

    # Disable KV cache to save GPU memory
    if getattr(base_model.config, "use_cache", False):
        logger.info("Disabling KV cache for training")
        base_model.config.use_cache = False

    # Enable gradient checkpointing to save GPU memory (non-reentrant for DDP compatibility)
    logger.info("Enabling gradient checkpointing (non-reentrant)")
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Apply LoRA to base model
    logger.info("[DEBUG] Applying LoRA to base model...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        # Include both language model and vision encoder attention layers
        # q_proj, k_proj, v_proj match in both LM and vision encoder
        # o_proj for LM, out_proj for vision encoder
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    # Log trainable modules to verify vision encoder is included
    trainable_modules = [
        name for name, param in base_model.named_parameters() if param.requires_grad
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

    logger.info("[DEBUG] LoRA applied successfully")
    log_gpu_memory("After applying LoRA")

    # Wrap with classifier head
    # Gemma-3 multimodal has text_config.hidden_size
    hidden_size = base_model.config.text_config.hidden_size
    logger.info(
        f"[DEBUG] Creating classifier with hidden_size={hidden_size}, num_tools={num_tools}"
    )
    model = GemmaToolClassifier(
        base_model=base_model, hidden_size=hidden_size, num_tools=num_tools
    )

    # Create collator
    collator = ToolClassificationCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        num_tools=num_tools,
    )

    # Calculate total steps for adaptive eval/plot frequency
    # With DDP, samples are split across GPUs, so we need to account for world_size
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    samples_per_gpu = len(train_dataset) // world_size
    steps_per_epoch = samples_per_gpu // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    # Eval at least once, at most every 50 steps, aim for ~5 evals
    eval_every = max(1, min(50, max(1, total_steps // 5)))
    # Also set logging_steps to be frequent enough
    log_every = max(1, total_steps // 10)
    logger.info(f"World size: {world_size}, samples/GPU: {samples_per_gpu}")
    logger.info(
        f"Total training steps: ~{total_steps}, eval every {eval_every} steps, log every {log_every} steps"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        # float16 causes NaN in Gemma hidden states.
        bf16=True,
        logging_steps=1,
        # Built-in eval would score the full validation set on every check.
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        # Vision-encoder LoRA leaves out_proj unused on some steps.
        ddp_find_unused_parameters=True,
    )

    # Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    class AccuracyEvaluationCallback(TrainerCallback):
        """Callback to compute train+val accuracy and save plots during training."""

        def __init__(
            self,
            output_dir: Path,
            train_frames: List[Path],
            val_frames: List[Path],
            processor,
            frame_path_to_label: Dict[str, torch.Tensor],
            valid_tools: List[str],
            prompt_text: str,
            num_samples: int = 100,
            eval_every_n_steps: int = 10,
        ):
            self.output_dir = output_dir
            self.train_frames = train_frames
            self.val_frames = val_frames
            self.processor = processor
            self.frame_path_to_label = frame_path_to_label
            self.valid_tools = valid_tools
            self.prompt_text = prompt_text
            self.num_samples = num_samples
            self.eval_every_n_steps = eval_every_n_steps
            self.last_eval_step = -1

            # History for train and val accuracy/jaccard/loss
            self.train_accuracy_history = []
            self.val_accuracy_history = []
            self.train_jaccard_history = []
            self.val_jaccard_history = []
            self.val_loss_history = []

            # Sample fixed indices for consistent evaluation
            random.seed(RANDOM_SEED)
            self.train_sample_indices = random.sample(
                range(len(train_frames)), min(num_samples, len(train_frames))
            )
            self.val_sample_indices = random.sample(
                range(len(val_frames)), min(num_samples, len(val_frames))
            )
            self.train_sample_frames = [
                train_frames[i] for i in self.train_sample_indices
            ]
            self.val_sample_frames = [val_frames[i] for i in self.val_sample_indices]

            logger.info(
                f"AccuracyCallback: will evaluate on {len(self.train_sample_frames)} train "
                f"and {len(self.val_sample_frames)} val samples every {eval_every_n_steps} steps"
            )

        def _compute_accuracy_on_samples(
            self, model, frames: List[Path], desc: str, threshold: float = 0.5
        ) -> Tuple[float, float]:
            """Run forward pass on samples and compute exact match + Jaccard."""
            model.eval()
            device = next(model.parameters()).device

            exact_matches = 0
            jaccard_sum = 0.0
            total = 0

            with torch.no_grad():
                for frame_path in tqdm(frames, desc=desc, leave=False):
                    # Get ground truth from dataset labels (key is frame path string)
                    gt_label = self.frame_path_to_label.get(
                        str(frame_path), torch.zeros(len(self.valid_tools))
                    )
                    gt_indices = (gt_label > 0.5).nonzero(as_tuple=True)[0].tolist()
                    gt_set = set(self.valid_tools[i] for i in gt_indices)

                    # Prepare input
                    image = Image.open(frame_path).convert("RGB")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": self.prompt_text},
                            ],
                        }
                    ]
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    fwd_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "pixel_values": inputs["pixel_values"],
                    }
                    if "token_type_ids" in inputs:
                        fwd_kwargs["token_type_ids"] = inputs["token_type_ids"]
                    out = model(**fwd_kwargs)
                    probs = torch.sigmoid(out["logits"]).squeeze(0)

                    pred_indices = (
                        (probs > threshold).nonzero(as_tuple=True)[0].tolist()
                    )
                    pred_set = set(self.valid_tools[i] for i in pred_indices)

                    if pred_set == gt_set:
                        exact_matches += 1

                    union = pred_set | gt_set
                    jaccard = len(pred_set & gt_set) / len(union) if union else 1.0
                    jaccard_sum += jaccard
                    total += 1

            model.train()
            exact_acc = (exact_matches / total * 100) if total > 0 else 0.0
            jaccard_acc = (jaccard_sum / total * 100) if total > 0 else 0.0
            return exact_acc, jaccard_acc

        def _compute_val_loss_on_samples(
            self, model, frames: List[Path], threshold: float = 0.5
        ) -> float:
            """Compute average validation loss (BCE) on sampled frames."""
            model.eval()
            device = next(model.parameters()).device

            total_loss = 0.0
            count = 0

            with torch.no_grad():
                for frame_path in frames:
                    gt_label = self.frame_path_to_label.get(
                        str(frame_path), torch.zeros(len(self.valid_tools))
                    ).to(device)

                    image = Image.open(frame_path).convert("RGB")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": self.prompt_text},
                            ],
                        }
                    ]
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    fwd_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                        "pixel_values": inputs["pixel_values"],
                    }
                    if "token_type_ids" in inputs:
                        fwd_kwargs["token_type_ids"] = inputs["token_type_ids"]
                    out = model(**fwd_kwargs)
                    logits = out["logits"].squeeze(0)

                    loss = F.binary_cross_entropy_with_logits(logits, gt_label)
                    total_loss += loss.item()
                    count += 1

            model.train()
            return total_loss / count if count > 0 else 0.0

        def on_step_end(self, args, state, control, model=None, **kwargs):
            """Called after each training step - evaluate accuracy periodically."""
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
            self.val_accuracy_history.append(
                {"step": current_step, "accuracy": val_exact}
            )
            self.train_jaccard_history.append(
                {"step": current_step, "jaccard": train_jaccard}
            )
            self.val_jaccard_history.append(
                {"step": current_step, "jaccard": val_jaccard}
            )

            logger.info(
                f"[Step {current_step}] Val Loss={val_loss:.4f} | "
                f"Train: Exact={train_exact:.2f}%, Jaccard={train_jaccard:.2f}% | "
                f"Val: Exact={val_exact:.2f}%, Jaccard={val_jaccard:.2f}%"
            )

            self._save_plots(state)

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called when trainer logs metrics - update loss plot."""
            if not is_main_process():
                return
            self._save_plots(state)

        def _save_plots(self, state):
            """Save loss and accuracy plots."""
            try:
                log_history = state.log_history
                current_step = state.global_step

                # Extract train logs (loss) - divide by GRADIENT_ACCUMULATION_STEPS
                train_logs = [
                    entry
                    for entry in log_history
                    if "loss" in entry and "eval_loss" not in entry
                ]
                train_loss_points = [
                    {
                        "step": entry.get("step", idx),
                        "loss": entry["loss"] / GRADIENT_ACCUMULATION_STEPS,
                    }
                    for idx, entry in enumerate(train_logs)
                    if "loss" in entry
                ]

                # Plot: Loss, Accuracy, Jaccard in 1x3 layout
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

                # Exact Match Accuracy plot (train + val)
                ax2 = axes[1]
                if self.train_accuracy_history:
                    steps = [p["step"] for p in self.train_accuracy_history]
                    accs = [p["accuracy"] for p in self.train_accuracy_history]
                    ax2.plot(
                        steps,
                        accs,
                        linewidth=1.5,
                        color="#1f77b4",
                        marker="s",
                        markersize=4,
                        label="Train",
                    )
                if self.val_accuracy_history:
                    steps = [p["step"] for p in self.val_accuracy_history]
                    accs = [p["accuracy"] for p in self.val_accuracy_history]
                    ax2.plot(
                        steps,
                        accs,
                        linewidth=1.5,
                        color="#ff7f0e",
                        linestyle="--",
                        marker="o",
                        markersize=4,
                        label="Val",
                    )
                ax2.set_xlabel("Step", fontsize=11)
                ax2.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
                ax2.set_title("Exact Match Accuracy", fontsize=12)
                ax2.set_ylim(0, 100)
                ax2.legend()
                ax2.grid(alpha=0.3)

                # Jaccard plot (train + val)
                ax3 = axes[2]
                if self.train_jaccard_history:
                    steps = [p["step"] for p in self.train_jaccard_history]
                    jacs = [p["jaccard"] for p in self.train_jaccard_history]
                    ax3.plot(
                        steps,
                        jacs,
                        linewidth=1.5,
                        color="#1f77b4",
                        marker="s",
                        markersize=4,
                        label="Train",
                    )
                if self.val_jaccard_history:
                    steps = [p["step"] for p in self.val_jaccard_history]
                    jacs = [p["jaccard"] for p in self.val_jaccard_history]
                    ax3.plot(
                        steps,
                        jacs,
                        linewidth=1.5,
                        color="#ff7f0e",
                        linestyle="--",
                        marker="o",
                        markersize=4,
                        label="Val",
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
                plt.savefig(
                    str(plot_path_png), dpi=150, bbox_inches="tight", facecolor="white"
                )
                plt.savefig(
                    str(plot_path_svg),
                    format="svg",
                    bbox_inches="tight",
                    facecolor="white",
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
                    step_val_loss_df.to_csv(
                        self.output_dir / "val_loss_data.csv", index=False
                    )
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

            except Exception as e:
                logger.warning(f"Could not create plots: {e}")
                logger.warning(
                    f"[DEBUG] Plot error traceback:\n{traceback.format_exc()}"
                )

    accuracy_callback = AccuracyEvaluationCallback(
        output_dir=OUTPUT_DIR,
        train_frames=train_frames,
        val_frames=val_frames,
        processor=processor,
        frame_path_to_label=frame_path_to_label,
        valid_tools=valid_tools,
        prompt_text=prompt,
        num_samples=NUM_ACCURACY_SAMPLES,
        eval_every_n_steps=eval_every,
    )

    logger.info("[DEBUG] Creating Trainer...")
    log_gpu_memory("Before Trainer creation")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[accuracy_callback],
        compute_metrics=compute_metrics,
    )

    logger.info("[DEBUG] Trainer created successfully")
    log_gpu_memory("After Trainer creation")

    # Train
    logger.info("[DEBUG] Starting trainer.train()...")
    try:
        trainer.train()
        logger.info("[DEBUG] trainer.train() completed successfully")
    except Exception as e:
        logger.error(f"[DEBUG] trainer.train() FAILED with error: {e}")
        log_gpu_memory("At training failure")
        logger.error(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
        raise

    log_gpu_memory("After training")

    # Run FULL validation evaluation at the end of training (only on main process)
    if is_main_process():
        logger.info("\n" + "=" * 80)
        logger.info("FINAL FULL VALIDATION EVALUATION (20k frames)")
        logger.info("=" * 80)

        model.eval()
        device = next(model.parameters()).device

        exact_matches = 0
        jaccard_sum = 0.0
        total = 0
        threshold = 0.5

        # Collect predictions and ground truths for ROC-AUC
        all_probs = []
        all_gt_labels = []

        with torch.no_grad():
            for frame_path in tqdm(val_frames, desc="Final validation"):
                gt_label = frame_path_to_label.get(
                    str(frame_path), torch.zeros(num_tools)
                )
                gt_indices = (gt_label > 0.5).nonzero(as_tuple=True)[0].tolist()
                gt_set = set(valid_tools[i] for i in gt_indices)

                image = Image.open(frame_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                fwd_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "pixel_values": inputs["pixel_values"],
                }
                if "token_type_ids" in inputs:
                    fwd_kwargs["token_type_ids"] = inputs["token_type_ids"]
                out = model(**fwd_kwargs)
                probs = torch.sigmoid(out["logits"]).squeeze(0).cpu()

                all_probs.append(probs.numpy())
                all_gt_labels.append(gt_label.numpy())

                pred_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
                pred_set = set(valid_tools[i] for i in pred_indices)

                if pred_set == gt_set:
                    exact_matches += 1

                if len(pred_set | gt_set) > 0:
                    jaccard = len(pred_set & gt_set) / len(pred_set | gt_set)
                else:
                    jaccard = 1.0
                jaccard_sum += jaccard
                total += 1

        final_exact_acc = 100.0 * exact_matches / total if total > 0 else 0.0
        final_jaccard = 100.0 * jaccard_sum / total if total > 0 else 0.0

        # Both arrays have shape (N, num_tools).
        all_probs_np = np.array(all_probs)
        all_gt_np = np.array(all_gt_labels)

        per_tool_auc = {}
        per_tool_auprc = {}
        valid_aucs = []
        valid_auprcs = []
        for i, tool in enumerate(valid_tools):
            gt_col = all_gt_np[:, i]
            prob_col = all_probs_np[:, i]
            # ROC-AUC and AUPRC require both classes to be present
            if len(np.unique(gt_col)) > 1:
                auc = roc_auc_score(gt_col, prob_col)
                auprc = average_precision_score(gt_col, prob_col)
                per_tool_auc[tool] = auc
                per_tool_auprc[tool] = auprc
                valid_aucs.append(auc)
                valid_auprcs.append(auprc)
            else:
                per_tool_auc[tool] = None
                per_tool_auprc[tool] = None

        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        macro_auprc = np.mean(valid_auprcs) if valid_auprcs else 0.0

        logger.info(f"\nFinal Validation Results (FULL {len(val_frames)} frames):")
        logger.info(f"  Total processed: {total}")
        logger.info(f"  Exact matches: {exact_matches}")
        logger.info(f"  Exact match accuracy: {final_exact_acc:.2f}%")
        logger.info(f"  Jaccard similarity: {final_jaccard:.2f}%")
        logger.info(f"  Macro ROC-AUC: {macro_auc:.4f}")
        logger.info(f"  Macro AUPRC: {macro_auprc:.4f}")

        logger.info("\nPer-Tool Metrics (sorted by AUPRC):")
        sorted_tools = sorted(
            valid_tools,
            key=lambda t: per_tool_auprc[t] if per_tool_auprc[t] is not None else -1,
            reverse=True,
        )
        for tool in sorted_tools:
            auc = per_tool_auc[tool]
            auprc = per_tool_auprc[tool]
            if auc is not None:
                logger.info(f"  {tool}: ROC-AUC={auc:.4f}, AUPRC={auprc:.4f}")
            else:
                logger.info(f"  {tool}: N/A (only one class in val set)")

        # Save final results to JSON
        final_results = {
            "total_processed": total,
            "exact_matches": exact_matches,
            "exact_match_accuracy": final_exact_acc,
            "jaccard_similarity": final_jaccard,
            "macro_roc_auc": macro_auc,
            "macro_auprc": macro_auprc,
            "per_tool_roc_auc": {
                k: v for k, v in per_tool_auc.items() if v is not None
            },
            "per_tool_auprc": {
                k: v for k, v in per_tool_auprc.items() if v is not None
            },
        }
        with (OUTPUT_DIR / "final_validation_results.json").open("w") as f:
            json.dump(final_results, f, indent=2)

    # Save final model manually (only rank 0) - avoid trainer.save_model() which corrupts with DDP
    if is_main_process():
        # Save to final_classifier (overwrites previous)
        final_model_path = OUTPUT_DIR / "final_classifier"
        if final_model_path.exists():
            shutil.rmtree(final_model_path)
        final_model_path.mkdir(parents=True, exist_ok=True)

        # Also save timestamped copy for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = OUTPUT_DIR / f"checkpoint_{timestamp}"
        timestamped_path.mkdir(parents=True, exist_ok=True)

        # Save only the classifier head weights
        classifier_state = model.classifier.state_dict()
        torch.save(classifier_state, final_model_path / "classifier.pt")
        torch.save(classifier_state, timestamped_path / "classifier.pt")

        # Save LoRA adapters
        model.base_model.save_pretrained(final_model_path / "lora_adapters")
        model.base_model.save_pretrained(timestamped_path / "lora_adapters")

        # Save comprehensive model config for robust loading
        model_config = {
            "base_model_path": str(MODEL_PATH),
            "hidden_size": hidden_size,
            "num_tools": num_tools,
            "valid_tools": valid_tools,
            "tool_to_idx": tool_to_idx,
            "lora_config": {
                "r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"],
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
            "data_config": {
                "labels_root": str(LABELS_ROOT),
                "dataset_csv": str(DATASET_CSV),
                "yolo_mapping_file": str(YOLO_MAPPING_FILE),
            },
            "saved_at": datetime.now().isoformat(),
        }

        with (final_model_path / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)
        with (timestamped_path / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"\nFinal model saved to: {final_model_path}")
        logger.info("  - classifier.pt")
        logger.info("  - lora_adapters/")
        logger.info("  - model_config.json")
        logger.info(f"\nTimestamped copy saved to: {timestamped_path}")

        # Final plots already generated by callback during training
        logger.info("\nFinal training plots saved to:")
        logger.info(f"  - {OUTPUT_DIR / 'training_loss.png'}")
        logger.info(f"  - {OUTPUT_DIR / 'training_loss.svg'}")

        # Evaluate on validation set
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON VALIDATION SET")
        logger.info("=" * 80)

        model.eval()

        val_predictions_file = OUTPUT_DIR / "val_predictions_after_training.jsonl"
        if val_predictions_file.exists():
            val_predictions_file.unlink()

        val_exact_acc, val_jaccard_acc, val_predictions = (
            evaluate_accuracy_from_predictions(
                model,
                processor,
                val_frames,
                yolo_mapping,
                prompt,
                valid_tools,
                val_predictions_file,
                "Validation inference",
            )
        )

        if val_frames:
            debug_frame = val_frames[0]
            debug_out_path = OUTPUT_DIR / "debug_single_frame.json"
            model.eval()
            gt = get_ground_truth_tools(debug_frame, yolo_mapping)
            pred = predict_tools(model, processor, debug_frame, prompt, valid_tools)
            with debug_out_path.open("w") as f:
                json.dump(
                    {
                        "frame_path": str(debug_frame),
                        "ground_truth": sorted(gt),
                        "predicted": sorted(pred),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved debug frame to {debug_out_path}")

        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(
            f"Validation - Exact Match: {val_exact_acc:.2f}%, Jaccard: {val_jaccard_acc:.2f}%"
        )

        # Save results
        results = {
            "val_exact_accuracy": val_exact_acc,
            "val_jaccard_accuracy": val_jaccard_acc,
            "num_val_samples": len(val_predictions),
            "num_tools": num_tools,
        }

        results_file = OUTPUT_DIR / "results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")

    logger.info("\nMulti-label classification complete!")


if __name__ == "__main__":
    main()
