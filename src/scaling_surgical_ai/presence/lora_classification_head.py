"""Train a Gemma 3 27B LoRA classification head on PitVis-2023.

Produces the PitVis-2023 fine-tuned multi-label tool-presence result for
"A Comparative Study in Surgical AI: Potential and Limitations of Data,
Compute, and Scaling". PitVis-2023 has 18 instrument classes. SurgVU reuses
this module by overriding ``PITVIS_ROOT``, ``DATASET_CSV``, ``OUTPUT_DIR``,
``INSTRUMENT_NAMES``, and ``MODEL_PATH`` before calling ``main``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
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
PITVIS_ROOT = paths.pitvis_root()
PITVIS_OLD_ROOT = paths.pitvis_recorded_root()
MODEL_PATH = paths.hf_hub_snapshot(
    "google/gemma-3-27b-it",
    "005ad3404e59d6023443cb575daa05336842228a",
)
DATASET_CSV = PITVIS_ROOT / "dataset.csv"

INSTRUMENT_NAMES = [
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
]

LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = paths.experiment_output_dir(
    "pitvis_lora_classification_head",
    SLURM_JOB_ID,
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
                f"[GPU_MEM] {prefix} GPU {i}: "
                f"{allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, "
                f"{total:.2f}GB total",
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


def is_main_process() -> bool:
    """Return True on global rank 0 (or single-process runs)."""
    return int(os.environ.get("RANK", "0")) == 0


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load PitVis-2023 dataset from CSV.

    Returns:
        frame_label_train_df: DataFrame with train split rows
        frame_label_val_df: DataFrame with val split rows
        frame_to_video: Dict mapping frame_path -> video_id
    """
    logger.info(f"Loading PitVis-2023 dataset from {DATASET_CSV}")
    frame_label_df = pd.read_csv(DATASET_CSV)

    frame_label_df["frame_path"] = frame_label_df["frame_path"].str.replace(
        str(PITVIS_OLD_ROOT),
        str(PITVIS_ROOT),
        regex=False,
    )

    frame_stride = int(os.environ.get("FRAME_STRIDE", "1"))
    if frame_stride > 1:
        before = len(frame_label_df)
        frame_label_df = frame_label_df[
            frame_label_df["frame_id"] % frame_stride == 0
        ].reset_index(drop=True)
        logger.info(
            f"FRAME_STRIDE={frame_stride}: filtered "
            f"{before} -> {len(frame_label_df)} frames"
        )

    frame_label_train_df = frame_label_df[frame_label_df["split"] == "train"].copy()
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"].copy()

    frame_to_video = dict(
        zip(
            frame_label_df["frame_path"],
            frame_label_df["video_id"].astype(str),
        )
    )

    logger.info(
        f"Loaded {len(frame_label_train_df)} train frames, {len(frame_label_val_df)} val frames"
    )
    return frame_label_train_df, frame_label_val_df, frame_to_video


def load_labels_from_dataset(
    frame_label_df: pd.DataFrame,
    valid_tools: List[str],
) -> Dict[str, torch.Tensor]:
    """Load labels from dataset CSV.

    Returns mapping from frame_path -> multi-hot label tensor.
    """
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    num_tools = len(valid_tools)

    frame_to_label = {}
    for _, row in frame_label_df.iterrows():
        frame_path = row["frame_path"]
        tools = json.loads(row["truth_label"])

        label = torch.zeros(num_tools, dtype=torch.float32)
        for tool in tools:
            if tool in tool_to_idx:
                label[tool_to_idx[tool]] = 1.0

        frame_to_label[frame_path] = label

    return frame_to_label


def get_ground_truth_tools_from_label(
    frame_path: Path,
    frame_path_to_label: Dict[str, torch.Tensor],
    valid_tools: List[str],
) -> List[str]:
    """Get ground truth tool names for a frame from label tensor."""
    label = frame_path_to_label.get(str(frame_path), torch.zeros(len(valid_tools)))
    indices = (label > 0.5).nonzero(as_tuple=True)[0].tolist()
    return sorted([valid_tools[i] for i in indices])


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
        """Store frames, labels, and the processor used to build inputs."""
        self.frames = frames
        self.frame_path_to_label = frame_path_to_label
        self.num_tools = num_tools
        self.prompt_text = prompt_text
        self.processor = processor

    def __len__(self):
        """Return the number of frames."""
        return len(self.frames)

    def __getitem__(self, idx):
        """Return tokenized inputs and the multi-hot label for one frame."""
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

        label = self.frame_path_to_label.get(
            str(frame_path),
            torch.zeros(self.num_tools, dtype=torch.float32),
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": label,
        }


class ToolClassificationCollator:
    """Collator for variable-length sequences."""

    def __init__(self, pad_token_id: int, num_tools: int):
        """Store the pad token and the number of tool classes."""
        self.pad_token_id = pad_token_id
        self.num_tools = num_tools

    def __call__(self, examples):
        """Pad a batch of variable-length examples into tensors."""
        input_ids = [ex["input_ids"] for ex in examples]
        attention_masks = [ex["attention_mask"] for ex in examples]
        labels = torch.stack([ex["labels"] for ex in examples], dim=0)

        input_ids_padded = pad_sequence_right(
            input_ids, padding_value=self.pad_token_id
        )
        attention_mask_padded = pad_sequence_right(attention_masks, padding_value=0)
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples], dim=0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "pixel_values": pixel_values,
            "labels": labels,
        }


class GemmaToolClassifier(nn.Module):
    """Gemma-3 with classification head for multi-label tool detection."""

    def __init__(self, base_model, hidden_size: int, num_tools: int):
        """Attach a linear classification head to the base model."""
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_tools)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """Pool the last hidden state and return logits and optional BCE loss."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[-1]

        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )
        probs = torch.sigmoid(out["logits"]).squeeze(0)

    pred_indices = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    return sorted([valid_tools[i] for i in pred_indices])


def evaluate_accuracy_from_predictions(
    model: GemmaToolClassifier,
    processor,
    frames: List[Path],
    frame_path_to_label: Dict[str, torch.Tensor],
    prompt: str,
    valid_tools: List[str],
    predictions_file: Path,
    desc: str,
) -> Tuple[float, float, Dict[str, Dict]]:
    """Evaluate accuracy on a set of frames.

    Returns:
        (exact_match_accuracy, jaccard_accuracy, predictions_dict)
    """
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
            pred_set = set(cached["predicted"])
            gt_set = set(cached["ground_truth"])
            union = pred_set | gt_set
            jaccard = cached.get(
                "jaccard",
                len(pred_set & gt_set) / len(union) if union else 1.0,
            )
            predictions[canonical_path] = {
                "ground_truth": cached["ground_truth"],
                "predicted": cached["predicted"],
                "is_correct": cached["is_correct"],
                "jaccard": jaccard,
            }
        else:
            ground_truth = get_ground_truth_tools_from_label(
                frame_path, frame_path_to_label, valid_tools
            )
            frames_to_process.append((frame_path, canonical_path, ground_truth))

    logger.info(f"Using {len(predictions)} cached predictions")
    logger.info(f"Running inference on {len(frames_to_process)} frames")

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

            union = pred_set | gt_set
            jaccard = len(pred_set & gt_set) / len(union) if union else 1.0

            predictions[canonical_path] = {
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "jaccard": jaccard,
            }

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

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)

    matches = np.all(preds == labels, axis=1)
    exact_accuracy = matches.mean()

    jaccard_scores = []
    for pred_row, label_row in zip(preds, labels):
        pred_set = set(np.where(pred_row == 1)[0])
        label_set = set(np.where(label_row == 1)[0])
        union = pred_set | label_set
        jaccard = len(pred_set & label_set) / len(union) if union else 1.0
        jaccard_scores.append(jaccard)
    jaccard_accuracy = np.mean(jaccard_scores)

    return {
        "accuracy": exact_accuracy * 100.0,
        "jaccard": jaccard_accuracy * 100.0,
    }


def main() -> None:
    """Run multi-label classification training."""
    parser = argparse.ArgumentParser(
        description="Multi-label classification for Gemma-3 27B on PitVis-2023"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Load existing checkpoint for evaluation (path to checkpoint dir)",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    logger.info("Starting PitVis-2023 Multi-label classification with Gemma-3 27B")
    logger.info(f"Load checkpoint: {args.load_checkpoint}")
    logger.info(f"LOCAL_RANK: {local_rank}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_tools = INSTRUMENT_NAMES
    num_tools = len(valid_tools)
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    logger.info(f"PitVis-2023 instrument classes ({num_tools}): {valid_tools}")

    frame_label_train_df, frame_label_val_df, frame_to_video = load_dataset()

    train_frames = [Path(p) for p in frame_label_train_df["frame_path"].tolist()]
    val_frames = [Path(p) for p in frame_label_val_df["frame_path"].tolist()]

    train_videos = set(frame_label_train_df["video_id"].astype(str).unique())
    val_videos = set(frame_label_val_df["video_id"].astype(str).unique())
    overlap = train_videos & val_videos

    if overlap:
        raise RuntimeError(
            f"VIDEO LEAKAGE DETECTED! {len(overlap)} videos in both splits"
        )

    logger.info(
        f"No video overlap between train ({len(train_videos)}) and val ({len(val_videos)})"
    )

    logger.info("\n" + "=" * 80)
    logger.info("ENVIRONMENT CHECK")
    logger.info("=" * 80)
    logger.info(f"MODEL_PATH   = {MODEL_PATH}")
    logger.info(f"DATASET_CSV  = {DATASET_CSV}")
    logger.info(f"num_tools    = {num_tools}")
    logger.info(f"valid_tools  = {valid_tools}")
    logger.info("=" * 80)

    prompt = create_prompt(valid_tools)

    logger.info("Loading labels from dataset...")
    frame_label_df = pd.concat(
        [frame_label_train_df, frame_label_val_df], ignore_index=True
    )
    frame_path_to_label = load_labels_from_dataset(frame_label_df, valid_tools)
    logger.info(f"Loaded labels for {len(frame_path_to_label)} frames")

    sample_labels = list(frame_path_to_label.values())[:100]
    avg_tools = sum(label.sum().item() for label in sample_labels) / len(sample_labels)
    logger.info(f"Average tools per frame (sample): {avg_tools:.2f}")

    logger.info(f"Loading processor from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )

    logger.info("=" * 80)
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

    logger.info(f"Loading base model from {MODEL_PATH}")

    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model = base_model.to(f"cuda:{local_rank}")
    log_gpu_memory("After loading base model")

    if getattr(base_model.config, "use_cache", False):
        logger.info("Disabling KV cache for training")
        base_model.config.use_cache = False

    logger.info("Enabling gradient checkpointing (non-reentrant)")
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    logger.info("Applying LoRA to base model...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

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

    log_gpu_memory("After applying LoRA")

    hidden_size = base_model.config.text_config.hidden_size
    logger.info(
        f"Creating classifier with hidden_size={hidden_size}, num_tools={num_tools}"
    )
    model = GemmaToolClassifier(
        base_model=base_model,
        hidden_size=hidden_size,
        num_tools=num_tools,
    )

    collator = ToolClassificationCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        num_tools=num_tools,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    samples_per_gpu = len(train_dataset) // world_size
    steps_per_epoch = samples_per_gpu // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    eval_every = max(1, min(50, max(1, total_steps // 5)))
    log_every = max(1, total_steps // 10)
    logger.info(f"World size: {world_size}, samples/GPU: {samples_per_gpu}")
    logger.info(
        f"Total training steps: ~{total_steps}, "
        f"eval every {eval_every} steps, log every {log_every} steps"
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
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
    )

    logger.info("=" * 80)
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
            """Store callback settings and sample frames for periodic evaluation."""
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
            self.train_sample_frames = [
                train_frames[i] for i in self.train_sample_indices
            ]
            self.val_sample_frames = [val_frames[i] for i in self.val_sample_indices]

            logger.info(
                f"AccuracyCallback: will evaluate on {len(self.train_sample_frames)} train "
                f"and {len(self.val_sample_frames)} val samples every {eval_every_n_steps} steps"
            )

        def _compute_accuracy_on_samples(
            self,
            model,
            frames: List[Path],
            desc: str,
            threshold: float = 0.5,
        ) -> Tuple[float, float]:
            """Run forward pass on samples and compute exact match + Jaccard."""
            model.eval()
            device = next(model.parameters()).device

            exact_matches = 0
            jaccard_sum = 0.0
            total = 0

            with torch.no_grad():
                for frame_path in tqdm(frames, desc=desc, leave=False):
                    gt_label = self.frame_path_to_label.get(
                        str(frame_path), torch.zeros(len(self.valid_tools))
                    )
                    gt_indices = (gt_label > 0.5).nonzero(as_tuple=True)[0].tolist()
                    gt_set = set(self.valid_tools[i] for i in gt_indices)

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

                    out = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                    )
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
            self,
            model,
            frames: List[Path],
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

                    out = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=inputs["pixel_values"],
                    )
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

            is_first_step = current_step == 1 and self.last_eval_step == -1
            is_eval_step = current_step - self.last_eval_step >= self.eval_every_n_steps

            if not (is_first_step or is_eval_step):
                return

            self.last_eval_step = current_step

            logger.info(
                f"\n[Step {current_step}] Computing val loss and accuracy "
                f"on {self.num_samples} samples..."
            )

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
            log_history = state.log_history
            current_step = state.global_step

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

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

            if self.val_loss_history:
                val_steps = [p["step"] for p in self.val_loss_history]
                val_losses = [p["loss"] for p in self.val_loss_history]
                ax1.plot(
                    val_steps,
                    val_losses,
                    linewidth=1.5,
                    color="#ff7f0e",
                    linestyle="--",
                    marker="o",
                    markersize=4,
                    label="Val Loss (100 samples)",
                )

            ax1.set_xlabel("Step", fontsize=11)
            ax1.set_ylabel("Loss", fontsize=11)
            ax1.set_title("Training Loss", fontsize=12)
            ax1.legend()
            ax1.grid(alpha=0.3)

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
                str(plot_path_svg), format="svg", bbox_inches="tight", facecolor="white"
            )
            plt.close()
            logger.info(
                f"Saved plots to {plot_path_png} and {plot_path_svg} (step {current_step})"
            )

            if train_loss_points:
                pd.DataFrame(train_loss_points).to_csv(
                    self.output_dir / "train_loss_data.csv",
                    index=False,
                )
            if self.val_loss_history:
                pd.DataFrame(self.val_loss_history).to_csv(
                    self.output_dir / "val_loss_data.csv",
                    index=False,
                )
            if self.train_accuracy_history:
                pd.DataFrame(self.train_accuracy_history).to_csv(
                    self.output_dir / "train_accuracy_data.csv",
                    index=False,
                )
            if self.val_accuracy_history:
                pd.DataFrame(self.val_accuracy_history).to_csv(
                    self.output_dir / "val_accuracy_data.csv",
                    index=False,
                )
            if self.train_jaccard_history:
                pd.DataFrame(self.train_jaccard_history).to_csv(
                    self.output_dir / "train_jaccard_data.csv",
                    index=False,
                )
            if self.val_jaccard_history:
                pd.DataFrame(self.val_jaccard_history).to_csv(
                    self.output_dir / "val_jaccard_data.csv",
                    index=False,
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

    logger.info("Creating Trainer...")
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

    logger.info("Trainer created successfully")
    log_gpu_memory("After Trainer creation")

    logger.info("Starting trainer.train()...")
    trainer.train()
    logger.info("trainer.train() completed successfully")

    log_gpu_memory("After training")

    if is_main_process():
        logger.info("\n" + "=" * 80)
        logger.info(f"FINAL FULL VALIDATION EVALUATION ({len(val_frames)} frames)")
        logger.info("=" * 80)

        model.eval()
        device = next(model.parameters()).device

        exact_matches = 0
        jaccard_sum = 0.0
        total = 0
        threshold = 0.5

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

                out = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                )
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

        all_probs_np = np.array(all_probs)
        all_gt_np = np.array(all_gt_labels)

        per_tool_auc = {}
        per_tool_auprc = {}
        valid_aucs = []
        valid_auprcs = []
        for i, tool in enumerate(valid_tools):
            gt_col = all_gt_np[:, i]
            prob_col = all_probs_np[:, i]
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

    if is_main_process():
        final_model_path = OUTPUT_DIR / "final_classifier"
        if final_model_path.exists():
            shutil.rmtree(final_model_path)
        final_model_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = OUTPUT_DIR / f"checkpoint_{timestamp}"
        timestamped_path.mkdir(parents=True, exist_ok=True)

        classifier_state = model.classifier.state_dict()
        torch.save(classifier_state, final_model_path / "classifier.pt")
        torch.save(classifier_state, timestamped_path / "classifier.pt")

        model.base_model.save_pretrained(final_model_path / "lora_adapters")
        model.base_model.save_pretrained(timestamped_path / "lora_adapters")

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
                "pitvis_root": str(PITVIS_ROOT),
                "dataset_csv": str(DATASET_CSV),
            },
            "saved_at": datetime.now().isoformat(),
        }

        with (final_model_path / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)
        with (timestamped_path / "model_config.json").open("w") as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"\nFinal model saved to: {final_model_path}")
        logger.info(f"Timestamped copy saved to: {timestamped_path}")

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
                frame_path_to_label,
                prompt,
                valid_tools,
                val_predictions_file,
                "Validation inference",
            )
        )

        logger.info("\n" + "=" * 80)
        logger.info("FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(
            f"Validation - Exact Match: {val_exact_acc:.2f}%, Jaccard: {val_jaccard_acc:.2f}%"
        )

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

    logger.info("Multi-label classification complete!")


if __name__ == "__main__":
    main()
