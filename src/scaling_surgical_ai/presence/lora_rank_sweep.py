"""Sweep LoRA rank for a Gemma 3 27B classification head on PitVis-2023.

Produces the PitVis-2023 LoRA-rank sweep of multi-label tool-presence
accuracy for "A Comparative Study in Surgical AI: Potential and Limitations
of Data, Compute, and Scaling". PitVis-2023 has 18 instrument classes.
SurgVU reuses this module by overriding ``PITVIS_ROOT``, ``DATASET_CSV``,
``INSTRUMENT_NAMES``, ``MODEL_PATH``, ``OUTPUT_DIR_FULL``,
``OUTPUT_DIR_DEBUG``, and ``OUTPUT_DIR`` before calling ``main``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Trainer,
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

DEBUG_MODE = False
DEBUG_MAX_STEPS = 2
DEBUG_EVAL_SAMPLES = 10

OUTPUT_DIR_FULL = paths.experiment_output_dir("pitvis_lora_rank_sweep")
OUTPUT_DIR_DEBUG = paths.experiment_output_dir("pitvis_lora_rank_sweep_debug")
OUTPUT_DIR = OUTPUT_DIR_DEBUG if DEBUG_MODE else OUTPUT_DIR_FULL

LORA_RANKS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DATASET_NAME = "PitVis-2023"
MAJORITY_BASELINE = 39.63

LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]

NUM_EPOCHS = 3
LEARNING_RATE = 5e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
BOOTSTRAP_ITERATIONS = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def log_gpu_memory(prefix: str = "") -> Dict[str, float]:
    """Log and return GPU memory usage on device 0."""
    mem_info = {}
    if not torch.cuda.is_available():
        return mem_info
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    mem_info["gpu_0_allocated_gb"] = allocated
    mem_info["gpu_0_reserved_gb"] = reserved
    mem_info["gpu_0_total_gb"] = total
    print(
        f"[GPU_MEM] {prefix} GPU 0: {allocated:.2f}GB allocated, "
        f"{reserved:.2f}GB reserved, {total:.2f}GB total",
        flush=True,
    )
    return mem_info


def get_peak_vram() -> float:
    """Get peak VRAM usage on device 0 in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(0) / 1024**3


def reset_peak_vram():
    """Reset peak VRAM tracking."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


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


def load_pitvis_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PitVis-2023 dataset from CSV.

    Returns:
        frame_label_train_df: DataFrame with train split
        frame_label_val_df: DataFrame with val split
    """
    logger.info(f"Loading {DATASET_NAME} dataset from {DATASET_CSV}")
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

    logger.info(
        f"Loaded {len(frame_label_train_df)} train frames, {len(frame_label_val_df)} val frames"
    )
    return frame_label_train_df, frame_label_val_df


def load_labels_from_dataset(
    frame_label_df: pd.DataFrame,
    valid_tools: List[str],
) -> Dict[str, torch.Tensor]:
    """Load labels from dataset CSV."""
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    filename_to_label = {}

    for _, row in frame_label_df.iterrows():
        frame_path = row["frame_path"]
        tools = json.loads(row["truth_label"])

        label = torch.zeros(len(valid_tools), dtype=torch.float32)
        for tool in tools:
            if tool in tool_to_idx:
                label[tool_to_idx[tool]] = 1.0

        filename_to_label[frame_path] = label

    return filename_to_label


def get_label_vector(
    frame_path: Path,
    frame_path_to_label: Dict[str, torch.Tensor],
    num_tools: int,
) -> torch.Tensor:
    """Get multi-hot label vector for a frame."""
    frame_path_str = str(frame_path)
    if frame_path_str in frame_path_to_label:
        return frame_path_to_label[frame_path_str]
    return torch.zeros(num_tools, dtype=torch.float32)


def is_main_process() -> bool:
    """Return True on global rank 0."""
    return int(os.environ.get("RANK", "0")) == 0


def get_world_size() -> int:
    """Return distributed world size, or 1 if non-distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Return distributed rank, or 0 if non-distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


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
        filename_to_label: Dict[str, torch.Tensor],
        num_tools: int,
        prompt_text: str,
        processor,
    ):
        """Store frames, labels, and the processor used to build inputs."""
        self.frames = frames
        self.filename_to_label = filename_to_label
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

        label = get_label_vector(frame_path, self.filename_to_label, self.num_tools)

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
        self.classifier = nn.Linear(hidden_size, num_tools, dtype=torch.bfloat16)

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


def bootstrap_ci(
    correct: List[int],
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for accuracy.

    Args:
        correct: List of 0/1 values indicating correct predictions.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        The 2.5 and 97.5 percentile accuracies, as percentages.
    """
    if len(correct) == 0:
        return 0.0, 0.0

    correct_arr = np.array(correct)
    n = len(correct_arr)

    bootstrap_accs = []
    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_acc = correct_arr[indices].mean() * 100
        bootstrap_accs.append(bootstrap_acc)

    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)

    return ci_lower, ci_upper


def compute_accuracy_with_ci(
    model: GemmaToolClassifier,
    processor,
    frames: List[Path],
    frame_path_to_label: Dict[str, torch.Tensor],
    prompt: str,
    valid_tools: List[str],
    desc: str,
) -> Tuple[float, float, float]:
    """Compute exact match accuracy with 95% bootstrap confidence interval.

    Inference is sharded across distributed ranks; per-frame correctness
    flags are all-gathered to every rank before computing accuracy and CI.

    Returns:
        Accuracy, lower CI, and upper CI, as percentages.
    """
    model.eval()

    world_size = get_world_size()
    rank = get_rank()

    rank_frames = frames[rank::world_size]
    rank_correct: List[int] = []

    with torch.no_grad():
        pbar = tqdm(
            rank_frames,
            desc=f"{desc} [rank {rank}]",
            leave=False,
            disable=(rank != 0),
        )
        for frame_path in pbar:
            gt_label = frame_path_to_label.get(
                str(frame_path), torch.zeros(len(valid_tools))
            )
            gt_indices = (gt_label > 0.5).nonzero(as_tuple=True)[0].tolist()
            gt_set = set(valid_tools[i] for i in gt_indices)

            predicted = predict_tools(model, processor, frame_path, prompt, valid_tools)
            pred_set = set(predicted)

            rank_correct.append(1 if pred_set == gt_set else 0)

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        # Pad each rank's tensor to max shard length so all_gather (which
        # requires equal sizes) can merge variable-length local results.
        rank_shard_lengths = [len(frames[r::world_size]) for r in range(world_size)]
        max_len = max(rank_shard_lengths)
        padding_value = -1
        padded = rank_correct + [padding_value] * (max_len - len(rank_correct))

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        local_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor)

        all_correct: List[int] = []
        for tensor in gathered:
            for value in tensor.tolist():
                if value != padding_value:
                    all_correct.append(value)
    else:
        all_correct = rank_correct

    n = len(all_correct)
    if n == 0:
        return 0.0, 0.0, 0.0

    accuracy = sum(all_correct) / n * 100
    ci_lower, ci_upper = bootstrap_ci(all_correct)

    return accuracy, ci_lower, ci_upper


def train_with_lora_rank(
    lora_r: int,
    train_frames: List[Path],
    val_frames: List[Path],
    filename_to_label: Dict[str, torch.Tensor],
    valid_tools: List[str],
    processor,
    prompt: str,
    output_subdir: Path,
) -> Dict:
    """Train model with specific LoRA rank and return results."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"TRAINING WITH LORA_R = {lora_r}")
    logger.info(f"{'=' * 80}")

    output_subdir.mkdir(parents=True, exist_ok=True)

    reset_peak_vram()
    start_time = time.time()

    num_tools = len(valid_tools)
    lora_alpha = lora_r * 2

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    base_model = None
    model = None
    trainer = None
    trainable_params = 0
    total_params = 0

    try:
        log_gpu_memory(f"[r={lora_r}] Before loading model")

        base_model = Gemma3ForConditionalGeneration.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model = base_model.to(f"cuda:{local_rank}")
        log_gpu_memory(f"[r={lora_r}] After loading base model")

        if getattr(base_model.config, "use_cache", False):
            base_model.config.use_cache = False

        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)

        trainable_params = sum(
            p.numel() for p in base_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in base_model.parameters())
        logger.info(
            f"LoRA r={lora_r}: trainable params: {trainable_params:,} / {total_params:,}"
        )

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

        log_gpu_memory(f"[r={lora_r}] After applying LoRA")

        hidden_size = base_model.config.text_config.hidden_size
        model = GemmaToolClassifier(
            base_model=base_model,
            hidden_size=hidden_size,
            num_tools=num_tools,
        )

        train_dataset = SurgicalToolClassificationDataset(
            frames=train_frames,
            filename_to_label=filename_to_label,
            num_tools=num_tools,
            prompt_text=prompt,
            processor=processor,
        )

        val_dataset = SurgicalToolClassificationDataset(
            frames=val_frames,
            filename_to_label=filename_to_label,
            num_tools=num_tools,
            prompt_text=prompt,
            processor=processor,
        )

        collator = ToolClassificationCollator(
            pad_token_id=processor.tokenizer.pad_token_id,
            num_tools=num_tools,
        )

        optim = "adafactor" if lora_r >= 256 else "adamw_torch"
        logger.info(f"[r={lora_r}] Using optimizer: {optim}, strategy: DDP")

        training_kwargs = {
            "output_dir": str(output_subdir / "checkpoints"),
            "num_train_epochs": NUM_EPOCHS,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "optim": optim,
            "bf16": True,
            "logging_steps": 1 if DEBUG_MODE else 10,
            "eval_strategy": "no" if DEBUG_MODE else "epoch",
            "save_strategy": "no" if (DEBUG_MODE or lora_r >= 1024) else "epoch",
            "save_total_limit": 1,
            "save_safetensors": False,
            "report_to": "none",
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
        }

        training_kwargs["ddp_find_unused_parameters"] = True

        if DEBUG_MODE:
            training_kwargs["max_steps"] = DEBUG_MAX_STEPS
            logger.info(
                f"[r={lora_r}] DEBUG MODE: Training for {DEBUG_MAX_STEPS} step(s)"
            )

        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        log_gpu_memory(f"[r={lora_r}] Before training")

        trainer.train()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"OOM with LoRA r={lora_r}: {e}")
            if trainer is not None:
                del trainer
            if model is not None:
                del model
            if base_model is not None:
                del base_model
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "lora_r": lora_r,
                "status": "OOM",
                "trainable_params": trainable_params,
                "total_params": total_params,
            }
        raise

    training_time = time.time() - start_time
    peak_vram = get_peak_vram()

    log_gpu_memory(f"[r={lora_r}] After training")

    if is_main_process():
        logger.info(
            f"[r={lora_r}] Saving LoRA adapter and classifier before inference..."
        )
        model.base_model.save_pretrained(output_subdir / "lora_adapters")
        classifier_state = model.classifier.state_dict()
        torch.save(classifier_state, output_subdir / "classifier.pt")

        config = {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }
        with (output_subdir / "config.json").open("w") as f:
            json.dump(config, f, indent=2)

        partial_result = {
            "lora_r": lora_r,
            "status": "training_complete",
            "trainable_params": trainable_params,
            "total_params": total_params,
            "training_time_hours": training_time / 3600,
            "peak_vram_gb": peak_vram,
        }
        with (output_subdir / "result_partial.json").open("w") as f:
            json.dump(partial_result, f, indent=2)
        logger.info(f"[r={lora_r}] Saved adapter, classifier, and partial result")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if DEBUG_MODE:
        eval_train_frames = train_frames[:DEBUG_EVAL_SAMPLES]
        eval_val_frames = val_frames[:DEBUG_EVAL_SAMPLES]
        logger.info(
            f"[r={lora_r}] DEBUG MODE: Evaluating on "
            f"{len(eval_train_frames)} train, {len(eval_val_frames)} val frames"
        )
    else:
        eval_train_frames = train_frames
        eval_val_frames = val_frames
        logger.info(
            f"[r={lora_r}] Computing final accuracies on FULL datasets with bootstrap CIs..."
        )
        logger.info(
            f"[r={lora_r}] Train frames: {len(eval_train_frames)}, "
            f"Val frames: {len(eval_val_frames)}"
        )

    train_acc, train_ci_lo, train_ci_hi = compute_accuracy_with_ci(
        model,
        processor,
        eval_train_frames,
        filename_to_label,
        prompt,
        valid_tools,
        f"Train acc (r={lora_r}, n={len(eval_train_frames)})",
    )

    val_acc, val_ci_lo, val_ci_hi = compute_accuracy_with_ci(
        model,
        processor,
        eval_val_frames,
        filename_to_label,
        prompt,
        valid_tools,
        f"Val acc (r={lora_r}, n={len(eval_val_frames)})",
    )

    logger.info(
        f"[r={lora_r}] Train: {train_acc:.2f}% [{train_ci_lo:.2f}, {train_ci_hi:.2f}]"
    )
    logger.info(f"[r={lora_r}] Val: {val_acc:.2f}% [{val_ci_lo:.2f}, {val_ci_hi:.2f}]")
    logger.info(
        f"[r={lora_r}] Training time: {training_time / 3600:.2f}h, "
        f"Peak VRAM: {peak_vram:.2f}GB"
    )

    result = {
        "lora_r": lora_r,
        "status": "success",
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_accuracy": train_acc,
        "train_ci_lower": train_ci_lo,
        "train_ci_upper": train_ci_hi,
        "val_accuracy": val_acc,
        "val_ci_lower": val_ci_lo,
        "val_ci_upper": val_ci_hi,
        "training_time_hours": training_time / 3600,
        "peak_vram_gb": peak_vram,
    }

    if is_main_process():
        with (output_subdir / "result.json").open("w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"[r={lora_r}] Saved final result.json")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    del model, trainer, base_model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def generate_summary_plots(output_dir: Path, all_ranks: List[int]):
    """Generate summary plots from all completed rank experiments."""
    results = []
    for r in all_ranks:
        result_path = output_dir / f"r_{r}" / "result.json"
        if result_path.exists():
            with result_path.open("r") as f:
                results.append(json.load(f))

    sweep_results_path = output_dir / "sweep_results.json"
    if sweep_results_path.exists():
        with sweep_results_path.open("r") as f:
            sweep_results = json.load(f)
            existing_ranks = {r["lora_r"] for r in results}
            for r in sweep_results:
                if r["lora_r"] not in existing_ranks and r.get("status") == "success":
                    results.append(r)

    if len(results) < 2:
        logger.info(f"Only {len(results)} completed experiments found, skipping plots")
        return

    results = sorted(results, key=lambda x: x["lora_r"])

    logger.info(f"Generating summary plots for {len(results)} experiments...")

    ranks = [r["lora_r"] for r in results]
    train_accs = [r["train_accuracy"] for r in results]
    train_ci_lo = [r["train_ci_lower"] for r in results]
    train_ci_hi = [r["train_ci_upper"] for r in results]
    val_accs = [r["val_accuracy"] for r in results]
    val_ci_lo = [r["val_ci_lower"] for r in results]
    val_ci_hi = [r["val_ci_upper"] for r in results]
    vram = [r["peak_vram_gb"] for r in results]
    times = [r["training_time_hours"] for r in results]
    params = [r["trainable_params"] for r in results]

    x = np.arange(len(ranks))
    BLUE = "#1f77b4"

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35

    train_err_lo = [train_accs[i] - train_ci_lo[i] for i in range(len(ranks))]
    train_err_hi = [train_ci_hi[i] - train_accs[i] for i in range(len(ranks))]
    ax.bar(
        x - width / 2,
        train_accs,
        width,
        label="Train",
        color="#1f77b4",
        yerr=[train_err_lo, train_err_hi],
        capsize=3,
        error_kw={"linewidth": 1},
    )

    val_err_lo = [val_accs[i] - val_ci_lo[i] for i in range(len(ranks))]
    val_err_hi = [val_ci_hi[i] - val_accs[i] for i in range(len(ranks))]
    ax.bar(
        x + width / 2,
        val_accs,
        width,
        label="Validation",
        color="#aec7e8",
        yerr=[val_err_lo, val_err_hi],
        capsize=3,
        error_kw={"linewidth": 1},
    )

    ax.axhline(
        y=MAJORITY_BASELINE,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Majority Class Baseline (Validation)",
    )

    ax.set_xlabel("LoRA Rank (r)", fontsize=12)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(train_accs), max(val_accs)) * 1.08)
    plt.tight_layout()
    plt.savefig(
        str(output_dir / "accuracy_vs_rank.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        str(output_dir / "accuracy_vs_rank.pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    logger.info("Saved accuracy_vs_rank.png/pdf")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, vram, color=BLUE, width=0.6)
    ax.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax.set_ylabel("Peak VRAM (GB)", fontsize=11)
    ax.set_title(
        f"{DATASET_NAME}: VRAM Usage vs LoRA Rank", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vram):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        str(output_dir / "vram_vs_rank.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    logger.info("Saved vram_vs_rank.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, times, color=BLUE, width=0.6)
    ax.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax.set_ylabel("Training Time (hours)", fontsize=11)
    ax.set_title(
        f"{DATASET_NAME}: Training Time vs LoRA Rank", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.grid(alpha=0.3, axis="y")
    for i, t in enumerate(times):
        ax.text(i, t + 0.1, f"{t:.2f}h", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        str(output_dir / "time_vs_rank.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    logger.info("Saved time_vs_rank.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    params_millions = [p / 1e6 for p in params]
    ax.bar(x, params_millions, color=BLUE, width=0.6)
    ax.set_xlabel("LoRA Rank (r)", fontsize=11)
    ax.set_ylabel("Trainable Parameters (M)", fontsize=11)
    ax.set_title(
        f"{DATASET_NAME}: Trainable Parameters vs LoRA Rank",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.grid(alpha=0.3, axis="y")
    for i, p in enumerate(params_millions):
        ax.text(i, p + 0.5, f"{p:.1f}M", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        str(output_dir / "params_vs_rank.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    logger.info("Saved params_vs_rank.png")

    summary_csv = output_dir / "sweep_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "lora_r",
                "train_acc",
                "train_ci_95",
                "val_acc",
                "val_ci_95",
                "vram_gb",
                "time_hours",
                "trainable_params_M",
            ]
        )
        for i, r in enumerate(results):
            writer.writerow(
                [
                    r["lora_r"],
                    f"{r['train_accuracy']:.2f}",
                    f"[{r['train_ci_lower']:.2f}, {r['train_ci_upper']:.2f}]",
                    f"{r['val_accuracy']:.2f}",
                    f"[{r['val_ci_lower']:.2f}, {r['val_ci_upper']:.2f}]",
                    f"{r['peak_vram_gb']:.2f}",
                    f"{r['training_time_hours']:.2f}",
                    f"{r['trainable_params'] / 1e6:.2f}",
                ]
            )
    logger.info(f"Saved summary CSV to {summary_csv}")


def main():
    """Run LoRA rank sweep."""
    parser = argparse.ArgumentParser(description="LoRA rank sweep for PitVis-2023")
    parser.add_argument(
        "--ranks",
        type=str,
        default=None,
        help="Comma-separated list of ranks to test",
    )
    args = parser.parse_args()

    if args.ranks:
        ranks_to_test = [int(r.strip()) for r in args.ranks.split(",")]
    else:
        ranks_to_test = LORA_RANKS

    logger.info("=" * 80)
    logger.info(f"{DATASET_NAME} LoRA RANK SWEEP (FULL TRAIN + FULL EVAL)")
    logger.info("=" * 80)
    logger.info(f"Ranks to test: {ranks_to_test}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Bootstrap iterations for CI: {BOOTSTRAP_ITERATIONS}")
    logger.info(f"LoRA target modules: {LORA_TARGET_MODULES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(RANDOM_SEED)
        logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available!")

    frame_label_train_df, frame_label_val_df = load_pitvis_dataset()

    train_frames = [Path(p) for p in frame_label_train_df["frame_path"].tolist()]
    val_frames = [Path(p) for p in frame_label_val_df["frame_path"].tolist()]

    logger.info(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    valid_tools = INSTRUMENT_NAMES
    logger.info(f"{DATASET_NAME} instruments ({len(valid_tools)}): {valid_tools}")

    frame_label_df = pd.concat(
        [frame_label_train_df, frame_label_val_df], ignore_index=True
    )
    filename_to_label = load_labels_from_dataset(frame_label_df, valid_tools)
    logger.info(f"Loaded labels for {len(filename_to_label)} frames")

    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )

    prompt = create_prompt(valid_tools)

    all_results = []

    for lora_r in ranks_to_test:
        output_subdir = OUTPUT_DIR / f"r_{lora_r}"

        result = train_with_lora_rank(
            lora_r=lora_r,
            train_frames=train_frames,
            val_frames=val_frames,
            filename_to_label=filename_to_label,
            valid_tools=valid_tools,
            processor=processor,
            prompt=prompt,
            output_subdir=output_subdir,
        )

        all_results.append(result)

        logger.info(f"Rank {lora_r} complete, saving intermediate results...")

    fieldnames = [
        "lora_r",
        "status",
        "trainable_params",
        "total_params",
        "train_accuracy",
        "train_ci_lower",
        "train_ci_upper",
        "val_accuracy",
        "val_ci_lower",
        "val_ci_upper",
        "training_time_hours",
        "peak_vram_gb",
    ]

    all_completed_results = []
    all_possible_ranks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for r in all_possible_ranks:
        result_path = OUTPUT_DIR / f"r_{r}" / "result.json"
        if result_path.exists():
            with result_path.open("r") as f:
                all_completed_results.append(json.load(f))

    all_completed_results = sorted(all_completed_results, key=lambda x: x["lora_r"])

    results_csv = OUTPUT_DIR / "sweep_results.csv"
    with results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_completed_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    logger.info(f"Aggregated results saved to {results_csv}")

    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    header = (
        f"{'r':>6} | {'Train Acc':>12} | {'Val Acc':>12} | "
        f"{'Time (h)':>10} | {'VRAM (GB)':>10} | {'Params':>12}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for r in all_completed_results:
        if r["status"] == "success":
            train_str = (
                f"{r['train_accuracy']:.1f}% "
                f"[{r['train_ci_lower']:.1f}-{r['train_ci_upper']:.1f}]"
            )
            val_str = (
                f"{r['val_accuracy']:.1f}% "
                f"[{r['val_ci_lower']:.1f}-{r['val_ci_upper']:.1f}]"
            )
            line = (
                f"{r['lora_r']:>6} | {train_str:>12} | {val_str:>12} | "
                f"{r['training_time_hours']:>10.2f} | {r['peak_vram_gb']:>10.2f} | "
                f"{r['trainable_params']:>12,}"
            )
        else:
            line = f"{r['lora_r']:>6} | OOM"
        logger.info(line)

    with (OUTPUT_DIR / "sweep_results.json").open("w") as f:
        json.dump(all_completed_results, f, indent=2)

    generate_summary_plots(OUTPUT_DIR, all_possible_ranks)

    logger.info(f"\nResults saved to {OUTPUT_DIR}")
    logger.info("LoRA rank sweep complete!")


if __name__ == "__main__":
    main()
