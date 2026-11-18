"""Train a ResNet-50 multi-label classifier on SDSC-EEA set labels.

The network sees the same set-level supervision as the vision-language
models, with no bounding boxes. Reports exact match, Jaccard, per-tool
F1, ROC-AUC, and AUPRC on the 31-class ontology.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
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
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models, transforms
from tqdm.auto import tqdm

from scaling_surgical_ai.paths import (
    eea_class_mapping_path,
    experiment_output_dir,
)

matplotlib.use("Agg")

RANDOM_SEED = 42

DATASET_CSV = experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
YOLO_MAPPING_FILE = eea_class_mapping_path()

NUM_EPOCHS = 100
BATCH_SIZE_PER_GPU = 64
BACKBONE_LR = 3e-5
HEAD_LR = 3e-4
WEIGHT_DECAY = 1e-2
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1
IMGSZ = 224
NUM_WORKERS = 4
EVAL_EVERY_N_EPOCHS = 1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = experiment_output_dir("sdsc_eea_resnet50", SLURM_JOB_ID)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_ddp() -> int:
    """Initialize distributed training.

    Returns:
        Local rank of this process.
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True on rank 0."""
    return int(os.environ.get("RANK", "0")) == 0


def log_gpu_memory(prefix: str = "") -> None:
    """Log allocated, reserved, and total memory on every visible GPU.

    Args:
        prefix: Text placed in front of each memory line.
    """
    if not is_main_process():
        return
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(
                f"[GPU_MEM] {prefix} GPU {i}: "
                f"{allocated:.2f}GB alloc, "
                f"{reserved:.2f}GB reserved, "
                f"{total:.2f}GB total"
            )


def load_yolo_mapping() -> Dict[int, str]:
    """Load the YOLO class id to tool name mapping.

    Returns:
        Class id keyed to the tool name.
    """
    with YOLO_MAPPING_FILE.open("r") as f:
        data = json.load(f)
    mapping = data.get("mapping", data)
    return {int(k): v for k, v in mapping.items()}


def get_tool_names_sorted(yolo_mapping: Dict[int, str]) -> List[str]:
    """Return tool names in a stable order for the multi-hot vector.

    Args:
        yolo_mapping: Class id keyed to the tool name.

    Returns:
        Sorted unique tool names.
    """
    return sorted(set(yolo_mapping.values()))


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the SDSC-EEA frame table and split it.

    Returns:
        Training frames and validation frames.
    """
    logger.info(f"Loading dataset from {DATASET_CSV}")
    frame_dataset_df = pd.read_csv(DATASET_CSV)
    frame_dataset_train_df = frame_dataset_df[
        frame_dataset_df["split"] == "train"
    ].copy()
    frame_dataset_val_df = frame_dataset_df[frame_dataset_df["split"] == "val"].copy()
    logger.info(
        f"Train: {len(frame_dataset_train_df)} frames, "
        f"Val: {len(frame_dataset_val_df)} frames"
    )
    return frame_dataset_train_df, frame_dataset_val_df


def load_labels(
    frame_dataset_df: pd.DataFrame,
    valid_tools: List[str],
) -> Dict[str, torch.Tensor]:
    """Map each frame path to a multi-hot label tensor.

    The ``truth_label`` column holds a JSON list of tool names.

    Args:
        frame_dataset_df: Frames whose labels should be encoded.
        valid_tools: Tool names in multi-hot order.

    Returns:
        Frame path keyed to its multi-hot label.
    """
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    num_tools = len(valid_tools)

    frame_to_label = {}
    for _, row in frame_dataset_df.iterrows():
        frame_path = row["frame_path"]
        tools = json.loads(row["truth_label"])
        label = torch.zeros(num_tools, dtype=torch.float32)
        for tool in tools:
            if tool in tool_to_idx:
                label[tool_to_idx[tool]] = 1.0
        frame_to_label[frame_path] = label

    return frame_to_label


class SDSCToolDataset(Dataset):
    """Multi-label classification dataset for SDSC-EEA surgical tools."""

    def __init__(
        self,
        frame_paths: List[str],
        frame_to_label: Dict[str, torch.Tensor],
        transform: transforms.Compose,
        num_tools: int,
    ) -> None:
        """Keep frames that exist on disk.

        Args:
            frame_paths: Candidate frame paths.
            frame_to_label: Frame path keyed to its multi-hot label.
            transform: Image transform applied in ``__getitem__``.
            num_tools: Width of the label vector.
        """
        self.frame_paths = [p for p in frame_paths if Path(p).exists()]
        self.frame_to_label = frame_to_label
        self.transform = transform
        self.num_tools = num_tools

        skipped = len(frame_paths) - len(self.frame_paths)
        if skipped > 0:
            logger.warning(f"Skipped {skipped} missing frame files")

    def __len__(self) -> int:
        """Return the number of frames that exist on disk."""
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load one image and its multi-hot label.

        Args:
            idx: Index into the filtered frame list.

        Returns:
            The transformed image and its label vector.
        """
        frame_path = self.frame_paths[idx]
        image = Image.open(frame_path).convert("RGB")
        image = self.transform(image)
        label = self.frame_to_label.get(
            frame_path,
            torch.zeros(self.num_tools, dtype=torch.float32),
        )
        return image, label


def create_resnet50_classifier(
    num_classes: int,
    dropout: float = DROPOUT,
) -> nn.Module:
    """Create a ResNet-50 with a dropout multi-label head.

    ImageNet-V2 weights initialize the backbone. The 1000-way layer is
    replaced by dropout and a linear layer.

    Args:
        num_classes: Number of tool outputs.
        dropout: Dropout probability in front of the linear layer.

    Returns:
        The classification model.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    classifier = nn.Linear(in_features, num_classes)
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        classifier,
    )
    return model


def get_train_transforms() -> transforms.Compose:
    """Return the training augmentations.

    Returns:
        The training transform.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMGSZ, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.15,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.2),
            ),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """Return the deterministic validation transform.

    Returns:
        The validation transform.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMGSZ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch.

    Args:
        model: Model in train mode, possibly wrapped in DDP.
        loader: Training loader.
        optimizer: Optimizer stepped once per batch.
        scaler: GradScaler used for fp16.
        device: Device that holds the batch.
        epoch: Epoch number shown in the progress bar.

    Returns:
        Loss averaged across batches and ranks.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        disable=not is_main_process(),
    )
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            smoothed = labels * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            loss = F.binary_cross_entropy_with_logits(logits, smoothed)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device, dtype=torch.float64)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor[0] / dist.get_world_size()).item()

    return avg_loss


@torch.no_grad()
def compute_val_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute the validation loss averaged across ranks.

    Args:
        model: Model in eval mode, possibly wrapped in DDP.
        loader: Validation loader.
        device: Device that holds the batch.

    Returns:
        Unsmooth binary-cross-entropy averaged across batches and ranks.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device, dtype=torch.float64)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor[0] / dist.get_world_size()).item()

    return avg_loss


@torch.no_grad()
def compute_full_metrics(
    model: nn.Module,
    loader: DataLoader,
    valid_tools: List[str],
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Compute exact match, Jaccard, per-tool P/R/F1, ROC-AUC, AUPRC.

    Args:
        model: Unwrapped model (not DDP).
        loader: DataLoader without DistributedSampler (full val set).
        valid_tools: Sorted list of tool names.
        device: CUDA device.
        output_dir: Where to save CSVs and JSONs.

    Returns:
        Dict with all computed metrics.
    """
    model.eval()

    all_probs_list = []
    all_labels_list = []

    for images, labels in tqdm(loader, desc="Computing metrics"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu()
        all_probs_list.append(probs)
        all_labels_list.append(labels)

    all_probs = torch.cat(all_probs_list, dim=0).numpy()
    all_gt = torch.cat(all_labels_list, dim=0).numpy().astype(int)

    threshold = 0.5
    preds = (all_probs > threshold).astype(int)

    exact_matches_mask = np.all(preds == all_gt, axis=1)
    exact_accuracy = exact_matches_mask.mean() * 100

    jaccard_scores = []
    for pred_row, gt_row in zip(preds, all_gt):
        pred_set = set(np.where(pred_row == 1)[0])
        gt_set = set(np.where(gt_row == 1)[0])
        union = pred_set | gt_set
        jaccard = len(pred_set & gt_set) / len(union) if union else 1.0
        jaccard_scores.append(jaccard)
    jaccard_accuracy = np.mean(jaccard_scores) * 100

    top1_correct = 0
    top1_total = 0
    for prob_row, gt_row in zip(all_probs, all_gt):
        gt_set = set(np.where(gt_row == 1)[0])
        top1_idx = int(np.argmax(prob_row))
        if gt_set:
            top1_total += 1
            if top1_idx in gt_set:
                top1_correct += 1
        else:
            top1_total += 1
            if prob_row.max() < threshold:
                top1_correct += 1
    top1_accuracy = (top1_correct / top1_total * 100) if top1_total > 0 else 0.0

    per_tool_tp = defaultdict(int)
    per_tool_fp = defaultdict(int)
    per_tool_fn = defaultdict(int)
    per_tool_tn = defaultdict(int)

    for pred_row, gt_row in zip(preds, all_gt):
        for i, tool in enumerate(valid_tools):
            if gt_row[i] == 1 and pred_row[i] == 1:
                per_tool_tp[tool] += 1
            elif gt_row[i] == 0 and pred_row[i] == 1:
                per_tool_fp[tool] += 1
            elif gt_row[i] == 1 and pred_row[i] == 0:
                per_tool_fn[tool] += 1
            else:
                per_tool_tn[tool] += 1

    logger.info("\n" + "-" * 70)
    logger.info("PER-TOOL METRICS")
    logger.info("-" * 70)
    header_cols = [
        f"{'Tool':<30}",
        f"{'Prec':>8}",
        f"{'Recall':>8}",
        f"{'F1':>8}",
        f"{'Support':>8}",
    ]
    logger.info(" ".join(header_cols))
    logger.info("-" * 70)

    total_tp, total_fp, total_fn = 0, 0, 0
    per_tool_metrics = {}
    per_tool_rows = []

    for tool in sorted(valid_tools):
        tp = per_tool_tp[tool]
        fp = per_tool_fp[tool]
        fn = per_tool_fn[tool]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        support = tp + fn

        per_tool_metrics[tool] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        per_tool_rows.append(
            {
                "tool": tool,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

        if support > 0:
            logger.info(
                f"{tool:<30} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f} {support:>8}"
            )

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_prec * micro_recall / (micro_prec + micro_recall)
        if (micro_prec + micro_recall) > 0
        else 0.0
    )

    logger.info("-" * 70)
    logger.info(
        f"{'MICRO AVG':<30} {micro_prec:>8.3f} {micro_recall:>8.3f} {micro_f1:>8.3f}"
    )

    per_tool_auc = {}
    per_tool_auprc = {}
    valid_aucs = []
    valid_auprcs = []

    for i, tool in enumerate(valid_tools):
        gt_col = all_gt[:, i]
        prob_col = all_probs[:, i]
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

    pd.DataFrame(per_tool_rows).to_csv(output_dir / "per_tool_metrics.csv", index=False)

    predictions_list = []
    for idx, (prob_row, gt_row, pred_row) in enumerate(zip(all_probs, all_gt, preds)):
        gt_tools = sorted(
            [valid_tools[i] for i in range(len(valid_tools)) if gt_row[i] == 1]
        )
        pred_tools = sorted(
            [valid_tools[i] for i in range(len(valid_tools)) if pred_row[i] == 1]
        )
        is_exact = set(gt_tools) == set(pred_tools)
        p_set = set(pred_tools)
        g_set = set(gt_tools)
        union = p_set | g_set
        jac = len(p_set & g_set) / len(union) if union else 1.0
        predictions_list.append(
            {
                "idx": idx,
                "ground_truth": gt_tools,
                "predicted": pred_tools,
                "exact_match": is_exact,
                "jaccard": jac,
            }
        )

    with (output_dir / "predictions_detailed.json").open("w") as f:
        json.dump(predictions_list, f, indent=2)

    logger.info(f"\nExact Match Accuracy: {exact_accuracy:.2f}%")
    logger.info(f"Jaccard Similarity:   {jaccard_accuracy:.2f}%")
    logger.info(f"Top-1 Accuracy:       {top1_accuracy:.2f}%")
    logger.info(f"Micro F1:             {micro_f1:.4f}")
    logger.info(f"Macro ROC-AUC:        {macro_auc:.4f}")
    logger.info(f"Macro AUPRC:          {macro_auprc:.4f}")

    return {
        "exact_match_accuracy": exact_accuracy,
        "jaccard_accuracy": jaccard_accuracy,
        "top1_accuracy": top1_accuracy,
        "total_images": len(all_probs),
        "exact_matches": int(exact_matches_mask.sum()),
        "micro_precision": micro_prec,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_roc_auc": macro_auc,
        "macro_auprc": macro_auprc,
        "per_tool_roc_auc": {k: v for k, v in per_tool_auc.items() if v is not None},
        "per_tool_auprc": {k: v for k, v in per_tool_auprc.items() if v is not None},
        "per_tool_metrics": per_tool_metrics,
    }


def save_plots(
    train_losses: List[float],
    val_losses: List[float],
    exact_accuracies: List[Tuple[int, float]],
    jaccard_accuracies: List[Tuple[int, float]],
    output_dir: Path,
) -> None:
    """Save the loss and accuracy curves and the tables behind them.

    Args:
        train_losses: Training loss after each epoch.
        val_losses: Validation loss after each epoch.
        exact_accuracies: Epoch and exact-match accuracy pairs.
        jaccard_accuracies: Epoch and Jaccard pairs.
        output_dir: Directory for the figures and CSVs.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax1 = axes[0]
    ax1.plot(
        epochs,
        train_losses,
        linewidth=1.5,
        color="#1f77b4",
        label="Train",
    )
    ax1.plot(
        epochs,
        val_losses,
        linewidth=1.5,
        color="#ff7f0e",
        linestyle="--",
        label="Val",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    if exact_accuracies:
        eval_epochs = [e for e, _ in exact_accuracies]
        eval_accs = [a for _, a in exact_accuracies]
        ax2.plot(
            eval_epochs,
            eval_accs,
            linewidth=1.5,
            color="#2ca02c",
            marker="o",
            markersize=4,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Exact Match Accuracy (%)")
    ax2.set_title("Exact Match Accuracy (Val)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    if jaccard_accuracies:
        eval_epochs = [e for e, _ in jaccard_accuracies]
        eval_jacs = [j for _, j in jaccard_accuracies]
        ax3.plot(
            eval_epochs,
            eval_jacs,
            linewidth=1.5,
            color="#d62728",
            marker="o",
            markersize=4,
        )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Jaccard Similarity (%)")
    ax3.set_title("Jaccard Similarity (Val)")
    ax3.set_ylim(0, 100)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        str(output_dir / "training_loss.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        str(output_dir / "training_loss.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    epoch_loss_df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    epoch_loss_df.to_csv(output_dir / "loss_data.csv", index=False)

    if exact_accuracies:
        epoch_accuracy_df = pd.DataFrame(
            exact_accuracies,
            columns=["epoch", "exact_accuracy"],
        )
        epoch_accuracy_df.to_csv(output_dir / "accuracy_data.csv", index=False)

    if jaccard_accuracies:
        epoch_jaccard_df = pd.DataFrame(
            jaccard_accuracies,
            columns=["epoch", "jaccard"],
        )
        epoch_jaccard_df.to_csv(output_dir / "jaccard_data.csv", index=False)

    logger.info(f"Saved plots to {output_dir / 'training_loss.png'}")


def main() -> None:
    """Train ResNet-50 for multi-label tool classification on SDSC data."""
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    yolo_mapping = load_yolo_mapping()
    valid_tools = get_tool_names_sorted(yolo_mapping)
    num_tools = len(valid_tools)

    if is_main_process():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 80)
        logger.info(
            "SDSC-EEA ResNet-50 multi-label classification "
            "(no bounding-box supervision)"
        )
        logger.info("=" * 80)
        logger.info(f"Job ID: {SLURM_JOB_ID}")
        logger.info(f"Output: {OUTPUT_DIR}")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {dist.get_world_size()}")
        logger.info(f"Num tool classes: {num_tools}")
        logger.info(f"Tools: {valid_tools}")

    frame_dataset_train_df, frame_dataset_val_df = load_dataset()
    frame_dataset_df = pd.concat(
        [frame_dataset_train_df, frame_dataset_val_df],
        ignore_index=True,
    )
    frame_to_label = load_labels(frame_dataset_df, valid_tools)

    train_paths = frame_dataset_train_df["frame_path"].tolist()
    val_paths = frame_dataset_val_df["frame_path"].tolist()

    train_procedures = set(frame_dataset_train_df["procedure_id"].astype(str).unique())
    val_procedures = set(frame_dataset_val_df["procedure_id"].astype(str).unique())
    overlap = train_procedures & val_procedures
    if overlap:
        raise RuntimeError(
            f"PROCEDURE LEAKAGE: {len(overlap)} procedures in both splits!"
        )

    if is_main_process():
        logger.info(
            f"No procedure overlap: train={len(train_procedures)} "
            f"procedures, val={len(val_procedures)} procedures"
        )
        sample_labels = list(frame_to_label.values())[:500]
        avg_tools = sum(label.sum().item() for label in sample_labels) / len(
            sample_labels
        )
        logger.info(f"Average tools per frame (sample): {avg_tools:.2f}")

        tool_counts = defaultdict(int)
        for label in frame_to_label.values():
            indices = (label > 0.5).nonzero(as_tuple=True)[0]
            for idx in indices:
                tool_counts[valid_tools[idx.item()]] += 1
        logger.info("\nTool distribution:")
        for tool in sorted(
            tool_counts.keys(),
            key=lambda t: tool_counts[t],
            reverse=True,
        ):
            logger.info(f"  {tool:<30} {tool_counts[tool]:>6}")

    train_dataset = SDSCToolDataset(
        frame_paths=train_paths,
        frame_to_label=frame_to_label,
        transform=get_train_transforms(),
        num_tools=num_tools,
    )
    val_dataset = SDSCToolDataset(
        frame_paths=val_paths,
        frame_to_label=frame_to_label,
        transform=get_val_transforms(),
        num_tools=num_tools,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        shuffle=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    model = create_resnet50_classifier(num_tools)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model: ResNet-50 (IMAGENET1K_V2)")
        logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
        logger.info(
            f"Trainable parameters: {trainable_params:,} "
            f"({trainable_params / 1e6:.1f}M)"
        )

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    backbone_params = [
        p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() if "fc" in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")

    if is_main_process():
        total_batch = BATCH_SIZE_PER_GPU * dist.get_world_size()
        steps_per_epoch = len(train_dataset) // total_batch
        n_backbone = sum(p.numel() for p in backbone_params)
        n_head = sum(p.numel() for p in head_params)
        logger.info("\nTraining config:")
        logger.info(f"  Epochs: {NUM_EPOCHS}")
        logger.info(f"  Batch size/GPU: {BATCH_SIZE_PER_GPU}")
        logger.info(f"  Total batch size: {total_batch}")
        logger.info(f"  Steps per epoch: ~{steps_per_epoch}")
        logger.info(f"  Backbone LR: {BACKBONE_LR} ({n_backbone:,} params)")
        logger.info(f"  Head LR:     {HEAD_LR} ({n_head:,} params)")
        logger.info(f"  Weight decay: {WEIGHT_DECAY}")
        logger.info(f"  Dropout: {DROPOUT}")
        logger.info(f"  Label smoothing: {LABEL_SMOOTHING}")
        logger.info("  Optimizer: AdamW")
        logger.info("  Scheduler: CosineAnnealingLR")
        logger.info(f"  Image size: {IMGSZ}")
        logger.info("  AMP: fp16")
        log_gpu_memory("Before training")

    train_losses = []
    val_losses = []
    exact_accuracies: List[Tuple[int, float]] = []
    jaccard_accuracies: List[Tuple[int, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
        )
        val_loss = compute_val_loss(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if is_main_process():
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    model.module.state_dict(),
                    OUTPUT_DIR / "best_model.pt",
                )
                logger.info(f"  -> New best model (val_loss={val_loss:.4f})")

            is_eval_epoch = epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == NUM_EPOCHS
            if is_eval_epoch:
                eval_loader = DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE_PER_GPU * 2,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )
                metrics = compute_full_metrics(
                    model.module,
                    eval_loader,
                    valid_tools,
                    device,
                    OUTPUT_DIR,
                )
                exact_accuracies.append((epoch, metrics["exact_match_accuracy"]))
                jaccard_accuracies.append((epoch, metrics["jaccard_accuracy"]))

            save_plots(
                train_losses,
                val_losses,
                exact_accuracies,
                jaccard_accuracies,
                OUTPUT_DIR,
            )

        dist.barrier()

    if is_main_process():
        logger.info("\n" + "=" * 80)
        logger.info(f"FINAL EVALUATION (best model from epoch {best_epoch})")
        logger.info("=" * 80)

        best_state = torch.load(
            OUTPUT_DIR / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
        model.module.load_state_dict(best_state)

        eval_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE_PER_GPU * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        final_metrics = compute_full_metrics(
            model.module,
            eval_loader,
            valid_tools,
            device,
            OUTPUT_DIR,
        )

        torch.save(
            model.module.state_dict(),
            OUTPUT_DIR / "final_model.pt",
        )

        total_params = sum(p.numel() for p in model.module.parameters())
        results_summary = {
            "job_id": SLURM_JOB_ID,
            "timestamp": datetime.now().isoformat(),
            "model": "ResNet-50 (IMAGENET1K_V2)",
            "total_params": total_params,
            "num_tools": num_tools,
            "tool_names": valid_tools,
            "num_epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "total_batch_size": (BATCH_SIZE_PER_GPU * dist.get_world_size()),
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "label_smoothing": LABEL_SMOOTHING,
            "imgsz": IMGSZ,
            "training_config": {
                "supervision": ("set-level labels (no bounding boxes)"),
                "loss": "binary_cross_entropy_with_logits",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "amp": "fp16",
            },
            "eval_results": {
                k: v for k, v in final_metrics.items() if k != "per_tool_metrics"
            },
        }

        with (OUTPUT_DIR / "results_summary.json").open("w") as f:
            json.dump(results_summary, f, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Model: ResNet-50 ({total_params:,} params, {total_params / 1e6:.1f}M)"
        )
        logger.info("Supervision: Set-level labels (no bounding boxes)")
        logger.info(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
        logger.info(
            f"Exact Match Accuracy: {final_metrics['exact_match_accuracy']:.2f}%"
        )
        logger.info(f"Jaccard Similarity:   {final_metrics['jaccard_accuracy']:.2f}%")
        logger.info(f"Top-1 Accuracy:       {final_metrics['top1_accuracy']:.2f}%")
        logger.info(f"Micro F1:             {final_metrics['micro_f1']:.4f}")
        logger.info(f"Macro ROC-AUC:        {final_metrics['macro_roc_auc']:.4f}")
        logger.info(f"Macro AUPRC:          {final_metrics['macro_auprc']:.4f}")
        logger.info(f"\nOutputs saved to: {OUTPUT_DIR}")

    cleanup_ddp()

    if is_main_process():
        logger.info("\nSDSC-EEA ResNet-50 run complete!")


if __name__ == "__main__":
    main()
