"""Compute per-tool AUPRC for the SDSC-EEA classification head.

Loads the Gemma 3 27B LoRA classifier trained by
``experiments/sdsc_eea/lora_classification_head.py`` and scores the full
validation split from ``experiments/sdsc_eea/create_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from scaling_surgical_ai import paths

matplotlib.use("Agg")

RANDOM_SEED = 42
MODEL_PATH = paths.models_root() / "gemma-3-27b-it"
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
OUTPUT_DIR = paths.experiment_output_dir("sdsc_eea_classification_head_auprc")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GemmaToolClassifier(nn.Module):
    """Gemma-3 with classification head for multi-label tool detection."""

    def __init__(self, base_model, hidden_size: int, num_tools: int):
        """Attach a linear head to a Gemma backbone.

        Args:
            base_model: Gemma model with LoRA adapters loaded.
            hidden_size: Width of the last hidden state.
            num_tools: Number of tool classes.
        """
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_tools, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        """Pool the last hidden state and return classification logits.

        Args:
            input_ids: Token ids.
            attention_mask: Padding mask.
            pixel_values: Vision input.
            labels: Unused. Present so the trainer can pass a batch through.

        Returns:
            A dict with key ``logits``.
        """
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
        return {"logits": logits}


def create_prompt(valid_tools: list[str]) -> str:
    """Create simple prompt for tool detection."""
    tools_list = "\n".join([f"- {tool}" for tool in valid_tools])
    return f"""Identify all surgical tools visible in this surgical video frame.

Valid tool names:
{tools_list}

Carefully examine the image and identify which tools are present."""


def load_trained_model(checkpoint_dir: Path, device: torch.device):
    """Load a trained classification-head checkpoint.

    Args:
        checkpoint_dir: Directory that contains ``final_classifier/``.
        device: Torch device the model is moved to.

    Returns:
        The classifier, its processor, the tool list, and the prompt.
    """
    final_model_path = checkpoint_dir / "final_classifier"

    if not final_model_path.exists():
        raise FileNotFoundError(f"Model not found at {final_model_path}")

    logger.info(f"Loading model from {final_model_path}")

    config_path = final_model_path / "model_config.json"
    with config_path.open("r") as f:
        model_config = json.load(f)

    valid_tools = model_config["valid_tools"]
    hidden_size = model_config["hidden_size"]
    num_tools = model_config["num_tools"]

    logger.info(f"Model config: {num_tools} tools, hidden_size={hidden_size}")

    logger.info(f"Loading processor from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH), trust_remote_code=True)

    logger.info(f"Loading base model from {MODEL_PATH}")
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapters")
    lora_path = final_model_path / "lora_adapters"
    base_model = PeftModel.from_pretrained(base_model, str(lora_path))

    model = GemmaToolClassifier(
        base_model=base_model,
        hidden_size=hidden_size,
        num_tools=num_tools,
    )

    logger.info("Loading classifier head")
    classifier_state = torch.load(
        final_model_path / "classifier.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.classifier.load_state_dict(classifier_state)

    model = model.to(device)
    model.eval()

    prompt = create_prompt(valid_tools)

    return model, processor, valid_tools, prompt


def load_labels_from_dataset(
    frame_label_df: pd.DataFrame, valid_tools: list[str]
) -> dict[str, torch.Tensor]:
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


def main() -> None:
    """Run AUPRC evaluation on the validation set."""
    parser = argparse.ArgumentParser(
        description="AUPRC evaluation for the classification head"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help=(
            "Path to a classification-head output directory "
            "(for example sdsc_eea_lora_classification_head_48466)"
        ),
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model, processor, valid_tools, prompt = load_trained_model(checkpoint_dir, device)
    num_tools = len(valid_tools)

    logger.info(f"Loading dataset from {DATASET_CSV}")
    frame_label_df = pd.read_csv(DATASET_CSV)
    frame_label_val_df = frame_label_df[frame_label_df["split"] == "val"].copy()
    val_frames = [Path(p) for p in frame_label_val_df["frame_path"].tolist()]
    logger.info(f"Loaded {len(val_frames)} validation frames")

    frame_path_to_label = load_labels_from_dataset(frame_label_val_df, valid_tools)

    logger.info("Running inference on validation set...")
    all_probs = []
    all_gt_labels = []

    with torch.no_grad():
        for frame_path in tqdm(val_frames, desc="Validation inference"):
            gt_label = frame_path_to_label.get(str(frame_path), torch.zeros(num_tools))

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
            probs = torch.sigmoid(out["logits"]).squeeze(0).cpu().numpy()

            all_probs.append(probs)
            all_gt_labels.append(gt_label.numpy())

    all_probs_np = np.vstack(all_probs)
    all_gt_labels_np = np.vstack(all_gt_labels)

    logger.info("\n" + "=" * 80)
    logger.info("AUPRC AND ROC-AUC RESULTS")
    logger.info("=" * 80)

    results = []
    for i, tool in enumerate(valid_tools):
        gt_col = all_gt_labels_np[:, i]
        prob_col = all_probs_np[:, i]

        n_positive = int(gt_col.sum())
        n_negative = len(gt_col) - n_positive

        if n_positive == 0 or n_negative == 0:
            auprc = float("nan")
            roc_auc = float("nan")
            status = "skipped (single class)"
        else:
            auprc = average_precision_score(gt_col, prob_col)
            roc_auc = roc_auc_score(gt_col, prob_col)
            status = "ok"

        results.append(
            {
                "tool": tool,
                "auprc": auprc,
                "roc_auc": roc_auc,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "status": status,
            }
        )

        if status == "ok":
            logger.info(
                f"  {tool}: AUPRC={auprc:.4f}, ROC-AUC={roc_auc:.4f} (pos={n_positive}, neg={n_negative})"
            )
        else:
            logger.info(f"  {tool}: {status} (pos={n_positive}, neg={n_negative})")

    valid_auprcs = [r["auprc"] for r in results if not np.isnan(r["auprc"])]
    valid_roc_aucs = [r["roc_auc"] for r in results if not np.isnan(r["roc_auc"])]

    macro_auprc = np.mean(valid_auprcs) if valid_auprcs else float("nan")
    macro_roc_auc = np.mean(valid_roc_aucs) if valid_roc_aucs else float("nan")

    logger.info(f"\nMacro AUPRC: {macro_auprc:.4f}")
    logger.info(f"Macro ROC-AUC: {macro_roc_auc:.4f}")

    tool_metric_df = pd.DataFrame(results)
    tool_metric_df.to_csv(OUTPUT_DIR / "per_tool_metrics.csv", index=False)
    logger.info(f"\nSaved per-tool metrics to {OUTPUT_DIR / 'per_tool_metrics.csv'}")

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "num_val_frames": len(val_frames),
        "num_tools": num_tools,
        "macro_auprc": macro_auprc,
        "macro_roc_auc": macro_roc_auc,
        "per_tool_results": results,
    }
    with (OUTPUT_DIR / "evaluation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    logger.info(f"Saved summary to {OUTPUT_DIR / 'evaluation_summary.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("PRECISION-RECALL CURVES")
    logger.info("=" * 80)

    try:
        fig, axes = plt.subplots(5, 6, figsize=(24, 20))
        axes = axes.flatten()

        for i, tool in enumerate(valid_tools):
            if i >= len(axes):
                break

            gt_col = all_gt_labels_np[:, i]
            prob_col = all_probs_np[:, i]

            ax = axes[i]

            if len(np.unique(gt_col)) > 1:
                precision, recall, _ = precision_recall_curve(gt_col, prob_col)
                auprc = average_precision_score(gt_col, prob_col)
                ax.plot(recall, precision, linewidth=2)
                ax.set_title(f"{tool}\nAUPRC={auprc:.3f}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "Single class", ha="center", va="center")
                ax.set_title(f"{tool}\n(skipped)", fontsize=9)

            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3)

        for j in range(len(valid_tools), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "precision_recall_curves.png", dpi=150, bbox_inches="tight"
        )
        plt.savefig(
            OUTPUT_DIR / "precision_recall_curves.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()
        logger.info(f"Saved PR curves to {OUTPUT_DIR / 'precision_recall_curves.png'}")

    except Exception as e:
        logger.warning(f"Could not create PR curve plots: {e}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
