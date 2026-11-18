"""Compute top-1 and top-k accuracy for the SDSC-EEA classification head.

Loads the Gemma 3 27B LoRA classifier trained by
``experiments/sdsc_eea/lora_classification_head.py``. Top-1 counts a frame as
correct when the highest-probability tool is in the ground-truth set. Also
reports top-3 and top-5 accuracy, with bootstrap intervals, on the validation
split from ``experiments/sdsc_eea/create_dataset.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from scaling_surgical_ai import paths

RANDOM_SEED = 42
MODEL_PATH = paths.models_root() / "gemma-3-27b-it"
DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
OUTPUT_DIR = paths.experiment_output_dir("sdsc_eea_top1_accuracy")

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


def compute_topk_correctness(
    probs: np.ndarray,
    gt_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute per-sample correctness for top-K accuracy.

    Args:
        probs: (N, num_tools) probability array
        gt_labels: (N, num_tools) binary ground truth array
        k: K value for top-K

    Returns:
        (N,) boolean array indicating if each sample is correct
    """
    n_samples = probs.shape[0]
    correct = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        gt_tool_indices = set(np.where(gt_labels[i] > 0.5)[0])
        if len(gt_tool_indices) == 0:
            continue
        top_k_indices = np.argsort(probs[i])[::-1][:k]
        correct[i] = any(idx in gt_tool_indices for idx in top_k_indices)

    return correct


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for mean.

    Args:
        data: 1D array of values (e.g., boolean correctness)
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level (e.g., 0.95 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        (mean, ci_lower, ci_upper) as percentages
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = data[indices].mean()

    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100) * 100
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100) * 100
    mean = data.mean() * 100

    return mean, ci_lower, ci_upper


def compute_topk_accuracy_with_ci(
    probs: np.ndarray,
    gt_labels: np.ndarray,
    k_values: list[int] = [1, 3, 5],
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
) -> dict[int, dict[str, float]]:
    """Compute top-K accuracy with bootstrap confidence intervals.

    Args:
        probs: (N, num_tools) probability array
        gt_labels: (N, num_tools) binary ground truth array
        k_values: list of K values to compute
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level

    Returns:
        dict mapping K -> {"accuracy": float, "ci_lower": float, "ci_upper": float}
    """
    results = {}
    for k in k_values:
        correct = compute_topk_correctness(probs, gt_labels, k)
        mean, ci_lower, ci_upper = bootstrap_ci(correct, n_bootstrap, ci_level)
        results[k] = {
            "accuracy": mean,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    return results


def main() -> None:
    """Run top-1 and top-k accuracy evaluation on the validation set."""
    parser = argparse.ArgumentParser(
        description="Top-1 accuracy evaluation for the classification head"
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
    per_frame_results = []

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
            probs = torch.sigmoid(out["logits"]).squeeze(0).float().cpu().numpy()

            all_probs.append(probs)
            all_gt_labels.append(gt_label.numpy())

            top1_idx = int(np.argmax(probs))
            top1_tool = valid_tools[top1_idx]
            top1_prob = float(probs[top1_idx])

            gt_indices = set(np.where(gt_label.numpy() > 0.5)[0])
            gt_tools = [valid_tools[i] for i in gt_indices]
            is_top1_correct = top1_idx in gt_indices

            per_frame_results.append(
                {
                    "frame_path": str(frame_path),
                    "top1_tool": top1_tool,
                    "top1_prob": top1_prob,
                    "top1_correct": is_top1_correct,
                    "ground_truth_tools": gt_tools,
                    "num_gt_tools": len(gt_tools),
                }
            )

    all_probs_np = np.vstack(all_probs)
    all_gt_labels_np = np.vstack(all_gt_labels)

    logger.info("\n" + "=" * 80)
    logger.info("TOP-K ACCURACY RESULTS (with 95% Bootstrap CI)")
    logger.info("=" * 80)

    k_values = [1, 3, 5, 10]
    logger.info("Computing bootstrap confidence intervals (10,000 samples)...")
    topk_results_with_ci = compute_topk_accuracy_with_ci(
        all_probs_np, all_gt_labels_np, k_values, n_bootstrap=10000, ci_level=0.95
    )

    for k, metrics in topk_results_with_ci.items():
        acc = metrics["accuracy"]
        ci_lo = metrics["ci_lower"]
        ci_hi = metrics["ci_upper"]
        logger.info(
            f"  Top-{k} Accuracy: {acc:.2f}% (95% CI: [{ci_lo:.2f}%, {ci_hi:.2f}%])"
        )

    top1_correct = sum(1 for r in per_frame_results if r["top1_correct"])
    top1_accuracy = (top1_correct / len(per_frame_results)) * 100
    logger.info(
        f"\nTop-1 Accuracy (direct count): {top1_accuracy:.2f}% ({top1_correct}/{len(per_frame_results)})"
    )

    logger.info("\n" + "=" * 80)
    logger.info("PER-TOOL TOP-1 ANALYSIS")
    logger.info("=" * 80)

    tool_top1_stats = {
        tool: {"predicted_as_top1": 0, "correct_as_top1": 0, "in_gt": 0}
        for tool in valid_tools
    }

    for result in per_frame_results:
        top1_tool = result["top1_tool"]
        tool_top1_stats[top1_tool]["predicted_as_top1"] += 1
        if result["top1_correct"]:
            tool_top1_stats[top1_tool]["correct_as_top1"] += 1
        for gt_tool in result["ground_truth_tools"]:
            tool_top1_stats[gt_tool]["in_gt"] += 1

    per_tool_metrics = []
    for tool in sorted(valid_tools):
        stats = tool_top1_stats[tool]
        precision = (
            stats["correct_as_top1"] / stats["predicted_as_top1"] * 100
            if stats["predicted_as_top1"] > 0
            else 0.0
        )
        recall = (
            stats["correct_as_top1"] / stats["in_gt"] * 100
            if stats["in_gt"] > 0
            else 0.0
        )

        per_tool_metrics.append(
            {
                "tool": tool,
                "predicted_as_top1": stats["predicted_as_top1"],
                "correct_as_top1": stats["correct_as_top1"],
                "in_ground_truth": stats["in_gt"],
                "top1_precision": precision,
                "top1_recall": recall,
            }
        )

        if stats["predicted_as_top1"] > 0 or stats["in_gt"] > 0:
            logger.info(
                f"  {tool}: predicted={stats['predicted_as_top1']}, "
                f"correct={stats['correct_as_top1']}, in_gt={stats['in_gt']}, "
                f"prec={precision:.1f}%, recall={recall:.1f}%"
            )

    tool_top1_df = pd.DataFrame(per_tool_metrics)
    tool_top1_df.to_csv(OUTPUT_DIR / "per_tool_top1_metrics.csv", index=False)
    logger.info(
        f"\nSaved per-tool metrics to {OUTPUT_DIR / 'per_tool_top1_metrics.csv'}"
    )

    frame_prediction_df = pd.DataFrame(per_frame_results)
    frame_prediction_df.to_csv(OUTPUT_DIR / "per_frame_predictions.csv", index=False)
    logger.info(
        f"Saved per-frame predictions to {OUTPUT_DIR / 'per_frame_predictions.csv'}"
    )

    logger.info("\n" + "=" * 80)
    logger.info("FULL SET MATCH ACCURACY (for comparison, with 95% CI)")
    logger.info("=" * 80)

    threshold = 0.5
    exact_correct_arr = np.zeros(len(all_probs_np), dtype=bool)
    jaccard_arr = np.zeros(len(all_probs_np))

    for i in range(len(all_probs_np)):
        pred_indices = set(np.where(all_probs_np[i] > threshold)[0])
        gt_indices = set(np.where(all_gt_labels_np[i] > 0.5)[0])

        exact_correct_arr[i] = pred_indices == gt_indices

        union = pred_indices | gt_indices
        if len(union) > 0:
            jaccard_arr[i] = len(pred_indices & gt_indices) / len(union)
        else:
            jaccard_arr[i] = 1.0

    exact_mean, exact_ci_lo, exact_ci_hi = bootstrap_ci(exact_correct_arr)
    jaccard_mean, jaccard_ci_lo, jaccard_ci_hi = bootstrap_ci(jaccard_arr)

    logger.info(
        f"  Exact Match Accuracy: {exact_mean:.2f}% (95% CI: [{exact_ci_lo:.2f}%, {exact_ci_hi:.2f}%])"
    )
    logger.info(
        f"  Jaccard Similarity: {jaccard_mean:.2f}% (95% CI: [{jaccard_ci_lo:.2f}%, {jaccard_ci_hi:.2f}%])"
    )

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "num_val_frames": len(val_frames),
        "num_tools": num_tools,
        "top_k_accuracy": {
            f"top_{k}": {
                "accuracy": m["accuracy"],
                "ci_lower": m["ci_lower"],
                "ci_upper": m["ci_upper"],
            }
            for k, m in topk_results_with_ci.items()
        },
        "exact_match_accuracy": {
            "accuracy": exact_mean,
            "ci_lower": exact_ci_lo,
            "ci_upper": exact_ci_hi,
        },
        "jaccard_similarity": {
            "accuracy": jaccard_mean,
            "ci_lower": jaccard_ci_lo,
            "ci_upper": jaccard_ci_hi,
        },
    }
    with (OUTPUT_DIR / "evaluation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {OUTPUT_DIR / 'evaluation_summary.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("TOP-1 CONFUSION ANALYSIS")
    logger.info("=" * 80)

    confusion_data = []
    for result in per_frame_results:
        if not result["top1_correct"] and result["num_gt_tools"] > 0:
            confusion_data.append(
                {
                    "frame_path": result["frame_path"],
                    "predicted_top1": result["top1_tool"],
                    "predicted_prob": result["top1_prob"],
                    "ground_truth": result["ground_truth_tools"],
                }
            )

    if confusion_data:
        misclassification_pairs = []
        for item in confusion_data:
            for gt_tool in item["ground_truth"]:
                misclassification_pairs.append((item["predicted_top1"], gt_tool))

        pair_counts = Counter(misclassification_pairs)
        top_confusions = pair_counts.most_common(20)

        logger.info("Top 20 confusion pairs (predicted -> actual):")
        for (pred, actual), count in top_confusions:
            logger.info(f"  {pred} -> {actual}: {count} times")

        tool_confusion_df = pd.DataFrame(
            [
                {"predicted": p, "actual": a, "count": c}
                for (p, a), c in pair_counts.items()
            ]
        )
        tool_confusion_df = tool_confusion_df.sort_values("count", ascending=False)
        tool_confusion_df.to_csv(OUTPUT_DIR / "top1_confusion_pairs.csv", index=False)
        logger.info(
            f"\nSaved confusion pairs to {OUTPUT_DIR / 'top1_confusion_pairs.csv'}"
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
