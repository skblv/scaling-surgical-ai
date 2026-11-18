"""Re-score the LoRA rank sweeps with micro-F1 (and macro-F1 / exact match).

The rank sweeps in ``experiments/sdsc_eea/lora_rank_sweep.py``,
``experiments/cholect50/lora_rank_sweep.py``,
``experiments/pitvis/lora_rank_sweep.py``, and
``experiments/surgvu/lora_rank_sweep.py`` trained a Gemma-3-27B + LoRA backbone
with a mean-pooled multi-label classification head and only persisted scalar
exact-match accuracy. This script reloads every saved checkpoint (LoRA adapter
+ classifier head), reruns the identical forward pass over the train and
validation frames, writes per-frame predictions to disk, and computes micro-F1,
macro-F1, and exact match with 95% bootstrap confidence intervals. The
validation micro-F1 values are Figure 3 of the camera-ready paper.

Tasks (dataset x rank) are distributed round-robin across the visible GPUs, one
worker process per GPU. Each worker loads the base model once per task.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from peft import PeftModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from scaling_surgical_ai import paths

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 1000
PREDICTION_THRESHOLD = 0.5

BASE_MODEL_PATH = paths.hf_hub_snapshot(
    "google/gemma-3-27b-it",
    "005ad3404e59d6023443cb575daa05336842228a",
)

OUTPUT_ROOT = paths.experiment_output_dir("lora_sweep_micro_f1")

SWEEP_RANKS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

EEA_DATASET_CSV = paths.experiment_output_dir("sdsc_eea_dataset") / "dataset.csv"
EEA_LABEL_CSV = paths.eea_labels_root() / "eea_09_24_2025.csv"

# The EEA sweep was run in two campaigns with different YOLO mappings: a 26-tool
# set (ranks 2/8/32/128/256) and a later 31-tool set (ranks 4/16/64/512/1024).
# Both orderings are sorted(set(yolo_mapping.values())) and were recovered
# verbatim from the experiments/sdsc_eea/lora_rank_sweep.py training logs. The
# classifier head size selects which list a given checkpoint uses.
EEA_TOOLS_26 = [
    "Aspirating dissector",
    "Backbiting rongeur",
    "Bipolar Forceps",
    "Collagen Matrix",
    "Cottle Elevator",
    "Cotton Patty",
    "Curette",
    "Doppler",
    "Drill",
    "Floseal Applicator",
    "Floseal Gel",
    "Grasper",
    "Irrigation",
    "Monopolar Electrocautery",
    "Needle",
    "Rhoton Dissector",
    "Rongeur",
    "Scissor",
    "Sonopet- pineapple tip",
    "Straight Forceps",
    "Suction",
    "Surgicel",
    "Through cutting forceps",
    "Tisseel Applicator",
    "Tissue shaver",
    "unknown",
]

EEA_TOOLS_31 = [
    "Aspirating dissector",
    "Backbiting rongeur",
    "Bipolar Forceps",
    "Collagen Matrix",
    "Cottle Elevator",
    "Cotton Patty",
    "Curette",
    "Doppler",
    "Drill",
    "Floseal Applicator",
    "Floseal Gel",
    "Grasper",
    "Irrigation",
    "Local Anesthesia Needle",
    "Monopolar Electrocautery",
    "Needle",
    "Rhoton Dissector",
    "Rongeur",
    "Scissor",
    "Sonopet- pineapple tip",
    "Straight Curette",
    "Straight Forceps",
    "Suction",
    "Suction Coagulator",
    "Suction microdebrider",
    "Surgical Knife",
    "Surgicel",
    "Through cutting forceps",
    "Tisseel Applicator",
    "Tissue shaver",
    "unknown",
]

EEA_TOOLS_BY_NUM = {len(EEA_TOOLS_26): EEA_TOOLS_26, len(EEA_TOOLS_31): EEA_TOOLS_31}

CHOLECT_DATASET_CSV = paths.experiment_output_dir("cholect50_dataset") / "dataset.csv"
CHOLECT_OLD_ROOT = str(paths.cholect50_root())
CHOLECT_NEW_ROOT = str(paths.cholect50_root())
PITVIS_ROOT = paths.pitvis_root()
PITVIS_RECORDED_ROOT = paths.pitvis_recorded_root()
PITVIS_DATASET_CSV = PITVIS_ROOT / "dataset.csv"
SURGVU_DATASET_CSV = paths.surgvu_root() / "dataset.csv"

CHOLECT_TOOLS = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]

PITVIS_TOOLS = [
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

SURGVU_TOOLS = [
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
]

OUTPUTS = paths.output_root()
SURGVU_SWEEP_OUTPUT_SUFFIX = os.environ.get("SURGVU_SWEEP_OUTPUT_SUFFIX", "")

CHECKPOINT_DIRS: Dict[Tuple[str, int], Path] = {
    **{
        ("eea", rank): OUTPUTS / "sdsc_eea_lora_rank_sweep" / f"r_{rank}"
        for rank in SWEEP_RANKS
    },
    **{
        ("cholect50", rank): OUTPUTS / "cholect50_lora_rank_sweep" / f"r_{rank}"
        for rank in SWEEP_RANKS
    },
    **{
        ("pitvis", rank): OUTPUTS / "pitvis_lora_rank_sweep" / f"r_{rank}"
        for rank in SWEEP_RANKS
    },
    **{
        ("surgvu", rank): OUTPUTS
        / f"surgvu_lora_rank_sweep{SURGVU_SWEEP_OUTPUT_SUFFIX}"
        / f"r_{rank}"
        for rank in SWEEP_RANKS
    },
}
for _pitvis_project_rank in (4, 16, 64, 512):
    CHECKPOINT_DIRS[("pitvis", _pitvis_project_rank)] = (
        OUTPUTS / "pitvis_lora_rank_sweep" / f"r_{_pitvis_project_rank}"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_NAMES = ["eea", "cholect50", "pitvis", "surgvu"]


@dataclass(frozen=True)
class SweepTask:
    """A single (dataset, rank) checkpoint to re-score.

    Attributes:
        dataset: Dataset name.
        rank: LoRA rank.
        checkpoint_dir: Directory holding ``lora_adapters/`` and ``classifier.pt``.
    """

    dataset: str
    rank: int
    checkpoint_dir: Path


@dataclass(frozen=True)
class DatasetData:
    """Loaded frames, labels, tools, and prompt for one dataset.

    Attributes:
        valid_tools: Tool names in the exact index order used at training time.
        prompt: Prompt text shown to the model.
        train_frames: Train split frame paths.
        val_frames: Validation split frame paths.
        frame_to_label: Mapping str(frame_path) -> multi-hot float tensor.
        frame_to_cluster: Mapping str(frame_path) -> cluster id (procedure/video).
    """

    valid_tools: List[str]
    prompt: str
    train_frames: List[Path]
    val_frames: List[Path]
    frame_to_label: Dict[str, torch.Tensor]
    frame_to_cluster: Dict[str, str]


def build_prompt(valid_tools: Sequence[str]) -> str:
    """Build the tool-detection prompt (identical to the sweep scripts).

    Args:
        valid_tools: Tool names to enumerate.

    Returns:
        Prompt string.
    """
    tools_list = "\n".join([f"- {tool}" for tool in valid_tools])
    return (
        "Identify all surgical tools visible in this surgical video frame.\n\n"
        f"Valid tool names:\n{tools_list}\n\n"
        "Carefully examine the image and identify which tools are present."
    )


def multi_hot(
    tools: Sequence[str], tool_to_idx: Dict[str, int], num_tools: int
) -> torch.Tensor:
    """Encode a tool-name list as a multi-hot float tensor.

    Args:
        tools: Present tool names.
        tool_to_idx: Mapping tool name -> index.
        num_tools: Number of tools.

    Returns:
        Float tensor of shape (num_tools,).
    """
    label = torch.zeros(num_tools, dtype=torch.float32)
    for tool in tools:
        if tool in tool_to_idx:
            label[tool_to_idx[tool]] = 1.0
    return label


def load_truth_label_dataset(
    dataset_csv: Path,
    valid_tools: List[str],
    cluster_column: str,
    path_replace: Tuple[str, str] | None = None,
) -> DatasetData:
    """Load a dataset whose CSV has split / frame_path / truth_label / cluster columns.

    Args:
        dataset_csv: Path to the dataset CSV.
        valid_tools: Tool order matching the trained classifier.
        cluster_column: Column holding the cluster id (e.g. video_id).
        path_replace: Optional (old_prefix, new_prefix) applied to frame_path.

    Returns:
        DatasetData for the dataset.
    """
    if not dataset_csv.exists():
        raise FileNotFoundError(dataset_csv)
    frame_label_df = pd.read_csv(dataset_csv)
    if path_replace is not None:
        old_prefix, new_prefix = path_replace
        frame_label_df["frame_path"] = frame_label_df["frame_path"].str.replace(
            old_prefix,
            new_prefix,
            regex=False,
        )

    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    num_tools = len(valid_tools)

    frame_to_label: Dict[str, torch.Tensor] = {}
    frame_to_cluster: Dict[str, str] = {}
    for row in frame_label_df.itertuples(index=False):
        frame_path = str(getattr(row, "frame_path"))
        tools = json.loads(getattr(row, "truth_label"))
        frame_to_label[frame_path] = multi_hot(tools, tool_to_idx, num_tools)
        frame_to_cluster[frame_path] = str(getattr(row, cluster_column))

    split_series = frame_label_df["split"]
    train_frames = [
        Path(p) for p in frame_label_df.loc[split_series == "train", "frame_path"]
    ]
    val_frames = [
        Path(p) for p in frame_label_df.loc[split_series == "val", "frame_path"]
    ]
    return DatasetData(
        valid_tools=valid_tools,
        prompt=build_prompt(valid_tools),
        train_frames=train_frames,
        val_frames=val_frames,
        frame_to_label=frame_to_label,
        frame_to_cluster=frame_to_cluster,
    )


def read_classifier_out_dim(checkpoint_dir: Path) -> int:
    """Return the output dimension of a checkpoint's classifier head.

    Args:
        checkpoint_dir: Directory holding ``classifier.pt``.

    Returns:
        Number of output classes of the saved linear head.
    """
    classifier_path = checkpoint_dir / "classifier.pt"
    if not classifier_path.exists():
        raise FileNotFoundError(classifier_path)
    state = torch.load(classifier_path, map_location="cpu")
    return int(state["bias"].shape[0])


def load_eea_dataset(valid_tools: List[str]) -> DatasetData:
    """Load the EEA dataset (split CSV plus per-tool label CSV) for a tool list.

    Args:
        valid_tools: Ordered tool list matching the target checkpoint head.

    Returns:
        DatasetData for EEA.
    """
    for required in (EEA_DATASET_CSV, EEA_LABEL_CSV):
        if not required.exists():
            raise FileNotFoundError(required)

    valid_tools = list(valid_tools)
    num_tools = len(valid_tools)

    frame_tool_label_df = pd.read_csv(EEA_LABEL_CSV)
    present_tool_columns = [
        tool for tool in valid_tools if tool in frame_tool_label_df.columns
    ]
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    present_col_indices = [tool_to_idx[tool] for tool in present_tool_columns]
    names = frame_tool_label_df["name"].tolist()
    present_values = frame_tool_label_df[present_tool_columns].to_numpy()
    filename_to_label: Dict[str, torch.Tensor] = {}
    for row_index, name in enumerate(names):
        label = torch.zeros(num_tools, dtype=torch.float32)
        row_values = present_values[row_index]
        for col_position, value in enumerate(row_values):
            if value == 1:
                label[present_col_indices[col_position]] = 1.0
        filename_to_label[name] = label

    frame_split_df = pd.read_csv(EEA_DATASET_CSV)
    frame_to_cluster: Dict[str, str] = {}
    frame_to_label: Dict[str, torch.Tensor] = {}
    zero_label = torch.zeros(num_tools, dtype=torch.float32)
    for row in frame_split_df.itertuples(index=False):
        frame_path = str(getattr(row, "frame_path"))
        filename = Path(frame_path).name
        frame_to_label[frame_path] = filename_to_label.get(filename, zero_label.clone())
        frame_to_cluster[frame_path] = str(getattr(row, "procedure_id"))

    split_series = frame_split_df["split"]
    train_frames = [
        Path(p) for p in frame_split_df.loc[split_series == "train", "frame_path"]
    ]
    val_frames = [
        Path(p) for p in frame_split_df.loc[split_series == "val", "frame_path"]
    ]
    return DatasetData(
        valid_tools=valid_tools,
        prompt=build_prompt(valid_tools),
        train_frames=train_frames,
        val_frames=val_frames,
        frame_to_label=frame_to_label,
        frame_to_cluster=frame_to_cluster,
    )


def load_dataset_data(dataset: str, num_tools: int | None = None) -> DatasetData:
    """Dispatch to the correct dataset loader.

    Args:
        dataset: One of DATASET_NAMES.
        num_tools: For EEA, the checkpoint head size selecting the tool list.

    Returns:
        DatasetData for the dataset.
    """
    if dataset == "eea":
        if num_tools not in EEA_TOOLS_BY_NUM:
            raise ValueError(f"No EEA tool list for head size {num_tools}")
        return load_eea_dataset(EEA_TOOLS_BY_NUM[num_tools])
    if dataset == "cholect50":
        return load_truth_label_dataset(
            CHOLECT_DATASET_CSV,
            CHOLECT_TOOLS,
            "video_id",
            path_replace=(CHOLECT_OLD_ROOT, CHOLECT_NEW_ROOT),
        )
    if dataset == "pitvis":
        return load_truth_label_dataset(
            PITVIS_DATASET_CSV,
            PITVIS_TOOLS,
            "video_id",
            path_replace=(str(PITVIS_RECORDED_ROOT), str(PITVIS_ROOT)),
        )
    if dataset == "surgvu":
        return load_truth_label_dataset(SURGVU_DATASET_CSV, SURGVU_TOOLS, "video_id")
    raise ValueError(f"Unknown dataset: {dataset}")


class GemmaToolClassifier(nn.Module):
    """Gemma-3 backbone with a mean-pooled multi-label head (matches the sweep)."""

    def __init__(self, base_model: nn.Module, hidden_size: int, num_tools: int):
        """Attach a linear tool head to a Gemma-3 backbone.

        Args:
            base_model: Gemma-3 model with the LoRA adapter merged in.
            hidden_size: Width of the last hidden state.
            num_tools: Number of tool classes.
        """
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_tools, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, pixel_values):
        """Mean-pool the last hidden state and apply the tool head.

        Args:
            input_ids: Token ids.
            attention_mask: Attention mask.
            pixel_values: Image tensor.

        Returns:
            Class logits of shape (batch, num_tools).
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
        return self.classifier(pooled)


def load_checkpoint_model(
    checkpoint_dir: Path, num_tools: int, device: torch.device, processor
) -> GemmaToolClassifier:
    """Load base model + LoRA adapter + classifier head onto a device in bf16.

    Args:
        checkpoint_dir: Directory with ``lora_adapters/`` and ``classifier.pt``.
        num_tools: Classifier output dimension.
        device: Target CUDA device.
        processor: Unused here but kept for symmetry with the inference signature.

    Returns:
        GemmaToolClassifier in eval mode on the device.
    """
    adapter_dir = checkpoint_dir / "lora_adapters"
    classifier_path = checkpoint_dir / "classifier.pt"
    for required in (adapter_dir / "adapter_config.json", classifier_path):
        if not required.exists():
            raise FileNotFoundError(required)

    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model = peft_model.merge_and_unload()

    hidden_size = peft_model.config.text_config.hidden_size
    model = GemmaToolClassifier(peft_model, hidden_size, num_tools)

    classifier_state = torch.load(classifier_path, map_location="cpu")
    model.classifier.load_state_dict(classifier_state)
    model.classifier.to(dtype=torch.bfloat16)

    model.to(device)
    model.eval()
    return model


def predict_frame(
    model: GemmaToolClassifier,
    processor,
    frame_path: Path,
    prompt: str,
    valid_tools: List[str],
    device: torch.device,
) -> List[str]:
    """Predict the tool set for one frame (sigmoid > threshold), matching the sweep.

    Args:
        model: Loaded classifier.
        processor: Gemma processor.
        frame_path: Image path.
        prompt: Prompt text.
        valid_tools: Tool order.
        device: CUDA device.

    Returns:
        Sorted predicted tool names.
    """
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
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )
        probs = torch.sigmoid(logits).squeeze(0)
    pred_indices = (probs > PREDICTION_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
    return sorted([valid_tools[i] for i in pred_indices])


def run_split_inference(
    model: GemmaToolClassifier,
    processor,
    frames: Sequence[Path],
    data: DatasetData,
    device: torch.device,
    desc: str,
    limit_frames: int = 0,
) -> List[dict]:
    """Run inference over a frame list, returning per-frame prediction records.

    Args:
        model: Loaded classifier.
        processor: Gemma processor.
        frames: Frame paths.
        data: DatasetData (labels, tools, prompt, clusters).
        device: CUDA device.
        desc: tqdm description.
        limit_frames: If > 0, only process the first N frames (smoke test).

    Returns:
        List of dicts with frame_path, cluster_id, predicted, truth.
    """
    if limit_frames > 0:
        frames = list(frames)[:limit_frames]
    zero_label = torch.zeros(len(data.valid_tools))
    records: List[dict] = []
    for frame_path in tqdm(frames, desc=desc, leave=False):
        frame_key = str(frame_path)
        gt_label = data.frame_to_label.get(frame_key, zero_label)
        gt_indices = (gt_label > 0.5).nonzero(as_tuple=True)[0].tolist()
        truth = sorted([data.valid_tools[i] for i in gt_indices])
        predicted = predict_frame(
            model, processor, frame_path, data.prompt, data.valid_tools, device
        )
        records.append(
            {
                "frame_path": frame_key,
                "cluster_id": data.frame_to_cluster.get(frame_key, ""),
                "predicted": predicted,
                "truth": truth,
            }
        )
    return records


def encode_records(
    records: Sequence[dict], valid_tools: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode prediction records as aligned (N, T) prediction / truth matrices.

    Args:
        records: Per-frame prediction records.
        valid_tools: Tool order.

    Returns:
        (pred_matrix, gt_matrix) as uint8 arrays.
    """
    tool_to_idx = {tool: i for i, tool in enumerate(valid_tools)}
    num_tools = len(valid_tools)
    pred_matrix = np.zeros((len(records), num_tools), dtype=np.uint8)
    gt_matrix = np.zeros((len(records), num_tools), dtype=np.uint8)
    for i, record in enumerate(records):
        for tool in record["predicted"]:
            pred_matrix[i, tool_to_idx[tool]] = 1
        for tool in record["truth"]:
            gt_matrix[i, tool_to_idx[tool]] = 1
    return pred_matrix, gt_matrix


def micro_f1(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    """Micro-averaged F1 over all (frame, tool) pairs.

    Args:
        pred_matrix: (N, T) predictions.
        gt_matrix: (N, T) ground truth.

    Returns:
        Micro-F1 as a percentage.
    """
    tp = float(np.sum((pred_matrix == 1) & (gt_matrix == 1)))
    fp = float(np.sum((pred_matrix == 1) & (gt_matrix == 0)))
    fn = float(np.sum((pred_matrix == 0) & (gt_matrix == 1)))
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom * 100


def macro_f1(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    """Macro-averaged F1 across tools.

    Args:
        pred_matrix: (N, T) predictions.
        gt_matrix: (N, T) ground truth.

    Returns:
        Macro-F1 as a percentage.
    """
    tp = np.sum((pred_matrix == 1) & (gt_matrix == 1), axis=0).astype(np.float64)
    fp = np.sum((pred_matrix == 1) & (gt_matrix == 0), axis=0).astype(np.float64)
    fn = np.sum((pred_matrix == 0) & (gt_matrix == 1), axis=0).astype(np.float64)
    denom = 2 * tp + fp + fn
    per_tool = np.where(denom > 0, (2 * tp) / np.maximum(denom, 1), 0.0)
    return float(per_tool.mean()) * 100 if per_tool.size > 0 else 0.0


def exact_match(pred_matrix: np.ndarray, gt_matrix: np.ndarray) -> float:
    """Exact-match set accuracy (matches the original sweep metric).

    Args:
        pred_matrix: (N, T) predictions.
        gt_matrix: (N, T) ground truth.

    Returns:
        Exact-match accuracy as a percentage.
    """
    if pred_matrix.shape[0] == 0:
        return 0.0
    matches = ~np.any(pred_matrix != gt_matrix, axis=1)
    return float(matches.mean()) * 100


def bootstrap_ci(
    pred_matrix: np.ndarray, gt_matrix: np.ndarray, metric_fn
) -> Tuple[float, float]:
    """Frame-level bootstrap 95% CI for a metric (matches the sweep CI style).

    Args:
        pred_matrix: (N, T) predictions.
        gt_matrix: (N, T) ground truth.
        metric_fn: Callable(pred_matrix, gt_matrix) -> float.

    Returns:
        (ci_lower, ci_upper) as percentages.
    """
    n = pred_matrix.shape[0]
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(RANDOM_SEED)
    values = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for b in range(BOOTSTRAP_ITERATIONS):
        idx = rng.integers(0, n, size=n)
        values[b] = metric_fn(pred_matrix[idx], gt_matrix[idx])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def split_metrics(records: Sequence[dict], valid_tools: List[str]) -> dict:
    """Compute all metrics + bootstrap CIs for one split.

    Args:
        records: Per-frame prediction records.
        valid_tools: Tool order.

    Returns:
        Dict of point estimates and CIs.
    """
    pred_matrix, gt_matrix = encode_records(records, valid_tools)
    micro = micro_f1(pred_matrix, gt_matrix)
    macro = macro_f1(pred_matrix, gt_matrix)
    exact = exact_match(pred_matrix, gt_matrix)
    micro_lo, micro_hi = bootstrap_ci(pred_matrix, gt_matrix, micro_f1)
    macro_lo, macro_hi = bootstrap_ci(pred_matrix, gt_matrix, macro_f1)
    exact_lo, exact_hi = bootstrap_ci(pred_matrix, gt_matrix, exact_match)
    return {
        "n_frames": int(pred_matrix.shape[0]),
        "micro_f1": micro,
        "micro_f1_ci_lower": micro_lo,
        "micro_f1_ci_upper": micro_hi,
        "macro_f1": macro,
        "macro_f1_ci_lower": macro_lo,
        "macro_f1_ci_upper": macro_hi,
        "exact_match": exact,
        "exact_match_ci_lower": exact_lo,
        "exact_match_ci_upper": exact_hi,
    }


def write_predictions(records: Sequence[dict], path: Path) -> None:
    """Write per-frame prediction records as JSONL.

    Args:
        records: Records to write.
        path: Output JSONL path.
    """
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def run_task(
    task: SweepTask,
    device: torch.device,
    processor,
    data: DatasetData,
    limit_frames: int = 0,
) -> dict:
    """Re-score one checkpoint: inference on train + val, metrics, predictions to disk.

    Args:
        task: SweepTask to run.
        device: CUDA device.
        processor: Gemma processor.
        data: Pre-loaded DatasetData for the task's dataset.
        limit_frames: If > 0, cap frames per split (smoke test).

    Returns:
        Result dict written to the task output dir.
    """
    output_dir = OUTPUT_ROOT / task.dataset / f"r_{task.rank}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint_model(
        task.checkpoint_dir, len(data.valid_tools), device, processor
    )

    tag = f"{task.dataset} r={task.rank}"
    val_records = run_split_inference(
        model,
        processor,
        data.val_frames,
        data,
        device,
        f"{tag} val",
        limit_frames,
    )
    train_records = run_split_inference(
        model,
        processor,
        data.train_frames,
        data,
        device,
        f"{tag} train",
        limit_frames,
    )

    write_predictions(val_records, output_dir / "val_predictions.jsonl")
    write_predictions(train_records, output_dir / "train_predictions.jsonl")

    result = {
        "dataset": task.dataset,
        "lora_r": task.rank,
        "checkpoint_dir": str(task.checkpoint_dir),
        "num_tools": len(data.valid_tools),
        "val": split_metrics(val_records, data.valid_tools),
        "train": split_metrics(train_records, data.valid_tools),
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()
    logger.info(
        f"[{tag}] val micro-F1 {result['val']['micro_f1']:.2f}%  "
        f"exact {result['val']['exact_match']:.2f}%  "
        f"(n_val={result['val']['n_frames']})"
    )
    return result


def worker(
    worker_index: int, world_size: int, tasks: List[SweepTask], limit_frames: int
) -> None:
    """Process the worker's slice of tasks on its assigned GPU.

    Args:
        worker_index: This worker's index.
        world_size: Total number of workers (= GPUs).
        tasks: Full task list (sliced round-robin by index).
        limit_frames: If > 0, cap frames per split (smoke test).
    """
    device = torch.device(f"cuda:{worker_index}")
    torch.cuda.set_device(device)
    torch.manual_seed(RANDOM_SEED)
    processor = AutoProcessor.from_pretrained(
        str(BASE_MODEL_PATH), trust_remote_code=True
    )

    my_tasks = tasks[worker_index::world_size]
    data_cache: Dict[Tuple[str, int], DatasetData] = {}
    logger.info(f"[worker {worker_index}] {len(my_tasks)} tasks on {device}")
    for task in my_tasks:
        output_metrics = OUTPUT_ROOT / task.dataset / f"r_{task.rank}" / "metrics.json"
        if output_metrics.exists() and limit_frames == 0:
            logger.info(
                f"[worker {worker_index}] skip done {task.dataset} r={task.rank}"
            )
            continue
        head_dim = read_classifier_out_dim(task.checkpoint_dir)
        cache_key = (task.dataset, head_dim)
        if cache_key not in data_cache:
            data_cache[cache_key] = load_dataset_data(task.dataset, head_dim)
        data = data_cache[cache_key]
        if len(data.valid_tools) != head_dim:
            raise ValueError(
                f"{task.dataset} r={task.rank}: head dim {head_dim} != "
                f"tool list size {len(data.valid_tools)}"
            )
        run_task(task, device, processor, data, limit_frames)


def build_tasks(datasets: List[str], ranks: List[int]) -> List[SweepTask]:
    """Build the full task list for the requested datasets and ranks.

    Args:
        datasets: Dataset names to include.
        ranks: Ranks to include.

    Returns:
        List of SweepTask, skipping checkpoints whose adapter is absent.
    """
    tasks: List[SweepTask] = []
    for dataset in datasets:
        for rank in ranks:
            checkpoint_dir = CHECKPOINT_DIRS[(dataset, rank)]
            adapter = checkpoint_dir / "lora_adapters" / "adapter_config.json"
            if not adapter.exists():
                raise FileNotFoundError(
                    f"Missing adapter for {dataset} r={rank}: {adapter}"
                )
            tasks.append(
                SweepTask(dataset=dataset, rank=rank, checkpoint_dir=checkpoint_dir)
            )
    return tasks


def aggregate_results(datasets: List[str], ranks: List[int]) -> None:
    """Collect all metrics.json files into a single summary CSV.

    Args:
        datasets: Dataset names included.
        ranks: Ranks included.
    """
    summary_columns = [
        "dataset",
        "lora_r",
        "n_val_frames",
        "val_micro_f1",
        "val_micro_f1_ci",
        "val_macro_f1",
        "val_exact_match",
        "train_micro_f1",
        "train_macro_f1",
        "train_exact_match",
    ]
    rows = []
    for dataset in datasets:
        for rank in ranks:
            metrics_path = OUTPUT_ROOT / dataset / f"r_{rank}" / "metrics.json"
            if not metrics_path.exists():
                continue
            with metrics_path.open("r") as f:
                result = json.load(f)
            val = result["val"]
            train = result["train"]
            rows.append(
                {
                    "dataset": dataset,
                    "lora_r": rank,
                    "n_val_frames": val["n_frames"],
                    "val_micro_f1": round(val["micro_f1"], 2),
                    "val_micro_f1_ci": f"[{val['micro_f1_ci_lower']:.2f}, {val['micro_f1_ci_upper']:.2f}]",
                    "val_macro_f1": round(val["macro_f1"], 2),
                    "val_exact_match": round(val["exact_match"], 2),
                    "train_micro_f1": round(train["micro_f1"], 2),
                    "train_macro_f1": round(train["macro_f1"], 2),
                    "train_exact_match": round(train["exact_match"], 2),
                }
            )
    rank_metric_df = pd.DataFrame(rows, columns=summary_columns)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rank_metric_df.to_csv(OUTPUT_ROOT / "sweep_microf1_summary.csv", index=False)
    logger.info(
        f"Wrote summary with {len(rank_metric_df)} rows to "
        f"{OUTPUT_ROOT / 'sweep_microf1_summary.csv'}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Re-score LoRA sweeps with micro-F1")
    parser.add_argument("--datasets", type=str, default=",".join(DATASET_NAMES))
    parser.add_argument(
        "--ranks", type=str, default=",".join(str(r) for r in SWEEP_RANKS)
    )
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument(
        "--limit-frames", type=int, default=0, help="Cap frames per split (smoke test)"
    )
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point: distribute tasks across GPUs, then aggregate."""
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]

    if args.aggregate_only:
        aggregate_results(datasets, ranks)
        return

    tasks = build_tasks(datasets, ranks)
    world_size = max(1, args.num_gpus)
    logger.info(f"Running {len(tasks)} tasks across {world_size} GPU(s)")

    if world_size == 1:
        worker(0, 1, tasks, args.limit_frames)
    else:
        mp.spawn(
            worker,
            args=(world_size, tasks, args.limit_frames),
            nprocs=world_size,
            join=True,
        )

    aggregate_results(datasets, ranks)
    logger.info("Done.")


if __name__ == "__main__":
    main()
