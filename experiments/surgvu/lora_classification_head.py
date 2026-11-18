"""Train a Gemma 3 27B LoRA classification head on SurgVU.

Produces the SurgVU fine-tuned multi-label tool-presence result by reusing
the PitVis training module with SurgVU paths, instrument names, and output
directory.
"""

from __future__ import annotations

import os

from scaling_surgical_ai import paths
from scaling_surgical_ai.presence import lora_classification_head as base

SURGVU_ROOT = paths.surgvu_root()
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

INSTRUMENT_NAMES = [
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

MODEL_PATH = paths.hf_hub_snapshot(
    "google/gemma-3-27b-it",
    "005ad3404e59d6023443cb575daa05336842228a",
)

base.PITVIS_ROOT = SURGVU_ROOT
base.DATASET_CSV = SURGVU_ROOT / "dataset.csv"
base.OUTPUT_DIR = paths.experiment_output_dir(
    "surgvu_lora_classification_head",
    SLURM_JOB_ID,
)
base.INSTRUMENT_NAMES = INSTRUMENT_NAMES
base.MODEL_PATH = MODEL_PATH


def main() -> None:
    """Run SurgVU LoRA classification-head training."""
    base.main()


if __name__ == "__main__":
    main()
