"""Sweep LoRA rank for a Gemma 3 27B classification head on SurgVU.

Produces the SurgVU LoRA-rank sweep of multi-label tool-presence accuracy
by reusing the PitVis sweep module with SurgVU paths, instrument names, and
output directories.
"""

from __future__ import annotations

import os

from scaling_surgical_ai import paths
from scaling_surgical_ai.presence import lora_rank_sweep as base

OUTPUT_SUFFIX = os.environ.get("SURGVU_SWEEP_OUTPUT_SUFFIX", "")

SURGVU_ROOT = paths.surgvu_root()
DATASET_NAME = "SurgVU"
MAJORITY_BASELINE = 16.94

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
base.INSTRUMENT_NAMES = INSTRUMENT_NAMES
base.MODEL_PATH = MODEL_PATH
base.DATASET_NAME = DATASET_NAME
base.MAJORITY_BASELINE = MAJORITY_BASELINE
base.OUTPUT_DIR_FULL = paths.experiment_output_dir(
    f"surgvu_lora_rank_sweep{OUTPUT_SUFFIX}"
)
base.OUTPUT_DIR_DEBUG = paths.experiment_output_dir(
    f"surgvu_lora_rank_sweep{OUTPUT_SUFFIX}_debug"
)
base.OUTPUT_DIR = base.OUTPUT_DIR_DEBUG if base.DEBUG_MODE else base.OUTPUT_DIR_FULL


def main() -> None:
    """Run the SurgVU LoRA rank sweep."""
    base.main()


if __name__ == "__main__":
    main()
