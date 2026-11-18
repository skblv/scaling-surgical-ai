"""Zero-shot vision-language tool detection on SurgVU.

Overrides the PitVis zero-shot module with the SurgVU dataset, the 17
instrument classes, and a four-worker layout, then runs the same
evaluation. Produces the SurgVU zero-shot VLM exact-match accuracy.
"""

from __future__ import annotations

from typing import List, Literal

import torch
from pydantic import BaseModel, Field

from scaling_surgical_ai import paths
from scaling_surgical_ai.presence import zero_shot_vlm as base

SURGVU_ROOT = paths.surgvu_root()

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

Tool = Literal[
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


class ToolDetectionResult(BaseModel):
    """Structured output for tool detection."""

    detected_tools: List[Tool] = Field(
        description="List of surgical tool names visible in the frame"
    )


base.PITVIS_ROOT = SURGVU_ROOT
base.DATASET_CSV = SURGVU_ROOT / "dataset.csv"
base.OUTPUT_BASE_DIR = paths.experiment_output_dir("surgvu_zero_shot_vlm")
base.INSTRUMENT_NAMES = INSTRUMENT_NAMES
base.NUM_WORKERS = 4
base.GPUS_PER_WORKER = 2
base.ToolDetectionResult = ToolDetectionResult
base.Tool = Tool

base.MODEL_CONFIGS = {
    "gemma-3-27b-it": {
        "path": "google/gemma-3-27b-it",
        "model_class": "Gemma3ForConditionalGeneration",
        "dtype": torch.bfloat16,
    },
}

if __name__ == "__main__":
    base.main()
