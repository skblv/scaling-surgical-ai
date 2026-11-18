"""Run PitVis-2023 Gemma 3 27B LoRA classification-head training.

Produces the PitVis-2023 fine-tuned multi-label tool-presence result.
"""

from __future__ import annotations

from scaling_surgical_ai.presence import lora_classification_head

if __name__ == "__main__":
    lora_classification_head.main()
