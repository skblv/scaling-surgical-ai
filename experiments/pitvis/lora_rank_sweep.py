"""Run the PitVis-2023 Gemma 3 27B LoRA rank sweep.

Produces the PitVis-2023 LoRA-rank sweep of multi-label tool-presence
accuracy.
"""

from __future__ import annotations

from scaling_surgical_ai.presence import lora_rank_sweep

if __name__ == "__main__":
    lora_rank_sweep.main()
