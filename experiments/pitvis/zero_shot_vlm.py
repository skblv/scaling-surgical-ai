"""Run zero-shot vision-language tool detection on PitVis-2023.

Produces the PitVis-2023 zero-shot VLM exact-match accuracy.
"""

from __future__ import annotations

from scaling_surgical_ai.presence import zero_shot_vlm

if __name__ == "__main__":
    zero_shot_vlm.main()
