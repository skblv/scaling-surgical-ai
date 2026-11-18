"""Train the YOLOv12m presence baseline on PitVis-2023.

Produces the PitVis-2023 YOLO exact-match, Jaccard, top-1, and
micro-F1 results.
"""

from __future__ import annotations

from scaling_surgical_ai.presence import yolo

if __name__ == "__main__":
    yolo.main()
