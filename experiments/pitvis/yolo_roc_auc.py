"""Compute ROC-AUC for the PitVis-2023 YOLOv12m model.

Produces the PitVis-2023 YOLO ROC-AUC result.
"""

from __future__ import annotations

from scaling_surgical_ai.presence import yolo_roc_auc

if __name__ == "__main__":
    yolo_roc_auc.main()
