"""Print train, validation, and full exact match from recalculated summaries.

Reads each model directory under the recalculated zero-shot output and
prints one CSV row. Train counts are the complement of the validation
counts stored in ``summary.csv``.
"""

from __future__ import annotations

import csv

from scaling_surgical_ai import paths

BASE = paths.experiment_output_dir("sdsc_eea_recalculate_zero_shot_metrics")

HEADER = [
    "model",
    "total",
    "parse_fail",
    "pf_rate",
    "train_n",
    "train_em",
    "train_pct",
    "val_n",
    "val_em",
    "val_pct",
    "val_ci_lo",
    "val_ci_hi",
    "all_em",
    "all_pct",
]


def main() -> None:
    """Print the breakdown for every model that has a summary CSV."""
    print(",".join(HEADER))

    for model_dir in sorted(BASE.iterdir(), key=lambda path: path.name):
        summary_path = model_dir / "summary.csv"
        if not summary_path.is_file():
            continue
        with open(summary_path) as f:
            summary = list(csv.DictReader(f))[0]
        total_attempted = int(summary["total_attempted"])
        val_frames = int(summary["val_frames"])
        all_exact = int(summary["all_exact_matches"])
        val_exact = int(summary["val_exact_matches"])
        train_frames = total_attempted - val_frames
        train_exact = all_exact - val_exact
        train_pct = train_exact / train_frames * 100 if train_frames > 0 else 0
        cols = [
            summary["model_name"],
            str(total_attempted),
            summary["parse_failures"],
            summary["parse_failure_rate_pct"],
            str(train_frames),
            str(train_exact),
            f"{train_pct:.2f}",
            str(val_frames),
            str(val_exact),
            summary["val_exact_match_accuracy_pct"],
            summary["val_exact_match_ci_low_pct"],
            summary["val_exact_match_ci_high_pct"],
            str(all_exact),
            summary["all_exact_match_accuracy_pct"],
        ]
        print(",".join(cols))


if __name__ == "__main__":
    main()
