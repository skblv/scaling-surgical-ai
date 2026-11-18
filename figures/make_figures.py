"""Render the figures of the paper from the bundled result tables.

PDF and 300 dpi PNG files are written to ``figures/output/``. The stems map
onto the paper as follows:

- ``setup_overview``: Figure 1, validation frames. Rendered only when
  ``--frames-dir`` supplies ``correct_{1..5}.png`` and ``incorrect_{1..5}.png``.
- ``zero_shot_scaling_micro_f1``: Figure 2, zero-shot scaling, micro-F1.
- ``lora_sweep_micro_f1``: Figure 3, LoRA rank sweep, micro-F1.
- ``cross_dataset_micro_f1``: Figure 4, cross-dataset replication, micro-F1.
- ``zero_shot_scaling_exact_match``: Supplementary Figure S1.
- ``lora_sweep_exact_match``: Supplementary Figure S2.
- ``cross_dataset_exact_match``: Supplementary Figure S3.

``figures/plot_temperature_sweep.py`` writes ``accuracy_vs_temperature``,
Supplementary Figure S7.

Style matches the camera-ready figures: embedded fonts (pdf/ps fonttype 42),
sans-serif (Helvetica/Arial, with DejaVu Sans fallback), lowercase 8 pt
panel letters, other text 5-7 pt, colour-blind-safe palettes, no coloured
text, no gridlines, and widths of 90 mm or 180 mm. Numbers are read from the
CSVs in ``figures/data/`` and from the constants transcribed below.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.image import imread

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
PLOTS_OUT = HERE / "output"
DATA_DIR = HERE / "data"

MM_PER_IN = 25.4
W_DOUBLE = 180 / MM_PER_IN

# NEJM palette (ggsci ``pal_nejm``) at full saturation, as in the camera-ready
# figures. Black outlines on every marker and bar keep the contrast crisp.
NEJM_RED = "#BC3C29"
NEJM_BLUE = "#0072B5"
NEJM_ORANGE = "#E18727"
NEJM_GREEN = "#20854E"
NEJM_PURPLE = "#7876B1"
NEJM_SLATE = "#6F99AD"
NEJM_YELLOW = "#FFDC91"
NEJM_PINK = "#EE4C97"
# The cross-dataset figure has nine series and the palette eight hues; the
# zero-shot reference bar is neutral grey.
NEUTRAL_GREY = "#7F7F7F"

# LoRA rank-sweep train/validation bars.
TRAIN_COLOR = NEJM_BLUE
VAL_COLOR = NEJM_ORANGE

PANEL_LABEL_GAP_IN = 0.04
PANEL_LABEL_TOP_IN = 0.05


def _add_panel_labels(fig: plt.Figure, axes_seq, labels: tuple[str, ...]) -> None:
    """Place lowercase panel labels at a uniform offset from each panel.

    Labels sit a fixed horizontal pixel-offset from each panel's leftmost
    rendered content.

    For panels that carry a y-axis title (e.g., the left column of a
    ``sharey=True`` grid), the leftmost element is that title. For panels
    that share their y-axis and have ``set_ylabel("")`` it is the plot
    frame. Using each panel's tight bounding box guarantees the gap
    between the letter and the leftmost content is identical for every
    panel of every figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    dpi = fig.dpi
    gap_px = PANEL_LABEL_GAP_IN * dpi
    top_px = PANEL_LABEL_TOP_IN * dpi
    for ax, lab in zip(axes_seq, labels):
        bbox = ax.get_tightbbox(renderer)
        x_fig, y_fig = inv.transform((bbox.x0 - gap_px, bbox.y1 + top_px))
        fig.text(
            x_fig,
            y_fig,
            lab,
            fontsize=8,
            fontweight="normal",
            fontstyle="normal",
            color="black",
            ha="left",
            va="bottom",
        )


# Zero-shot families are distinguished by marker shape and hue. The two
# trained models keep their hue in the cross-dataset figure (fine-tuned Gemma
# red, YOLOv12-m green).
PALETTE_FAMILIES = {
    "Qwen3-VL": NEJM_ORANGE,
    "Qwen2.5-VL": NEJM_PURPLE,
    "Qwen2-VL": NEJM_PINK,
    "Gemma 3": NEJM_BLUE,
    "Gemma 4": NEJM_SLATE,
    "MedGemma 3": NEJM_YELLOW,
    "LLaMA 3.2 Vision": NEJM_ORANGE,
    "LLaVA": NEJM_PINK,
    "Fine-tuned Gemma 3 (ours)": NEJM_RED,
    "YOLOv12-m (ours)": NEJM_GREEN,
}

MARKERS_FAMILIES = {
    "Qwen3-VL": "o",
    "Qwen2.5-VL": "s",
    "Qwen2-VL": "^",
    "Gemma 3": "D",
    "Gemma 4": "d",
    "MedGemma 3": "p",
    "LLaMA 3.2 Vision": "v",
    "LLaVA": "X",
    "Fine-tuned Gemma 3 (ours)": "*",
    "YOLOv12-m (ours)": "P",
}

# Each row: (name, family, params_B, mmbench, exact_match_%, micro_f1_15tool_%).
# Micro F1 over the 15 evaluable SDSC-EEA tools, computed by
# experiments/sdsc_eea/zero_shot_micro_f1.py (val split, n = 20,016 frames;
# LLaVA read from its results.csv, which already encodes parse failures).
ZS_MODELS = [
    ("Qwen3-VL-235B-A22B-Thinking", "Qwen3-VL", 235, 90.6, 14.52, 39.01),
    ("Qwen3-VL-32B-Instruct", "Qwen3-VL", 32, 88.9, 11.04, 39.72),
    ("Qwen3-VL-8B-Instruct", "Qwen3-VL", 8, 85.0, 10.73, 21.87),
    ("Qwen3-VL-4B-Instruct", "Qwen3-VL", 4, 85.1, 6.37, 27.94),
    ("Qwen3-VL-2B-Instruct", "Qwen3-VL", 2, 77.8, 0.11, 22.14),
    ("Qwen2.5-VL-72B-Instruct", "Qwen2.5-VL", 72, 87.8, 4.56, 18.72),
    ("Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL", 32, 84.0, 5.43, 18.85),
    ("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL", 7, 82.2, 3.48, 7.61),
    ("Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL", 3, 76.8, 2.90, 7.92),
    ("Qwen2-VL-72B-Instruct", "Qwen2-VL", 72, 85.9, 5.70, 16.27),
    ("Qwen2-VL-7B-Instruct", "Qwen2-VL", 7, 81.0, 3.35, 23.55),
    ("Qwen2-VL-2B-Instruct", "Qwen2-VL", 2, 72.2, 1.48, 4.76),
    ("Gemma 3 27B-it", "Gemma 3", 27, 78.9, 9.83, 44.87),
    ("Gemma 3 12B-it", "Gemma 3", 12, 74.6, 4.95, 27.65),
    ("Gemma 3 4B-it", "Gemma 3", 4, 66.4, 0.23, 27.53),
    ("MedGemma 3 27B-it", "MedGemma 3", 27, None, 6.36, 43.44),
    ("Llama-3.2-90B-Vision", "LLaMA 3.2 Vision", 90, 79.5, 8.06, 39.68),
    ("Llama-3.2-11B-Vision", "LLaMA 3.2 Vision", 11, 67.5, 0.19, 1.04),
    ("LLaVA-1.5-13B", "LLaVA", 13, 65.8, 0.53, 13.57),
    ("Gemma 4 31B-it", "Gemma 4", 31, 90.9, 10.05, 39.18),
]

# (name, params, exact_match_%, micro_f1_15tool_%).
OURS = [
    ("Fine-tuned Gemma 3 (ours)", 27e9, 51.08, 81.40),
    ("YOLOv12-m (ours)", 26e6, 54.73, 80.48),
]

BASELINE_VAL_SDSC = 13.41
BASELINE_VAL_SDSC_F1 = 47.27
BASELINE_VAL_PITVIS = 39.63
BASELINE_VAL_SURGVU = 16.94
BASELINE_VAL_CHOLECT50 = 34.76


CROSS_DATASET_EM = {
    "Gemma 3 27B + LoRA": [51.08, 83.02, 84.77, 50.61],
    "YOLOv12-m": [54.73, 81.37, 82.78, 51.75],
    "Gemini 3.1 Pro Preview": [None, 66.21, 57.65, 22.46],
    "Gemini 3 Flash Preview": [None, 69.15, 52.62, 11.69],
    "Claude Opus 4.6": [None, 52.37, 50.74, 17.22],
    "Claude Sonnet 4.6": [None, 30.73, 50.03, 23.05],
    "Claude Fable 5": [None, 54.70, 57.20, 20.09],
    "GPT-5.4": [None, 32.09, 49.35, 17.64],
    "Gemma 3 27B-it (zero-shot)": [9.83, 6.87, 28.46, 2.90],
}
CROSS_DATASET_EM_CI = {
    "Gemma 3 27B + LoRA": [
        (50.39, 51.81),
        (82.52, 83.56),
        (84.36, 85.16),
        (49.84, 51.39),
    ],
    "YOLOv12-m": [(54.03, 55.44), (80.87, 81.92), (82.36, 83.20), (50.97, 52.51)],
    "Gemini 3.1 Pro Preview": [None, (65.58, 66.88), (57.11, 58.20), (21.86, 23.02)],
    "Gemini 3 Flash Preview": [None, (68.49, 69.73), (52.08, 53.18), (11.23, 12.16)],
    "Claude Opus 4.6": [None, (51.67, 53.03), (50.18, 51.31), (16.73, 17.77)],
    "Claude Sonnet 4.6": [None, (30.07, 31.37), (49.49, 50.55), (22.46, 23.65)],
    "Claude Fable 5": [None, (51.60, 57.70), (54.30, 60.50), (17.75, 22.34)],
    "GPT-5.4": [None, (31.40, 32.72), (48.81, 49.92), (17.10, 18.24)],
    "Gemma 3 27B-it (zero-shot)": [
        (9.43, 10.21),
        (6.55, 7.22),
        (27.97, 28.94),
        (2.66, 3.16),
    ],
}
CROSS_DATASET_BASELINES = [13.41, 34.76, 39.63, 16.94]
CROSS_DATASET_NAMES = ["SDSC-EEA", "CholecT50", "PitVis-2023", "SurgVU"]

# Micro-averaged F1 point estimates (Figure 2's definition: 15 evaluable
# v4.2.4 tools for SDSC-EEA, full ontology for the public datasets). Computed by
# experiments/cross_dataset/cross_dataset_f1.py; exact-match values above
# reproduce verbatim, which validates that the same predictions back both.
CROSS_DATASET_F1 = {
    "Gemma 3 27B + LoRA": [81.40, 92.83, 85.65, 74.86],
    "YOLOv12-m": [80.48, 92.37, 82.75, 75.34],
    "Gemini 3.1 Pro Preview": [None, 80.01, 50.41, 43.65],
    "Gemini 3 Flash Preview": [None, 82.88, 48.91, 36.89],
    "Claude Opus 4.6": [None, 71.33, 46.89, 27.38],
    "Claude Sonnet 4.6": [None, 51.87, 46.84, 31.27],
    "Claude Fable 5": [None, 69.87, 53.08, 30.83],
    "GPT-5.4": [None, 48.61, 45.69, 25.99],
    "Gemma 3 27B-it (zero-shot)": [44.87, 33.70, 35.59, 20.69],
}
# Majority-tool micro F1 baseline (predict the single most frequent tool on every
# frame): SDSC-EEA=Suction, CholecT50=grasper, PitVis=suction, SurgVU=needle driver.
CROSS_DATASET_F1_BASELINES = [47.27, 54.03, 43.88, 25.93]

# Cross-dataset per-model colours, in CROSS_DATASET_EM key order.
CROSS_DATASET_PALETTE = [
    NEJM_RED,
    NEJM_GREEN,
    NEJM_BLUE,
    NEJM_SLATE,
    NEJM_ORANGE,
    NEJM_PURPLE,
    NEJM_PINK,
    NEJM_YELLOW,
    NEUTRAL_GREY,
]


def configure_matplotlib() -> None:
    """Apply the camera-ready matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "mathtext.default": "regular",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    """Write one figure as PDF and 300 dpi PNG.

    Args:
        fig: Figure to save.
        stem: File name without an extension.
    """
    fig.savefig(PLOTS_OUT / f"{stem}.pdf")
    fig.savefig(PLOTS_OUT / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {stem}.pdf / {stem}.png")


def fig1_setup_overview(frames_dir: Path) -> None:
    """Ten representative SDSC-EEA frames: five correct and five incorrect.

    Args:
        frames_dir: Directory containing ``correct_{1..5}.png`` and
            ``incorrect_{1..5}.png``.
    """
    print("Figure 1 setup overview")
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.46))
    gs = fig.add_gridspec(
        2, 5, hspace=0.55, wspace=0.06, left=0.03, right=0.99, top=0.94, bottom=0.05
    )
    frames_correct = [
        ("correct_1.png", "Drill; Suction"),
        ("correct_2.png", "Suction"),
        ("correct_3.png", "Drill; Suction"),
        ("correct_4.png", "no tools"),
        ("correct_5.png", "no tools"),
    ]
    frames_incorrect = [
        (
            "incorrect_1.png",
            "Predicted: Curette, Grasper,\nIrrigation, Monopolar, Suction\n"
            "Truth: Drill, Suction",
        ),
        (
            "incorrect_2.png",
            "Predicted: Grasper,\nMonopolar, Suction\n"
            "Truth: Cotton Patty,\nRhoton Dissector, Suction",
        ),
        (
            "incorrect_3.png",
            "Predicted: Curette, Drill,\nSuction, Tissue shaver\n"
            "Truth: Bipolar Forceps, Suction",
        ),
        ("incorrect_4.png", "Predicted: Grasper,\nMonopolar, Suction\nTruth: Suction"),
        ("incorrect_5.png", "Predicted: Monopolar, Suction\nTruth: Rhoton Dissector"),
    ]
    for col, (fname, label) in enumerate(frames_correct):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(imread(frames_dir / fname))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(label, fontsize=6.5, color="black", labelpad=2)
    for col, (fname, label) in enumerate(frames_incorrect):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(imread(frames_dir / fname))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(label, fontsize=6.5, color="black", labelpad=2)
    fig.text(0.005, 0.96, "a", fontsize=8, fontweight="normal", color="black")
    fig.text(0.005, 0.48, "b", fontsize=8, fontweight="normal", color="black")
    fig.savefig(PLOTS_OUT / "setup_overview.pdf")
    fig.savefig(PLOTS_OUT / "setup_overview.png", dpi=300)
    plt.close(fig)
    print("  saved setup_overview.pdf / setup_overview.png (300 dpi)")


def _zero_shot_scatter(
    ax: plt.Axes,
    x_attr: str,
    xlabel: str,
    baseline: float,
    metric: str = "f1",
    xscale: str = "linear",
    include_specialised: bool = True,
) -> None:
    """Scatter zero-shot (and optionally specialist) models on one panel.

    Args:
        ax: Target axis.
        x_attr: "params" or "mmbench" for the x coordinate.
        xlabel: x-axis label.
        baseline: Majority-class baseline level for the horizontal line.
        metric: "f1" (micro-averaged F1, 15 tools) or "em" (exact match).
        xscale: "linear" or "log".
        include_specialised: Whether to overlay the two specialist models.
    """
    y_idx = 5 if metric == "f1" else 4
    plotted_families: set[str] = set()
    for row in ZS_MODELS:
        family = row[1]
        x = row[2] * 1e9 if x_attr == "params" else row[3]
        if x is None:
            continue
        ax.scatter(
            x,
            row[y_idx],
            color=PALETTE_FAMILIES[family],
            marker=MARKERS_FAMILIES[family],
            s=28,
            edgecolor="black",
            linewidth=0.4,
            label=family if family not in plotted_families else None,
        )
        plotted_families.add(family)
    if include_specialised:
        ours_y = 3 if metric == "f1" else 2
        for row in OURS:
            name = row[0]
            ax.scatter(
                row[1],
                row[ours_y],
                color=PALETTE_FAMILIES[name],
                marker=MARKERS_FAMILIES[name],
                s=70 if MARKERS_FAMILIES[name] == "*" else 50,
                edgecolor="black",
                linewidth=0.6,
                label=name,
            )
    ax.axhline(
        baseline,
        linestyle="--",
        color="black",
        linewidth=0.6,
        zorder=0,
        label="SDSC-EEA majority-class baseline",
    )
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if metric == "f1":
        ax.set_ylabel("Micro F1 (15-tool, %)")
        ax.set_ylim(-2, 88)
    else:
        ax.set_ylabel("Exact-match accuracy (%)")
        ax.set_ylim(-2, 60) if include_specialised else ax.set_ylim(-1, 20)


def _fig2_two_panel(metric: str, baseline: float, stem: str) -> None:
    """Render the two-panel zero-shot scaling figure for one metric.

    Args:
        metric: "f1" or "em".
        baseline: Majority-class baseline level for that metric.
        stem: Output filename stem.
    """
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.36))
    _zero_shot_scatter(
        axes[0],
        "params",
        "Model parameters (log scale)",
        baseline,
        metric=metric,
        xscale="log",
        include_specialised=True,
    )
    _zero_shot_scatter(
        axes[1],
        "mmbench",
        "MMBench accuracy (%)",
        baseline,
        metric=metric,
        include_specialised=False,
    )
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    seen = {}
    for handle, legend_label in zip(handles_a, labels_a):
        seen.setdefault(legend_label, handle)
    base_labels = [
        legend_label for legend_label in seen if not legend_label.endswith("(ours)")
    ]
    ours_labels = [
        legend_label for legend_label in seen if legend_label.endswith("(ours)")
    ]
    axes[1].legend(
        [seen[legend_label] for legend_label in base_labels + ours_labels],
        base_labels + ours_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.4,
    )
    axes[0].xaxis.set_minor_locator(mticker.NullLocator())
    log_ticks = [1e8, 1e9, 1e10, 1e11, 1e12]
    axes[0].set_xticks(log_ticks)
    axes[0].set_xticklabels(
        ["100M", "1B", "10B", "100B", "1T"],
        fontsize=7,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.30)
    _add_panel_labels(fig, axes, ("a", "b"))
    save(fig, stem)


def fig2_zero_shot_scaling() -> None:
    """Render the two-panel zero-shot scaling figure for both metrics.

    Micro-F1 is Figure 2. Exact-match accuracy is Supplementary Figure S1.
    """
    print("Figure 2 zero-shot scaling")
    _fig2_two_panel(
        "f1",
        BASELINE_VAL_SDSC_F1,
        "zero_shot_scaling_micro_f1",
    )
    _fig2_two_panel(
        "em",
        BASELINE_VAL_SDSC,
        "zero_shot_scaling_exact_match",
    )


def fig3_cross_dataset() -> None:
    """Grouped bar chart: 4 datasets x 8 models, baselines as horizontal lines per panel."""
    print("Supplementary Figure S3 cross-dataset exact match")
    fig, ax = plt.subplots(figsize=(W_DOUBLE, W_DOUBLE * 0.45))
    n_models = len(CROSS_DATASET_EM)
    n_datasets = len(CROSS_DATASET_NAMES)
    width = 0.85 / n_models
    x_base = np.arange(n_datasets)
    for i, (model, vals) in enumerate(CROSS_DATASET_EM.items()):
        x = x_base + (i - (n_models - 1) / 2) * width
        bars_y = [v if v is not None else 0 for v in vals]
        cis = CROSS_DATASET_EM_CI[model]
        err_low: list[float] = []
        err_high: list[float] = []
        for v, ci in zip(vals, cis):
            if v is None or ci is None:
                err_low.append(0.0)
                err_high.append(0.0)
            else:
                lo, hi = ci
                err_low.append(max(0.0, v - lo))
                err_high.append(max(0.0, hi - v))
        bars = ax.bar(
            x,
            bars_y,
            width=width * 0.95,
            color=CROSS_DATASET_PALETTE[i],
            edgecolor="black",
            linewidth=0.3,
            label=model,
            yerr=[err_low, err_high],
            error_kw=dict(ecolor="black", elinewidth=0.5, capsize=1.2, capthick=0.5),
        )
        for xi, v, bar in zip(x, vals, bars):
            if v is None:
                ax.text(
                    xi,
                    1.5,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="black",
                )
                bar.set_edgecolor("none")
    for j, b in enumerate(CROSS_DATASET_BASELINES):
        line = ax.hlines(
            b,
            x_base[j] - 0.5,
            x_base[j] + 0.5,
            colors="black",
            linestyles="--",
            linewidth=0.8,
        )
        if j == 0:
            line.set_label("Per-dataset majority-class baseline")
    ax.set_xticks(x_base)
    ax.set_xticklabels(CROSS_DATASET_NAMES)
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_ylim(0, 95)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=0.6,
    )
    plt.tight_layout()
    fig.savefig(PLOTS_OUT / "cross_dataset_exact_match.pdf")
    fig.savefig(PLOTS_OUT / "cross_dataset_exact_match.png", dpi=300)
    plt.close(fig)
    print(
        "  saved cross_dataset_exact_match.pdf / cross_dataset_exact_match.png (300 dpi)"
    )


def fig3_cross_dataset_f1() -> None:
    """Grouped bar chart of micro F1 (point estimates): 4 datasets x 9 models.

    Micro-F1 counterpart of ``fig3_cross_dataset`` (Figure 4). No error bars:
    these are point estimates (bootstrap CIs not computed for the F1 metric).
    """
    print("Figure 4 cross-dataset micro-F1")
    fig, ax = plt.subplots(figsize=(W_DOUBLE, W_DOUBLE * 0.45))
    n_models = len(CROSS_DATASET_F1)
    n_datasets = len(CROSS_DATASET_NAMES)
    width = 0.85 / n_models
    x_base = np.arange(n_datasets)
    for i, (model, vals) in enumerate(CROSS_DATASET_F1.items()):
        x = x_base + (i - (n_models - 1) / 2) * width
        bars_y = [v if v is not None else 0 for v in vals]
        bars = ax.bar(
            x,
            bars_y,
            width=width * 0.95,
            color=CROSS_DATASET_PALETTE[i],
            edgecolor="black",
            linewidth=0.3,
            label=model,
        )
        for xi, v, bar in zip(x, vals, bars):
            if v is None:
                ax.text(
                    xi,
                    1.5,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="black",
                )
                bar.set_edgecolor("none")
    for j, b in enumerate(CROSS_DATASET_F1_BASELINES):
        line = ax.hlines(
            b,
            x_base[j] - 0.5,
            x_base[j] + 0.5,
            colors="black",
            linestyles="--",
            linewidth=0.8,
        )
        if j == 0:
            line.set_label("Per-dataset majority-tool F1 baseline")
    ax.set_xticks(x_base)
    ax.set_xticklabels(CROSS_DATASET_NAMES)
    ax.set_ylabel("Micro-averaged F1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=0.6,
    )
    plt.tight_layout()
    fig.savefig(PLOTS_OUT / "cross_dataset_micro_f1.pdf")
    fig.savefig(PLOTS_OUT / "cross_dataset_micro_f1.png", dpi=300)
    plt.close(fig)
    print("  saved cross_dataset_micro_f1.pdf / cross_dataset_micro_f1.png (300 dpi)")


def _wilson_ci(p_pct: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson interval for a percentage.

    Args:
        p_pct: Point estimate in percent.
        n: Sample size.
        z: Normal quantile. Defaults to 1.96.

    Returns:
        Lower and upper bounds in percent.
    """
    p = p_pct / 100.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (centre - margin) * 100.0, (centre + margin) * 100.0


def _read_sweep_csv(path: Path) -> dict:
    """Read a sweep CSV. Returns dict with ranks, train_acc, val_acc, and CIs.

    If explicit ``*_ci_lower`` / ``*_ci_upper`` columns are absent, falls back
    to a Wilson-style 95% CI computed from per-row sample sizes (``n_train``,
    ``n_val``).
    """
    rows: list[dict] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "success":
                continue
            rows.append(row)
    rows.sort(key=lambda r: int(r["lora_r"]))
    out = {
        "ranks": [int(r["lora_r"]) for r in rows],
        "train_acc": [float(r["train_accuracy"]) for r in rows],
        "val_acc": [float(r["val_accuracy"]) for r in rows],
    }
    has_ci = "train_ci_lower" in rows[0] and rows[0].get("train_ci_lower")
    if has_ci:
        out["train_lo"] = [float(r["train_ci_lower"]) for r in rows]
        out["train_hi"] = [float(r["train_ci_upper"]) for r in rows]
        out["val_lo"] = [float(r["val_ci_lower"]) for r in rows]
        out["val_hi"] = [float(r["val_ci_upper"]) for r in rows]
    elif rows[0].get("n_train") and rows[0].get("n_val"):
        train_lo, train_hi, val_lo, val_hi = [], [], [], []
        for r in rows:
            n_t = int(r["n_train"])
            n_v = int(r["n_val"])
            tlo, thi = _wilson_ci(float(r["train_accuracy"]), n_t)
            vlo, vhi = _wilson_ci(float(r["val_accuracy"]), n_v)
            train_lo.append(tlo)
            train_hi.append(thi)
            val_lo.append(vlo)
            val_hi.append(vhi)
        out["train_lo"] = train_lo
        out["train_hi"] = train_hi
        out["val_lo"] = val_lo
        out["val_hi"] = val_hi
    return out


def _sweep_panel(ax: plt.Axes, csv_path: Path, baseline: float, title: str) -> None:
    """Draw one exact-match LoRA-rank panel.

    Args:
        ax: Target axis.
        csv_path: Sweep CSV for one dataset.
        baseline: Majority-class baseline in percent.
        title: Panel title.
    """
    d = _read_sweep_csv(csv_path)
    ranks = d["ranks"]
    train_acc = d["train_acc"]
    val_acc = d["val_acc"]
    x = np.arange(len(ranks))
    width = 0.4
    train_kw = {}
    val_kw = {}
    if "train_lo" in d:
        train_err = [
            [a - lo for a, lo in zip(train_acc, d["train_lo"])],
            [hi - a for a, hi in zip(train_acc, d["train_hi"])],
        ]
        val_err = [
            [a - lo for a, lo in zip(val_acc, d["val_lo"])],
            [hi - a for a, hi in zip(val_acc, d["val_hi"])],
        ]
        train_kw = {"yerr": train_err, "capsize": 1.5, "error_kw": {"elinewidth": 0.5}}
        val_kw = {"yerr": val_err, "capsize": 1.5, "error_kw": {"elinewidth": 0.5}}
    ax.bar(
        x - width / 2,
        train_acc,
        width,
        color=TRAIN_COLOR,
        edgecolor="black",
        linewidth=0.3,
        label="Train",
        **train_kw,
    )
    ax.bar(
        x + width / 2,
        val_acc,
        width,
        color=VAL_COLOR,
        edgecolor="black",
        linewidth=0.3,
        label="Validation",
        **val_kw,
    )
    ax.axhline(
        baseline,
        linestyle="--",
        color="black",
        linewidth=0.6,
        label="Majority-class baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks], rotation=0)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=7)


def fig4_lora_sweep() -> None:
    """Four-panel LoRA rank sweep across all four datasets."""
    print("Supplementary Figure S2 LoRA rank sweep exact match (2x2)")
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.66), sharey=True)
    _sweep_panel(
        axes[0, 0], DATA_DIR / "eea_sweep_results.csv", BASELINE_VAL_SDSC, "SDSC-EEA"
    )
    _sweep_panel(
        axes[0, 1],
        DATA_DIR / "cholect50_sweep_results.csv",
        BASELINE_VAL_CHOLECT50,
        "CholecT50",
    )
    _sweep_panel(
        axes[1, 0],
        DATA_DIR / "pitvis_sweep_results.csv",
        BASELINE_VAL_PITVIS,
        "PitVis-2023",
    )
    _sweep_panel(
        axes[1, 1], DATA_DIR / "surgvu_sweep_results.csv", BASELINE_VAL_SURGVU, "SurgVU"
    )
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    # Force the right column to render its y-axis tick labels even
    # though it shares the y-axis with the left column. Left panels
    # already keep them via the y-axis label; right panels need the
    # explicit override so a reader scanning a single panel can still
    # read off the scale.
    for ax in (axes[0, 1], axes[1, 1]):
        ax.tick_params(axis="y", labelleft=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
        handletextpad=0.4,
        borderpad=0.3,
        columnspacing=1.6,
    )
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    _add_panel_labels(fig, list(axes.flat), ("a", "b", "c", "d"))
    save(fig, "lora_sweep_exact_match")


def _read_sweep_microf1_csv() -> dict:
    """Read the re-scored micro-F1 LoRA sweep summary, grouped by dataset.

    Returns:
        Mapping dataset_name -> dict with ranks, train_f1, val_f1 (all sorted
        by rank). Source: experiments/cross_dataset/lora_sweep_micro_f1.py.
    """
    path = DATA_DIR / "sweep_microf1_summary.csv"
    rows_by_dataset: dict = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows_by_dataset.setdefault(row["dataset"], []).append(row)
    out: dict = {}
    for dataset, rows in rows_by_dataset.items():
        rows.sort(key=lambda r: int(r["lora_r"]))
        out[dataset] = {
            "ranks": [int(r["lora_r"]) for r in rows],
            "train_f1": [float(r["train_micro_f1"]) for r in rows],
            "val_f1": [float(r["val_micro_f1"]) for r in rows],
        }
    return out


def _sweep_panel_f1(ax: plt.Axes, d: dict, baseline: float, title: str) -> None:
    """One micro-F1 sweep panel: train vs validation bars across LoRA ranks."""
    ranks = d["ranks"]
    x = np.arange(len(ranks))
    width = 0.4
    ax.bar(
        x - width / 2,
        d["train_f1"],
        width,
        color=TRAIN_COLOR,
        edgecolor="black",
        linewidth=0.3,
        label="Train",
    )
    ax.bar(
        x + width / 2,
        d["val_f1"],
        width,
        color=VAL_COLOR,
        edgecolor="black",
        linewidth=0.3,
        label="Validation",
    )
    ax.axhline(
        baseline,
        linestyle="--",
        color="black",
        linewidth=0.6,
        label="Majority-tool F1 baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks], rotation=0)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Micro-averaged F1 (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=7)


def fig4_lora_sweep_f1() -> None:
    """Micro-F1 counterpart of the LoRA rank sweep across all four datasets.

    Point estimates (no error bars), matching Figure 4. Baselines are the
    per-dataset majority-tool micro F1 values used in that figure. This panel
    is Figure 3 of the camera-ready paper.
    """
    print("Figure 3 LoRA rank sweep micro-F1 (2x2)")
    sweep = _read_sweep_microf1_csv()
    baselines = dict(
        zip(
            ["SDSC-EEA", "CholecT50", "PitVis-2023", "SurgVU"],
            CROSS_DATASET_F1_BASELINES,
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.66), sharey=True)
    _sweep_panel_f1(axes[0, 0], sweep["eea"], baselines["SDSC-EEA"], "SDSC-EEA")
    _sweep_panel_f1(axes[0, 1], sweep["cholect50"], baselines["CholecT50"], "CholecT50")
    _sweep_panel_f1(
        axes[1, 0], sweep["pitvis"], baselines["PitVis-2023"], "PitVis-2023"
    )
    _sweep_panel_f1(axes[1, 1], sweep["surgvu"], baselines["SurgVU"], "SurgVU")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    for ax in (axes[0, 1], axes[1, 1]):
        ax.tick_params(axis="y", labelleft=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
        handletextpad=0.4,
        borderpad=0.3,
        columnspacing=1.6,
    )
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    _add_panel_labels(fig, list(axes.flat), ("a", "b", "c", "d"))
    save(fig, "lora_sweep_micro_f1")


def main() -> None:
    """Render the bundled figures, skipping Figure 1 unless frames are given."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help=(
            "Directory of Figure 1 frames named correct_{1..5}.png and "
            "incorrect_{1..5}.png. Omit to skip Figure 1."
        ),
    )
    args = parser.parse_args()
    PLOTS_OUT.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    if args.frames_dir is None:
        print(
            "Skipping Figure 1 (setup_overview): pass --frames-dir with "
            "correct_{1..5}.png and incorrect_{1..5}.png. Those frames come "
            "from a private dataset and are not shipped. The other figures "
            "are rendered from the bundled tables."
        )
    else:
        fig1_setup_overview(args.frames_dir)
    fig2_zero_shot_scaling()
    fig3_cross_dataset()
    fig3_cross_dataset_f1()
    fig4_lora_sweep()
    fig4_lora_sweep_f1()
    print("\nAll figures written to", PLOTS_OUT)


if __name__ == "__main__":
    main()
