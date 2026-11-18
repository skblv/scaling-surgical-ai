"""Plot the hierarchical paired bootstrap and write its LaTeX tables.

Reads the per-dataset outputs of ``experiments/cross_dataset/clustered_bootstrap.py``.

Plots produced per dataset:

    forest_<metric>.{png,pdf}         Headline metric forest plot, models sorted by point estimate.
    paired_delta_<metric>.{png,pdf}   Top-K paired Delta-CIs (largest-magnitude differences).
    per_tool_heatmap_<metric>.png     Per-tool F1 heatmap (model x tool).
    block_length_distribution.png     Per-cluster block lengths histogram.

LaTeX tables:

    headline_table.tex                Per-model headline metrics with [low, high] CIs.
    paired_table.tex                  Paired Delta-CIs and p-values, one row per pair.
    per_tool_table.tex                Per-tool F1 with CIs (long table).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaling_surgical_ai import paths

DEFAULT_OUTPUT_ROOT = paths.experiment_output_dir("clustered_bootstrap")
HEADLINE_METRICS = ("exact_match", "jaccard", "micro_f1", "macro_f1")


def _format_ci(value: float, low: float, high: float, scale: float = 100.0) -> str:
    """Format a value with a percentile CI as ``v.vv [l.ll, h.hh]``."""
    return f"{value * scale:.2f} [{low * scale:.2f}, {high * scale:.2f}]"


def plot_forest(
    headline_metric_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Forest plot of one headline metric across models, sorted by point estimate.

    Args:
        headline_metric_df: Long-form headline summary DataFrame.
        metric: Metric column to plot.
        output_path: Path to write PNG; PDF is written next to it.
        title: Optional title.
    """
    model_metric_df = headline_metric_df[headline_metric_df["metric"] == metric].copy()
    model_metric_df = model_metric_df.sort_values(
        "point_estimate", ascending=True
    ).reset_index(drop=True)
    if len(model_metric_df) == 0:
        return

    fig_height = max(3.0, 0.32 * len(model_metric_df) + 1.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    y_positions = np.arange(len(model_metric_df))
    ax.errorbar(
        model_metric_df["point_estimate"] * 100,
        y_positions,
        xerr=[
            (model_metric_df["point_estimate"] - model_metric_df["percentile_ci_low"])
            * 100,
            (model_metric_df["percentile_ci_high"] - model_metric_df["point_estimate"])
            * 100,
        ],
        fmt="o",
        color="#1f4e79",
        ecolor="#1f4e79",
        elinewidth=1.2,
        capsize=3,
        markersize=4,
    )

    family_colors = {"zero_shot": "#000000", "openrouter": "#7b3294", "lora": "#1a9850"}
    label_colors = [family_colors.get(f, "#000000") for f in model_metric_df["family"]]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_metric_df["model_name"], fontsize=8)
    for tick_label, color in zip(ax.get_yticklabels(), label_colors):
        tick_label.set_color(color)

    ax.set_xlabel(f"{metric} (%)")
    if title is None:
        title = f"Hierarchical paired MBB 95% CIs — {metric}"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", linestyle=":", linewidth=0.5)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=family,
        )
        for family, color in family_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paired_delta(
    paired_delta_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    top_k: int = 30,
) -> None:
    """Forest plot of paired Delta-CIs (top-K by absolute point delta).

    Args:
        paired_delta_df: Long-form paired comparison DataFrame.
        metric: Metric to filter.
        output_path: PNG path; PDF written alongside.
        top_k: Show the top-K pairs by |point_delta|.
    """
    pair_delta_df = paired_delta_df[paired_delta_df["metric"] == metric].copy()
    if len(pair_delta_df) == 0:
        return
    pair_delta_df["abs_delta"] = pair_delta_df["point_delta"].abs()
    pair_delta_df = pair_delta_df.sort_values("abs_delta", ascending=False).head(top_k)
    pair_delta_df = pair_delta_df.sort_values(
        "point_delta", ascending=True
    ).reset_index(drop=True)

    fig_height = max(3.0, 0.30 * len(pair_delta_df) + 1.0)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))

    y_positions = np.arange(len(pair_delta_df))
    ax.errorbar(
        pair_delta_df["point_delta"] * 100,
        y_positions,
        xerr=[
            (pair_delta_df["point_delta"] - pair_delta_df["delta_ci_low"]) * 100,
            (pair_delta_df["delta_ci_high"] - pair_delta_df["point_delta"]) * 100,
        ],
        fmt="o",
        color="#cc4c02",
        ecolor="#cc4c02",
        elinewidth=1.2,
        capsize=3,
        markersize=4,
    )

    labels = [
        f"{row.model_a}  −  {row.model_b}  (p={row.p_value_two_sided:.3g})"
        for row in pair_delta_df.itertuples(index=False)
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Δ {metric} = A − B (%)")
    ax.set_title(
        f"Paired hierarchical MBB Δ-CIs (top {len(pair_delta_df)} by |Δ|) — {metric}",
        fontsize=10,
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_per_tool_heatmap(
    per_tool_f1_df: pd.DataFrame,
    output_path: Path,
    metric_label: str = "F1",
) -> None:
    """Heatmap of per-tool F1 with model rows sorted by mean F1.

    Args:
        per_tool_f1_df: Long-form per-tool F1 DataFrame.
        output_path: PNG path.
        metric_label: Label for the colorbar.
    """
    if per_tool_f1_df.empty:
        return
    model_tool_f1_df = per_tool_f1_df.pivot_table(
        index="model_name",
        columns="tool_name",
        values="f1",
        aggfunc="mean",
    )
    model_tool_f1_df = model_tool_f1_df.loc[
        model_tool_f1_df.mean(axis=1).sort_values(ascending=False).index
    ]

    n_rows, n_cols = model_tool_f1_df.shape
    fig_h = max(4.0, 0.30 * n_rows + 2.0)
    fig_w = max(8.0, 0.42 * n_cols + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        model_tool_f1_df.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0
    )
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(model_tool_f1_df.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(model_tool_f1_df.index, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(metric_label)
    ax.set_title("Per-tool F1 (paired MBB point estimate)", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_block_length_distribution(
    block_length_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Histogram of per-cluster block lengths.

    Args:
        block_length_df: DataFrame with a `block_length` column.
        output_path: PNG path.
    """
    if block_length_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.hist(
        block_length_df["block_length"], bins=20, color="#1f4e79", edgecolor="white"
    )
    ax.set_xlabel("Block length L_k (frames)")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Politis-White block-length distribution per cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_headline_latex(headline_metric_df: pd.DataFrame, output_path: Path) -> None:
    """Write a long-form LaTeX table of headline metrics.

    Args:
        headline_metric_df: Long-form headline summary.
        output_path: Path to .tex file.
    """
    if headline_metric_df.empty:
        return
    model_metric_long_df = headline_metric_df.copy()
    model_metric_long_df["formatted"] = model_metric_long_df.apply(
        lambda r: _format_ci(
            r["point_estimate"], r["percentile_ci_low"], r["percentile_ci_high"]
        ),
        axis=1,
    )
    model_metric_wide_df = model_metric_long_df.pivot(
        index=["model_name", "family"],
        columns="metric",
        values="formatted",
    )
    model_metric_wide_df = model_metric_wide_df.reindex(columns=list(HEADLINE_METRICS))
    model_metric_wide_df = model_metric_wide_df.reset_index()

    with output_path.open("w") as f:
        f.write("% Hierarchical paired MBB headline summary\n")
        f.write("\\begin{tabular}{ll" + "r" * len(HEADLINE_METRICS) + "}\n")
        f.write("\\toprule\n")
        header_cells = ["Model", "Family"] + [
            m.replace("_", " ") for m in HEADLINE_METRICS
        ]
        f.write(" & ".join(header_cells) + " \\\\\n")
        f.write("\\midrule\n")
        for _, row in model_metric_wide_df.iterrows():
            cells = [
                str(row["model_name"]).replace("_", "\\_"),
                str(row["family"]).replace("_", "\\_"),
            ]
            cells.extend(
                str(row[m]) if pd.notna(row[m]) else "--" for m in HEADLINE_METRICS
            )
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_paired_latex(
    paired_delta_df: pd.DataFrame, output_path: Path, top_k: int = 50
) -> None:
    """Write a LaTeX table of paired Delta-CIs (top-K by |Δ|).

    Args:
        paired_delta_df: Long-form paired comparison DataFrame.
        output_path: Path to .tex file.
        top_k: Maximum number of rows to include.
    """
    if paired_delta_df.empty:
        return
    pair_delta_top_df = paired_delta_df.copy()
    pair_delta_top_df["abs_delta"] = pair_delta_top_df["point_delta"].abs()
    pair_delta_top_df = pair_delta_top_df.sort_values(
        "abs_delta", ascending=False
    ).head(top_k)
    pair_delta_top_df["delta_str"] = pair_delta_top_df.apply(
        lambda r: _format_ci(r["point_delta"], r["delta_ci_low"], r["delta_ci_high"]),
        axis=1,
    )

    with output_path.open("w") as f:
        f.write(
            "% Paired hierarchical MBB Delta-CIs (top {} by |Delta|)\n".format(top_k)
        )
        f.write("\\begin{tabular}{lllrr}\n\\toprule\n")
        f.write(
            "Model A & Model B & Metric & Δ (\\%) [95\\% CI] & p-value \\\\\n\\midrule\n"
        )
        for _, row in pair_delta_top_df.iterrows():
            cells = [
                str(row["model_a"]).replace("_", "\\_"),
                str(row["model_b"]).replace("_", "\\_"),
                str(row["metric"]).replace("_", " "),
                row["delta_str"],
                f"{row['p_value_two_sided']:.3g}",
            ]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_per_tool_latex(per_tool_f1_df: pd.DataFrame, output_path: Path) -> None:
    """Write a long-form LaTeX table of per-tool F1 with CIs.

    Args:
        per_tool_f1_df: Long-form per-tool F1 DataFrame.
        output_path: Path to .tex file.
    """
    if per_tool_f1_df.empty:
        return
    tool_f1_long_df = per_tool_f1_df.copy()
    tool_f1_long_df["formatted"] = tool_f1_long_df.apply(
        lambda r: _format_ci(r["f1"], r["percentile_ci_low"], r["percentile_ci_high"]),
        axis=1,
    )
    tool_f1_wide_df = tool_f1_long_df.pivot(
        index="tool_name",
        columns="model_name",
        values="formatted",
    )
    tool_f1_wide_df = tool_f1_wide_df.reset_index()

    with output_path.open("w") as f:
        f.write("% Per-tool F1 with hierarchical paired MBB 95% CIs\n")
        f.write(
            "\\begin{tabular}{l"
            + "r" * (tool_f1_wide_df.shape[1] - 1)
            + "}\n\\toprule\n"
        )
        header_cells = ["Tool"] + [
            str(c).replace("_", "\\_") for c in tool_f1_wide_df.columns[1:]
        ]
        f.write(" & ".join(header_cells) + " \\\\\n\\midrule\n")
        for _, row in tool_f1_wide_df.iterrows():
            cells = [str(row.iloc[0]).replace("_", "\\_")]
            cells.extend(str(v) if pd.notna(v) else "--" for v in row.iloc[1:])
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def render_for_dataset(dataset_dir: Path) -> None:
    """Generate all plots and LaTeX tables for one dataset directory.

    Args:
        dataset_dir: Directory containing headline_summary.csv, per_tool_f1.csv,
            paired_deltas.csv, block_lengths.csv (output of run_dataset).
    """
    headline_metric_df = pd.read_csv(dataset_dir / "headline_summary.csv")
    per_tool_f1_df = pd.read_csv(dataset_dir / "per_tool_f1.csv")
    paired_delta_df = pd.read_csv(dataset_dir / "paired_deltas.csv")
    block_length_df = pd.read_csv(dataset_dir / "block_lengths.csv")

    plots_dir = dataset_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    tables_dir = dataset_dir / "latex"
    tables_dir.mkdir(exist_ok=True)

    for metric in HEADLINE_METRICS:
        plot_forest(headline_metric_df, metric, plots_dir / f"forest_{metric}.png")
        plot_paired_delta(
            paired_delta_df, metric, plots_dir / f"paired_delta_{metric}.png"
        )

    plot_per_tool_heatmap(per_tool_f1_df, plots_dir / "per_tool_heatmap_f1.png")
    plot_block_length_distribution(
        block_length_df, plots_dir / "block_length_distribution.png"
    )

    write_headline_latex(headline_metric_df, tables_dir / "headline_table.tex")
    write_paired_latex(paired_delta_df, tables_dir / "paired_table.tex")
    write_per_tool_latex(per_tool_f1_df, tables_dir / "per_tool_table.tex")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["eea", "pitvis", "cholect50", "surgvu", "all"],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing per-dataset subfolders.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "all":
        for name in ("eea", "pitvis", "cholect50", "surgvu"):
            d = args.output_root / name
            if d.exists():
                print(f"Rendering {name}...")
                render_for_dataset(d)
    else:
        render_for_dataset(args.output_root / args.dataset)
