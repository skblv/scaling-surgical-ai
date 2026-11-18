"""Render Supplementary Figure S7, exact match versus sampling temperature.

The curve is Gemma 3 27B-it zero-shot exact-match accuracy on the SDSC-EEA
validation set, with a 95% Wilson binomial confidence interval. The PDF and
PNG are written to ``figures/output/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PLOTS_DIR = Path(__file__).parent / "output"

RESULTS = [
    (0.0, 20016, 1943),
    (0.1, 20016, 2007),
    (0.2, 20016, 2016),
    (0.3, 20016, 2001),
    (0.4, 20016, 1981),
    (0.5, 20016, 2008),
    (0.6, 20016, 2026),
    (0.7, 20016, 1994),
    (0.8, 20016, 1976),
    (0.9, 20016, 1982),
    (1.0, 20016, 1949),
    (1.1, 20016, 1968),
    (1.2, 20016, 1886),
    (1.3, 20016, 1944),
    (1.4, 20016, 1879),
    (1.5, 20010, 1855),
    (1.6, 20008, 1828),
    (1.7, 20001, 1828),
    (1.8, 20003, 1828),
    (1.9, 19997, 1811),
    (2.0, 19978, 1796),
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    Args:
        k: Number of exact matches.
        n: Number of frames.
        z: Normal quantile for the interval. Defaults to 1.96.

    Returns:
        Lower and upper bounds as percentages.
    """
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (centre - margin) * 100, (centre + margin) * 100


def main() -> None:
    """Write ``accuracy_vs_temperature`` as PDF and PNG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    temps = np.array([r[0] for r in RESULTS])
    accs = np.array([r[2] / r[1] * 100 for r in RESULTS])
    ci_bounds = [wilson_ci(r[2], r[1]) for r in RESULTS]
    ci_lo = np.array([b[0] for b in ci_bounds])
    ci_hi = np.array([b[1] for b in ci_bounds])

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        temps,
        accs,
        marker="o",
        markersize=5,
        linewidth=1.5,
        color="#1f77b4",
        zorder=3,
    )
    ax.fill_between(
        temps,
        ci_lo,
        ci_hi,
        alpha=0.2,
        color="#1f77b4",
        zorder=2,
    )

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Exact match accuracy (%)")
    ax.set_xticks(np.arange(0, 2.1, 0.2))
    ax.set_xlim(-0.05, 2.05)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            str(PLOTS_DIR / f"accuracy_vs_temperature.{ext}"),
            bbox_inches="tight",
            dpi=150,
        )
    plt.close(fig)
    print("Saved accuracy_vs_temperature.pdf")


if __name__ == "__main__":
    main()
