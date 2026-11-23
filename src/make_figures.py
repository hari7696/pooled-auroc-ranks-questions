"""Paper figures from outputs*/results/*.csv.  Usage: python src/make_figures.py [--res outputs_h100/results]

Palette: reference categorical slots 1-2 (blue #2a78d6, orange #eb6834; validated all-pairs in
light mode). Identity is never colour-alone: filled vs hollow markers and a legend.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze import DETECTORS

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "mathtext.fontset": "stix", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _style(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def fig_dumbbell(res: Path):
    """Pooled vs within-question AUROC per detector (mean over runs)."""
    m = pd.read_csv(res / "metrics_mean.csv", index_col=0)
    m = m.sort_values("auroc_w")
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    _style(ax)
    ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.plot([r.auroc_w, r.auroc], [yi, yi], color=GRID, lw=1.6, zorder=2, solid_capstyle="round")
    ax.scatter(m.auroc, y, s=22, color=BLUE, zorder=3, label="Pooled AUROC", edgecolor="white", lw=0.6)
    ax.scatter(m.auroc_w, y, s=22, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=4,
               label=r"Within-question AUROC $A_W$")
    labels = [DETECTORS[d][0] + ("$^{\\dagger}$" if DETECTORS[d][2] else "") for d in m.index]
    ax.set_yticks(y, labels)
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel("AUROC (mean over 18 model-dataset runs)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.2)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "dumbbell.pdf")
    fig.savefig(FIG / "dumbbell.png", dpi=200)
    plt.close(fig)


def fig_validity(res: Path):
    """Best-of-n gain on mixed questions vs pooled AUROC and vs A_W, one point per (run, detector)."""
    met = pd.read_csv(res / "metrics_per_run.csv")
    cells = met[~met.detector.map(lambda d: DETECTORS[d][2])]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
    for ax, col, name in [(axes[0], "auroc", "Pooled AUROC"), (axes[1], "auroc_w", r"Within-question $A_W$")]:
        _style(ax, "both")
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.scatter(cells[col], 100 * cells.gain_mixed, s=9, color=BLUE, alpha=0.75, edgecolor="white",
                   lw=0.3, zorder=3)
        rho = spearmanr(cells[col], cells.gain_mixed)[0]
        ax.text(0.04, 0.95, rf"$\rho={rho:.2f}$", transform=ax.transAxes, va="top", color=INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel(name)
    axes[0].set_ylabel("Best-of-10 gain on\nmixed questions (pts)")
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(FIG / "validity.pdf")
    fig.savefig(FIG / "validity.png", dpi=200)
    plt.close(fig)


def fig_majority(res: Path):
    """A_W on mixed questions whose plurality answer is correct vs wrong (answer-level detectors)."""
    mj = pd.read_csv(res / "majority_split_mean.csv", index_col=0)
    mj = mj.loc[[d for d in DETECTORS if not DETECTORS[d][2] and d in mj.index]]
    mj = mj.sort_values("majority_wrong")
    y = np.arange(len(mj))
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    _style(ax)
    ax.axvline(0.5, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for yi, (_, r) in zip(y, mj.iterrows()):
        ax.plot([r.majority_wrong, r.majority_correct], [yi, yi], color=GRID, lw=1.6, zorder=2)
    ax.scatter(mj.majority_correct, y, s=22, color=BLUE, zorder=3, edgecolor="white", lw=0.6,
               label="Plurality answer correct")
    ax.scatter(mj.majority_wrong, y, s=22, facecolor="white", edgecolor=ORANGE, lw=1.3, zorder=4,
               label="Plurality answer wrong")
    ax.set_yticks(y, [DETECTORS[d][0] for d in mj.index])
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel(r"Within-question AUROC $A_W$ (mean over runs)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0, borderaxespad=0.2)
    ax.tick_params(axis="y", length=0)
    fig.savefig(FIG / "majority.pdf")
    fig.savefig(FIG / "majority.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(ROOT / "outputs_h100" / "results"))
    res = Path(ap.parse_args().res)
    FIG.mkdir(parents=True, exist_ok=True)
    fig_dumbbell(res)
    fig_validity(res)
    fig_majority(res)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
