"""Revision figures (IJIES 20267522). Output: manuscript/figs/*.pdf + 300 dpi *.png.

fig3_attainment: two-objective projections (SVI-TH and TD-TH) of the final fronts, median (50 %)
attainment curve over 30 runs with the 25-75 % attainment band, per instance; constructive heuristics
as markers (R2.6).
"""
import glob
import sys
from pathlib import Path

import numpy as np
from figures import COLOR, LABEL, MARK, STYLE, TITLE, plt, save

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "sharding_all"
ALGS = ["NSGA2", "BFE-S", "BFE-D", "SDE", "NSGA3"]
HEUR = {"H-RR": ("RR", "v"), "H-KM": ("KM", "<"), "H-BW": ("BW", ">")}


def best_curve(F2, grid):
    """For each x in grid: best (smallest) second objective among points with first objective <= x."""
    o = np.argsort(F2[:, 0])
    xs, ys = F2[o, 0], np.minimum.accumulate(F2[o, 1])
    idx = np.searchsorted(xs, grid, side="right") - 1
    return np.where(idx >= 0, ys[np.clip(idx, 0, None)], np.nan)


def fig_attainment():
    pairs = [((0, 2), "Expected compromised shards SVI", "Throughput TH (tx/s)"),
             ((1, 2), "Confirmation latency TD (s)", "Throughput TH (tx/s)")]
    probs = list(TITLE)
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for c, p in enumerate(probs):
        data = {a: [np.load(f)["F"] for f in sorted(glob.glob(str(RAW / p / a / "run_*.npz")))] for a in ALGS}
        heur = {h: np.load(sorted(glob.glob(str(RAW / p / h / "run_*.npz")))[0])["F"][0] for h in HEUR}
        for r, ((i, j), xl, yl) in enumerate(pairs):
            ax = axes[r, c]
            allx = np.concatenate([F[:, i] for a in data for F in data[a] if len(F)])
            grid = np.linspace(np.percentile(allx, 0.5), np.percentile(allx, 99.5), 300)
            for a in ALGS:
                if not data[a]:
                    continue
                curves = np.array([best_curve(F[:, [i, j]], grid) for F in data[a] if len(F)])
                q25, q50, q75 = (np.nanpercentile(curves, q, axis=0) for q in (25, 50, 75))
                ax.fill_between(grid, -q25, -q75, color=COLOR[a], alpha=0.12, lw=0, step="post")
                ax.step(grid, -q50, where="post", color=COLOR[a], ls=STYLE[a], lw=1.1, label=LABEL[a])
                ax.plot(grid[::60], -q50[::60], ls="none", marker=MARK[a], ms=3, color=COLOR[a])
            for h, (lab, mk) in HEUR.items():
                ax.plot(heur[h][i], -heur[h][j], marker=mk, ms=4.5, ls="none", mfc="white", mec="#52514e", mew=0.8,
                        label=f"heuristic {lab}")
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel(yl)
            if r == 0:
                ax.set_title(TITLE[p])
            ax.grid(color="#e4e3df", lw=0.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig3_attainment")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    fig_attainment()
