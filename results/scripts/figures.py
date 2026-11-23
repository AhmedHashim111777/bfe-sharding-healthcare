"""Journal figures from results/metrics/*.csv (R8). Output: manuscript/figs/*.pdf + 300 dpi *.png.

Fig. 2 (replacement): normalised HV of the feasible non-dominated set versus evaluations,
mean over 30 runs with +-1 SD band, one panel per (N, Ns) instance.
Palette: reference categorical order (validated: adjacent CVD dE >= 9.1); every series also has its own
line style and marker because three slots are below 3:1 contrast on white.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

for _f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(_f))      # Times-metric fallback where Times New Roman is absent

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "manuscript" / "figs"
ORDER = ["NSGA2", "BFE-S", "BFE-D", "SDE", "NSGA3", "MOEAD"]   # SA dropped (Checkpoint C); SDE added (IJIES R2.1)
LABEL = {"NSGA2": "AC (NSGA-II)", "BFE-S": "AS", "BFE-D": "AD", "BFE-S+SA": "AS(SA)",
         "BFE-D+SA": "AD(SA)", "NSGA3": "NSGA-III", "MOEAD": "MOEA/D", "SDE": "NSGA-II+SDE"}
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#8a5a2b", "#008300", "#6250d6"]))
STYLE = dict(zip(ORDER, ["-", "--", "-.", (0, (5, 2, 1, 2)), (0, (3, 1)), (0, (1, 1))]))
MARK = dict(zip(ORDER, ["o", "s", "^", "D", "P", "X"]))
CK = 2500  # checkpoint_every in code/configs/sharding.yaml
TITLE = {"N100_S4": "$N=100,\\ N_s=4$", "N200_S8": "$N=200,\\ N_s=8$", "N500_S16": "$N=500,\\ N_s=16$"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def fig_hv_curves():
    c = pd.read_csv(ROOT / "results" / "metrics" / "sharding_hv_curves.csv")
    probs = [p for p in TITLE if p in set(c.problem)]
    fig, axes = plt.subplots(1, len(probs), figsize=(7.0, 2.3), sharey=False)
    for ax, p in zip(axes, probs):
        for a in ORDER:
            d = c[(c.problem == p) & (c.alg == a)]
            if d.empty:
                continue
            d = d.assign(n_eval=(d.n_eval // CK) * CK)          # runs log at 2500k or 2500k+99 evaluations
            g = d.groupby("n_eval").HV
            m, s = g.mean(), g.std(ddof=1)
            x = m.index.values / 1000
            ax.fill_between(x, m - s, m + s, color=COLOR[a], alpha=0.12, linewidth=0)
            ax.plot(x, m.values, color=COLOR[a], ls=STYLE[a], lw=1.2, marker=MARK[a], ms=3,
                    markevery=max(1, len(x) // 5), label=LABEL[a])
        ax.set_title(TITLE[p])
        ax.set_xlabel("Function evaluations ($\\times10^3$)")
        ax.grid(axis="y", color="#e4e3df", lw=0.5)
    axes[0].set_ylabel("Normalised hypervolume")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(ORDER), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_hv_convergence")


if __name__ == "__main__":
    fig_hv_curves()
