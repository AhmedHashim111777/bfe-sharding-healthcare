"""Metrics and statistics for all experiments (R5, R6, R7, R8, R10).

Inputs : results/raw/{sharding,dtlz}/<problem>/<alg>/run_XX.{npz,json}
Outputs: results/metrics/*.csv (per run), results/tables/*.csv (summaries + Holm-corrected tests)

Metrics
- sharding: objectives normalised by the ideal/nadir of the reference set R = non-dominated union of all
  final fronts (all algorithms, all runs, heuristics); HV with reference point 1.1 (exact, pymoo);
  IGD+ to R. Per-objective best values of each final front.
- DTLZ1-4: reference front = analytical front on Das-Dennis points; ideal 0, nadir 0.5 (DTLZ1) / 1.
- DTLZ5-6: analytical curve (theta_i = pi/4, i >= 2) plus the non-dominated union of all runs, because the
  unconstrained DTLZ5/6 front is not degenerate for m >= 4 (Ishibuchi et al., 2016).
- HV exact for m <= 5, Monte-Carlo (2e5 samples, fixed seed, common to all runs) for m >= 8.
Statistics: two-sided Wilcoxon rank-sum (Mann-Whitney U) for every pair of algorithms, Holm correction
within each (problem, metric) family, Vargha-Delaney A12.

Usage: python results/scripts/analyse.py [sharding|dtlz|all|tests]
("tests" recomputes the statistics from results/metrics/*_per_run.csv without recomputing indicators.)
"""

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUT_M = ROOT / "results" / "metrics"
OUT_T = ROOT / "results" / "tables"
NDS = NonDominatedSorting()
MAIN = ["NSGA2", "BFE-S", "BFE-D", "SDE", "NSGA3", "MOEAD"]   # SA dropped (Checkpoint C); SDE added (IJIES R2.1)
PEN = ["MOEAD", "MOEAD-P0.1", "MOEAD-P1", "MOEAD-P10"]       # IJIES R1.4 penalty sensitivity
SHARD_DIR = "sharding_all"   # union view: results/raw/sharding + revision algorithms from results/raw/sharding_rev
SA = [("BFE-S", "BFE-S+SA"), ("BFE-D", "BFE-D+SA")]       # reported in the Discussion only
SENS = ["BFE-S0", "BFE-S", "BFE-S1"]
HEUR = ["H-RAND", "H-RR", "H-KM", "H-BW"]
OBJ = ["SVI", "TD", "TH", "SSI", "RI"]


def nd(F, chunk=4000):
    """Non-dominated subset. Chunked (exact): a globally non-dominated point is non-dominated in its chunk, and a
    point of the union of chunk-wise non-dominated sets is globally dominated only by a point of that union.
    Avoids the O(n^2) dominance matrix of pymoo's sorting for the ~3e4 pooled vectors of the revision."""
    F = np.unique(F, axis=0)
    while len(F) > chunk:
        parts = [F[i:i + chunk] for i in range(0, len(F), chunk)]
        G = np.vstack([P[NDS.do(P, only_non_dominated_front=True)] for P in parts])
        if len(G) == len(F):
            break
        F = G
    return F[NDS.do(F, only_non_dominated_front=True)]


def load(exp, prob):
    runs = {}
    for f in sorted(glob.glob(str(RAW / exp / prob / "*" / "run_*.npz"))):
        alg = Path(f).parent.name
        d = np.load(f)
        meta = json.load(open(f[:-4] + ".json"))
        ck = alg in MAIN and "ck_evals" in d          # checkpoints are only needed for Fig. 2 (memory)
        runs.setdefault(alg, []).append(dict(run=meta["run"], F=d["F"], meta=meta,
                                             ck_evals=d["ck_evals"] if ck else None,
                                             ck_sizes=d["ck_sizes"] if ck else None,
                                             ck_F=d["ck_F"] if ck else None))
    for a in runs:
        runs[a].sort(key=lambda r: r["run"])
    return runs


class Indicator:
    def __init__(self, ref_front, ideal, nadir, m, mc=False, seed=12345):
        self.ideal, self.span = ideal, np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        self.Rn = self.norm(ref_front)
        self.igd = IGDPlus(self.Rn)
        self.m, self.mc = m, mc
        if mc:
            self.samples = np.random.default_rng(seed).uniform(0, 1.1, size=(200_000, m))
        else:
            self.hv = HV(ref_point=np.full(m, 1.1))

    def norm(self, F):
        return (np.asarray(F) - self.ideal) / self.span

    def __call__(self, F):
        if len(F) == 0:
            return 0.0, np.inf
        Fn = self.norm(F)
        igd = float(self.igd(Fn))
        Fn = Fn[np.all(Fn <= 1.1, axis=1)]
        if len(Fn) == 0:
            return 0.0, igd
        if self.mc:
            dom = np.zeros(len(self.samples), bool)
            for f in Fn:
                dom |= np.all(self.samples >= f, axis=1)
            hv = dom.mean() * 1.1 ** self.m
        else:
            hv = float(self.hv(Fn))
        return hv, igd


def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()
    return gt / (len(x) * len(y))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (len(p) - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_tests(df, metric, algs, larger_better):
    rows = []
    for a, b in combinations(algs, 2):
        x, y = df.loc[df.alg == a, metric].values, df.loc[df.alg == b, metric].values
        if len(x) < 2 or len(y) < 2:
            continue
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            p = 1.0
        else:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        e = a12(x, y) if larger_better else a12(-x, -y)
        rows.append(dict(A=a, B=b, metric=metric, p=p, A12=e))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm(t.p.values)
        t["better"] = np.where(t.p_holm >= 0.05, "=", np.where(t.A12 > 0.5, t.A, t.B))
    return t


def summary(df, cols, by=("problem", "alg")):
    g = df.groupby(list(by))[cols]
    s = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.size().rename("n"))
    return s.reset_index()


def front_extremes(F):
    """Best value of each objective in a front (TH reported positive)."""
    return dict(SVI=F[:, 0].min(), TD=F[:, 1].min(), TH=-F[:, 2].min(), SSI=F[:, 3].min(), RI=F[:, 4].min())


def knee(Fn_front):
    """Solution closest to the ideal point in the normalised space."""
    return int(np.argmin(np.linalg.norm(Fn_front, axis=1)))


def _one_sharding(prob):
    per_run, curves, knees, cover = [], [], [], []
    runs = load(SHARD_DIR, prob)
    allF = np.vstack([r["F"] for a in runs for r in runs[a] if len(r["F"])])
    R = nd(allF)
    ind = Indicator(R, R.min(0), R.max(0), 5)
    for a, rs in runs.items():
        for r in rs:
            hv, igd = ind(r["F"])
            row = dict(problem=prob, alg=a, run=r["run"], HV=hv, IGDp=igd, n_front=len(r["F"]),
                       n_eval=r["meta"]["n_eval"], runtime_s=r["meta"]["runtime_s"],
                       feasible=r["meta"].get("feasible", True))
            if len(r["F"]):
                row.update(front_extremes(r["F"]))
                k = knee(ind.norm(r["F"]))
                kf = r["F"][k]
                knees.append(dict(problem=prob, alg=a, run=r["run"], SVI=kf[0], TD=kf[1], TH=-kf[2],
                                  SSI=kf[3], RI=kf[4]))
            per_run.append(row)
            if r["ck_evals"] is not None and len(r["ck_evals"]) and a in MAIN:   # Fig. 2 algorithms only
                off = np.concatenate([[0], np.cumsum(r["ck_sizes"])])
                for i, e in enumerate(r["ck_evals"]):
                    Fi = r["ck_F"][off[i]:off[i + 1]]
                    curves.append(dict(problem=prob, alg=a, run=r["run"], n_eval=int(e), HV=ind(Fi)[0]))
    # set coverage C(A,B), runs paired by index
    for A, B in [(x, y) for x in MAIN for y in MAIN if x != y and x in runs and y in runs]:
        vals = [coverage(ra["F"], rb["F"]) for ra, rb in zip(runs[A], runs[B])]
        cover.append(dict(problem=prob, A=A, B=B, C_mean=np.mean(vals), C_sd=np.std(vals, ddof=1)))
    print(prob, "done", flush=True)
    return per_run, curves, knees, cover


def analyse_sharding():
    """One worker process per instance (the exact 5-D HV routine accumulates memory over ~10^4 calls)."""
    from multiprocessing import get_context
    OUT_M.mkdir(parents=True, exist_ok=True); OUT_T.mkdir(parents=True, exist_ok=True)
    probs = sorted(p.name for p in (RAW / SHARD_DIR).iterdir())
    with get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        res = pool.map(_one_sharding, probs, chunksize=1)
    per_run = [x for r in res for x in r[0]]; curves = [x for r in res for x in r[1]]
    knees = [x for r in res for x in r[2]]; cover = [x for r in res for x in r[3]]
    df = pd.DataFrame(per_run)
    df.to_csv(OUT_M / "sharding_per_run.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_M / "sharding_hv_curves.csv", index=False)
    pd.DataFrame(knees).to_csv(OUT_M / "sharding_knee.csv", index=False)
    summary(df, ["HV", "IGDp", "n_front", "runtime_s", "n_eval"] + OBJ).to_csv(OUT_T / "sharding_summary.csv", index=False)
    summary(pd.DataFrame(knees), OBJ).to_csv(OUT_T / "sharding_knee_summary.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT_T / "sharding_set_coverage.csv", index=False)
    print("sharding done:", len(df), "runs")
    sharding_tests()


def coverage(FA, FB):
    """C(A,B): fraction of B weakly dominated by at least one member of A (Zitzler & Thiele 1999)."""
    if len(FB) == 0:
        return np.nan
    if len(FA) == 0:
        return 0.0
    dom = np.all(FA[:, None, :] <= FB[None, :, :], axis=2).any(axis=0)
    return dom.mean()


def dtlz_front(name, m):
    parts = {3: 40, 5: 12, 8: 6, 10: 5}[m]
    if name in ("dtlz1", "dtlz2", "dtlz3", "dtlz4"):
        rd = get_reference_directions("das-dennis", m, n_partitions=parts)
        if name == "dtlz1":
            return 0.5 * rd
        return rd / np.linalg.norm(rd, axis=1, keepdims=True)
    t = np.linspace(0, np.pi / 2, 2000)
    theta = np.column_stack([t] + [np.full_like(t, np.pi / 4)] * (m - 2))
    F = np.empty((len(t), m))
    for i in range(m):
        f = np.prod(np.cos(theta[:, : m - 1 - i]), axis=1)
        if i > 0:
            f = f * np.sin(theta[:, m - 1 - i])
        F[:, i] = f
    return F


def analyse_dtlz():
    OUT_M.mkdir(parents=True, exist_ok=True); OUT_T.mkdir(parents=True, exist_ok=True)
    per_run, tests, cover = [], [], []
    for prob in sorted(p.name for p in (RAW / "dtlz").iterdir()):
        name, m = prob.split("_m"); m = int(m)
        runs = load("dtlz", prob)
        PF = dtlz_front(name, m)
        if name in ("dtlz5", "dtlz6"):
            allF = np.vstack([r["F"] for a in runs for r in runs[a]] + [PF])
            R = nd(allF)
            ideal, nadir = np.zeros(m), PF.max(0)
        else:
            R = PF
            ideal, nadir = np.zeros(m), np.full(m, 0.5 if name == "dtlz1" else 1.0)
        ind = Indicator(R, ideal, nadir, m, mc=m >= 8)
        for a, rs in runs.items():
            for r in rs:
                hv, igd = ind(r["F"])
                per_run.append(dict(problem=prob, dtlz=name, m=m, alg=a, run=r["run"], HV=hv, IGDp=igd,
                                    n_front=len(r["F"]), runtime_s=r["meta"]["runtime_s"]))
        for A, B in [(x, y) for x in MAIN for y in MAIN if x != y and x in runs and y in runs]:
            vals = [coverage(ra["F"], rb["F"]) for ra, rb in zip(runs[A], runs[B])]
            cover.append(dict(problem=prob, A=A, B=B, C_mean=np.mean(vals), C_sd=np.std(vals, ddof=1)))
        print(prob, "done", flush=True)
    df = pd.DataFrame(per_run)
    df.to_csv(OUT_M / "dtlz_per_run.csv", index=False)
    summary(df, ["HV", "IGDp", "n_front", "runtime_s"]).to_csv(OUT_T / "dtlz_summary.csv", index=False)
    pd.DataFrame(cover).to_csv(OUT_T / "dtlz_set_coverage.csv", index=False)
    print("dtlz done:", len(df), "runs")
    dtlz_tests()


def sa_tests(df, metric, lb):
    """Each BFE variant against its SA counterpart; Holm over the two comparisons."""
    rows = [pairwise_tests(df, metric, [a, b], lb) for a, b in SA if a in set(df.alg) and b in set(df.alg)]
    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows, ignore_index=True)
    t["p_holm"] = holm(t.p.values)
    t["better"] = np.where(t.p_holm >= 0.05, "=", np.where(t.A12 > 0.5, t.A, t.B))
    return t


def sharding_tests():
    df = pd.read_csv(OUT_M / "sharding_per_run.csv")
    tests = []
    for prob, d in df.groupby("problem"):
        algs = set(d.alg)
        fams = [("main", MAIN), ("sensitivity", SENS), ("penalty", PEN)]
        for metric, lb in [("HV", True), ("IGDp", False)]:
            for fam, A in fams:
                t = pairwise_tests(d, metric, [a for a in A if a in algs], lb)
                t.insert(0, "family", fam); t.insert(0, "problem", prob); tests.append(t)
            t = sa_tests(d, metric, lb)
            t.insert(0, "family", "sa"); t.insert(0, "problem", prob); tests.append(t)
        for o, lb in [("SVI", False), ("TD", False), ("TH", True), ("SSI", False), ("RI", False)]:
            t = pairwise_tests(d, o, [a for a in MAIN if a in algs], lb)
            t.insert(0, "family", "objective"); t.insert(0, "problem", prob); tests.append(t)
    pd.concat(tests).to_csv(OUT_T / "sharding_tests.csv", index=False)
    print("sharding tests written")


def dtlz_tests():
    df = pd.read_csv(OUT_M / "dtlz_per_run.csv")
    tests = []
    for prob, d in df.groupby("problem"):
        algs = set(d.alg)
        for metric, lb in [("HV", True), ("IGDp", False)]:
            t = pairwise_tests(d, metric, [a for a in MAIN if a in algs], lb)
            t.insert(0, "family", "main"); t.insert(0, "problem", prob); tests.append(t)
            t = sa_tests(d, metric, lb)
            t.insert(0, "family", "sa"); t.insert(0, "problem", prob); tests.append(t)
    pd.concat(tests).to_csv(OUT_T / "dtlz_tests.csv", index=False)
    print("dtlz tests written")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("sharding", "all"):
        analyse_sharding()
    if what in ("dtlz", "all"):
        analyse_dtlz()
    if what == "tests":
        sharding_tests()
        dtlz_tests()
