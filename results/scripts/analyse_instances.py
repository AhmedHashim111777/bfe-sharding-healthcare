"""Revision R2.2/R1.7: 18 additional networks (code/configs/rev_instances.yaml), 30 runs per algorithm.

Per instance: reference set = non-dominated union of all final fronts of all algorithms and heuristics on that
instance; normalised HV (reference point 1.1) and IGD+ exactly as in analyse.py. Run-level: pairwise
Wilcoxon rank-sum tests with Holm correction within each instance (wins/ties/losses over instances).
Instance-level: mean HV per (instance, algorithm); Friedman test over instances; paired Wilcoxon signed-rank
tests over the 18 instances with Holm correction; between-instance SD of the per-instance mean HV difference
against the pooled within-instance (run-to-run) SD.
Outputs: results/metrics/inst_per_run.csv, results/tables/inst_*.csv
"""
import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "results" / "scripts")); sys.path.insert(0, str(ROOT / "code"))
import analyse as A                                             # noqa: E402
from bfeshard.runner import load_config, build_problem          # noqa: E402

RAW = ROOT / "results" / "raw" / "sharding_inst"
CFG = load_config(ROOT / "code" / "configs" / "rev_instances.yaml")
ALGS = ["NSGA2", "BFE-S", "BFE-D", "BFE-S1", "SDE", "NSGA3"]
HEUR = ["H-RAND", "H-RR", "H-KM", "H-BW"]
PAIRS = [("BFE-S", "NSGA2"), ("BFE-D", "NSGA2"), ("BFE-S", "NSGA3"), ("BFE-D", "NSGA3"), ("BFE-S", "SDE"),
         ("BFE-D", "SDE"), ("BFE-S1", "BFE-S"), ("SDE", "NSGA2"), ("SDE", "NSGA3"), ("BFE-S", "BFE-D")]


def main():
    rows = []
    for ic in CFG["instances"]:
        prob = ic["name"]
        model = build_problem(CFG, prob)[0].model
        runs = {}
        for f in sorted(glob.glob(str(RAW / prob / "*" / "run_*.npz"))):
            d = np.load(f)
            runs.setdefault(Path(f).parent.name, []).append(dict(run=int(f[-6:-4]), F=d["F"], X=d["X"].astype(int)))
        if not runs:
            continue
        R = A.nd(np.vstack([r["F"] for a in runs for r in runs[a] if len(r["F"])]))
        ind = A.Indicator(R, R.min(0), R.max(0), 5)
        hb = [model.details(runs[h][0]["X"][0]) for h in ("H-RR", "H-KM", "H-BW") if h in runs]
        p_bal = min(d["p_sys"] for d in hb) if hb else np.nan
        th_bw = model.details(runs["H-BW"][0]["X"][0])["th"] if "H-BW" in runs else np.nan
        for a, rs in runs.items():
            for r in rs:
                hv, igd = ind(r["F"])
                row = dict(problem=prob, N=ic["N"], Ns=ic["Ns"], p=model.inst.params.p, eps=model.mp.eps,
                           rho=model.mp.rho, hosp=model.inst.params.hospital_share,
                           clusters=int(model.inst.params.n_clusters), alg=a, run=r["run"], HV=hv, IGDp=igd,
                           n_front=len(r["F"]), p_bal=p_bal, TH_BW=th_bw)
                if len(r["F"]) and a in ALGS:
                    pf = model.pf_table
                    ps = np.array([1 - np.prod(1 - pf[np.bincount(x, minlength=model.Ns)]) for x in r["X"]])
                    ok = ps <= p_bal * (1 + 1e-9)
                    cand = np.flatnonzero(ok) if ok.any() else np.array([np.argmin(ps)])
                    i = cand[np.lexsort((r["F"][cand, 1], r["F"][cand, 2]))[0]]
                    kn = A.knee(ind.norm(r["F"]))
                    row.update(rule_adm=bool(ok.any()), rule_TH=-r["F"][i, 2], rule_TD=r["F"][i, 1], rule_psys=ps[i],
                               knee_psys=ps[kn])
                rows.append(row)
        print(prob, "done", flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "results" / "metrics").mkdir(exist_ok=True, parents=True)
    df.to_csv(ROOT / "results" / "metrics" / "inst_per_run.csv", index=False)
    T = ROOT / "results" / "tables"
    s = A.summary(df, ["HV", "IGDp", "n_front", "rule_TH", "rule_TD", "rule_psys", "knee_psys"])
    s.to_csv(T / "inst_summary.csv", index=False)
    # run-level tests within each instance
    tests = []
    for prob, d in df.groupby("problem"):
        for m, lb in [("HV", True), ("IGDp", False)]:
            t = A.pairwise_tests(d, m, [a for a in ALGS if a in set(d.alg)], lb)
            t.insert(0, "problem", prob); tests.append(t)
    tests = pd.concat(tests); tests.to_csv(T / "inst_tests.csv", index=False)
    wtl = []
    for A_, B_ in PAIRS:
        for m in ("HV", "IGDp"):
            t = tests[((tests.A == A_) & (tests.B == B_)) | ((tests.A == B_) & (tests.B == A_))]
            t = t[t.metric == m]
            w = int((t.better == A_).sum()); l = int((t.better == B_).sum())
            wtl.append(dict(A=A_, B=B_, metric=m, wins=w, ties=len(t) - w - l, losses=l))
    pd.DataFrame(wtl).to_csv(T / "inst_wtl.csv", index=False)
    # instance-level analysis on mean HV
    M = df[df.alg.isin(ALGS)].groupby(["problem", "alg"]).HV.mean().unstack()
    W = df[df.alg.isin(ALGS)].groupby(["problem", "alg"]).HV.std(ddof=1).unstack()
    fr = friedmanchisquare(*[M[a].values for a in ALGS])
    ranks = M[ALGS].rank(axis=1, ascending=False).mean()
    inst = []
    for A_, B_ in PAIRS:
        diff = M[A_] - M[B_]
        p = wilcoxon(M[A_], M[B_]).pvalue if (diff != 0).any() else 1.0
        inst.append(dict(A=A_, B=B_, n_inst=len(diff), mean_diff=diff.mean(), between_sd=diff.std(ddof=1),
                         within_sd=np.sqrt(((W[A_] ** 2 + W[B_] ** 2) / 2).mean()), n_A_better=int((diff > 0).sum()),
                         p=p))
    inst = pd.DataFrame(inst); inst["p_holm"] = A.holm(inst.p.values)
    inst.to_csv(T / "inst_instance_level.csv", index=False)
    pd.DataFrame(dict(alg=ranks.index, mean_rank=ranks.values)).assign(
        friedman_chi2=fr.statistic, friedman_p=fr.pvalue).to_csv(T / "inst_friedman.csv", index=False)
    M.to_csv(T / "inst_mean_hv.csv")
    print(M.round(3).to_string()); print(inst.round(4).to_string()); print(ranks, fr)


if __name__ == "__main__":
    main()
