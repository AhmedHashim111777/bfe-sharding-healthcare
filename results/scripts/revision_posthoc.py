"""IJIES 20267522 revision: post-hoc analyses of the final fronts (no new optimisation runs).

  psys      R1.7  system compromise probability of knee solutions and of every front member
  sim       R1.2, R2.5  message-level PBFT simulation and offered-load model for the knee solutions;
                  rank agreement between analytical and simulated metrics on the reference set
  rule      R2.6  operational selection rule on every final front
  loo       R1.6  leave-one-algorithm-out reference sets for the sharding HV ranking
  refstats  R1.6  reference-set sizes, ideal/nadir, Das-Dennis resolution, MC error of the HV estimate

Inputs: results/raw/sharding_all/<problem>/<alg>/run_XX.npz (union view of results/raw/sharding and the
revision runs), code/configs/sharding.yaml (instances). Outputs: results/tables/rev_*.csv
Usage: python results/scripts/revision_posthoc.py [psys|sim|rule|loo|refstats|all]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "results" / "scripts"))
from bfeshard.runner import load_config, build_problem          # noqa: E402
from bfeshard.validation import simulate_config, offered_load   # noqa: E402
from bfeshard.heuristics import heuristic_solution, h_contrib   # noqa: E402
from bfeshard.operators import repair_sizes                     # noqa: E402
from bfeshard.model import admissible_sizes                     # noqa: E402
import analyse as A                                             # noqa: E402

RAW = ROOT / "results" / "raw" / "sharding_all"
OUT = ROOT / "results" / "tables"
CFG = load_config(ROOT / "code" / "configs" / "sharding.yaml")
ALGS = ["NSGA2", "BFE-S", "BFE-D", "BFE-S1", "SDE", "NSGA3", "MOEAD"]
HEUR = ["H-RAND", "H-RR", "H-KM", "H-BW"]
SIZES = dict(A=(8e6, 8e6), B=(8e6, 8e3))     # (pre-prepare, vote) message sizes [bit]


def load_runs(prob):
    runs = {}
    for f in sorted(glob.glob(str(RAW / prob / "*" / "run_*.npz"))):
        alg = Path(f).parent.name
        d = np.load(f)
        runs.setdefault(alg, []).append(dict(run=int(f[-6:-4]), F=d["F"], X=d["X"].astype(int)))
    return runs


def reference(runs, exclude=()):
    allF = np.vstack([r["F"] for a in runs if a not in exclude for r in runs[a] if len(r["F"])])
    R = A.nd(allF)
    return R, A.Indicator(R, R.min(0), R.max(0), 5)


def knee_index(ind, F):
    return A.knee(ind.norm(F))


def psys_of(model, X):
    pf = model.pf_table
    out = []
    for x in np.atleast_2d(X):
        k = np.bincount(x, minlength=model.Ns)
        out.append(1 - np.prod(1 - pf[k]))
    return np.array(out)


def hcc(model, seed=0):
    """ContribChain-style allocation (R2.4), repaired to admissible sizes like the other heuristics."""
    inst = model.inst
    x = h_contrib(inst, np.random.default_rng(seed), model)
    return repair_sizes(x, inst.n_shards, admissible_sizes(inst.n_nodes, inst.params.p, model.mp.eps),
                        np.random.default_rng(seed))


def problems():
    return [i["name"] for i in CFG["instances"]]


def run_psys():
    rows = []
    for prob in problems():
        model = build_problem(CFG, prob)[0].model
        runs = load_runs(prob)
        R, ind = reference(runs)
        for a, rs in runs.items():
            for r in rs:
                if not len(r["F"]):
                    continue
                ps = psys_of(model, r["X"])
                kn = knee_index(ind, r["F"])
                rows.append(dict(problem=prob, alg=a, run=r["run"], psys_knee=ps[kn], svi_knee=r["F"][kn, 0],
                                 psys_min=ps.min(), psys_max=ps.max()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rev_psys_per_run.csv", index=False)
    s = A.summary(df, ["psys_knee", "svi_knee", "psys_min", "psys_max"])
    s.to_csv(OUT / "rev_psys_summary.csv", index=False)
    # tolerance table: smallest admissible size and the largest feasible shard count per network size
    from bfeshard.model import p_fail
    tol = []
    for p in (0.1, 0.15, 0.2):
        for eps in (0.1, 0.05, 0.01, 0.001):
            k = np.arange(1, 501)
            ok = p_fail(k, p) <= eps
            kall = next(kk for kk in range(1, 501) if ok[kk - 1:].all())
            for N in (100, 200, 500):
                ns = N // kall
                pbal = 1 - (1 - p_fail(N // max(ns, 1), p)) ** ns if ns >= 1 else np.nan
                tol.append(dict(p=p, eps=eps, k_min_all=kall, N=N, max_shards=ns, psys_balanced=pbal))
    pd.DataFrame(tol).to_csv(OUT / "rev_tolerance.csv", index=False)
    print(s.to_string())


def run_sim(n_rank=300):
    rows, agree = [], []
    rng = np.random.default_rng(20261116)
    for prob in problems():
        model = build_problem(CFG, prob)[0].model
        runs = load_runs(prob)
        R, ind = reference(runs)
        xr = heuristic_solution("H-RR", model.inst, model.mp.eps, np.random.default_rng(0))
        lam_load = {s: 0.5 * simulate_config(xr, model, *SIZES[s])["lam_star"] for s in SIZES}
        for a in [a for a in ALGS + HEUR if a in runs]:
            for r in runs[a]:
                if not len(r["F"]):
                    continue
                kn = knee_index(ind, r["F"])
                x = r["X"][kn]
                d = model.details(x)
                row = dict(problem=prob, alg=a, run=r["run"], TD=d["td"], TH=d["th"], SVI=d["svi"], psys=d["p_sys"])
                for s, (spp, sv) in SIZES.items():
                    sim = simulate_config(x, model, spp, sv)
                    ol = offered_load(sim, model, lam_load[s], np.random.default_rng(r["run"]))
                    row.update({f"TDsim_{s}": sim["td_sim"], f"lamstar_{s}": sim["lam_star"],
                                f"p95_{s}": ol["p95"], f"stable_{s}": ol["stable"]})
                rows.append(row)
        xc = hcc(model); d = model.details(xc)
        row = dict(problem=prob, alg="H-CC", run=0, TD=d["td"], TH=d["th"], SVI=d["svi"], psys=d["p_sys"])
        for s_, (spp, sv) in SIZES.items():
            sim = simulate_config(xc, model, spp, sv)
            ol = offered_load(sim, model, lam_load[s_], np.random.default_rng(0))
            row.update({f"TDsim_{s_}": sim["td_sim"], f"lamstar_{s_}": sim["lam_star"], f"p95_{s_}": ol["p95"],
                        f"stable_{s_}": ol["stable"]})
        rows.append(row)
        # rank agreement on solutions of the pooled final fronts (one random sample per instance)
        allX = np.vstack([r["X"] for a in runs for r in runs[a] if len(r["F"])])
        allF = np.vstack([r["F"] for a in runs for r in runs[a] if len(r["F"])])
        _, idx = np.unique(allF, axis=0, return_index=True)
        idx = rng.choice(idx, size=min(n_rank, len(idx)), replace=False)
        for s, (spp, sv) in SIZES.items():
            sims = [simulate_config(allX[i], model, spp, sv) for i in idx]
            rho_td = spearmanr(allF[idx, 1], [q["td_sim"] for q in sims]).correlation
            rho_th = spearmanr(-allF[idx, 2], [q["lam_star"] for q in sims]).correlation
            agree.append(dict(problem=prob, sizes=s, n=len(idx), spearman_TD=rho_td, spearman_TH_lamstar=rho_th,
                              lam_load=lam_load[s]))
        print(prob, "sim done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rev_sim_knee_per_run.csv", index=False)
    cols = [c for c in df.columns if c not in ("problem", "alg", "run") and not c.startswith("stable")]
    s = A.summary(df.replace([np.inf], np.nan), cols)
    for st in SIZES:
        s = s.merge(df.groupby(["problem", "alg"])[f"stable_{st}"].mean().rename(f"stable_frac_{st}").reset_index())
    s.to_csv(OUT / "rev_sim_knee_summary.csv", index=False)
    pd.DataFrame(agree).to_csv(OUT / "rev_sim_rank_agreement.csv", index=False)
    tests = []
    for prob, d in df.groupby("problem"):
        algs = [a for a in ALGS if a in set(d.alg) and a != "MOEAD"]
        for m, lb in [("TDsim_B", False), ("lamstar_B", True), ("TDsim_A", False), ("lamstar_A", True)]:
            t = A.pairwise_tests(d, m, algs, lb)
            t.insert(0, "problem", prob); tests.append(t)
    pd.concat(tests).to_csv(OUT / "rev_sim_tests.csv", index=False)
    print(pd.DataFrame(agree).to_string())


def run_rule():
    """Security-first operational rule (R2.6): admit only front members whose system compromise probability
    does not exceed that of the balanced configuration of the instance (p_bal, attained by the round-robin,
    k-means and bandwidth-aware heuristics, whose shard sizes differ by at most one node); among them choose
    the largest throughput TH, ties broken by the smaller TD. If no member is admissible, the member of smallest
    p_sys is chosen and the run is counted as 'relaxed'."""
    rows = []
    for prob in problems():
        model = build_problem(CFG, prob)[0].model
        runs = load_runs(prob)
        hdet = {h: model.details(runs[h][0]["X"][0]) for h in HEUR if h in runs}
        p_bal = min(hdet[h]["p_sys"] for h in ("H-RR", "H-KM", "H-BW"))
        for a in [a for a in ALGS if a in runs]:
            for r in runs[a]:
                F, X = r["F"], r["X"]
                if not len(F):
                    continue
                ps = psys_of(model, X)
                ok = ps <= p_bal * (1 + 1e-9)
                cand = np.flatnonzero(ok) if ok.any() else np.array([np.argmin(ps)])
                i = cand[np.lexsort((F[cand, 1], F[cand, 2]))[0]]       # max TH (F stores -TH), then min TD
                rows.append(dict(problem=prob, alg=a, run=r["run"], admissible=bool(ok.any()), n_adm=int(ok.sum()),
                                 psys=ps[i], SVI=F[i, 0], TD=F[i, 1], TH=-F[i, 2], SSI=F[i, 3], RI=F[i, 4]))
        hdet["H-CC"] = model.details(hcc(model))
        for h, d in hdet.items():
            rows.append(dict(problem=prob, alg=h, run=0, admissible=bool(d["p_sys"] <= p_bal * (1 + 1e-9)), n_adm=1,
                             psys=d["p_sys"], SVI=d["svi"], TD=d["td"], TH=d["th"], SSI=d["ssi"], RI=d["ri"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rev_rule_per_run.csv", index=False)
    s = A.summary(df, ["psys", "SVI", "TD", "TH", "SSI", "RI", "n_adm"])
    s = s.merge(df.groupby(["problem", "alg"]).admissible.mean().rename("adm_frac").reset_index())
    s.to_csv(OUT / "rev_rule_summary.csv", index=False)
    tests = []
    for prob, d in df.groupby("problem"):
        algs = [a for a in ALGS if a in set(d.alg) and a != "MOEAD"]
        for m, lb in [("TH", True), ("TD", False), ("psys", False)]:
            t = A.pairwise_tests(d, m, algs, lb); t.insert(0, "problem", prob); tests.append(t)
    pd.concat(tests).to_csv(OUT / "rev_rule_tests.csv", index=False)
    print(s[["problem", "alg", "psys_mean", "TH_mean", "TH_sd", "TD_mean", "SSI_mean", "RI_mean", "adm_frac"]].to_string())


def run_loo():
    rows = []
    main = [a for a in ALGS if a != "BFE-S1"]
    for prob in problems():
        runs = load_runs(prob)
        variants = [("none", ())] + [(a, (a,)) for a in main if a in runs]
        for name, excl in variants:
            R, ind = reference(runs, excl)
            per = []
            for a in [a for a in main if a in runs]:
                for r in runs[a]:
                    per.append(dict(alg=a, run=r["run"], HV=ind(r["F"])[0]))
            d = pd.DataFrame(per)
            t = A.pairwise_tests(d, "HV", [a for a in main if a in runs], True)
            t.insert(0, "excluded", name); t.insert(0, "problem", prob)
            t["R_size"] = len(R)
            rows.append(t)
        print(prob, "loo done", flush=True)
    df = pd.concat(rows)
    df.to_csv(OUT / "rev_loo_tests.csv", index=False)
    base = df[df.excluded == "none"].set_index(["problem", "A", "B"]).better
    chg = df[df.excluded != "none"].copy()
    chg["base"] = [base.loc[(p, a, b)] for p, a, b in zip(chg.problem, chg.A, chg.B)]
    chg["changed"] = chg.better != chg.base
    chg[chg.changed].to_csv(OUT / "rev_loo_changes.csv", index=False)
    print("verdict changes:", int(chg.changed.sum()), "of", len(chg))


def run_refstats():
    rows = []
    for prob in problems():
        runs = load_runs(prob)
        allF = np.vstack([r["F"] for a in runs for r in runs[a] if len(r["F"])])
        R, _ = reference(runs)
        row = dict(problem=prob, n_pooled=len(allF), n_unique=len(np.unique(allF, axis=0)), R_size=len(R))
        for j, o in enumerate(A.OBJ):
            v = R[:, j] * (-1 if o == "TH" else 1)
            row[f"{o}_ideal"] = v.max() if o == "TH" else v.min()
            row[f"{o}_nadir"] = v.min() if o == "TH" else v.max()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "rev_refstats_sharding.csv", index=False)
    # DTLZ: Das-Dennis resolution and Monte-Carlo standard error of each HV estimate (m >= 8)
    d = pd.read_csv(ROOT / "results" / "metrics" / "dtlz_per_run.csv")
    parts = {3: 40, 5: 12, 8: 6, 10: 5}
    from math import comb
    dd = [dict(m=m, partitions=h, n_points=comb(h + m - 1, m - 1)) for m, h in parts.items()]
    pd.DataFrame(dd).to_csv(OUT / "rev_refstats_dtlz.csv", index=False)
    mc = d[d.m >= 8].copy()
    n = 200_000
    box = 1.1 ** mc.m
    pfrac = (mc.HV / box).clip(0, 1)
    mc["mc_se"] = box * np.sqrt(pfrac * (1 - pfrac) / n)
    g = mc.groupby(["problem", "alg"]).agg(HV_sd=("HV", "std"), mc_se_mean=("mc_se", "mean"), mc_se_max=("mc_se", "max"))
    g["ratio_se_to_sd"] = g.mc_se_max / g.HV_sd.replace(0, np.nan)
    g.reset_index().to_csv(OUT / "rev_mc_error.csv", index=False)
    print(pd.DataFrame(rows).to_string()); print(g.describe().to_string())


def run_repair():
    """R1.3: repair effort and infeasibility rates from the instrumented reruns (results/raw/sharding_rev)."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "raw" / "sharding_rev" / "*" / "*" / "run_*.json"))):
        j = json.load(open(f))
        rs, es = j.get("repair_stats") or {}, j.get("eval_stats") or {}
        rows.append(dict(problem=j["problem"], alg=j["algorithm"], run=j["run"], n_eval=j["n_eval"],
                         repair_calls=rs.get("calls", 0), repair_needed=rs.get("needed", 0),
                         repair_moves=rs.get("moves", 0), repair_failed=rs.get("failed", 0),
                         eval_n=es.get("n", 0), infeasible=es.get("infeasible", 0), inf_security=es.get("security", 0),
                         inf_comm=es.get("communication", 0), inf_resources=es.get("resources", 0),
                         inf_empty=es.get("empty", 0)))
    df = pd.DataFrame(rows)
    df["needed_frac"] = df.repair_needed / df.repair_calls.replace(0, np.nan)
    df["moves_per_repair"] = df.repair_moves / df.repair_needed.replace(0, np.nan)
    df["infeasible_frac"] = df.infeasible / df.eval_n.replace(0, np.nan)
    for c in ("inf_security", "inf_comm", "inf_resources", "inf_empty"):
        df[c + "_frac"] = df[c] / df.eval_n.replace(0, np.nan)
    df.to_csv(OUT / "rev_repair_per_run.csv", index=False)
    cols = ["repair_calls", "needed_frac", "moves_per_repair", "repair_failed", "infeasible_frac",
            "inf_security_frac", "inf_comm_frac", "inf_resources_frac", "inf_empty_frac"]
    s = A.summary(df, cols); s.to_csv(OUT / "rev_repair_summary.csv", index=False)
    print(s[["problem", "alg"] + [c + "_mean" for c in cols]].round(4).to_string())


def run_dtlz_mc(raw=None, seeds=(1, 2, 3, 4, 5)):
    """R1.6: Monte-Carlo HV with independent sample sets (m >= 8) and leave-one-algorithm-out pooled reference
    sets for DTLZ5/DTLZ6. Uses the archived DTLZ fronts available locally (RAW_DTLZ)."""
    import os
    raw = Path(raw or os.environ.get("RAW_DTLZ", ROOT / "results" / "raw" / "dtlz"))
    main = ["NSGA2", "BFE-S", "BFE-D", "NSGA3", "MOEAD"]
    rows, verdicts = [], []
    for pdir in sorted(raw.iterdir()):
        prob = pdir.name; name, m = prob.split("_m"); m = int(m)
        try:
            runs = {a: [np.load(f)["F"] for f in sorted(glob.glob(str(pdir / a / "run_*.npz")))] for a in main}
        except Exception:
            continue                                          # truncated archive member
        if any(len(v) < 30 for v in runs.values()):
            continue
        PF = A.dtlz_front(name, m)
        variants = []
        if m >= 8:
            if name in ("dtlz5", "dtlz6"):
                R = A.nd(np.vstack([F for a in runs for F in runs[a]] + [PF])); ideal, nadir = np.zeros(m), PF.max(0)
            else:
                R = PF; ideal, nadir = np.zeros(m), np.full(m, 0.5 if name == "dtlz1" else 1.0)
            variants += [(f"mc_seed{sd}", A.Indicator(R, ideal, nadir, m, mc=True, seed=sd)) for sd in (12345,) + tuple(seeds)]
        if name in ("dtlz5", "dtlz6"):
            for ex in [None] + main:
                R = A.nd(np.vstack([F for a in runs if a != ex for F in runs[a]] + [PF]))
                variants.append((f"loo_{ex}", A.Indicator(R, np.zeros(m), PF.max(0), m, mc=m >= 8)))
        for vname, ind in variants:
            per = [dict(alg=a, run=i, HV=ind(F)[0]) for a in main for i, F in enumerate(runs[a])]
            d = pd.DataFrame(per)
            for r in per:
                rows.append(dict(problem=prob, variant=vname, **r))
            t = A.pairwise_tests(d, "HV", main, True); t.insert(0, "variant", vname); t.insert(0, "problem", prob)
            verdicts.append(t)
        print(prob, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / "rev_dtlz_mc_per_run.csv", index=False)
    v = pd.concat(verdicts); v.to_csv(OUT / "rev_dtlz_mc_tests.csv", index=False)
    mc = df[df.variant.str.startswith("mc_")]
    if len(mc):
        est = mc.groupby(["problem", "alg", "run"]).HV.std(ddof=1).groupby(["problem", "alg"]).mean().rename("mc_sd")
        runsd = mc[mc.variant == "mc_seed12345"].groupby(["problem", "alg"]).HV.std(ddof=1).rename("run_sd")
        pd.concat([est, runsd], axis=1).reset_index().to_csv(OUT / "rev_dtlz_mc_error.csv", index=False)
    for kind in ("mc_", "loo_"):
        vv = v[v.variant.str.startswith(kind)]
        base = vv[vv.variant == ("mc_seed12345" if kind == "mc_" else "loo_None")].set_index(["problem", "A", "B"]).better
        vv = vv[vv.variant != base.name]
        ch = sum(b != base.loc[(p, a, bb)] for p, a, bb, b in zip(vv.problem, vv.A, vv.B, vv.better)
                 if (p, a, bb) in base.index and vv is not None)
        print(kind, "verdict changes:", ch, "of", len(vv) - len(vv[vv.variant.isin(["mc_seed12345", "loo_None"])]))


def check_identical():
    """The instrumented reruns must reproduce the archived fronts exactly."""
    bad, n = [], 0
    for f in sorted(glob.glob(str(ROOT / "results" / "raw" / "sharding_rev" / "*" / "*" / "run_*.npz"))):
        g = f.replace("sharding_rev", "sharding")
        if not Path(g).exists():
            continue
        n += 1
        a, b = np.load(f)["F"], np.load(g)["F"]
        if a.shape != b.shape or not np.allclose(np.sort(a, 0), np.sort(b, 0)):
            bad.append(f)
    print(f"identical fronts: {n - len(bad)} / {n}"); return bad


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("dtlzmc", run_dtlz_mc), ("check", check_identical), ("repair", run_repair), ("refstats", run_refstats), ("psys", run_psys), ("rule", run_rule), ("loo", run_loo), ("sim", run_sim)]:
        if what in (name, "all"):
            fn()
