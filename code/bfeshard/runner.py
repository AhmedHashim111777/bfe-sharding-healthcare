"""Run experiments from a YAML config; one .npz + .json per (problem, algorithm, run).

Usage:
    python -m bfeshard.runner configs/sharding.yaml [--runs 30] [--algs NSGA2,BFE-D] [--jobs 8]
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .algorithms import NormPenaltyProblem, PenaltyProblem, make_algorithm
from .instance import NetworkParams, make_instance
from .model import ModelParams, ShardingModel, ShardingProblem, admissible_sizes
from .heuristics import HEURISTICS, SeededSampling, heuristic_solution
from .operators import ShardRepair, shard_neighbour

ROOT = Path(__file__).resolve().parents[2]


def git_state():
    try:
        h = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "code/"],
                                             text=True).strip())
        return h, dirty
    except Exception:
        return "unknown", True


class Checkpoints(Callback):
    def __init__(self, every):
        super().__init__()
        self.every, self.next = every, every
        self.evals, self.fronts = [], []

    def notify(self, algorithm):
        n = algorithm.evaluator.n_eval
        while n >= self.next:
            self.evals.append(n)
            self.fronts.append(feasible_front(algorithm.pop)[0])
            self.next += self.every


def feasible_front(pop):
    F, X = pop.get("F"), pop.get("X")
    cv = pop.get("CV")
    ok = cv[:, 0] <= 0 if cv is not None and cv.size else np.ones(len(F), bool)
    F, X = F[ok], X[ok]
    if len(F) == 0:
        return np.empty((0, pop[0].F.shape[0])), np.empty((0, X.shape[1] if X.ndim > 1 else 0))
    F, idx = np.unique(F, axis=0, return_index=True)
    X = X[idx]
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return F[nd], X[nd]


def build_problem(cfg, key):
    """Return (problem, repair, neighbour, meta) for problem key."""
    if cfg["experiment"] == "sharding":
        ic = next(i for i in cfg["instances"] if i["name"] == key)
        net = {**cfg.get("network", {}), **ic.get("network", {})}       # per-instance overrides (R2.2)
        mod = {**cfg.get("model", {}), **ic.get("model", {})}
        prm = NetworkParams(**{k: float(v) for k, v in net.items()})
        mp = ModelParams(**{k: (int(v) if k == "block_tx" else float(v)) for k, v in mod.items()})
        inst = make_instance(ic["N"], ic["Ns"], ic["seed"], prm, name=key)
        model = ShardingModel(inst, mp)
        prob = ShardingProblem(model)
        adm = admissible_sizes(inst.n_nodes, prm.p, mp.eps)
        repair = ShardRepair(inst.n_nodes, inst.n_shards, prm.p, mp.eps, seed=ic["seed"])
        neighbour = lambda x, rng: shard_neighbour(x, inst.n_shards, adm, rng)
        return prob, repair, neighbour, dict(instance=inst.summary(), model=asdict(mp))
    name, m = key.split("_m")
    m = int(m)
    k = 5 if name == "dtlz1" else 10            # Deb et al. (2005): n_var = m + k - 1
    prob = get_problem(name, n_var=m + k - 1, n_obj=m)
    span = prob.xu - prob.xl

    def neighbour(x, rng):
        y = np.array(x, dtype=float)
        i = rng.integers(len(y))
        y[i] = np.clip(y[i] + rng.normal(0, 0.1 * span[i]), prob.xl[i], prob.xu[i])
        return y
    return prob, None, neighbour, dict(dtlz=name, n_obj=m, n_var=prob.n_var)


def problem_keys(cfg):
    if cfg["experiment"] == "sharding":
        return [i["name"] for i in cfg["instances"]]
    return [f"{p}_m{m}" for p in cfg["problems"] for m in cfg["n_obj"]]


def run_one(args):
    cfg, key, alg, run = args
    out_dir = ROOT / cfg["out_dir"] / key / alg
    out_dir.mkdir(parents=True, exist_ok=True)
    f_npz = out_dir / f"run_{run:02d}.npz"
    if f_npz.exists():
        return str(f_npz), "skipped"
    seed = int(cfg["seed_base"]) + run
    prob, repair, neighbour, meta = build_problem(cfg, key)
    budget = int(cfg["max_evals"])
    commit, dirty = git_state()
    if alg in HEURISTICS:
        inst = prob.model.inst
        t0 = time.time()
        x = heuristic_solution(alg, inst, prob.model.mp.eps, np.random.default_rng(seed))
        d = prob.model.details(x)
        np.savez_compressed(f_npz, F=d["F"][None, :], X=x[None, :].astype(np.int16), G=d["G"])
        info = dict(problem=key, algorithm=alg, run=run, seed=seed, n_eval=1, runtime_s=time.time() - t0,
                    feasible=bool(np.all(d["G"] <= 0)), k=d["k"].tolist(), p_sys=float(d["p_sys"]),
                    commit=commit, dirty=dirty, config_sha1=cfg["_sha1"], config_file=cfg["_file"], **meta)
        with open(out_dir / f"run_{run:02d}.json", "w") as fh:
            json.dump(info, fh, indent=1, default=float)
        return str(f_npz), "heuristic"
    sampling = None
    if cfg["experiment"] == "sharding" and int(cfg.get("init_seeded", 0)) > 0:
        sampling = SeededSampling(prob.model.inst, prob.model.mp.eps, int(cfg["init_seeded"]), seed)
    algo = make_algorithm(alg, "sharding" if cfg["experiment"] == "sharding" else "dtlz",
                          prob.n_obj, int(cfg["pop_size"]), repair=repair, neighbour=neighbour,
                          max_evals=budget, sa_cfg=cfg.get("sa"), seed=seed,
                          sampling=sampling)
    run_prob = PenaltyProblem(prob) if (alg == "MOEAD" and prob.has_constraints()) else prob
    if alg.startswith("MOEAD-P"):
        inst = prob.model.inst
        HF = np.array([prob.model.details(heuristic_solution(h, inst, prob.model.mp.eps,
                                                             np.random.default_rng(0)))["F"] for h in HEURISTICS])
        scale = HF.max(0) - HF.min(0)
        scale = np.where(scale > 1e-12, scale, np.maximum(np.abs(HF.mean(0)), 1e-12))
        run_prob = NormPenaltyProblem(prob, scale, float(alg[len("MOEAD-P"):]))
    cb = Checkpoints(int(cfg["checkpoint_every"]))
    t0 = time.time()
    res = minimize(run_prob, algo, ("n_eval", budget), seed=seed, callback=cb, verbose=False,
                   copy_algorithm=False)
    runtime = time.time() - t0
    pop = res.algorithm.pop
    if run_prob is not prob:   # re-evaluate MOEA/D population on the constrained problem
        from pymoo.core.evaluator import Evaluator
        from pymoo.core.population import Population
        pop = Population.new(X=pop.get("X"))
        Evaluator().eval(prob, pop)
    F, X = feasible_front(pop)
    np.savez_compressed(f_npz, F=F, X=X.astype(np.int16) if cfg["experiment"] == "sharding" else X,
                        ck_evals=np.array(cb.evals),
                        ck_sizes=np.array([len(f) for f in cb.fronts]),
                        ck_F=np.concatenate(cb.fronts) if cb.fronts else np.empty((0, prob.n_obj)))
    info = dict(problem=key, algorithm=alg, run=run, seed=seed, n_eval=int(res.algorithm.evaluator.n_eval),
                n_gen=int(res.algorithm.n_gen), runtime_s=runtime, n_front=int(len(F)),
                commit=commit, dirty=dirty, config_sha1=cfg["_sha1"], config_file=cfg["_file"],
                sa_log=getattr(res.algorithm, "sa_log", None),
                repair_stats=dict(repair.stats) if repair is not None and hasattr(repair, "stats") else None,
                eval_stats=dict(getattr(prob, "eval_stats", {})) or None, **meta)
    with open(out_dir / f"run_{run:02d}.json", "w") as fh:
        json.dump(info, fh, indent=1, default=float)
    return str(f_npz), f"{runtime:.1f}s n_eval={info['n_eval']} |front|={len(F)}"


def load_config(path):
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    cfg["_sha1"] = hashlib.sha1(text.encode()).hexdigest()
    cfg["_file"] = str(Path(path))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--algs", default=None)
    ap.add_argument("--problems", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    cfg = load_config(a.config)
    runs = a.runs or int(cfg["runs"])
    algs = a.algs.split(",") if a.algs else cfg["algorithms"]
    keys = a.problems.split(",") if a.problems else problem_keys(cfg)
    tasks = [(cfg, k, al, r) for k in keys for al in algs for r in range(runs)]
    print(f"{len(tasks)} runs, {a.jobs} workers", flush=True)
    with get_context("spawn").Pool(a.jobs) as pool:
        for i, (f, msg) in enumerate(pool.imap_unordered(run_one, tasks), 1):
            print(f"[{i}/{len(tasks)}] {Path(f).relative_to(ROOT)} {msg}", flush=True)


if __name__ == "__main__":
    main()
