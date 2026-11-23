"""Algorithm factory. Labels follow the manuscript: AC = NSGA-II (crowding distance),
AS = NSGA-II + static BFE, AD = NSGA-II + dynamic BFE, (SA) = with simulated annealing."""

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from .bfe import RankAndBFESurvival, RankAndSDESurvival
from .sa import NSGA2BFESA

LABELS = {
    "NSGA2": "AC", "BFE-S": "AS", "BFE-D": "AD", "BFE-S+SA": "AS(SA)", "BFE-D+SA": "AD(SA)",
    "BFE-S0": "AS(a=0)", "BFE-S1": "AS(a=1)", "NSGA3": "NSGA-III", "MOEAD": "MOEA/D",
    "SDE": "NSGA-II+SDE", "MOEAD-P0.1": "MOEA/D(l=0.1)", "MOEAD-P1": "MOEA/D(l=1)", "MOEAD-P10": "MOEA/D(l=10)",
}


class PenaltyProblem(Problem):
    """Unconstrained wrapper for MOEA/D: F + 1e3 * CV. Identical to F for feasible solutions."""

    def __init__(self, base):
        self.base = base
        super().__init__(n_var=base.n_var, n_obj=base.n_obj, n_ieq_constr=0, xl=base.xl, xu=base.xu,
                         vtype=getattr(base, "vtype", None))

    def _evaluate(self, X, out, *args, **kwargs):
        r = self.base.evaluate(X, return_values_of=["F", "G"])
        F, G = r
        cv = np.maximum(G, 0).sum(1) if G is not None and G.size else np.zeros(len(X))
        out["F"] = F + 1e3 * cv[:, None]


class NormPenaltyProblem(Problem):
    """Revision R1.4: F_i / s_i + lam * CV, with s_i a per-instance objective scale and
    CV = sum of the positive parts of the dimensionless constraint functions g (eqs. (6), (8), (9), (10)).

    s_i is the range of objective i over the four constructive heuristic solutions of the instance
    (absolute mean if the range is zero), so the scaling uses no information from any optimiser run.
    """

    def __init__(self, base, scale, lam):
        self.base, self.scale, self.lam = base, np.asarray(scale, float), float(lam)
        super().__init__(n_var=base.n_var, n_obj=base.n_obj, n_ieq_constr=0, xl=base.xl, xu=base.xu,
                         vtype=getattr(base, "vtype", None))

    def _evaluate(self, X, out, *args, **kwargs):
        F, G = self.base.evaluate(X, return_values_of=["F", "G"])
        cv = np.maximum(G, 0).sum(1)
        out["F"] = F / self.scale + self.lam * cv[:, None]


def operators(kind):
    if kind == "sharding":
        from .operators import LabelSampling, RandomResetMutation, UniformLabelCrossover
        return dict(sampling=LabelSampling(), crossover=UniformLabelCrossover(prob=0.9),
                    mutation=RandomResetMutation())
    return dict(sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20))


def make_algorithm(name, kind, n_obj, pop_size, repair=None, neighbour=None, max_evals=None,
                   sa_cfg=None, seed=0, sampling=None):
    ops = operators(kind)
    if sampling is not None:
        ops["sampling"] = sampling
    common = dict(pop_size=pop_size, repair=repair, eliminate_duplicates=True, **ops)
    sa_cfg = sa_cfg or {}
    if name == "NSGA2":
        return NSGA2(**common)
    if name in ("BFE-S", "BFE-S0", "BFE-S1", "BFE-D"):
        alpha = {"BFE-S": 0.5, "BFE-S0": 0.0, "BFE-S1": 1.0, "BFE-D": None}[name]
        return NSGA2(survival=RankAndBFESurvival(alpha=alpha, rng_seed=seed), **common)
    if name in ("BFE-S+SA", "BFE-D+SA"):
        alpha = 0.5 if name == "BFE-S+SA" else None
        return NSGA2BFESA(neighbour=neighbour, max_evals=max_evals, sa_seed=seed,
                          survival=RankAndBFESurvival(alpha=alpha, rng_seed=seed), **sa_cfg, **common)
    if name == "SDE":
        return NSGA2(survival=RankAndSDESurvival(rng_seed=seed), **common)
    ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=1)
    if name == "NSGA3":
        return NSGA3(ref_dirs=ref_dirs, **common)
    if name == "MOEAD" or name.startswith("MOEAD-P"):
        common.pop("pop_size"); common.pop("eliminate_duplicates")
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=20, prob_neighbor_mating=0.9, **common)
    raise KeyError(name)
