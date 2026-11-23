"""NSGA-II + BFE + simulated annealing (Algorithms 1-2, eq. (23)), budget-aware.

Fixes relative to the imported original (pymoo fork, nsga2.py):
- acceptance uses fit(.,P) (eq. (23)) instead of the raw sum of objectives;
- the inverted dominance test (which accepted dominated candidates) is removed;
- every SA evaluation is counted by the evaluator, so all algorithms share one budget;
- SA refines the K individuals of highest fit for L moves per generation;
- the temperature is on the scale of fit differences (T0 = 0.1 -> Tmin = 1e-3 over L moves);
- the refined individual replaces its parent only if the parent does not dominate it
  (constraint-domination), so SA cannot discard a better solution.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.dominator import Dominator

from .bfe import BFEReference


def constrained_dominates(a, b):
    """True if individual a constraint-dominates b."""
    ca, cb = a.CV[0], b.CV[0]
    if ca > 0 or cb > 0:
        return ca < cb
    return Dominator.get_relation(a.F, b.F) == 1


class NSGA2BFESA(NSGA2):
    def __init__(self, neighbour, sa_k=10, sa_steps=20, t0=0.1, t_min=1e-3, max_evals=None,
                 sa_seed=None, **kwargs):
        super().__init__(**kwargs)
        self.neighbour = neighbour            # callable(x, rng) -> x'
        self.sa_k, self.sa_steps = sa_k, sa_steps
        self.t0, self.t_min = t0, t_min
        self.r = (t_min / t0) ** (1.0 / max(sa_steps - 1, 1))
        self.max_evals = max_evals
        self.sa_rng = np.random.default_rng(sa_seed)
        self.sa_log = []                      # (generation, n_replaced, accept_rate)

    def _advance(self, infills=None, **kwargs):
        super()._advance(infills=infills, **kwargs)
        self._sa_phase()

    def _sa_phase(self):
        pop = self.pop
        F = pop.get("F")
        ref = BFEReference(F, self.survival.alpha, self.sa_rng)
        fit = ref.fit_of_members()
        feasible = pop.get("CV")[:, 0] <= 0
        order = np.argsort(-np.where(feasible, fit, -np.inf))
        chosen = [i for i in order if feasible[i]][: self.sa_k]
        n_rep, n_acc, n_mv = 0, 0, 0
        for idx in chosen:
            budget_left = None if self.max_evals is None else self.max_evals - self.evaluator.n_eval
            if budget_left is not None and budget_left <= 0:
                break
            steps = self.sa_steps if budget_left is None else min(self.sa_steps, budget_left)
            parent = pop[idx]
            cur_x, cur_fit = parent.X.copy(), fit[idx]
            best_ind, best_fit, T = None, -np.inf, self.t0
            for _ in range(steps):
                cand = Individual(X=self.neighbour(np.asarray(cur_x).astype(int), self.sa_rng))
                self.evaluator.eval(self.problem, Population.create(cand), algorithm=self)
                if cand.CV[0] > 0:
                    T *= self.r
                    continue
                f_c = ref.fit_of_new(cand.F, exclude=[idx])[0]
                n_mv += 1
                if f_c > best_fit:
                    best_ind, best_fit = cand, f_c
                if f_c >= cur_fit or self.sa_rng.random() < np.exp((f_c - cur_fit) / T):
                    cur_x, cur_fit = cand.X.copy(), f_c
                    n_acc += 1
                T *= self.r
            if best_ind is not None and not constrained_dominates(parent, best_ind):
                best_ind.set("rank", parent.get("rank"))
                best_ind.set("crowding", best_fit)
                pop[idx] = best_ind
                n_rep += 1
        self.pop = pop
        self.sa_log.append((self.n_gen, n_rep, n_acc / max(n_mv, 1)))
