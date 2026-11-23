"""Balanceable fitness estimation (BFE), eqs. (19)-(23), and BFE-based NSGA-II survival.

Fixes relative to the imported original (pymoo fork, metrics.py):
- convergence Cv and diversity Cd are no longer swapped;
- objectives are min-max normalised on the non-dominated set of the reference set
  (no absolute value about a fixed ideal point 0, which inverted negated objectives);
- d1, d2 are computed on the normalised objectives;
- the static weight alpha is a parameter (the original hard-coded alpha = 0);
- the dynamic alpha/beta case rule follows Lin et al. (2018) as implemented in PlatEMO
  (NMPSO/UpdateArchive.m, CalBFE), with alpha and beta drawn per individual.
"""

import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort


class BFEReference:
    """Frozen reference set used to evaluate fit(p, P): normalisation bounds and component statistics."""

    def __init__(self, F_ref, alpha=None, rng=None):
        F_ref = np.asarray(F_ref, dtype=float)
        nd = NonDominatedSorting().do(F_ref, only_non_dominated_front=True)
        self.fmin = F_ref[nd].min(axis=0)
        span = F_ref[nd].max(axis=0) - self.fmin
        self.span = np.where(span > 1e-12, span, 1.0)
        self.alpha = alpha                       # None -> dynamic case rule
        self.rng = rng if rng is not None else np.random.default_rng()
        self.P = self.normalise(F_ref)
        self.m = F_ref.shape[1]
        self.sde = sde_values(self.P)
        self.sde_min, self.sde_max = self.sde.min(), self.sde.max()
        self.cv = self._cv(self.P)
        self.cd = self._cd(self.sde)
        self.d1, self.d2 = d1_d2(self.P)
        self.mean = dict(cv=self.cv.mean(), cd=self.cd.mean(), d1=self.d1.mean(), d2=self.d2.mean())

    def normalise(self, F):
        return (np.asarray(F, dtype=float) - self.fmin) / self.span

    def _cv(self, P):                              # eq. (21)
        return 1.0 - np.linalg.norm(P, axis=1) / np.sqrt(P.shape[1])

    def _cd(self, sde):                            # eq. (20)
        rng_ = self.sde_max - self.sde_min
        if rng_ <= 1e-12:
            return np.zeros_like(sde)
        return (sde - self.sde_min) / rng_

    def weights(self, cv, cd, d1, d2):
        n = len(cv)
        if self.alpha is not None:                 # static variant
            return np.full(n, self.alpha), np.full(n, 1.0 - self.alpha)
        return dynamic_weights(cv, cd, d1, d2, self.mean, self.rng)

    def fit_of_members(self):
        a, b = self.weights(self.cv, self.cd, self.d1, self.d2)
        return a * self.cd + b * self.cv           # eq. (22)

    def fit_of_new(self, F_new, exclude=None):
        """fit of new objective vectors w.r.t. the frozen reference set (used inside SA, eq. (23))."""
        Pn = self.normalise(np.atleast_2d(F_new))
        ref = self.P if exclude is None else np.delete(self.P, exclude, axis=0)
        sde = sde_to_set(Pn, ref)
        cd = np.clip(self._cd(sde), 0.0, 1.0)
        cv = self._cv(Pn)
        d1, d2 = d1_d2(Pn)
        a, b = self.weights(cv, cd, d1, d2)
        return a * cd + b * cv


def sde_values(P):
    """Shift-based density (Li et al. 2014): distance to the nearest shifted neighbour."""
    n = len(P)
    if n < 2:
        return np.zeros(n)
    shift = np.maximum(P[None, :, :] - P[:, None, :], 0.0)    # [i, j, k] = max(0, P_jk - P_ik)
    d = np.sqrt((shift ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def sde_to_set(Q, P):
    if len(P) == 0:
        return np.zeros(len(Q))
    shift = np.maximum(P[None, :, :] - Q[:, None, :], 0.0)
    return np.sqrt((shift ** 2).sum(-1)).min(axis=1)


def d1_d2(P):
    """Projection on, and distance to, the line from the normalised ideal point through (1,...,1)."""
    u = np.ones(P.shape[1]) / np.sqrt(P.shape[1])
    d1 = P @ u
    d2 = np.sqrt(np.maximum((P ** 2).sum(1) - d1 ** 2, 0.0))
    return d1, d2


def dynamic_weights(cv, cd, d1, d2, mean, rng):
    """Case rule of Lin et al. (2018) as implemented in PlatEMO NMPSO (CalBFE)."""
    n = len(cv)
    alpha, beta = np.empty(n), np.empty(n)
    for i in range(n):
        low_cd = cd[i] <= mean["cd"]
        if cv[i] > mean["cv"]:                                     # well converged
            if d1[i] <= mean["d1"]:
                a, b = (rng.uniform(0.8, 1.1), 1.0) if low_cd else (1.0, 1.0)
            else:
                a, b = (0.6, 1.0) if low_cd else (0.9, 1.0)
        else:                                                      # poorly converged
            if d1[i] <= mean["d1"] and d2[i] > mean["d2"]:
                a, b = (rng.uniform(0.8, 1.1), rng.uniform(0.8, 1.1)) if low_cd else (1.0, 1.0)
            else:
                a, b = (0.2, 0.2) if low_cd else (1.0, 0.2)
        alpha[i], beta[i] = a, b
    return alpha, beta


class RankAndBFESurvival(Survival):
    """NSGA-II environmental selection with BFE replacing crowding distance (Algorithm 1, lines 5-7).

    The reference set for BFE is the union of the admitted fronts and the split front; normalisation
    bounds come from its non-dominated individuals. After selection, fit is recomputed on the
    survivors and stored as 'crowding' so that binary tournament mating selection also uses fit.
    """

    def __init__(self, alpha=None, rng_seed=None):
        super().__init__(filter_infeasible=True)
        self.alpha = alpha
        self.nds = NonDominatedSorting()
        self.rng = np.random.default_rng(rng_seed)
        self.last_reference = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        survivors, considered = [], []
        for k, front in enumerate(fronts):
            for i in front:
                pop[i].set("rank", k)
            if len(survivors) + len(front) > n_survive:
                considered = np.concatenate(survivors + [front]) if survivors else front
                ref = BFEReference(F[considered], self.alpha, self.rng)
                fit = ref.fit_of_members()[-len(front):]
                n_keep = n_survive - sum(len(s) for s in survivors)
                order = randomized_argsort(fit, order="descending", method="numpy")
                survivors.append(front[order[:n_keep]])
                break
            survivors.append(front)
        S = np.concatenate(survivors)
        out = pop[S]
        ref = BFEReference(out.get("F"), self.alpha, self.rng)
        fit = ref.fit_of_members()
        for ind, f in zip(out, fit):
            ind.set("crowding", f)
        self.last_reference = ref
        return out


def sde_matrix(P):
    """D[i, j] = || max(0, P_j - P_i) ||: distance of the shifted neighbour j as seen from i (Li et al. 2014)."""
    shift = np.maximum(P[None, :, :] - P[:, None, :], 0.0)
    D = np.sqrt((shift ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def sde_truncate(D, candidates, n_remove, rng):
    """SPEA2-style iterative truncation with shift-based density (SPEA2+SDE, Li et al. 2014).

    Repeatedly removes the candidate whose nearest shifted neighbour (among the individuals still
    present) is closest; ties are broken by the second-nearest distance and then at random.
    Returns the boolean mask of the individuals that are kept.
    """
    keep = np.ones(D.shape[0], bool)
    cand = np.zeros(D.shape[0], bool)
    cand[candidates] = True
    for _ in range(n_remove):
        idx = np.flatnonzero(cand & keep)
        sub = D[np.ix_(idx, np.flatnonzero(keep))]
        if sub.shape[1] >= 2:
            two = np.partition(sub, 1, axis=1)[:, :2]
        else:
            two = np.column_stack([sub.min(1), sub.min(1)])
        key = np.lexsort((rng.random(len(idx)), two[:, 1], two[:, 0]))
        keep[idx[key[0]]] = False
    return keep


class RankAndSDESurvival(Survival):
    """NSGA-II environmental selection with SDE truncation of the split front (revision R2.1 baseline).

    Objectives are normalised as in BFE (bounds of the non-dominated members of the considered set);
    the split front is truncated iteratively by shift-based density; the stored 'crowding' value for
    the binary tournament is the distance to the nearest shifted neighbour among the survivors.
    """

    def __init__(self, rng_seed=None):
        super().__init__(filter_infeasible=True)
        self.nds = NonDominatedSorting()
        self.rng = np.random.default_rng(rng_seed)

    def _norm(self, F):
        nd = self.nds.do(F, only_non_dominated_front=True)
        fmin = F[nd].min(0)
        span = F[nd].max(0) - fmin
        return (F - fmin) / np.where(span > 1e-12, span, 1.0)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        survivors = []
        for k, front in enumerate(fronts):
            for i in front:
                pop[i].set("rank", k)
            if len(survivors) + len(front) > n_survive:
                admitted = np.concatenate(survivors) if survivors else np.empty(0, int)
                considered = np.concatenate([admitted, front])
                D = sde_matrix(self._norm(F[considered]))
                n_remove = len(considered) - n_survive
                keep = sde_truncate(D, np.arange(len(admitted), len(considered)), n_remove, self.rng)
                survivors.append(considered[len(admitted):][keep[len(admitted):]])
                break
            survivors.append(front)
        S = np.concatenate(survivors)
        out = pop[S]
        Fo = out.get("F").astype(float)
        dens = sde_matrix(self._norm(Fo)).min(1) if len(Fo) > 1 else np.zeros(len(Fo))
        for ind, f in zip(out, np.where(np.isfinite(dens), dens, 0.0)):
            ind.set("crowding", f)
        return out
