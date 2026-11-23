"""Constructive sharding heuristics: baselines and seeds for the initial population.

All outputs are repaired to admissible shard sizes (eq. (6)).
"""

import numpy as np
from pymoo.core.sampling import Sampling

from .model import admissible_sizes
from .operators import repair_sizes


def _balanced_assign(points, centroids, cap):
    """Assign points to the nearest centroid with capacity cap (greedy on global distance order)."""
    d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    lab = np.full(len(points), -1)
    load = np.zeros(len(centroids), int)
    for flat in np.argsort(d, axis=None):
        i, j = divmod(flat, len(centroids))
        if lab[i] < 0 and load[j] < cap[j]:
            lab[i] = j
            load[j] += 1
    return lab


def balanced_kmeans(points, k, rng, iters=20):
    n = len(points)
    cap = np.full(k, n // k)
    cap[: n % k] += 1
    cent = points[rng.choice(n, k, replace=False)]
    lab = _balanced_assign(points, cent, cap)
    for _ in range(iters):
        new = np.array([points[lab == j].mean(0) for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
        lab = _balanced_assign(points, cent, cap)
    return lab


def h_random(inst, rng):
    return rng.integers(0, inst.n_shards, inst.n_nodes)


def h_round_robin(inst, rng):
    return rng.permutation(np.arange(inst.n_nodes) % inst.n_shards)


def h_kmeans(inst, rng):
    return balanced_kmeans(inst.pos, inst.n_shards, rng)


def h_bandwidth(inst, rng):
    """Bandwidth-aware: fill whole shards with geographically clustered hospitals, then clinics by k-means."""
    N, Ns = inst.n_nodes, inst.n_shards
    size = N // Ns
    hosp = np.flatnonzero(inst.is_hospital)
    n_hosp_shards = min(len(hosp) // size, Ns - 1)
    x = np.empty(N, int)
    rest = np.arange(N)
    if n_hosp_shards > 0:
        lab = balanced_kmeans(inst.pos[hosp], max(n_hosp_shards, 1), rng) if n_hosp_shards > 1 else np.zeros(len(hosp), int)
        # keep the size nearest hospitals of each hospital cluster; leftovers join the clinics
        chosen = []
        for j in range(n_hosp_shards):
            members = hosp[lab == j]
            c = inst.pos[members].mean(0)
            members = members[np.argsort(np.linalg.norm(inst.pos[members] - c, axis=1))][:size]
            x[members] = j
            chosen.extend(members)
        rest = np.setdiff1d(np.arange(N), chosen)
    lab = balanced_kmeans(inst.pos[rest], Ns - n_hosp_shards, rng)
    x[rest] = lab + n_hosp_shards
    return x


HEURISTICS = {"H-RAND": h_random, "H-RR": h_round_robin, "H-KM": h_kmeans, "H-BW": h_bandwidth}


def heuristic_solution(name, inst, eps, rng):
    adm = admissible_sizes(inst.n_nodes, inst.params.p, eps)
    return repair_sizes(HEURISTICS[name](inst, rng), inst.n_shards, adm, rng)


class SeededSampling(Sampling):
    """n_seed individuals from the RR, KM and BW heuristics (cycled), the rest uniform random labels."""

    def __init__(self, inst, eps, n_seed=10, seed=0):
        super().__init__()
        self.inst, self.eps, self.n_seed, self.seed = inst, eps, n_seed, seed

    def _do(self, problem, n_samples, **kwargs):
        rng = np.random.default_rng(self.seed)
        X = np.random.randint(0, self.inst.n_shards, size=(n_samples, problem.n_var))
        names = ["H-RR", "H-KM", "H-BW"]
        for i in range(min(self.n_seed, n_samples)):
            X[i] = heuristic_solution(names[i % 3], self.inst, self.eps, rng)
        return X


def h_contrib(inst, rng, model=None, iters=2000):
    """Contribution-aware allocation after the node-allocation step of ContribChain (Huang et al., INFOCOM 2025),
    revision R2.4. Under the homogeneous Byzantine model every node has the same security contribution, so only
    the performance loop applies: nodes are dealt in decreasing order of computation (snake order), then a node of
    high computation in the fastest shard is swapped with a node of low computation in the slowest shard while the
    variance of the shard block intervals tau_j (eq. (14)) decreases."""
    N, Ns = inst.n_nodes, inst.n_shards
    order = np.argsort(-inst.cpu, kind="stable")
    snake = np.concatenate([np.arange(Ns), np.arange(Ns)[::-1]])
    x = np.empty(N, int)
    x[order] = snake[np.arange(N) % (2 * Ns)]
    if model is None:
        return x

    def taus(x):
        return model.details(x)["tau"]
    t = taus(x); var = np.var(t)
    for _ in range(iters):
        fast, slow = int(np.argmin(t)), int(np.argmax(t))
        a = np.flatnonzero(x == fast); b = np.flatnonzero(x == slow)
        i = rng.choice(a[inst.cpu[a] >= np.median(inst.cpu[a])])
        j = rng.choice(b[inst.cpu[b] <= np.median(inst.cpu[b])])
        y = x.copy(); y[i], y[j] = slow, fast
        ty = taus(y)
        if np.var(ty) < var:
            x, t, var = y, ty, np.var(ty)
    return x
