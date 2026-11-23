"""Integer-label operators and repair for the sharding encoding (eq. (4)).

All algorithms share the same sampling, variation and repair, so differences in results
are attributable to selection (crowding distance / BFE / reference directions) and SA.
"""

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling

from .model import admissible_sizes


def repair_sizes(x, n_shards, admissible, rng, stats=None):
    """Move nodes from the largest shard into shards whose size violates eq. (6) or eq. (10).

    Procedure (revision R1.3): while some shard size k_j is not admissible (Pfail(k_j) > eps or k_j = 0),
    the recipient is the inadmissible shard of smallest size (lowest index on ties), the donor is the
    largest shard (lowest index on ties), and one node drawn uniformly from the donor is moved to the
    recipient. The loop stops when all sizes are admissible, when donor = recipient, or after N * Ns moves.
    A solution that is still inadmissible is returned unchanged in that state and is handled by
    constraint-domination (its security constraint is violated). Resource and communication
    constraints are left to constraint-domination. Repair evaluates no objective function.
    """
    x = np.array(x, dtype=int)
    k = np.bincount(x, minlength=n_shards)
    moves = 0
    needed = bool((~admissible[k]).any())
    for _ in range(len(x) * n_shards):
        bad = np.flatnonzero(~admissible[k])
        if bad.size == 0:
            break
        j = bad[np.argmin(k[bad])]
        donor = np.argmax(k)
        if donor == j:
            break
        i = rng.choice(np.flatnonzero(x == donor))
        x[i] = j
        k[donor] -= 1
        k[j] += 1
        moves += 1
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
        stats["needed"] = stats.get("needed", 0) + int(needed)
        stats["moves"] = stats.get("moves", 0) + moves
        stats["failed"] = stats.get("failed", 0) + int(bool((~admissible[k]).any()))
    return x


class ShardRepair(Repair):
    def __init__(self, n_nodes, n_shards, p, eps, seed=None):
        super().__init__()
        self.Ns = n_shards
        self.adm = admissible_sizes(n_nodes, p, eps)
        self.rng = np.random.default_rng(seed)
        self.stats = {}

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X).round().astype(int)
        return np.array([repair_sizes(x, self.Ns, self.adm, self.rng, self.stats) for x in X])


class LabelSampling(Sampling):
    """Uniform random labels; the repair operator then enforces admissible shard sizes."""

    def _do(self, problem, n_samples, **kwargs):
        return np.random.randint(0, problem.xu[0] + 1, size=(n_samples, problem.n_var))


class UniformLabelCrossover(Crossover):
    def __init__(self, prob=0.9, **kwargs):
        super().__init__(2, 2, prob=prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        mask = np.random.random((n_matings, n_var)) < 0.5
        Y = X.copy()
        Y[0][mask], Y[1][mask] = X[1][mask], X[0][mask]
        return Y


class RandomResetMutation(Mutation):
    """Each gene is reset to a random other label with probability 1/N."""

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X).astype(int).copy()
        n, n_var = X.shape
        ns = int(problem.xu[0]) + 1
        mask = np.random.random((n, n_var)) < 1.0 / n_var
        shift = np.random.randint(1, ns, size=mask.sum())
        X[mask] = (X[mask] + shift) % ns
        return X


def shard_neighbour(x, n_shards, admissible, rng):
    """SA move: reassign one node to another shard; if that breaks eq. (6), swap two nodes instead."""
    y = x.copy()
    i = rng.integers(len(y))
    src = y[i]
    dst = (src + rng.integers(1, n_shards)) % n_shards
    k = np.bincount(y, minlength=n_shards)
    if admissible[k[src] - 1] and admissible[k[dst] + 1]:
        y[i] = dst
    else:
        cand = np.flatnonzero(y == dst)
        i2 = rng.choice(cand)
        y[i], y[i2] = dst, src
    return y
