"""Corrected sharding model: eqs. (3)-(18) of manuscript/Main_corrected.tex.

Encoding: x_i in {0, ..., Ns-1} (the manuscript writes 1..Ns).
Objectives (all minimised): f1 = SVI, f2 = TD [s], f3 = -TH [tx/s], f4 = SSI [nodes], f5 = RI [-].
"""

from dataclasses import dataclass

import numpy as np
from pymoo.core.problem import Problem
from scipy.stats import binom

from .instance import Instance

OBJ_NAMES = ("SVI", "TD", "TH", "SSI", "RI")   # TH is stored negated in F
OBJ_UNITS = ("-", "s", "tx/s", "nodes", "-")


def p_fail(k, p):
    """Eq. (5): Pr(m >= floor((k-1)/3) + 1), m ~ Bin(k, p). Pfail(0) := 1 (an empty shard cannot run BFT)."""
    k = np.asarray(k)
    t = (k - 1) // 3 + 1
    out = binom.sf(t - 1, k, p)
    return np.where(k <= 0, 1.0, out)


def admissible_sizes(n_max, p, eps):
    """Boolean table a[k] = (Pfail(k) <= eps) for k = 0..n_max."""
    k = np.arange(n_max + 1)
    return p_fail(k, p) <= eps


@dataclass(frozen=True)
class ModelParams:
    eps: float = 0.1             # chance-constraint tolerance, eq. (6)
    rho: float = 0.2             # cross-shard transaction fraction
    block_tx: int = 1000         # b, transactions per block
    w_gflop: float = 0.02        # w, validation cost per transaction [GFLOP]
    d_max: float = 0.15          # maximum admissible one-way delay [s], eq. (8)
    res_frac: float = 0.5        # C_min = res_frac * sum(C)/Ns, same for M, eq. (9)


class ShardingModel:
    """Evaluates objectives and constraints of one assignment vector."""

    def __init__(self, inst: Instance, mp: ModelParams = ModelParams()):
        self.inst, self.mp = inst, mp
        self.N, self.Ns = inst.n_nodes, inst.n_shards
        self.pf_table = p_fail(np.arange(self.N + 1), inst.params.p)
        self.c_min = mp.res_frac * inst.cpu.sum() / self.Ns
        self.m_min = mp.res_frac * inst.mem.sum() / self.Ns
        self.c_bar = inst.cpu.sum() / self.Ns
        self.m_bar = inst.mem.sum() / self.Ns
        self.n_constr = 5 * self.Ns

    def details(self, x):
        """Return a dict with per-shard quantities, objectives and constraints."""
        inst, mp, Ns = self.inst, self.mp, self.Ns
        x = np.asarray(x, dtype=int)
        D = inst.delay
        k = np.bincount(x, minlength=Ns)
        members = [np.flatnonzero(x == j) for j in range(Ns)]

        T = np.zeros(Ns)              # eq. (11)
        dmax_in = np.zeros(Ns)
        cmin = np.full(Ns, np.nan)
        for j, S in enumerate(members):
            if S.size == 0:
                continue
            sub = D[np.ix_(S, S)]
            leader = S[np.argmax(inst.cpu[S])]
            dmax_in[j] = sub.max()
            T[j] = D[leader, S].max() + 2.0 * dmax_in[j]
            cmin[j] = inst.cpu[S].min()

        # eq. (12): T_ab = T_a + max_{i in S_a, i' in S_b} d(i,i') + T_b
        nonempty = [j for j in range(Ns) if members[j].size > 0]
        node_to_shard = np.zeros((self.N, Ns))
        for j in nonempty:
            node_to_shard[:, j] = D[:, members[j]].max(axis=1)
        cross = np.zeros((Ns, Ns))
        for a in nonempty:
            cross[a] = node_to_shard[members[a]].max(axis=0)
        Tab = T[:, None] + cross + T[None, :]
        off = ~np.eye(Ns, dtype=bool)
        td = (1 - mp.rho) * T.mean() + mp.rho * Tab[off].mean()       # eq. (13)

        with np.errstate(divide="ignore", invalid="ignore"):
            tau = T + mp.block_tx * mp.w_gflop / cmin                  # eq. (14)
            th_terms = np.where(np.isfinite(tau) & (tau > 0), mp.block_tx / tau, 0.0)
        th = th_terms.sum() / (1 + mp.rho)                             # eq. (15)

        svi = self.pf_table[k].sum()                                   # eq. (10)
        ssi = np.sqrt(np.mean((k - self.N / Ns) ** 2))                 # eq. (16)
        C = np.bincount(x, weights=inst.cpu, minlength=Ns)
        M = np.bincount(x, weights=inst.mem, minlength=Ns)
        ri = 0.5 * (np.sqrt(np.mean((C - self.c_bar) ** 2)) / self.c_bar
                    + np.sqrt(np.mean((M - self.m_bar) ** 2)) / self.m_bar)  # eq. (17)

        g = np.concatenate([
            (self.pf_table[k] - mp.eps) / mp.eps,      # eq. (6) security
            (dmax_in - mp.d_max) / mp.d_max,           # eq. (8) communication
            (self.c_min - C) / self.c_min,             # eq. (9) computation
            (self.m_min - M) / self.m_min,             # eq. (9) memory
            (1 - k).astype(float),                     # eq. (10) non-empty shards
        ])
        F = np.array([svi, td, -th, ssi, ri])
        p_sys = 1.0 - np.prod(1.0 - self.pf_table[k])
        return dict(F=F, G=g, k=k, T=T, tau=tau, cmin=cmin, C=C, M=M, dmax_in=dmax_in,
                    svi=svi, td=td, th=th, ssi=ssi, ri=ri, p_sys=p_sys,
                    realised_compromised=int(sum(
                        inst.byzantine[S].sum() >= (S.size - 1) // 3 + 1 for S in members if S.size)))

    def evaluate(self, x):
        d = self.details(x)
        return d["F"], d["G"]


class ShardingProblem(Problem):
    def __init__(self, model: ShardingModel):
        self.model = model
        super().__init__(n_var=model.N, n_obj=5, n_ieq_constr=model.n_constr,
                         xl=0, xu=model.Ns - 1, vtype=int)

        self.eval_stats = dict(n=0, infeasible=0, security=0, communication=0, resources=0, empty=0)

    def _evaluate(self, X, out, *args, **kwargs):
        res = [self.model.evaluate(np.asarray(x).round().astype(int)) for x in X]
        out["F"] = np.array([r[0] for r in res])
        out["G"] = np.array([r[1] for r in res])
        G, Ns, st = out["G"], self.model.Ns, self.eval_stats      # revision R1.3: infeasibility rates
        st["n"] += len(G)
        st["infeasible"] += int((G > 0).any(1).sum())
        st["security"] += int((G[:, :Ns] > 0).any(1).sum())
        st["communication"] += int((G[:, Ns:2 * Ns] > 0).any(1).sum())
        st["resources"] += int((G[:, 2 * Ns:4 * Ns] > 0).any(1).sum())
        st["empty"] += int((G[:, 4 * Ns:] > 0).any(1).sum())
