"""Seeded generation of heterogeneous healthcare blockchain network instances.

All physical quantities are in SI units unless stated otherwise:
positions in metres, bandwidth in bit/s, message size in bits, delays in seconds,
computation in GFLOPS, memory in GB, validation cost in GFLOP per transaction.
"""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class NetworkParams:
    region_m: float = 500e3          # side of the square deployment region [m]
    v: float = 2e8                   # propagation speed in fibre [m/s]
    sigma_bits: float = 8e6          # consensus message size, 1 MB [bit]
    b_hospital: float = 1e9          # access bandwidth of hospital nodes [bit/s]
    b_clinic: float = 1e8            # access bandwidth of clinic nodes [bit/s]
    hospital_share: float = 0.3      # fraction of hospital nodes
    delta: float = 1e-3              # per-message processing delay [s]
    mu_c: float = 50.0               # mean computation [GFLOPS]
    sd_c: float = 15.0
    c_floor: float = 5.0             # truncation of computation [GFLOPS]
    mu_m: float = 8.0                # mean memory [GB]
    sd_m: float = 2.0
    m_floor: float = 1.0             # truncation of memory [GB]
    p: float = 0.2                   # Byzantine probability per node
    n_clusters: float = 0.0          # 0: uniform positions; >0: Gaussian clusters around regional centres (revision R2.2)
    cluster_sd_m: float = 40e3       # spread of a cluster [m]


@dataclass
class Instance:
    name: str
    n_nodes: int
    n_shards: int
    seed: int
    pos: np.ndarray          # (N, 2) [m]
    bandwidth: np.ndarray    # (N,) [bit/s]
    is_hospital: np.ndarray  # (N,) bool
    cpu: np.ndarray          # (N,) [GFLOPS]
    mem: np.ndarray          # (N,) [GB]
    byzantine: np.ndarray    # (N,) bool, sampled from Bernoulli(p); never given to the optimizer
    delay: np.ndarray        # (N, N) one-way link delay d(i,i') [s], eq. (9)
    params: NetworkParams

    def summary(self):
        d = self.delay[np.triu_indices(self.n_nodes, 1)]
        return {
            "name": self.name, "N": self.n_nodes, "Ns": self.n_shards, "seed": self.seed,
            "hospitals": int(self.is_hospital.sum()),
            "byzantine_realised": int(self.byzantine.sum()),
            "cpu_mean": float(self.cpu.mean()), "cpu_min": float(self.cpu.min()),
            "mem_mean": float(self.mem.mean()),
            "delay_min_s": float(d.min()), "delay_max_s": float(d.max()),
            "params": asdict(self.params),
        }


def _truncated_normal(rng, mu, sd, floor, size):
    x = rng.normal(mu, sd, size)
    while np.any(x < floor):
        bad = x < floor
        x[bad] = rng.normal(mu, sd, bad.sum())
    return x


def link_delay_matrix(pos, bandwidth, prm: NetworkParams):
    """Eq. (9): d(i,i') = l(i,i')/v + sigma/B(i,i') + delta, B(i,i') = min(B_i, B_i')."""
    diff = pos[:, None, :] - pos[None, :, :]
    length = np.sqrt((diff ** 2).sum(-1))
    b_link = np.minimum(bandwidth[:, None], bandwidth[None, :])
    d = length / prm.v + prm.sigma_bits / b_link + prm.delta
    np.fill_diagonal(d, 0.0)
    return d


def make_instance(n_nodes, n_shards, seed, prm: NetworkParams = NetworkParams(), name=None):
    rng = np.random.default_rng(seed)
    if int(prm.n_clusters) > 0:
        centres = rng.uniform(0.1 * prm.region_m, 0.9 * prm.region_m, size=(int(prm.n_clusters), 2))
        pos = centres[rng.integers(0, int(prm.n_clusters), n_nodes)] + rng.normal(0.0, prm.cluster_sd_m, (n_nodes, 2))
        pos = np.clip(pos, 0.0, prm.region_m)
    else:
        pos = rng.uniform(0.0, prm.region_m, size=(n_nodes, 2))
    n_hosp = int(round(prm.hospital_share * n_nodes))
    is_hospital = np.zeros(n_nodes, dtype=bool)
    is_hospital[rng.choice(n_nodes, n_hosp, replace=False)] = True
    bandwidth = np.where(is_hospital, prm.b_hospital, prm.b_clinic)
    cpu = _truncated_normal(rng, prm.mu_c, prm.sd_c, prm.c_floor, n_nodes)
    mem = _truncated_normal(rng, prm.mu_m, prm.sd_m, prm.m_floor, n_nodes)
    byz = rng.random(n_nodes) < prm.p
    delay = link_delay_matrix(pos, bandwidth, prm)
    return Instance(name or f"N{n_nodes}_S{n_shards}_s{seed}", n_nodes, n_shards, seed,
                    pos, bandwidth, is_hospital, cpu, mem, byz, delay, prm)
