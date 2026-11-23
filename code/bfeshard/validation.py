"""Revision R1.2 / R2.5: message-level check of the analytical latency and throughput surrogates.

1. PBFT message-level discrete-event simulation of one shard with shared access links.
   A node's uplink transmits its outgoing messages one at a time (FIFO) at its access bandwidth B_s; a
   node's downlink receives incoming messages one at a time (FIFO) at B_r. A message whose uplink
   transmission starts at t_u is fully received at max(t_u + size/B_s, max(t_u, free_r) + size/B_r) + l/v,
   where free_r is the time at which the receiver's downlink becomes free, and is processed after delta.
   Phases: the leader (largest computation) sends PRE-PREPARE to every replica; a replica that has the
   PRE-PREPARE sends PREPARE to all other members; a node that holds the PRE-PREPARE (or is the leader)
   and 2f PREPAREs from distinct members sends COMMIT to all; a node commits after 2f COMMITs from others
   (2f + 1 with its own). Shard latency = time at which 2f + 1 members have committed (quorum) and the
   time at which all members have committed. Messages are sent in increasing order of link delay.
2. Cross-shard relay: the leader of the source shard forwards the commit certificate to the leader of the
   destination shard, which then runs its own consensus: T_ab = T_a + relay(a, b) + T_b.
3. Offered-load model: transactions originate uniformly at nodes; a transaction is processed by its home
   shard and, with probability rho, by one other shard chosen uniformly. Each shard serves blocks of up to
   b transactions; a block occupies the shard for tau_j = T_j + b * w / C_min,j (eq. (14) with the simulated
   T_j). Sustainable throughput lambda* = min_j mu_j / a_j, mu_j = b / tau_j, a_j = share of the offered
   load reaching shard j. Confirmation times under a given offered load come from a bulk-service queue
   simulation per shard (Poisson arrivals); a cross-shard transaction is confirmed when both shards have
   processed it and the relay has arrived.
"""

import heapq

import numpy as np


def simulate_shard(S, inst, size_pp, size_vote):
    S = np.asarray(S)
    k = len(S)
    if k == 1:
        return 0.0, 0.0
    f = (k - 1) // 3
    B = inst.bandwidth[S]
    pos = inst.pos[S]
    L = int(np.argmax(inst.cpu[S]))
    prm = inst.params
    prop = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)) / prm.v
    up = np.zeros(k); down = np.zeros(k)
    has_pp = np.zeros(k, bool); has_pp[L] = True
    n_prep = np.zeros(k, int); n_comm = np.zeros(k, int)
    sent_commit = np.zeros(k, bool); committed = np.full(k, np.inf)
    ev = []                       # (time, seq, kind, s, r, size, msg)
    seq = 0

    def broadcast(s, t, size, msg):
        nonlocal seq
        for r in np.argsort(prop[s]):
            if r != s:
                heapq.heappush(ev, (t, seq, 0, s, int(r), size, msg)); seq += 1

    def maybe_commit(i, t):
        if not sent_commit[i] and has_pp[i] and n_prep[i] >= 2 * f:
            sent_commit[i] = True
            broadcast(i, t, size_vote, 2)

    broadcast(L, 0.0, size_pp, 0)
    while ev:
        t, _, kind, s, r, size, msg = heapq.heappop(ev)
        if kind == 0:                                  # transmission
            tu = max(t, up[s])
            up[s] = tu + size / B[s]
            done = max(up[s], max(tu, down[r]) + size / B[r])
            down[r] = done
            heapq.heappush(ev, (done + prop[s, r] + prm.delta, seq, 1, s, r, size, msg)); seq += 1
        else:                                          # reception
            if msg == 0:
                has_pp[r] = True
                broadcast(r, t, size_vote, 1)
                maybe_commit(r, t)
            elif msg == 1:
                n_prep[r] += 1
                maybe_commit(r, t)
            else:
                n_comm[r] += 1
                if n_comm[r] >= 2 * f and not np.isfinite(committed[r]):
                    committed[r] = t
    c = np.sort(committed)
    return float(c[2 * f]), float(c[-1])


def simulate_config(x, model, size_pp, size_vote):
    inst, mp = model.inst, model.mp
    x = np.asarray(x, int)
    Ns = model.Ns
    members = [np.flatnonzero(x == j) for j in range(Ns)]
    Tq = np.zeros(Ns); Ta = np.zeros(Ns); leaders = np.zeros(Ns, int); cmin = np.zeros(Ns)
    for j, S in enumerate(members):
        Tq[j], Ta[j] = simulate_shard(S, inst, size_pp, size_vote)
        leaders[j] = S[np.argmax(inst.cpu[S])]
        cmin[j] = inst.cpu[S].min()
    pl = inst.pos[leaders]
    relay = (np.sqrt(((pl[:, None] - pl[None]) ** 2).sum(-1)) / inst.params.v
             + size_vote / np.minimum(inst.bandwidth[leaders][:, None], inst.bandwidth[leaders][None]) + inst.params.delta)
    off = ~np.eye(Ns, dtype=bool)
    Tab = Tq[:, None] + relay + Tq[None, :]
    td = (1 - mp.rho) * Tq.mean() + mp.rho * Tab[off].mean()
    tau = Tq + mp.block_tx * mp.w_gflop / cmin
    mu = mp.block_tx / tau
    share = np.array([len(S) for S in members]) / model.N
    a = share + mp.rho * (1 - share) / (Ns - 1)
    lam_star = float((mu / a).min())
    return dict(Tq=Tq, Tall=Ta, tau=tau, mu=mu, a=a, share=share, relay=relay, td_sim=float(td),
                td_all=float((1 - mp.rho) * Ta.mean() + mp.rho * (Ta[:, None] + relay + Ta[None, :])[off].mean()),
                lam_star=lam_star, th_sum=float(mu.sum() / (1 + mp.rho)))


def bulk_queue(arrivals, b, tau):
    n = len(arrivals)
    soj = np.empty(n)
    i, free = 0, 0.0
    while i < n:
        t = max(free, arrivals[i])
        j = min(n, i + b, int(np.searchsorted(arrivals, t, side="right")))
        free = t + tau
        soj[i:j] = free - arrivals[i:j]
        i = j
    return soj


def offered_load(sim, model, lam, rng, n_tx=20000):
    """p50/p95/p99 confirmation time [s] of transactions at total offered load lam [tx/s]."""
    mp, Ns = model.mp, model.Ns
    if lam >= sim["lam_star"]:
        return dict(stable=False, p50=np.inf, p95=np.inf, p99=np.inf)
    horizon = n_tx / lam
    soj = []
    for j in range(Ns):
        lj = lam * sim["a"][j]
        m = rng.poisson(lj * horizon)
        arr = np.sort(rng.uniform(0, horizon, m))
        soj.append(bulk_queue(arr, mp.block_tx, sim["tau"][j]))
    home = rng.choice(Ns, size=n_tx, p=sim["share"])
    cross = rng.random(n_tx) < mp.rho
    other = (home + rng.integers(1, Ns, size=n_tx)) % Ns
    t = np.array([rng.choice(soj[h]) if len(soj[h]) else 0.0 for h in home])
    t2 = np.array([rng.choice(soj[o]) if len(soj[o]) else 0.0 for o in other])
    conf = np.where(cross, np.maximum(t, t2) + sim["relay"][home, other], t)
    q = np.percentile(conf, [50, 95, 99])
    return dict(stable=True, p50=q[0], p95=q[1], p99=q[2])
