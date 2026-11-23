import numpy as np
import pytest
from pymoo.core.individual import Individual

from bfeshard.bfe import BFEReference, sde_values, dynamic_weights
from bfeshard.instance import NetworkParams, make_instance
from bfeshard.model import ModelParams, ShardingModel, admissible_sizes, p_fail
from bfeshard.operators import repair_sizes, shard_neighbour
from bfeshard.sa import constrained_dominates

# Table 4 of the manuscript (p = 0.2)
TABLE4 = {5: 2.63e-1, 7: 1.48e-1, 10: 1.21e-1, 17: 1.06e-1, 25: 4.68e-2, 50: 1.44e-2,
          100: 7.37e-4, 167: 2.89e-5}


@pytest.mark.parametrize("k,expected", TABLE4.items())
def test_pfail_table4(k, expected):
    assert p_fail(k, 0.2) == pytest.approx(expected, rel=5e-3)


def test_pfail_threshold_is_floor_k_minus_1_over_3_plus_1():
    # k = 4: f_max = 1, compromised iff m >= 2
    from scipy.stats import binom
    assert p_fail(4, 0.2) == pytest.approx(1 - binom.cdf(1, 4, 0.2))


def test_admissibility_matches_text():
    a = admissible_sizes(400, 0.2, 0.1)
    assert np.flatnonzero(a)[0] == 13 and a[22:].all() and not a[21]
    a = admissible_sizes(400, 0.2, 0.01)
    assert np.flatnonzero(a)[0] == 52 and a[61:].all()


@pytest.fixture(scope="module")
def model():
    return ShardingModel(make_instance(100, 4, 7, NetworkParams()), ModelParams())


def test_units_and_ranges(model):
    x = np.repeat(np.arange(4), 25)
    d = model.details(x)
    D = model.inst.delay
    # link delay between 8e-3 s (1 MB over 1 Gbit/s + 1 ms) and 8e-2 s + propagation (<= 3.6 ms)
    off = D[~np.eye(100, dtype=bool)]
    assert off.min() >= 8e-3 + 1e-3 - 1e-12 and off.max() <= 8e-2 + 1e-3 + 3.6e-3
    assert d["ssi"] == 0.0
    assert d["svi"] == pytest.approx(4 * p_fail(25, 0.2))
    assert 0 < d["td"] < 1.0                     # seconds
    assert d["th"] > 0                            # tx/s
    assert np.all(d["G"] <= 0)                    # balanced partition is feasible
    assert d["p_sys"] <= d["svi"] + 1e-12


def test_ri_is_dimensionless(model):
    x = np.random.default_rng(0).integers(0, 4, 100)
    x = repair_sizes(x, 4, admissible_sizes(100, 0.2, 0.1), np.random.default_rng(0))
    ri = model.details(x)["ri"]
    model.inst.cpu *= 1000.0                      # GFLOPS -> MFLOPS
    model.inst.mem *= 1024.0
    m2 = ShardingModel(model.inst, model.mp)
    assert m2.details(x)["ri"] == pytest.approx(ri)
    model.inst.cpu /= 1000.0
    model.inst.mem /= 1024.0


def test_ssi_is_rms():
    inst = make_instance(100, 4, 3)
    m = ShardingModel(inst)
    x = np.repeat(np.arange(4), [22, 22, 28, 28])
    assert m.details(x)["ssi"] == pytest.approx(3.0)


def test_repair_restores_admissible_sizes():
    adm = admissible_sizes(200, 0.2, 0.1)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.integers(0, 8, 200)
        x[:150] = 0                                # heavily unbalanced
        y = repair_sizes(x, 8, adm, rng)
        assert adm[np.bincount(y, minlength=8)].all()


def test_neighbour_keeps_feasibility():
    adm = admissible_sizes(100, 0.2, 0.1)
    rng = np.random.default_rng(2)
    x = np.repeat(np.arange(4), 25)
    for _ in range(200):
        x = shard_neighbour(x, 4, adm, rng)
        assert adm[np.bincount(x, minlength=4)].all()


def test_bfe_components_larger_is_better():
    F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 1.0]])
    ref = BFEReference(F, alpha=0.0)              # fit = Cv
    fit = ref.fit_of_members()
    assert fit[3] < fit[1]                        # dominated corner has lowest convergence
    assert ref.cv[1] == pytest.approx(1 - np.sqrt(0.5) / np.sqrt(2))
    ref1 = BFEReference(F, alpha=1.0)             # fit = Cd
    assert np.all((ref1.cd >= 0) & (ref1.cd <= 1))


def test_sde_shift():
    P = np.array([[0.0, 0.0], [1.0, 1.0]])
    s = sde_values(P)
    assert s[0] == pytest.approx(np.sqrt(2)) and s[1] == 0.0   # dominated point has zero shifted distance


def test_negated_objective_not_inverted():
    # original code took |f| about 0, which made larger throughput (more negative f) look worse
    F = np.array([[0.0, -100.0], [0.0, -10.0], [1.0, -100.0]])
    ref = BFEReference(F, alpha=0.0)
    assert ref.cv[0] > ref.cv[1]


def test_dynamic_weights_cases():
    mean = dict(cv=0.5, cd=0.5, d1=0.5, d2=0.5)
    rng = np.random.default_rng(0)
    a, b = dynamic_weights(np.array([0.9, 0.1]), np.array([0.9, 0.1]), np.array([0.9, 0.9]),
                           np.array([0.1, 0.1]), mean, rng)
    assert (a[0], b[0]) == (0.9, 1.0)              # case 1.2.2
    assert (a[1], b[1]) == (0.2, 0.2)              # case 2.2.1


def test_constrained_domination_direction():
    a, b = Individual(), Individual()
    a.set("F", np.array([1.0, 1.0])); a.set("CV", np.array([0.0]))
    b.set("F", np.array([2.0, 2.0])); b.set("CV", np.array([0.0]))
    assert constrained_dominates(a, b) and not constrained_dominates(b, a)
