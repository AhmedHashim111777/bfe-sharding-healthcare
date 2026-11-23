"""Revision R1.1: executable specification of BFE in the edge cases named by Reviewer 1.

Each test documents the implemented behaviour of bfeshard.bfe (the code that produced every result).
Run: pytest -q tests/test_bfe_edge.py -s   (prints the values quoted in the response letter)
"""
import numpy as np
from bfeshard.bfe import BFEReference, sde_values, RankAndSDESurvival, sde_matrix, sde_truncate


def show(tag, **kw):
    print(f"\n[{tag}] " + "; ".join(f"{k}={np.round(np.asarray(v, float), 4).tolist()}" for k, v in kw.items()))


def test_dominated_individual_outside_bounds():
    # first front {A,B}, dominated C beyond the nadir of the first front in both objectives
    F = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    ref = BFEReference(F, alpha=0.5)
    show("outside-bounds", P=ref.P, cv=ref.cv, cd=ref.cd, fit=ref.fit_of_members())
    assert np.allclose(ref.P[2], [2.0, 2.0])          # normalised coordinates above one
    assert ref.cv[2] < 0                               # eq. (21) is not bounded below by 0 for such individuals
    assert np.all(ref.cv[:2] >= 0) and np.all(ref.cv <= 1)
    assert np.all((ref.cd >= 0) & (ref.cd <= 1))       # eq. (20) stays in [0, 1] by construction


def test_zero_range_objective():
    F = np.array([[0.0, 3.0, 1.0], [1.0, 3.0, 0.0]])  # second objective constant on the non-dominated set
    ref = BFEReference(F, alpha=0.5)
    show("zero-range", span=ref.span, P=ref.P)
    assert ref.span[1] == 1.0 and np.allclose(ref.P[:, 1], 0.0)


def test_zero_range_density():
    F = np.array([[0.0, 1.0], [1.0, 0.0]])            # two symmetric points: equal shifted distances
    ref = BFEReference(F, alpha=0.5)
    show("zero-range-Cd", sde=ref.sde, cd=ref.cd)
    assert np.allclose(ref.cd, 0.0)                    # sde_max = sde_min -> Cd := 0 for all


def test_duplicate_objective_vectors():
    F = np.array([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    ref = BFEReference(F, alpha=0.5)
    show("duplicates", sde=ref.sde, cd=ref.cd, fit=ref.fit_of_members())
    assert ref.sde[1] == 0 and ref.sde[2] == 0         # a duplicate is its own nearest shifted neighbour
    assert ref.cd[1] == ref.cd[2] == ref.cd.min()      # both copies receive the smallest diversity value


def test_single_point_front():
    ref = BFEReference(np.array([[0.3, 0.7]]), alpha=0.5)
    show("single-point", P=ref.P, sde=ref.sde, cd=ref.cd, cv=ref.cv, fit=ref.fit_of_members())
    assert ref.sde[0] == 0 and ref.cd[0] == 0 and ref.cv[0] == 1.0   # span := 1, P = 0, Cv = 1


def test_tie_breaking_is_random_but_seeded():
    from pymoo.util.randomized_argsort import randomized_argsort
    fit = np.array([0.5, 0.5, 0.5, 0.1])
    np.random.seed(0); o1 = randomized_argsort(fit, order="descending", method="numpy")
    np.random.seed(0); o2 = randomized_argsort(fit, order="descending", method="numpy")
    show("ties", order=o1)
    assert list(o1) == list(o2) and o1[-1] == 3        # equal fit: random order, reproducible under the run seed


def test_sde_truncation_removes_duplicate_first():
    P = np.array([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    keep = sde_truncate(sde_matrix(P), np.arange(4), 1, np.random.default_rng(0))
    show("sde-truncation", keep=keep)
    assert keep.sum() == 3 and keep[[1, 2]].sum() == 1
