# Security-constrained multi-objective blockchain sharding for digital healthcare

Code and data for the article *Security-Constrained Multi-Objective Blockchain Sharding for Digital
Healthcare Using NSGA-II with Balanceable Fitness Estimation* (A. Hashim, K. Chellappan).

The repository contains the five-objective sharding model (SVI, TD, TH, SSI, RI) with the binomial
Byzantine chance constraint, NSGA-II with static and dynamic balanceable fitness estimation (BFE),
NSGA-II with shift-based density estimation (NSGA-II+SDE), the repair operator, the constructive
heuristics, the message-level PBFT validation, the configurations with all parameters and seeds, the unit
tests, the per-run metrics and the scripts that produce every table and figure of the article.

Archived version with the raw per-run non-dominated sets: https://doi.org/10.5281/zenodo.23428915

## Layout

| Path | Content |
|---|---|
| `code/bfeshard/` | model (`model.py`, `instance.py`), BFE and SDE selection (`bfe.py`), algorithms (`algorithms.py`), repair and variation (`operators.py`), heuristics (`heuristics.py`), PBFT simulation and offered-load model (`validation.py`), experiment runner (`runner.py`) |
| `code/configs/` | `sharding.yaml` (3 base networks), `rev_base.yaml` (NSGA-II+SDE, normalized MOEA/D, alpha = 1), `rev_instances.yaml` (18 additional networks, seeds 2001-2018), `dtlz.yaml` (DTLZ1-6, m = 3, 5, 8, 10) |
| `code/tests/` | unit tests, including the BFE edge cases (`test_bfe_edge.py`) |
| `results/scripts/` | indicators, statistical tests, tables and figures |
| `results/metrics/`, `results/tables/` | per-run metrics and the tables reported in the article |

## Reproduction

Python 3.12; the versions used are pinned in `code/requirements-lock.txt` (pymoo 0.6.1.5, numpy 1.26.4).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements-lock.txt

# unit tests
(cd code && python -m pytest -q)

# experiments: 30 independent runs per algorithm and network (seeds in the configuration files)
cd code
python -m bfeshard.runner configs/sharding.yaml        # -> results/raw/sharding
python -m bfeshard.runner configs/rev_base.yaml        # -> results/raw/sharding_rev
python -m bfeshard.runner configs/rev_instances.yaml   # -> results/raw/sharding_inst
python -m bfeshard.runner configs/dtlz.yaml            # -> results/raw/dtlz
cd ..

# indicators, tests, tables and figures
bash results/scripts/make_union.sh
python results/scripts/analyse.py
python results/scripts/analyse_instances.py
python results/scripts/revision_posthoc.py
python results/scripts/figures.py
python results/scripts/figures_rev.py
```

Options of the runner: `--runs`, `--algs`, `--problems`, `--jobs` (e.g. `--problems N100_S4 --runs 2`
for a quick check). Statistics: 30 runs per configuration, mean and standard deviation, Wilcoxon rank-sum
tests with Holm correction, Vargha-Delaney A12; Friedman and paired signed-rank tests across networks.

The raw runs (about 335 MB) are not stored in this repository; they are in the Zenodo archive above.
Results from the x86-64 platform and Apple M1 were statistically equivalent (all p >= 0.18).

## Licence

MIT (see `LICENSE`).
