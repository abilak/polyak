# Sparse ECP experiments

This repository implements support-first Sparse ECP and a reproducible experiment scaffold
for the September 2026 draft. It treats the attached papers as source material, not as
instructions, and keeps theorem tests separate from retrospective applications.

The central noiseless rate is

\[
R_n^*=\Theta\!\left(LB\left(\binom ds/n\right)^{1/s}\right)
\]

in the local regime for exact sparsity, with the corresponding slice diameter and finite
tolerance burn-in. The support factor is combinatorial; it is separate from the exponent of
\(n\). The current draft also covers unknown exact sparsity, approximate sparsity, filtering
gain, noisy evaluations, and proposal complexity. See `docs/THEORY_AUDIT.md` for the
claim-to-experiment map.

## Install

```bash
git clone https://github.com/abilak/polyak.git
cd polyak
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,dev]'
```

## Verify

```bash
python -m pytest
sparse-ecp theory-check -d 12 -s 2 -n 100000 --target-regret 0.1
```

## Synthetic scaling study

```bash
sparse-ecp synthetic --config configs/synthetic_scaling.yaml
```

This writes per-evaluation trajectories, per-run endpoints, bootstrap summaries,
log-log-slope checks, and publication-ready PNG figures.

The main-paper benchmark has 30 sparse problems and 100 repetitions, matching ECP's scale:

```bash
sparse-ecp synthetic --config configs/paper_main_benchmark.yaml
```

The optional extended 168-setting benchmark is:

```bash
sparse-ecp synthetic --config configs/synthetic_benchmark_100.yaml
```

## Theorem-validation battery

```bash
sparse-ecp theory-experiments --config configs/theory_validation_smoke.yaml
sparse-ecp theory-experiments --config configs/theory_validation.yaml
```

The full run produces separate noiseless, filtering, sparsity-adaptation,
approximate-sparsity, noisy, and proposal-complexity tables plus figures and a compact coverage
summary. The smoke configuration is an engineering check only.

After reviewing the completed full battery, the only unresolved empirical claim was precise
finite-budget convergence to the asymptotic exponents. The targeted, endpoint-only follow-up
extends the tractable `s=2,3` noisy regimes and the `s=2,3,4` noiseless regimes without
rerunning the already decisive oracle-bound study:

```bash
./scripts/run_server_suite.sh theory-rate-followup
```

This follow-up is needed only if the manuscript makes an empirical rate-matching claim. The
finite-bound coverage claims are already evaluated by the full battery.

The manuscript-ready experiment section is in `docs/PAPER_EXPERIMENTS.md`; the shorter
preregistered protocol is in `docs/EXPERIMENTS.md`.

The matched ECP hyperparameter and rejection-patience ablation is:

```bash
sparse-ecp synthetic --config configs/ecp_hyperparameter_ablation.yaml
```

For a Linux compute server, `scripts/run_server_suite.sh` provides setup, verification,
theory, targeted rate follow-up, benchmark, ablation, data preparation, and real-data phases. These
implementations are CPU-bound;
the current NumPy/SciPy/scikit-learn stack does not use the GPU.

Raw and prepared third-party tables are intentionally excluded from Git. Prepare all public
inputs explicitly with `./scripts/run_server_suite.sh data`. The `real` phase also calls this
idempotent preparation step automatically, so a fresh server checkout downloads and converts
only files that are missing. The ALMANAC archive is roughly 583 MB; allow additional space for
its four prepared panel tables. The CADS OCM source terms prohibit redistribution, so that
table remains local to the server.

If a long real-data suite is interrupted, resume it with
`./scripts/run_server_suite.sh real-resume`. A study is skipped only when its trajectories,
run summary, aggregate summary, and paired comparison table are all present and nonempty.
Use the ordinary `real` phase when every study should be recomputed from scratch.

All experiment runners parallelize independent task/seed/algorithm runs with
`--workers N`. Use `--workers auto` for a conservative automatic choice (half the detected
logical CPUs, capped at four), or set `workers` in a YAML configuration. The bundled
biology configurations use two workers because their Gaussian-process baseline is
resource-intensive. Output ordering and random seeds are unchanged, so serial and
parallel runs are reproducible. Each worker limits numerical libraries to one internal
thread to avoid oversubscribing the CPU.

## Public example data and triple-malaria smoke study

```bash
sparse-ecp fetch-examples --raw-dir data/raw
sparse-ecp prepare \
  --input data/raw/NCATS_screening_data.csv \
  --output data/processed/NCATS_screening_data.csv \
  --response-kind inhibition
sparse-ecp biology --config configs/biology_ncats_example.yaml
```

The downloaded two-block O'Neil illustration is run separately with
`configs/biology_oneil_example.yaml`; it is deliberately labeled as an example rather
than the full 583-pair replication dataset. Its two blocks belong to different cell lines;
the server data phase passes `--context cell_line_name` so they remain two independent tasks
rather than one inconsistent finite objective.

## Full biological studies

Convert any SynergyFinder-style table with `prepare`, passing `--context` once for each
strain/cell-line column that defines an independent task.  The biology runner simulates
strict sequential revelation of historical measurements.  It reports query counts to
90%/95%/99%, top-1% discovery, simple regret, curve area, rejection overhead, and bootstrap
uncertainty.

For ALMANAC, stream the full dose-level table directly into repeated d = 8, 12, 16, and 20
panels.  Restricting the first run to a few named cell lines keeps the prepared files small:

```bash
sparse-ecp fetch-dataset --name nci_almanac_growth --raw-dir data/raw
sparse-ecp prepare-almanac-panels \
  --input data/raw/ComboDrugGrowth_Nov2017.csv \
  --output-dir data/processed/almanac_panels \
  --sizes 8 12 16 20 --count 20 \
  --cell-line MCF7 --cell-line A549/ATCC --cell-line K-562
```

The resulting tables are consumed by `configs/biology_almanac_d8.yaml` through
`configs/biology_almanac_d20.yaml`.  Each configuration runs all 20 panels across the
three selected cell lines with paired seeds and the same hidden response table for every
method.

Dataset availability, exact scope, and honest limitations are in `docs/DATASETS.md`; the
preregisterable protocol is in `docs/EXPERIMENTS.md`.  Completed engineering validation
runs and their deliberately limited interpretation are summarized in `docs/RESULTS.md`.

## Main implementation choices

- `sparse_ecp`: ECP acceptance filtering with support-first proposals (the proved version).
- `ecp_uniform`: ordinary candidate-uniform ECP on finite assay tables.
- `ecp_hard_threshold`: the draft paper's dense-draw/threshold heuristic, retained as a
  synthetic ablation but not covered by the corrected minimax proof.
- `structured_sparse_ecp`: the same acceptance rule with preregistered support weights.
- Baselines: support-balanced and candidate-uniform random search, maximin space filling,
  and Gaussian-process UCB.
- Continuous synthetic and finite historical-oracle spaces share one trace interface, so
  budgets and metrics are applied consistently.

The biology code makes no clinical claims.  It is retrospective algorithm evaluation on
published measurements and does not demonstrate safety, efficacy in patients, or
prospective assay savings.

## Materials, catalyst, and process studies

The additional pipeline covers experimental steel strength, target experimental band gap,
computed perovskite formation energy, and joint catalyst/process optimization for oxidative
coupling of methane. The OCM table is the most closely matched application because each of
59 catalyst formulations has many measured process settings.

```bash
.venv/bin/sparse-ecp materials --config configs/materials_smoke.yaml
.venv/bin/sparse-ecp materials --config configs/materials_ocm_pilot.yaml
```

Acquisition, preparation, full-run commands, exact encodings, and interpretation limits are
in `docs/MATERIALS.md`. These configurations default to one worker so they can be run after
the current ALMANAC job without repeating the earlier laptop overload.
