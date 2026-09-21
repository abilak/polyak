# Materials and process-optimization benchmarks

These experiments implement the materials, catalyst, and process-control directions proposed
in the project chat. They are finite historical-oracle tests: every property is hidden until
the optimizer selects that row, but no new material is synthesized. The same paired seeds,
budgets, candidates, and evaluation metrics are used for every method.

## Strongest matched application: catalyst plus process conditions

The CADS high-throughput oxidative-coupling-of-methane (OCM) data contain 12,708 experiments
for 59 catalysts, with up to 216 reaction conditions per catalyst. The optimizer maximizes
measured C2 yield while choosing both:

- the catalyst's one-to-three active metals and molar fractions;
- the catalyst support;
- temperature, contact time, methane/oxygen ratio, and argon fraction.

The resulting vector has 43 coordinates and at most eight nonzero entries. A `support_id` is
one fixed catalyst composition/support (`Name` in the source); the within-support candidates
are its measured process conditions. This is the closest real-data match to the support-first
algorithm because it contains a genuine discrete choice of formulation followed by repeated
condition optimization.

The CADS source page requires citation of Nguyen et al., *ACS Catalysis* 2020,
DOI 10.1021/acscatal.9b04293, and marks the data “Not for redistribution.” Raw and processed
tables are therefore ignored by version control and must never be committed or rehosted.

## Matbench composition benchmarks

Three official Matbench v0.1 tables provide independent materials tests:

| Dataset | Prepared candidates | Geometry | Objective |
| --- | ---: | --- | --- |
| `matbench_steels` | 312 | 14 elemental-fraction coordinates; 10–13 active | maximize measured yield strength |
| `matbench_expt_gap` | 4,601 after exact-composition aggregation | 81 elemental-fraction coordinates; 2–4 active | minimize absolute distance to the preregistered 1.34 eV target |
| `matbench_perovskites` | 18,928 | 112 role-specific one-hot coordinates; exactly 5 active | minimize computed formation energy |

For formula data, decimal stoichiometries and nested parentheses are parsed, converted to
atomic fractions, and exact duplicate compositions are averaged before optimization. The
perovskite table contains mixed-anion structures, so A, B, X1, X2, and X3 crystallographic
roles are kept distinct instead of being collapsed into an incorrect single-X formula.

These are useful negative controls as well as benchmarks. The steel table is only moderately
sparse. Most band-gap supports have one candidate, and every role-specific perovskite support
has one candidate; those tasks primarily test support discovery, not continuous refinement
within a support. The perovskite candidate set is categorical and is not the continuous sparse
Euclidean ball assumed by the minimax theorem. Any success on these tables is empirical and
must not be described as verifying the theorem.

## Acquisition and preparation

```bash
.venv/bin/sparse-ecp fetch-dataset --name matbench_steels --raw-dir data/raw
.venv/bin/sparse-ecp fetch-dataset --name matbench_expt_gap --raw-dir data/raw
.venv/bin/sparse-ecp fetch-dataset --name matbench_perovskites --raw-dir data/raw
.venv/bin/sparse-ecp fetch-dataset --name cads_ocm --raw-dir data/raw

.venv/bin/sparse-ecp prepare-materials \
  --input data/raw/matbench_steels.json.gz \
  --output data/processed/matbench_steels.csv \
  --composition composition --property "yield strength" \
  --objective maximize --task-id steel_yield_strength

.venv/bin/sparse-ecp prepare-materials \
  --input data/raw/matbench_expt_gap.json.gz \
  --output data/processed/matbench_expt_gap_target.csv \
  --composition composition --property "gap expt" \
  --objective target --target 1.34 --task-id experimental_band_gap_target

.venv/bin/sparse-ecp prepare-perovskites \
  --input data/raw/matbench_perovskites.json.gz \
  --output data/processed/matbench_perovskites.csv

.venv/bin/sparse-ecp prepare-ocm \
  --input data/raw/CADS_high_throughput_OCM.csv \
  --output data/processed/cads_ocm.csv
```

Downloads with registered checksums are verified before use. If an upstream provider changes
a file, acquisition fails rather than silently changing the benchmark.

## Run order on a laptop

All supplied materials configurations default to one worker because GP-UCB can be memory- and
CPU-intensive. Finish any large ALMANAC run first, then use:

```bash
.venv/bin/sparse-ecp materials --config configs/materials_smoke.yaml
.venv/bin/sparse-ecp materials --config configs/materials_ocm_pilot.yaml
.venv/bin/sparse-ecp materials --config configs/materials_steels.yaml
.venv/bin/sparse-ecp materials --config configs/materials_bandgap.yaml
.venv/bin/sparse-ecp materials --config configs/materials_perovskites.yaml
.venv/bin/sparse-ecp materials --config configs/materials_ocm.yaml
```

The first two commands are validation pilots. The last four are the paired ten-seed studies.
Only increase `--workers` after watching memory use on a pilot.
