# Compute-server runbook

## Resource note

The current implementation is CPU-bound. NumPy, SciPy, scikit-learn, and the multiprocessing
runners do not use CUDA, so reserving a GPU does not make these experiments faster. Use a
CPU-heavy node when the scheduler permits it. Eight experiment workers are a safe starting
point; GP-UCB can make larger worker counts memory-intensive.

ECP's rejected candidates are cheap proposals rather than black-box evaluations. The runner
allows up to 50 million cumulative proposals per method/seed job so that deliberately slow
settings such as the `tau=1.001` ablation are not incorrectly terminated by the computational
safety guard. These arms can take much longer than the main `tau=1.01` setting; this proposal
cost is itself an ablation outcome.

## One-time setup and verification

From the repository root:

```bash
git clone https://github.com/abilak/polyak.git
cd polyak
chmod +x scripts/run_server_suite.sh
./scripts/run_server_suite.sh setup
./scripts/run_server_suite.sh verify
```

The verification phase runs all tests and the theorem smoke battery. Do not start the full
jobs unless it completes successfully.

## Full runs

Use separate terminal multiplexor or scheduler jobs so a failure in one family does not lose
the others:

```bash
./scripts/run_server_suite.sh theory
SPARSE_ECP_WORKERS=8 ./scripts/run_server_suite.sh benchmark
SPARSE_ECP_WORKERS=8 ./scripts/run_server_suite.sh ablation
SPARSE_ECP_REAL_WORKERS=2 SPARSE_ECP_MATERIALS_WORKERS=1 \
  ./scripts/run_server_suite.sh real
SPARSE_ECP_HPO_WORKERS=2 ./scripts/run_server_suite.sh ecp-hpo
```

The theory phase is the primary theorem-validation battery. The benchmark phase is the broad
30-problem/100-repetition main-paper comparison. The ablation phase tests epsilon_1, tau, and
C. The real phase expects the prepared CSVs named by the YAML files to exist under
`data/processed/`.

The `ecp-hpo` phase downloads four small UCI files and runs the paper's dense 2-D Gaussian
kernel-ridge protocol with 100 paired repetitions. It is CPU-bound. Start with two workers:
kernel-ridge fits create dense matrices, and larger worker counts can increase memory pressure.
This control is intentionally separate from the sparse evidence.

The optional 168-setting stress suite is substantially larger and should be launched only
after the main runs finish:

```bash
SPARSE_ECP_WORKERS=8 ./scripts/run_server_suite.sh benchmark-extended
```

Expect the extended benchmark to generate tens of millions of trajectory rows. Keep at least
30-50 GB free if running every phase and retaining all CSV outputs.

If the server has at least 128 GB RAM, try 12-16 workers for the benchmark and ablation. Keep
the real-data settings conservative until the GP-UCB processes have been observed with a
memory monitor. The numerical thread variables in the script prevent each process from
spawning another full set of BLAS threads.

## Completion checks

```bash
test -f results/theory_validation/coverage_summary.csv
test -f results/paper_main_benchmark/aggregate_summary.csv
test -f results/ecp_hyperparameter_ablation/aggregate_summary.csv
test -f results/nci_almanac_d20_panels/aggregate_summary.csv
test -f results/cads_ocm/aggregate_summary.csv
test -f results/ecp_uci_hpo/aggregate_summary.csv
```

Preserve the entire `results/` directory. The aggregate files alone are not sufficient for
paired reanalysis because the per-query trajectories and per-run summaries are also required.
