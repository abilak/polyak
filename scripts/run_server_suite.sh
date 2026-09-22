#!/usr/bin/env bash
set -euo pipefail

# Sparse ECP currently uses NumPy/SciPy/scikit-learn and is CPU-bound. A GPU is
# neither required nor used. Keep numerical libraries single-threaded inside
# each experiment worker to avoid oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/sparse-ecp-mpl-${USER:-runner}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/sparse-ecp-cache-${USER:-runner}}"

phase="${1:-all}"
workers="${SPARSE_ECP_WORKERS:-8}"
real_workers="${SPARSE_ECP_REAL_WORKERS:-2}"
materials_workers="${SPARSE_ECP_MATERIALS_WORKERS:-1}"
python_bin="${PYTHON_BIN:-python3}"

setup_environment() {
  if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv .venv
  fi
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[data,dev]'
}

verify_installation() {
  .venv/bin/python -m pytest -q
  .venv/bin/sparse-ecp theory-experiments \
    --config configs/theory_validation_smoke.yaml
}

run_theory() {
  .venv/bin/sparse-ecp theory-experiments \
    --config configs/theory_validation.yaml
}

run_theory_rate_followup() {
  .venv/bin/sparse-ecp theory-experiments \
    --config configs/theory_rate_followup.yaml
}

run_benchmark() {
  .venv/bin/sparse-ecp synthetic \
    --config configs/paper_main_benchmark.yaml \
    --workers "$workers"
}

run_extended_benchmark() {
  .venv/bin/sparse-ecp synthetic \
    --config configs/synthetic_benchmark_100.yaml \
    --workers "$workers"
}

run_ablation() {
  .venv/bin/sparse-ecp synthetic \
    --config configs/ecp_hyperparameter_ablation.yaml \
    --workers "$workers"
}

run_real_data() {
  .venv/bin/sparse-ecp biology --config configs/biology_ncats_example.yaml --workers "$real_workers"
  .venv/bin/sparse-ecp biology --config configs/biology_oneil_example.yaml --workers "$real_workers"
  .venv/bin/sparse-ecp biology --config configs/biology_almanac_d8.yaml --workers "$real_workers"
  .venv/bin/sparse-ecp biology --config configs/biology_almanac_d12.yaml --workers "$real_workers"
  .venv/bin/sparse-ecp biology --config configs/biology_almanac_d16.yaml --workers "$real_workers"
  .venv/bin/sparse-ecp biology --config configs/biology_almanac_d20.yaml --workers "$real_workers"
  .venv/bin/sparse-ecp materials --config configs/materials_steels.yaml --workers "$materials_workers"
  .venv/bin/sparse-ecp materials --config configs/materials_bandgap.yaml --workers "$materials_workers"
  .venv/bin/sparse-ecp materials --config configs/materials_perovskites.yaml --workers "$materials_workers"
  .venv/bin/sparse-ecp materials --config configs/materials_ocm.yaml --workers "$materials_workers"
}

case "$phase" in
  setup)
    setup_environment
    ;;
  verify)
    verify_installation
    ;;
  theory)
    run_theory
    ;;
  theory-rate-followup)
    run_theory_rate_followup
    ;;
  benchmark)
    run_benchmark
    ;;
  benchmark-extended)
    run_extended_benchmark
    ;;
  ablation)
    run_ablation
    ;;
  real)
    run_real_data
    ;;
  all)
    setup_environment
    verify_installation
    run_theory
    run_benchmark
    run_ablation
    run_real_data
    ;;
  *)
    echo "usage: $0 {setup|verify|theory|theory-rate-followup|benchmark|benchmark-extended|ablation|real|all}" >&2
    exit 2
    ;;
esac
