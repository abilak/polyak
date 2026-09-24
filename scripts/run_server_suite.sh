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
hpo_workers="${SPARSE_ECP_HPO_WORKERS:-2}"
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

prepare_ecp_hpo_data() {
  mkdir -p data/raw
  .venv/bin/sparse-ecp fetch-dataset --name ecp_auto_mpg --raw-dir data/raw
  .venv/bin/sparse-ecp fetch-dataset --name ecp_breast_cancer_wisconsin --raw-dir data/raw
  .venv/bin/sparse-ecp fetch-dataset --name ecp_concrete_slump --raw-dir data/raw
  .venv/bin/sparse-ecp fetch-dataset --name ecp_yacht_hydrodynamics --raw-dir data/raw
}

run_ecp_hpo() {
  prepare_ecp_hpo_data
  .venv/bin/sparse-ecp ecp-hpo \
    --config configs/ecp_uci_hpo.yaml \
    --workers "$hpo_workers"
}

prepare_real_data() {
  mkdir -p data/raw data/processed data/processed/almanac_panels

  if [[ ! -s data/raw/NCATS_screening_data.csv || ! -s data/raw/ONEIL_screening_data.csv ]]; then
    .venv/bin/sparse-ecp fetch-examples --raw-dir data/raw
  fi
  if [[ ! -s data/processed/NCATS_screening_data.csv ]]; then
    .venv/bin/sparse-ecp prepare \
      --input data/raw/NCATS_screening_data.csv \
      --output data/processed/NCATS_screening_data.csv \
      --response-kind inhibition
  fi
  # The two released O'Neil blocks use different cell lines. Rebuild this
  # small table on every data pass so older one-task preparations are repaired.
  .venv/bin/sparse-ecp prepare \
    --input data/raw/ONEIL_screening_data.csv \
    --output data/processed/ONEIL_screening_data.csv \
    --response-kind inhibition \
    --context cell_line_name

  if [[ ! -s data/processed/almanac_panels/almanac_panels_d8.csv \
     || ! -s data/processed/almanac_panels/almanac_panels_d12.csv \
     || ! -s data/processed/almanac_panels/almanac_panels_d16.csv \
     || ! -s data/processed/almanac_panels/almanac_panels_d20.csv ]]; then
    .venv/bin/sparse-ecp fetch-dataset --name nci_almanac_growth --raw-dir data/raw
    .venv/bin/sparse-ecp prepare-almanac-panels \
      --input data/raw/ComboDrugGrowth_Nov2017.csv \
      --output-dir data/processed/almanac_panels \
      --sizes 8 12 16 20 \
      --count 20 \
      --cell-line MCF7 \
      --cell-line A549/ATCC \
      --cell-line K-562
  fi

  if [[ ! -s data/processed/matbench_steels.csv ]]; then
    .venv/bin/sparse-ecp fetch-dataset --name matbench_steels --raw-dir data/raw
    .venv/bin/sparse-ecp prepare-materials \
      --input data/raw/matbench_steels.json.gz \
      --output data/processed/matbench_steels.csv \
      --composition composition \
      --property "yield strength" \
      --objective maximize \
      --task-id steel_yield_strength
  fi
  if [[ ! -s data/processed/matbench_expt_gap_target.csv ]]; then
    .venv/bin/sparse-ecp fetch-dataset --name matbench_expt_gap --raw-dir data/raw
    .venv/bin/sparse-ecp prepare-materials \
      --input data/raw/matbench_expt_gap.json.gz \
      --output data/processed/matbench_expt_gap_target.csv \
      --composition composition \
      --property "gap expt" \
      --objective target \
      --target 1.34 \
      --task-id experimental_band_gap_target
  fi
  if [[ ! -s data/processed/matbench_perovskites.csv ]]; then
    .venv/bin/sparse-ecp fetch-dataset --name matbench_perovskites --raw-dir data/raw
    .venv/bin/sparse-ecp prepare-perovskites \
      --input data/raw/matbench_perovskites.json.gz \
      --output data/processed/matbench_perovskites.csv
  fi
  if [[ ! -s data/processed/cads_ocm.csv ]]; then
    .venv/bin/sparse-ecp fetch-dataset --name cads_ocm --raw-dir data/raw
    .venv/bin/sparse-ecp prepare-ocm \
      --input data/raw/CADS_high_throughput_OCM.csv \
      --output data/processed/cads_ocm.csv
  fi
}

run_real_data() {
  prepare_real_data
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

result_complete() {
  local output_dir="$1"
  [[ -s "$output_dir/trajectories.csv" \
     && -s "$output_dir/run_summary.csv" \
     && -s "$output_dir/aggregate_summary.csv" \
     && -s "$output_dir/paired_algorithm_summary.csv" ]]
}

run_biology_unless_complete() {
  local config="$1"
  local output_dir="$2"
  if result_complete "$output_dir"; then
    echo "complete; skipping $output_dir"
    return
  fi
  .venv/bin/sparse-ecp biology --config "$config" --workers "$real_workers"
}

run_materials_unless_complete() {
  local config="$1"
  local output_dir="$2"
  if result_complete "$output_dir"; then
    echo "complete; skipping $output_dir"
    return
  fi
  .venv/bin/sparse-ecp materials --config "$config" --workers "$materials_workers"
}

resume_real_data() {
  prepare_real_data
  run_biology_unless_complete configs/biology_ncats_example.yaml results/ncats_triple_example
  run_biology_unless_complete configs/biology_oneil_example.yaml results/oneil_two_block_example
  run_biology_unless_complete configs/biology_almanac_d8.yaml results/nci_almanac_d8_panels
  run_biology_unless_complete configs/biology_almanac_d12.yaml results/nci_almanac_d12_panels
  run_biology_unless_complete configs/biology_almanac_d16.yaml results/nci_almanac_d16_panels
  run_biology_unless_complete configs/biology_almanac_d20.yaml results/nci_almanac_d20_panels
  run_materials_unless_complete configs/materials_steels.yaml results/matbench_steels
  run_materials_unless_complete configs/materials_bandgap.yaml results/matbench_expt_gap_target
  run_materials_unless_complete configs/materials_perovskites.yaml results/matbench_perovskites
  run_materials_unless_complete configs/materials_ocm.yaml results/cads_ocm
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
  ecp-data)
    prepare_ecp_hpo_data
    ;;
  ecp-hpo)
    run_ecp_hpo
    ;;
  data)
    prepare_real_data
    ;;
  real)
    run_real_data
    ;;
  real-resume)
    resume_real_data
    ;;
  all)
    setup_environment
    verify_installation
    run_theory
    run_benchmark
    run_ablation
    run_real_data
    run_ecp_hpo
    ;;
  *)
    echo "usage: $0 {setup|verify|theory|theory-rate-followup|benchmark|benchmark-extended|ablation|ecp-data|ecp-hpo|data|real|real-resume|all}" >&2
    exit 2
    ;;
esac
