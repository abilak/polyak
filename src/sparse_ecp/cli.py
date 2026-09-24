from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    fetch_public_dataset,
    fetch_synergyfinder_examples,
    load_prepared,
    make_random_panels,
    prepare_almanac_panels,
    prepare_matbench_compositions,
    prepare_matbench_perovskites,
    prepare_materials_table,
    prepare_ocm_table,
    prepare_pair_score_table,
    prepare_synergy_table,
    read_table,
    save_prepared,
)
from .theory import minimax_bounds, sample_complexity


def _print_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        print(f"{label}: {path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparse-ecp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    theory = subparsers.add_parser("theory-check", help="evaluate the proved finite-n bounds")
    theory.add_argument("--dimension", "-d", type=int, required=True)
    theory.add_argument("--sparsity", "-s", type=int, required=True)
    theory.add_argument("--evaluations", "-n", type=int, required=True)
    theory.add_argument("--lipschitz", type=float, default=1.0)
    theory.add_argument("--radius", type=float, default=1.0)
    theory.add_argument("--target-regret", type=float)

    synthetic = subparsers.add_parser("synthetic", help="run matched synthetic experiments")
    synthetic.add_argument("--config", required=True)
    synthetic.add_argument(
        "--workers",
        default=None,
        help="worker processes (positive integer or conservative 'auto'; overrides the config)",
    )

    theory_experiments = subparsers.add_parser(
        "theory-experiments",
        help="run noiseless, filtering, adaptation, noise, and proposal-complexity studies",
    )
    theory_experiments.add_argument("--config", required=True)

    biology = subparsers.add_parser("biology", help="run retrospective drug-screen experiments")
    biology.add_argument("--config", required=True)
    biology.add_argument(
        "--workers",
        default=None,
        help="worker processes (positive integer or conservative 'auto'; overrides the config)",
    )

    materials = subparsers.add_parser(
        "materials", help="run retrospective materials and process experiments"
    )
    materials.add_argument("--config", required=True)
    materials.add_argument(
        "--workers",
        default=None,
        help="worker processes (positive integer or conservative 'auto'; overrides the config)",
    )

    hpo = subparsers.add_parser(
        "ecp-hpo",
        help="run the defensible UCI kernel-ridge controls from the ECP paper",
    )
    hpo.add_argument("--config", required=True)
    hpo.add_argument(
        "--workers",
        default=None,
        help="worker processes (positive integer or conservative 'auto'; overrides the config)",
    )

    prepare = subparsers.add_parser("prepare", help="prepare a SynergyFinder-style table")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--response-kind", choices=["inhibition", "viability"], default="inhibition")
    prepare.add_argument("--context", action="append", default=[])
    prepare.add_argument("--dose-penalty", type=float, default=0.0)

    pair_scores = subparsers.add_parser(
        "prepare-pair-scores",
        help="prepare one-score-per-pair support-discovery data",
    )
    pair_scores.add_argument("--input", required=True)
    pair_scores.add_argument("--output", required=True)
    pair_scores.add_argument("--drug1", required=True)
    pair_scores.add_argument("--drug2", required=True)
    pair_scores.add_argument("--score", required=True)
    pair_scores.add_argument("--minimize", action="store_true")

    material_data = subparsers.add_parser(
        "prepare-materials", help="prepare a composition/property optimization table"
    )
    material_data.add_argument("--input", required=True)
    material_data.add_argument("--output", required=True)
    material_data.add_argument("--composition", required=True)
    material_data.add_argument("--property", required=True)
    material_data.add_argument(
        "--objective", choices=["maximize", "minimize", "target"], default="maximize"
    )
    material_data.add_argument("--target", type=float)
    material_data.add_argument("--task-id", default="materials")

    perovskites = subparsers.add_parser(
        "prepare-perovskites", help="prepare the ordered Matbench ABX3 benchmark"
    )
    perovskites.add_argument("--input", required=True)
    perovskites.add_argument("--output", required=True)

    ocm = subparsers.add_parser(
        "prepare-ocm", help="prepare the CADS catalyst and process-condition benchmark"
    )
    ocm.add_argument("--input", required=True)
    ocm.add_argument("--output", required=True)

    fetch = subparsers.add_parser("fetch-examples", help="download documented public example data")
    fetch.add_argument("--raw-dir", default="data/raw")
    fetch.add_argument("--version", default="3.21.0")

    panels = subparsers.add_parser("panels", help="make repeated random drug-library panels")
    panels.add_argument("--input", required=True)
    panels.add_argument("--output-dir", required=True)
    panels.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20])
    panels.add_argument("--count", type=int, default=20)
    panels.add_argument("--seed", type=int, default=0)
    panels.add_argument("--min-candidates", type=int, default=20)

    fetch_data = subparsers.add_parser("fetch-dataset", help="download a registered public dataset")
    fetch_data.add_argument(
        "--name",
        required=True,
        choices=[
            "ncats_malaria_pair_scores",
            "nci_almanac_growth",
            "nci_almanac_combo_scores",
            "matbench_steels",
            "matbench_expt_gap",
            "matbench_perovskites",
            "cads_ocm",
            "ecp_auto_mpg",
            "ecp_breast_cancer_wisconsin",
            "ecp_concrete_slump",
            "ecp_yacht_hydrodynamics",
        ],
    )
    fetch_data.add_argument("--raw-dir", default="data/raw")

    almanac = subparsers.add_parser(
        "prepare-almanac-panels",
        help="stream the full ALMANAC dose table into repeated drug panels",
    )
    almanac.add_argument("--input", required=True)
    almanac.add_argument("--output-dir", required=True)
    almanac.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20])
    almanac.add_argument("--count", type=int, default=20)
    almanac.add_argument("--seed", type=int, default=0)
    almanac.add_argument("--cell-line", action="append")
    almanac.add_argument("--objective", choices=["inhibition", "synergy"], default="inhibition")
    almanac.add_argument("--dose-penalty", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "theory-check":
        result = minimax_bounds(
            args.dimension,
            args.sparsity,
            args.evaluations,
            args.lipschitz,
            args.radius,
        )
        payload = result.__dict__.copy()
        if args.target_regret is not None:
            payload["rate_level_sample_complexity"] = sample_complexity(
                args.dimension,
                args.sparsity,
                args.target_regret,
                args.lipschitz,
                args.radius,
            )
        print(json.dumps(payload, indent=2))
    elif args.command == "synthetic":
        from .experiments import run_synthetic

        _print_paths(run_synthetic(args.config, workers=args.workers))
    elif args.command == "theory-experiments":
        from .theory_experiments import run_theory_validation

        _print_paths(run_theory_validation(args.config))
    elif args.command == "biology":
        from .experiments import run_biology

        _print_paths(run_biology(args.config, workers=args.workers))
    elif args.command == "materials":
        from .experiments import run_materials

        _print_paths(run_materials(args.config, workers=args.workers))
    elif args.command == "ecp-hpo":
        from .experiments import run_ecp_hpo

        _print_paths(run_ecp_hpo(args.config, workers=args.workers))
    elif args.command == "prepare":
        data = prepare_synergy_table(
            read_table(args.input),
            response_kind=args.response_kind,
            context_columns=args.context,
            dose_penalty=args.dose_penalty,
        )
        output = save_prepared(data, args.output)
        print(f"prepared: {output.resolve()}")
        print(f"candidates: {len(data.table)}; tasks: {len(data.task_ids)}; drugs: {len(data.drug_names)}")
    elif args.command == "fetch-examples":
        for path in fetch_synergyfinder_examples(args.raw_dir, args.version):
            print(path.resolve())
    elif args.command == "prepare-pair-scores":
        data = prepare_pair_score_table(
            read_table(args.input),
            drug1_column=args.drug1,
            drug2_column=args.drug2,
            score_column=args.score,
            minimize_score=args.minimize,
        )
        output = save_prepared(data, args.output)
        print(f"prepared: {output.resolve()}")
        print(f"pairs: {len(data.table)}; drugs: {len(data.drug_names)}")
    elif args.command == "prepare-materials":
        input_path = Path(args.input)
        if input_path.name.endswith((".json", ".json.gz")):
            data = prepare_matbench_compositions(
                input_path,
                composition_column=args.composition,
                property_column=args.property,
                objective=args.objective,
                target=args.target,
                task_id=args.task_id,
            )
        else:
            data = prepare_materials_table(
                read_table(input_path),
                composition_column=args.composition,
                property_column=args.property,
                objective=args.objective,
                target=args.target,
                task_id=args.task_id,
            )
        output = save_prepared(data, args.output)
        print(f"prepared: {output.resolve()}")
        print(
            f"candidates: {len(data.table)}; supports: "
            f"{data.table['support_id'].nunique()}; features: {len(data.feature_columns)}"
        )
    elif args.command == "prepare-perovskites":
        data = prepare_matbench_perovskites(args.input)
        output = save_prepared(data, args.output)
        print(f"prepared: {output.resolve()}")
        print(
            f"candidates: {len(data.table)}; supports: "
            f"{data.table['support_id'].nunique()}; features: {len(data.feature_columns)}"
        )
    elif args.command == "prepare-ocm":
        data = prepare_ocm_table(read_table(args.input))
        output = save_prepared(data, args.output)
        print(f"prepared: {output.resolve()}")
        print(
            f"candidates: {len(data.table)}; catalysts: "
            f"{data.table['support_id'].nunique()}; features: {len(data.feature_columns)}"
        )
    elif args.command == "panels":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        panels = make_random_panels(
            load_prepared(args.input),
            args.sizes,
            panels_per_size=args.count,
            seed=args.seed,
            min_candidates=args.min_candidates,
        )
        for size, data in panels.items():
            output = save_prepared(data, output_dir / f"panels_d{size}.csv")
            print(output.resolve())
    elif args.command == "fetch-dataset":
        print(fetch_public_dataset(args.name, args.raw_dir).resolve())
    elif args.command == "prepare-almanac-panels":
        for output in prepare_almanac_panels(
            args.input,
            args.output_dir,
            args.sizes,
            panels_per_size=args.count,
            seed=args.seed,
            cell_lines=args.cell_line,
            objective_kind=args.objective,
            dose_penalty=args.dose_penalty,
        ):
            print(output.resolve())


if __name__ == "__main__":
    main()
