import gzip
import json

import numpy as np
import pandas as pd
import pytest

from sparse_ecp.data import (
    parse_composition,
    prepare_almanac_panels,
    prepare_matbench_perovskites,
    prepare_materials_table,
    prepare_ocm_table,
    prepare_pair_score_table,
    prepare_synergy_table,
)


def test_prepare_synergy_table_aggregates_replicates_and_inverts_viability():
    frame = pd.DataFrame(
        {
            "block_id": [1, 1, 1],
            "drug1": ["A", "A", "A"],
            "drug2": ["B", "B", "B"],
            "conc1": [0.1, 0.1, 1.0],
            "conc2": [0.0, 0.0, 1.0],
            "response": [80.0, 60.0, 10.0],
            "cell_line": ["C", "C", "C"],
        }
    )
    prepared = prepare_synergy_table(
        frame,
        response_kind="viability",
        context_columns=["cell_line"],
    )
    assert len(prepared.table) == 2
    assert prepared.task_ids == ["C"]
    assert prepared.table["replicates"].max() == 2
    assert prepared.table["response"].max() == 90.0
    assert len(prepared.feature_columns) == 2


def test_prepare_almanac_panels_streams_raw_format(tmp_path):
    rows = []
    sequence = 0
    for first in [1, 2, 3, 4]:
        for second in range(first + 1, 5):
            for index1 in [1, 2]:
                for index2 in [1, 2]:
                    sequence += 1
                    rows.append(
                        {
                            "COMBODRUGSEQ": sequence,
                            "NSC1": first,
                            "CONCINDEX1": index1,
                            "CONC1": 10 ** (-index1),
                            "NSC2": second,
                            "CONCINDEX2": index2,
                            "CONC2": 10 ** (-index2),
                            "PERCENTGROWTH": 100 - first - second - index1 - index2,
                            "SCORE": first + second,
                            "VALID": "Y",
                            "PANEL": "test",
                            "CELLNAME": "C",
                        }
                    )
    rows.append(
        {
            "COMBODRUGSEQ": sequence + 1,
            "NSC1": 1,
            "CONCINDEX1": 1,
            "CONC1": 0.1,
            "NSC2": None,
            "CONCINDEX2": 1,
            "CONC2": 0.1,
            "PERCENTGROWTH": 5,
            "SCORE": 100,
            "VALID": "Y",
            "PANEL": "test",
            "CELLNAME": "C",
        }
    )
    raw = tmp_path / "almanac.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    outputs = prepare_almanac_panels(
        raw,
        tmp_path / "out",
        [4],
        panels_per_size=1,
        cell_lines=["C"],
        chunk_size=5,
    )
    prepared = pd.read_csv(outputs[0])
    assert len(prepared) == 24
    assert {f"x_{i}" for i in range(4)}.issubset(prepared.columns)
    assert prepared["objective"].max() > 0
    feature_values = prepared[[f"x_{i}" for i in range(4)]].to_numpy()
    assert feature_values[feature_values > 0].min() == 0.05
    assert feature_values.max() == 1.0


def test_prepare_pair_scores_canonicalizes_pairs_and_minimization():
    frame = pd.DataFrame(
        {
            "left": ["A", "B", "A"],
            "right": ["B", "A", "C"],
            "score": [-2.0, -4.0, -1.0],
        }
    )
    prepared = prepare_pair_score_table(
        frame,
        drug1_column="left",
        drug2_column="right",
        score_column="score",
        minimize_score=True,
    )
    assert len(prepared.table) == 2
    assert prepared.table.loc[prepared.table["support_id"] == "A|B", "objective"].iloc[0] == 3.0
    assert set(prepared.table[prepared.feature_columns].sum(axis=1)) == {2.0}


def test_parse_composition_handles_decimals_and_parentheses():
    assert parse_composition("Ag(AuS)2") == {"Ag": 1.0, "Au": 2.0, "S": 2.0}
    assert parse_composition("Fe0.7Ni0.2Cr0.1") == {"Fe": 0.7, "Ni": 0.2, "Cr": 0.1}
    with pytest.raises(ValueError):
        parse_composition("CuSO4·5H2O")


def test_prepare_materials_aggregates_compositions_and_targets_property():
    frame = pd.DataFrame(
        {
            "formula": ["FeNi", "NiFe", "Fe2O3"],
            "gap": [1.0, 1.4, 2.0],
        }
    )
    prepared = prepare_materials_table(
        frame,
        composition_column="formula",
        property_column="gap",
        objective="target",
        target=1.2,
        task_id="target_gap",
    )
    assert len(prepared.table) == 2
    assert prepared.table["replicates"].max() == 2
    assert np.allclose(prepared.table[prepared.feature_columns].sum(axis=1), 1.0)
    assert prepared.table["objective"].max() == pytest.approx(0.0)
    assert prepared.task_ids == ["target_gap"]


def test_prepare_perovskites_preserves_site_roles(tmp_path):
    def site(element):
        return {"species": [{"element": element, "occu": 1.0}]}

    payload = {
        "index": [0, 1],
        "columns": ["structure", "e_form"],
        "data": [
            [{"sites": [site("Ca"), site("Ti"), site("O"), site("N"), site("O")]}, -1.0],
            [{"sites": [site("Ti"), site("Ca"), site("O"), site("N"), site("O")]}, -0.5],
        ],
    }
    path = tmp_path / "perovskites.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    prepared = prepare_matbench_perovskites(path)
    assert len(prepared.table) == 2
    assert set((prepared.table[prepared.feature_columns] != 0).sum(axis=1)) == {5}
    assert prepared.table["support_id"].nunique() == 2
    assert prepared.table["objective"].max() == 1.0


def test_prepare_ocm_uses_catalyst_supports_and_positive_conditions():
    frame = pd.DataFrame(
        {
            "Name": ["A-B/SiO2", "A/Al2O3"],
            "M1": ["A", "A"],
            "M2": ["B", "n.a."],
            "M3": ["n.a.", "n.a."],
            "M1_mol%": [60, 100],
            "M2_mol%": [40, 0],
            "M3_mol%": [0, 0],
            "Support ": ["SiO2", "Al2O3"],
            "Temp": [700, 900],
            "Total_flow": [10, 20],
            "Ar_flow": [1.5, 14],
            "CT": [0.38, 0.75],
            "CH4/O2": [2, 6],
            "C2y": [5.0, 10.0],
        }
    )
    prepared = prepare_ocm_table(frame)
    assert prepared.task_ids == ["ocm_c2_yield"]
    assert prepared.table["support_id"].nunique() == 2
    assert prepared.table["objective"].tolist() == [5.0, 10.0]
    condition_columns = [
        column
        for column, label in zip(prepared.feature_columns, prepared.drug_names)
        if label.startswith("condition:")
    ]
    assert prepared.table[condition_columns].to_numpy().min() == pytest.approx(0.05)
