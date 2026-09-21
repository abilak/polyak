from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .spaces import FiniteOracle

COLUMN_ALIASES = {
    "pairindex": "block_id",
    "blockid": "block_id",
    "drugrow": "drug1",
    "drug_row": "drug1",
    "drugcol": "drug2",
    "drug_col": "drug2",
    "concrow": "conc1",
    "conc_row": "conc1",
    "conc_r": "conc1",
    "conccol": "conc2",
    "conc_col": "conc2",
    "conc_c": "conc2",
    "inhibition": "response",
    "viability": "response",
}


@dataclass(frozen=True)
class PreparedData:
    table: pd.DataFrame
    drug_names: list[str]
    feature_columns: list[str]

    @property
    def task_ids(self) -> list[str]:
        return sorted(self.table["task_id"].astype(str).unique().tolist())

    def oracle(self, task_id: str | None = None) -> FiniteOracle:
        table = self.table
        if task_id is not None:
            table = table[table["task_id"].astype(str) == str(task_id)]
        if table.empty:
            raise ValueError(f"no candidates for task_id={task_id!r}")
        return FiniteOracle(
            features=table[self.feature_columns].to_numpy(dtype=float),
            values=table["objective"].to_numpy(dtype=float),
            support_ids=table["support_id"].astype(str).to_numpy(dtype=object),
            metadata=table.reset_index(drop=True),
        )


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for column in frame.columns:
        key = re.sub(r"[^a-z0-9_]", "", str(column).strip().lower())
        rename[column] = COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=rename)


def _numbered_columns(columns: Iterable[str], prefix: str) -> list[str]:
    found = [column for column in columns if re.fullmatch(rf"{prefix}\d+", column)]
    return sorted(found, key=lambda name: int(name[len(prefix) :]))


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {suffix}")


_FORMULA_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
_FORMULA_ELEMENT = re.compile(r"[A-Z][a-z]?")


def parse_composition(formula: str) -> dict[str, float]:
    """Parse an ordinary chemical formula into element amounts.

    Decimal stoichiometries and nested parentheses are supported. Charges,
    hydrate separators, and variable-composition symbols are intentionally
    rejected instead of being guessed.
    """
    source = str(formula).strip().replace(" ", "")
    if not source:
        raise ValueError("chemical formula is empty")
    position = 0

    def number() -> float:
        nonlocal position
        match = _FORMULA_NUMBER.match(source, position)
        if match is None:
            return 1.0
        position = match.end()
        value = float(match.group())
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid multiplier in chemical formula {formula!r}")
        return value

    def sequence(*, inside_parentheses: bool) -> dict[str, float]:
        nonlocal position
        amounts: dict[str, float] = {}
        while position < len(source):
            if source[position] == ")":
                if not inside_parentheses:
                    raise ValueError(f"unmatched ')' in chemical formula {formula!r}")
                break
            if source[position] == "(":
                position += 1
                nested = sequence(inside_parentheses=True)
                if position >= len(source) or source[position] != ")":
                    raise ValueError(f"unclosed '(' in chemical formula {formula!r}")
                position += 1
                multiplier = number()
                for element, amount in nested.items():
                    amounts[element] = amounts.get(element, 0.0) + multiplier * amount
                continue
            match = _FORMULA_ELEMENT.match(source, position)
            if match is None:
                raise ValueError(
                    f"unsupported token {source[position]!r} in chemical formula {formula!r}"
                )
            position = match.end()
            element = match.group()
            amounts[element] = amounts.get(element, 0.0) + number()
        if not amounts:
            raise ValueError(f"empty group in chemical formula {formula!r}")
        return amounts

    result = sequence(inside_parentheses=False)
    if position != len(source):
        raise ValueError(f"could not fully parse chemical formula {formula!r}")
    return result


def read_matbench_table(path: str | Path) -> pd.DataFrame:
    """Read Matbench's column-oriented JSON or JSON.GZ dataset format."""
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not {"columns", "data"}.issubset(payload):
        raise ValueError("expected a Matbench JSON object with columns and data")
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    if "index" in payload and len(payload["index"]) == len(frame):
        frame.insert(0, "source_index", payload["index"])
    return frame


def _material_objective(values: pd.Series, mode: str, target: float | None) -> pd.Series:
    if mode == "maximize":
        return values
    if mode == "minimize":
        return -values
    if mode == "target":
        if target is None or not np.isfinite(target):
            raise ValueError("a finite target is required for objective='target'")
        return -(values - float(target)).abs()
    raise ValueError("objective must be 'maximize', 'minimize', or 'target'")


def prepare_materials_table(
    frame: pd.DataFrame,
    *,
    composition_column: str,
    property_column: str,
    objective: str = "maximize",
    target: float | None = None,
    task_id: str = "materials",
) -> PreparedData:
    """Encode composition/property rows as a finite sparse optimization task."""
    required = {composition_column, property_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    source = frame[[composition_column, property_column]].copy()
    source[property_column] = pd.to_numeric(source[property_column], errors="coerce")
    source = source.dropna().reset_index(drop=True)
    if source.empty:
        raise ValueError("no finite material-property rows remain")

    parsed: list[dict[str, float]] = []
    keys: list[str] = []
    for formula in source[composition_column].astype(str):
        amounts = parse_composition(formula)
        total = float(sum(amounts.values()))
        fractions = {element: amount / total for element, amount in amounts.items()}
        parsed.append(fractions)
        keys.append(
            "|".join(f"{element}:{fractions[element]:.12g}" for element in sorted(fractions))
        )
    source["composition_key"] = keys
    source["formula"] = source[composition_column].astype(str)
    source["material_property"] = source[property_column].astype(float)
    source = source.groupby("composition_key", as_index=False).agg(
        formula=("formula", "first"),
        material_property=("material_property", "mean"),
        property_std=("material_property", "std"),
        replicates=("material_property", "size"),
    )

    fraction_by_key = dict(zip(keys, parsed))
    elements = sorted({element for fractions in parsed for element in fractions})
    element_index = {element: index for index, element in enumerate(elements)}
    features = np.zeros((len(source), len(elements)), dtype=float)
    support_ids: list[str] = []
    for row, key in enumerate(source["composition_key"]):
        fractions = fraction_by_key[key]
        for element, fraction in fractions.items():
            features[row, element_index[element]] = fraction
        support_ids.append("|".join(sorted(fractions)))

    feature_columns = [f"x_{index}" for index in range(len(elements))]
    source = pd.concat(
        [source.reset_index(drop=True), pd.DataFrame(features, columns=feature_columns)], axis=1
    )
    source["response"] = source["material_property"]
    source["objective"] = _material_objective(source["material_property"], objective, target)
    source["objective_kind"] = objective
    source["target"] = np.nan if target is None else float(target)
    source["support_id"] = support_ids
    source["task_id"] = str(task_id)
    source["candidate_id"] = np.arange(len(source), dtype=int)
    return PreparedData(source, elements, feature_columns)


def prepare_matbench_compositions(
    path: str | Path,
    *,
    composition_column: str,
    property_column: str,
    objective: str = "maximize",
    target: float | None = None,
    task_id: str = "materials",
) -> PreparedData:
    return prepare_materials_table(
        read_matbench_table(path),
        composition_column=composition_column,
        property_column=property_column,
        objective=objective,
        target=target,
        task_id=task_id,
    )


def _ordered_site_element(site: dict, row_number: int, role: str) -> str:
    species = site.get("species", [])
    if len(species) != 1 or float(species[0].get("occu", 0.0)) != 1.0:
        raise ValueError(f"row {row_number}: {role} site is not a single fully occupied species")
    element = str(species[0].get("element", ""))
    if not _FORMULA_ELEMENT.fullmatch(element):
        raise ValueError(f"row {row_number}: invalid {role}-site element {element!r}")
    return element


def prepare_matbench_perovskites(path: str | Path) -> PreparedData:
    """Encode Matbench perovskites with five distinct crystallographic roles."""
    frame = read_matbench_table(path)
    if not {"structure", "e_form"}.issubset(frame.columns):
        raise ValueError("perovskite data needs structure and e_form columns")
    records = []
    for row_number, row in frame.iterrows():
        structure = row["structure"]
        sites = structure.get("sites", []) if isinstance(structure, dict) else []
        if len(sites) != 5:
            raise ValueError(f"row {row_number}: expected five ABX3 sites")
        a = _ordered_site_element(sites[0], row_number, "A")
        b = _ordered_site_element(sites[1], row_number, "B")
        x_sites = [
            _ordered_site_element(site, row_number, f"X{site_number}")
            for site_number, site in enumerate(sites[2:], start=1)
        ]
        value = float(row["e_form"])
        if not np.isfinite(value):
            continue
        records.append(
            {
                "A": a,
                "B": b,
                "X1": x_sites[0],
                "X2": x_sites[1],
                "X3": x_sites[2],
                "formation_energy": value,
            }
        )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        raise ValueError("no valid perovskite rows remain")
    role_columns = ["A", "B", "X1", "X2", "X3"]
    table = table.groupby(role_columns, as_index=False).agg(
        formation_energy=("formation_energy", "mean"),
        property_std=("formation_energy", "std"),
        replicates=("formation_energy", "size"),
    )
    labels = [
        f"{role}:{value}"
        for role in role_columns
        for value in sorted(table[role].unique())
    ]
    label_index = {label: index for index, label in enumerate(labels)}
    features = np.zeros((len(table), len(labels)), dtype=float)
    for row, values in enumerate(table[role_columns].itertuples(index=False, name=None)):
        for role, element in zip(role_columns, values):
            features[row, label_index[f"{role}:{element}"]] = 1.0
    feature_columns = [f"x_{index}" for index in range(len(labels))]
    table = pd.concat(
        [table.reset_index(drop=True), pd.DataFrame(features, columns=feature_columns)], axis=1
    )
    table["formula"] = table[role_columns].astype(str).agg("".join, axis=1)
    table["response"] = table["formation_energy"]
    table["objective"] = -table["formation_energy"]
    table["objective_kind"] = "minimize"
    table["support_id"] = [
        "|".join(f"{role}={element}" for role, element in zip(role_columns, values))
        for values in table[role_columns].itertuples(index=False, name=None)
    ]
    table["task_id"] = "perovskite_formation_energy"
    table["candidate_id"] = np.arange(len(table), dtype=int)
    return PreparedData(table, labels, feature_columns)


def _scale_positive(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return pd.Series(np.ones(len(values)), index=values.index)
    return 0.05 + 0.95 * (values - low) / (high - low)


def prepare_ocm_table(frame: pd.DataFrame) -> PreparedData:
    """Prepare the CADS high-throughput oxidative-coupling-of-methane table."""
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {
        "Name",
        "M1",
        "M2",
        "M3",
        "M1_mol%",
        "M2_mol%",
        "M3_mol%",
        "Support",
        "Temp",
        "Total_flow",
        "Ar_flow",
        "CT",
        "CH4/O2",
        "C2y",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required OCM columns: {sorted(missing)}")
    numeric = [
        "M1_mol%",
        "M2_mol%",
        "M3_mol%",
        "Temp",
        "Total_flow",
        "Ar_flow",
        "CT",
        "CH4/O2",
        "C2y",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).reset_index(drop=True)
    if frame.empty:
        raise ValueError("no complete OCM rows remain")
    if (frame["Total_flow"] <= 0).any():
        raise ValueError("OCM total flow must be positive")

    elements = sorted(
        {
            str(element)
            for column in ["M1", "M2", "M3"]
            for element in frame[column].unique()
            if str(element).lower() not in {"n.a.", "n/a", "none", "nan", "-"}
        }
    )
    supports = sorted(frame["Support"].astype(str).unique())
    labels = [
        *[f"metal:{element}" for element in elements],
        *[f"support:{support}" for support in supports],
        "condition:temperature",
        "condition:contact_time",
        "condition:ch4_o2_ratio",
        "condition:argon_fraction",
    ]
    label_index = {label: index for index, label in enumerate(labels)}
    features = np.zeros((len(frame), len(labels)), dtype=float)
    for component, amount_column in zip(
        ["M1", "M2", "M3"], ["M1_mol%", "M2_mol%", "M3_mol%"]
    ):
        for row, (element, amount) in enumerate(
            frame[[component, amount_column]].itertuples(index=False, name=None)
        ):
            label = f"metal:{element}"
            if label in label_index and float(amount) > 0:
                features[row, label_index[label]] += float(amount) / 100.0
    for row, support in enumerate(frame["Support"].astype(str)):
        features[row, label_index[f"support:{support}"]] = 1.0
    frame["argon_fraction"] = (frame["Ar_flow"] / frame["Total_flow"]).round(2)
    conditions = {
        "condition:temperature": frame["Temp"],
        "condition:contact_time": frame["CT"],
        "condition:ch4_o2_ratio": frame["CH4/O2"],
        "condition:argon_fraction": frame["argon_fraction"],
    }
    for label, values in conditions.items():
        features[:, label_index[label]] = _scale_positive(values).to_numpy(dtype=float)

    feature_columns = [f"x_{index}" for index in range(len(labels))]
    output = frame.copy()
    output = pd.concat(
        [output.reset_index(drop=True), pd.DataFrame(features, columns=feature_columns)], axis=1
    )
    output["response"] = output["C2y"]
    output["objective"] = output["C2y"]
    output["objective_kind"] = "maximize"
    output["support_id"] = output["Name"].astype(str)
    output["task_id"] = "ocm_c2_yield"
    output["candidate_id"] = np.arange(len(output), dtype=int)
    return PreparedData(output, labels, feature_columns)


def _dose_scalers(frame: pd.DataFrame, drug_columns: list[str], dose_columns: list[str]):
    observations: dict[str, list[float]] = {}
    for drug_col, dose_col in zip(drug_columns, dose_columns):
        for drug, dose in frame[[drug_col, dose_col]].itertuples(index=False, name=None):
            if pd.isna(drug) or pd.isna(dose) or float(dose) <= 0:
                continue
            observations.setdefault(str(drug), []).append(float(dose))
    bounds = {}
    for drug, doses in observations.items():
        logs = np.log10(np.asarray(doses, dtype=float))
        bounds[drug] = (float(np.min(logs)), float(np.max(logs)))
    return bounds


def _normalize_dose(drug: str, dose: float, bounds: dict[str, tuple[float, float]]) -> float:
    if not np.isfinite(dose) or dose <= 0:
        return 0.0
    low, high = bounds[str(drug)]
    if high <= low:
        return 1.0
    scaled = (np.log10(dose) - low) / (high - low)
    return float(0.05 + 0.95 * np.clip(scaled, 0.0, 1.0))


def prepare_synergy_table(
    frame: pd.DataFrame,
    *,
    response_kind: str = "inhibition",
    context_columns: list[str] | None = None,
    dose_penalty: float = 0.0,
) -> PreparedData:
    """Convert SynergyFinder-style long data into finite sparse search tasks.

    Replicates are aggregated before optimization.  Concentrations are mapped to
    per-drug log-dose coordinates in [0.05, 1], while true zero dose stays zero.
    """
    frame = _canonicalize_columns(frame.copy())
    drug_columns = _numbered_columns(frame.columns, "drug")
    dose_columns = _numbered_columns(frame.columns, "conc")
    if not drug_columns or len(drug_columns) != len(dose_columns):
        raise ValueError("expected paired drug1..drugK and conc1..concK columns")
    required = {"response", *drug_columns, *dose_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    context_columns = [str(column).lower() for column in (context_columns or [])]
    missing_context = set(context_columns) - set(frame.columns)
    if missing_context:
        raise ValueError(f"missing context columns: {sorted(missing_context)}")

    frame["response"] = pd.to_numeric(frame["response"], errors="coerce")
    for dose_column in dose_columns:
        frame[dose_column] = pd.to_numeric(frame[dose_column], errors="coerce")
    frame = frame.dropna(subset=["response", *drug_columns, *dose_columns]).copy()
    if response_kind not in {"inhibition", "viability"}:
        raise ValueError("response_kind must be 'inhibition' or 'viability'")
    if response_kind == "viability":
        frame["response"] = 100.0 - frame["response"]

    grouping = [*context_columns]
    if "block_id" in frame:
        grouping.append("block_id")
    grouping.extend(drug_columns)
    grouping.extend(dose_columns)
    aggregated = frame.groupby(grouping, dropna=False, as_index=False).agg(
        response=("response", "mean"),
        response_std=("response", "std"),
        replicates=("response", "count"),
    )

    drug_names = sorted(
        {str(value) for column in drug_columns for value in aggregated[column].dropna().unique()}
    )
    drug_to_index = {drug: index for index, drug in enumerate(drug_names)}
    dose_bounds = _dose_scalers(aggregated, drug_columns, dose_columns)
    x = np.zeros((len(aggregated), len(drug_names)), dtype=float)
    support_ids = []
    total_normalized_dose = np.zeros(len(aggregated), dtype=float)
    for row_index, row in aggregated.iterrows():
        support = []
        for drug_col, dose_col in zip(drug_columns, dose_columns):
            drug = str(row[drug_col])
            support.append(drug)
            normalized = _normalize_dose(drug, float(row[dose_col]), dose_bounds)
            x[row_index, drug_to_index[drug]] = normalized
            total_normalized_dose[row_index] += normalized
        support_ids.append("|".join(sorted(support)))

    if context_columns:
        aggregated["task_id"] = aggregated[context_columns].astype(str).agg("|".join, axis=1)
    else:
        aggregated["task_id"] = "all"
    aggregated["support_id"] = support_ids
    aggregated["total_normalized_dose"] = total_normalized_dose
    aggregated["objective"] = aggregated["response"] - dose_penalty * total_normalized_dose
    feature_columns = [f"x_{index}" for index in range(len(drug_names))]
    aggregated[feature_columns] = x
    aggregated["candidate_id"] = np.arange(len(aggregated), dtype=int)
    return PreparedData(aggregated, drug_names, feature_columns)


def save_prepared(data: PreparedData, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.table.to_csv(output, index=False)
    names_path = output.with_suffix(".drugs.txt")
    names_path.write_text("\n".join(data.drug_names) + "\n", encoding="utf-8")
    return output


def load_prepared(path: str | Path) -> PreparedData:
    path = Path(path)
    table = pd.read_csv(path)
    feature_columns = _numbered_columns(table.columns, "x_")
    if not feature_columns:
        feature_columns = sorted(
            [column for column in table if re.fullmatch(r"x_\d+", column)],
            key=lambda name: int(name.split("_")[1]),
        )
    names_path = path.with_suffix(".drugs.txt")
    drug_names = (
        names_path.read_text(encoding="utf-8").splitlines()
        if names_path.exists()
        else feature_columns
    )
    return PreparedData(table, drug_names, feature_columns)


def prepare_pair_score_table(
    frame: pd.DataFrame,
    *,
    drug1_column: str,
    drug2_column: str,
    score_column: str,
    minimize_score: bool = False,
) -> PreparedData:
    """Prepare one-score-per-pair data as a support-discovery-only benchmark."""
    required = {drug1_column, drug2_column, score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    table = frame[[drug1_column, drug2_column, score_column]].copy()
    table[score_column] = pd.to_numeric(table[score_column], errors="coerce")
    table = table.dropna().copy()
    table[drug1_column] = table[drug1_column].astype(str)
    table[drug2_column] = table[drug2_column].astype(str)
    canonical = table[[drug1_column, drug2_column]].apply(
        lambda row: tuple(sorted((row.iloc[0], row.iloc[1]))),
        axis=1,
    )
    table["drug1"] = [pair[0] for pair in canonical]
    table["drug2"] = [pair[1] for pair in canonical]
    table["support_id"] = table["drug1"] + "|" + table["drug2"]
    table = table.groupby(["support_id", "drug1", "drug2"], as_index=False).agg(
        response=(score_column, "mean"),
        response_std=(score_column, "std"),
        replicates=(score_column, "size"),
    )
    drug_names = sorted(set(table["drug1"]) | set(table["drug2"]))
    drug_index = {drug: index for index, drug in enumerate(drug_names)}
    features = np.zeros((len(table), len(drug_names)), dtype=float)
    for row, (first, second) in enumerate(table[["drug1", "drug2"]].itertuples(index=False)):
        features[row, drug_index[first]] = 1.0
        features[row, drug_index[second]] = 1.0
    feature_columns = [f"x_{index}" for index in range(len(drug_names))]
    table[feature_columns] = features
    table["objective"] = -table["response"] if minimize_score else table["response"]
    table["task_id"] = "all"
    table["candidate_id"] = np.arange(len(table), dtype=int)
    return PreparedData(table, drug_names, feature_columns)


def robust_across_tasks(data: PreparedData, task_ids: list[str] | None = None) -> PreparedData:
    """Create a worst-context objective by taking the minimum response per candidate."""
    table = data.table.copy()
    selected = data.task_ids if task_ids is None else [str(task) for task in task_ids]
    table = table[table["task_id"].astype(str).isin(selected)]
    if not len(table):
        raise ValueError("no rows remain for the requested robust objective")
    keys = ["support_id", *data.feature_columns]
    grouped = table.groupby(keys, as_index=False, dropna=False)
    robust = grouped.agg(
        objective=("objective", "min"),
        response=("response", "min"),
        contexts=("task_id", "nunique"),
    )
    robust = robust[robust["contexts"] == len(selected)].copy()
    if robust.empty:
        raise ValueError("tasks do not share an exactly aligned candidate grid")
    robust["task_id"] = "robust:" + "+".join(selected)
    robust["candidate_id"] = np.arange(len(robust), dtype=int)
    return PreparedData(robust, data.drug_names, data.feature_columns)


def make_random_panels(
    data: PreparedData,
    panel_sizes: list[int],
    *,
    panels_per_size: int = 20,
    seed: int = 0,
    min_candidates: int = 20,
) -> dict[int, PreparedData]:
    """Build repeated d-drug subpanels for ALMANAC-style scaling experiments."""
    rng = np.random.default_rng(seed)
    drug_index = {drug: index for index, drug in enumerate(data.drug_names)}
    outputs: dict[int, PreparedData] = {}
    for size in panel_sizes:
        if not 1 <= size <= len(data.drug_names):
            raise ValueError(f"invalid panel size {size}")
        panel_frames = []
        attempts = 0
        while len(panel_frames) < panels_per_size and attempts < panels_per_size * 100:
            attempts += 1
            drugs = sorted(rng.choice(data.drug_names, size=size, replace=False).tolist())
            allowed = set(drugs)
            mask = data.table["support_id"].astype(str).map(
                lambda value, allowed=allowed: set(value.split("|")).issubset(allowed)
            )
            candidate_table = data.table[mask].copy()
            if len(candidate_table) < min_candidates:
                continue
            source_features = [data.feature_columns[drug_index[drug]] for drug in drugs]
            target_features = [f"x_{index}" for index in range(size)]
            candidate_table[target_features] = candidate_table[source_features].to_numpy()
            original_task = candidate_table["task_id"].astype(str)
            panel_number = len(panel_frames)
            candidate_table["task_id"] = original_task + f"|panel_d{size}_{panel_number:03d}"
            candidate_table["panel_drugs"] = "|".join(drugs)
            candidate_table["panel_size"] = size
            keep = [
                column
                for column in candidate_table.columns
                if not re.fullmatch(r"x_\d+", column) or column in target_features
            ]
            panel_frames.append(candidate_table[keep])
        if len(panel_frames) < panels_per_size:
            raise RuntimeError(
                f"could construct only {len(panel_frames)}/{panels_per_size} panels of size {size}"
            )
        combined = pd.concat(panel_frames, ignore_index=True)
        combined["candidate_id"] = np.arange(len(combined), dtype=int)
        outputs[size] = PreparedData(combined, [f"panel_coordinate_{i}" for i in range(size)], target_features)
    return outputs


def fetch_synergyfinder_examples(raw_dir: str | Path, version: str = "3.21.0") -> list[Path]:
    """Download and extract the three documented SynergyFinder example datasets."""
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("install the 'data' extra to convert R data: pip install -e '.[data]'") from exc
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://bioc.r-universe.dev/src/contrib/synergyfinder_{version}.tar.gz"
    archive = raw_dir / f"synergyfinder_{version}.tar.gz"
    if not archive.exists():
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
    wanted = {
        "NCATS_screening_data.rda",
        "ONEIL_screening_data.rda",
        "mathews_screening_data.rda",
    }
    extracted = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            name = Path(member.name).name
            if name not in wanted:
                continue
            rda_path = raw_dir / name
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read {member.name} from archive")
            with rda_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            objects = pyreadr.read_r(rda_path)
            if len(objects) != 1:
                raise RuntimeError(f"expected one object in {rda_path}")
            frame = next(iter(objects.values()))
            csv_path = raw_dir / f"{rda_path.stem}.csv"
            frame.to_csv(csv_path, index=False)
            extracted.append(csv_path)
    if len(extracted) != len(wanted):
        raise RuntimeError("the SynergyFinder archive did not contain all expected example datasets")
    return sorted(extracted)


PUBLIC_DOWNLOADS = {
    "ncats_malaria_pair_scores": (
        (
            "https://raw.githubusercontent.com/richlewis42/synergy-maps/"
            "5c5a4d97aeba6286283adcddc311808de6d82e97/backend/synergy_maps/examples/"
            "NCATS-Malaria/combinations.csv"
        ),
        "NCATS_Malaria_pair_scores.csv",
    ),
    "nci_almanac_growth": (
        (
            "https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/"
            "ComboDrugGrowth_Nov2017.csv"
        ),
        "ComboDrugGrowth_Nov2017.csv",
    ),
    "nci_almanac_combo_scores": (
        (
            "https://discover.nci.nih.gov/cellminer/download/processeddataset/"
            "DTP_NCI60_ALMANAC_COMBO_SCORE.zip"
        ),
        "DTP_NCI60_ALMANAC_COMBO_SCORE.zip",
    ),
    "matbench_steels": (
        "https://ml.materialsproject.org/projects/matbench_steels.json.gz",
        "matbench_steels.json.gz",
    ),
    "matbench_expt_gap": (
        "https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz",
        "matbench_expt_gap.json.gz",
    ),
    "matbench_perovskites": (
        "https://ml.materialsproject.org/projects/matbench_perovskites.json.gz",
        "matbench_perovskites.json.gz",
    ),
    "cads_ocm": (
        (
            "https://cads.eng.hokudai.ac.jp/private-media/"
            "ce5f5ab3-86a2-4ee1-a386-bf41092a973c/"
            "21010bbe-0a5c-4d12-a5fa-84eea540e4be.csv"
        ),
        "CADS_high_throughput_OCM.csv",
    ),
}


PUBLIC_SHA256 = {
    "matbench_steels": "473bc4957b2ea5e6465aef84bc29bb48ac34db27d69ea4ec5f508745c6fae252",
    "matbench_expt_gap": "783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2",
    "matbench_perovskites": "4641e2417f8ec8b50096d2230864468dfa08278dc9d257c327f65d0305278483",
    "cads_ocm": "b942693b1269abb2113f1ab5c82f798ff68d44b66aadf1b7f99e565a7cb905b2",
}


def _verify_download(name: str, path: Path) -> None:
    expected = PUBLIC_SHA256.get(name)
    if expected is None:
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(
            f"checksum mismatch for {name}; delete {path} and retry, or update the pinned source"
        )


def fetch_public_dataset(name: str, raw_dir: str | Path) -> Path:
    import requests

    if name not in PUBLIC_DOWNLOADS:
        raise ValueError(f"no direct public download registered for {name!r}")
    url, filename = PUBLIC_DOWNLOADS[name]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        _verify_download(name, destination)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)
    _verify_download(name, destination)
    return destination


def prepare_almanac_panels(
    raw_csv: str | Path,
    output_dir: str | Path,
    panel_sizes: list[int],
    *,
    panels_per_size: int = 20,
    seed: int = 0,
    cell_lines: list[str] | None = None,
    objective_kind: str = "inhibition",
    dose_penalty: float = 0.0,
    chunk_size: int = 200_000,
) -> list[Path]:
    """Stream the 583 MB ALMANAC table into repeated sparse drug panels."""
    raw_csv = Path(raw_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if objective_kind not in {"inhibition", "synergy"}:
        raise ValueError("objective_kind must be inhibition or synergy")
    discovery_columns = ["NSC1", "NSC2", "CONC1", "CONC2"]
    drugs: set[int] = set()
    dose_bounds: dict[str, tuple[float, float]] = {}
    for chunk in pd.read_csv(raw_csv, usecols=discovery_columns, chunksize=chunk_size):
        for drug_col, dose_col in [("NSC1", "CONC1"), ("NSC2", "CONC2")]:
            valid = chunk[[drug_col, dose_col]].dropna()
            valid = valid[valid[dose_col] > 0].copy()
            valid["log_dose"] = np.log10(valid[dose_col].astype(float))
            grouped = valid.groupby(drug_col)["log_dose"].agg(["min", "max"])
            for drug, row in grouped.iterrows():
                drug_int = int(drug)
                drugs.add(drug_int)
                key = str(drug_int)
                old_low, old_high = dose_bounds.get(key, (float("inf"), float("-inf")))
                dose_bounds[key] = (
                    min(old_low, float(row["min"])),
                    max(old_high, float(row["max"])),
                )
    drug_list = sorted(drugs)
    rng = np.random.default_rng(seed)
    pair_rows = []
    panel_definitions: dict[tuple[int, int], list[int]] = {}
    for size in panel_sizes:
        if size > len(drug_list):
            raise ValueError(f"panel size {size} exceeds the {len(drug_list)} available drugs")
        seen: set[tuple[int, ...]] = set()
        for panel_number in range(panels_per_size):
            panel = tuple(sorted(int(x) for x in rng.choice(drug_list, size=size, replace=False)))
            while panel in seen:
                panel = tuple(sorted(int(x) for x in rng.choice(drug_list, size=size, replace=False)))
            seen.add(panel)
            panel_definitions[(size, panel_number)] = list(panel)
            position = {drug: index for index, drug in enumerate(panel)}
            panel_id = f"d{size}_p{panel_number:03d}"
            for first, second in combinations(panel, 2):
                pair_rows.append(
                    {
                        "pair_a": first,
                        "pair_b": second,
                        "panel_size": size,
                        "panel_id": panel_id,
                        "panel_drugs": "|".join(map(str, panel)),
                        "coord_a": position[first],
                        "coord_b": position[second],
                    }
                )
    allowed_pairs = pd.DataFrame(pair_rows)
    read_columns = [
        "COMBODRUGSEQ",
        "NSC1",
        "CONCINDEX1",
        "CONC1",
        "NSC2",
        "CONCINDEX2",
        "CONC2",
        "PERCENTGROWTH",
        "SCORE",
        "VALID",
        "PANEL",
        "CELLNAME",
    ]
    by_size: dict[int, list[pd.DataFrame]] = {size: [] for size in panel_sizes}
    for chunk in pd.read_csv(raw_csv, usecols=read_columns, chunksize=chunk_size):
        chunk = chunk[chunk["VALID"].astype(str).str.upper() == "Y"].copy()
        if cell_lines:
            chunk = chunk[chunk["CELLNAME"].astype(str).isin(cell_lines)]
        required_values = ["NSC1", "NSC2", "CONC1", "CONC2", "CELLNAME"]
        required_values.append("PERCENTGROWTH" if objective_kind == "inhibition" else "SCORE")
        chunk = chunk.dropna(subset=required_values)
        if chunk.empty:
            continue
        swap = chunk["NSC1"].astype(int) > chunk["NSC2"].astype(int)
        chunk["pair_a"] = np.where(swap, chunk["NSC2"], chunk["NSC1"]).astype(int)
        chunk["pair_b"] = np.where(swap, chunk["NSC1"], chunk["NSC2"]).astype(int)
        chunk["index_a"] = np.where(swap, chunk["CONCINDEX2"], chunk["CONCINDEX1"])
        chunk["index_b"] = np.where(swap, chunk["CONCINDEX1"], chunk["CONCINDEX2"])
        chunk["conc_a"] = np.where(swap, chunk["CONC2"], chunk["CONC1"])
        chunk["conc_b"] = np.where(swap, chunk["CONC1"], chunk["CONC2"])
        joined = chunk.merge(allowed_pairs, on=["pair_a", "pair_b"], how="inner")
        if joined.empty:
            continue
        joined["dose_a"] = [
            _normalize_dose(str(int(drug)), float(dose), dose_bounds)
            for dose, drug in zip(joined["conc_a"], joined["pair_a"])
        ]
        joined["dose_b"] = [
            _normalize_dose(str(int(drug)), float(dose), dose_bounds)
            for dose, drug in zip(joined["conc_b"], joined["pair_b"])
        ]
        joined["response"] = (
            100.0 - joined["PERCENTGROWTH"]
            if objective_kind == "inhibition"
            else joined["SCORE"]
        )
        joined["support_id"] = joined["pair_a"].astype(str) + "|" + joined["pair_b"].astype(str)
        joined["task_id"] = joined["CELLNAME"].astype(str) + "|" + joined["panel_id"]
        joined["total_normalized_dose"] = joined["dose_a"] + joined["dose_b"]
        joined["objective"] = joined["response"] - dose_penalty * joined["total_normalized_dose"]
        for size, group in joined.groupby("panel_size"):
            x = np.zeros((len(group), int(size)), dtype=float)
            rows = np.arange(len(group))
            x[rows, group["coord_a"].to_numpy(dtype=int)] = group["dose_a"].to_numpy()
            x[rows, group["coord_b"].to_numpy(dtype=int)] = group["dose_b"].to_numpy()
            output = group[
                [
                    "COMBODRUGSEQ",
                    "CELLNAME",
                    "PANEL",
                    "panel_id",
                    "panel_drugs",
                    "task_id",
                    "support_id",
                    "pair_a",
                    "pair_b",
                    "CONC1",
                    "CONC2",
                    "response",
                    "objective",
                    "total_normalized_dose",
                ]
            ].reset_index(drop=True)
            feature_columns = [f"x_{index}" for index in range(int(size))]
            output[feature_columns] = x
            by_size[int(size)].append(output)
    paths = []
    for size, chunks in by_size.items():
        if not chunks:
            raise RuntimeError(f"no valid ALMANAC rows were found for d={size}")
        table = pd.concat(chunks, ignore_index=True)
        feature_columns = [f"x_{index}" for index in range(size)]
        group_columns = ["task_id", "support_id", *feature_columns]
        metadata_columns = ["panel_id", "panel_drugs", "pair_a", "pair_b", "CELLNAME", "PANEL"]
        aggregated = table.groupby(group_columns, as_index=False).agg(
            objective=("objective", "mean"),
            response=("response", "mean"),
            response_std=("response", "std"),
            replicates=("response", "size"),
            total_normalized_dose=("total_normalized_dose", "first"),
            **{column: (column, "first") for column in metadata_columns},
        )
        aggregated["candidate_id"] = np.arange(len(aggregated), dtype=int)
        prepared = PreparedData(aggregated, [f"panel_coordinate_{i}" for i in range(size)], feature_columns)
        paths.append(save_prepared(prepared, output_dir / f"almanac_panels_d{size}.csv"))
    manifest = output_dir / "panel_manifest.csv"
    pd.DataFrame(
        [
            {
                "panel_size": size,
                "panel_number": number,
                "panel_drugs": "|".join(map(str, panel)),
            }
            for (size, number), panel in panel_definitions.items()
        ]
    ).to_csv(manifest, index=False)
    return paths
