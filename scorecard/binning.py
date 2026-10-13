"""Coarse classification via optbinning's monotonic optimal binning.

Monotonicity is enforced wherever business logic demands it (e.g. higher DTI/utilization
must not reduce risk) via `trend` in MONOTONIC_TREND below. Anything not listed is fit with
optbinning's auto-selected trend (still constrained to monotonic, just direction-free).
"""
from __future__ import annotations

import pathlib
import pickle

# NOTE: optbinning must be imported before pandas on this Windows/Python 3.12 setup.
# The reverse order (pandas first) reliably segfaults at import time -- a native
# DLL-loading-order conflict between pandas' and optbinning's/ortools' bundled libraries,
# not a bug in either library's Python code. Every module that uses both must preserve
# this import order.
from optbinning import OptimalBinning

import pandas as pd

# characteristic -> forced monotonic trend ("ascending" risk increases with value,
# "descending" risk decreases with value). Business-logic constraints, not statistical fits.
MONOTONIC_TREND = {
    "bureau_debt_to_income": "ascending",
    "bureau_avg_utilization": "ascending",
    "cc_avg_utilization": "ascending",
    "bureau_pct_months_delinquent": "ascending",
    "bureau_max_dpd_ever": "ascending",
    "instalment_pct_late": "ascending",
    "prev_refusal_rate": "ascending",
    "EXT_SOURCE_1": "descending",
    "EXT_SOURCE_2": "descending",
    "EXT_SOURCE_3": "descending",
    "prev_approval_rate": "descending",
    "instalment_pct_early": "descending",
}

OUT_PATH = pathlib.Path("scorecard/artifacts/binning_objects.pkl")


def fit_binning(df: pd.DataFrame, target_col: str, candidate_cols: list[str]) -> dict[str, OptimalBinning]:
    binning_objects: dict[str, OptimalBinning] = {}
    y = df[target_col].values

    for col in candidate_cols:
        x = df[col].values
        trend = MONOTONIC_TREND.get(col, "auto")
        try:
            ob = OptimalBinning(name=col, dtype="numerical", solver="cp", monotonic_trend=trend)
            ob.fit(x, y)
            if ob.status != "OPTIMAL" and ob.status != "FEASIBLE":
                print(f"  [skip] {col}: binning status={ob.status}")
                continue
            binning_objects[col] = ob
        except Exception as exc:  # noqa: BLE001 - report and continue across ~20 characteristics
            print(f"  [skip] {col}: {exc}")

    return binning_objects


def save(binning_objects: dict[str, OptimalBinning]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(binning_objects, f)


def load() -> dict[str, OptimalBinning]:
    with open(OUT_PATH, "rb") as f:
        return pickle.load(f)


def main() -> None:
    df = pd.read_parquet("marts/feature_table.parquet")
    candidate_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")
                      and df[c].dtype.kind in "fi"]

    print(f"Fitting optimal binning on {len(candidate_cols)} numeric candidates ...")
    binning_objects = fit_binning(df, "TARGET", candidate_cols)
    save(binning_objects)

    print(f"\nFitted {len(binning_objects)} / {len(candidate_cols)} characteristics.")
    for col, ob in binning_objects.items():
        table = ob.binning_table.build()
        iv = table["IV"].sum()
        print(f"  {col:<40} IV={iv:.4f}  n_bins={len(ob.splits) + 1}")


if __name__ == "__main__":
    main()
