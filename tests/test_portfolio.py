"""Regression guards on the portfolio/ layer. These specifically exist because the hazard
model and ECL engine have twice quietly produced backwards results during development
(class-weight rebalancing miscalibrating PD; unclipped months_on_book extrapolating to
PD ~= 1; a wrong-signed HPI coefficient making stress scenarios show LOWER expected loss
than baseline) — see git history / dev notes. These checks make sure that can't silently
recur.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _require(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    return path


def test_lgd_monotonic_in_ltv_band():
    path = _require(ROOT / "portfolio" / "artifacts" / "lgd_table.parquet")
    lgd = pd.read_parquet(path)
    band_order = ["<60", "60-70", "70-80", "80-90", "90+"]
    avg_by_band = lgd.groupby("ltv_band")["lgd"].mean().reindex(band_order)
    assert avg_by_band.is_monotonic_increasing, "LGD should rise monotonically with LTV band"


def test_ecl_scenario_ordering():
    path = _require(ROOT / "marts" / "mart_ecl.parquet")
    ecl = pd.read_parquet(path)
    totals = ecl.groupby("scenario")["ecl"].sum()
    assert totals["baseline"] < totals["adverse"] < totals["severely_adverse"], (
        f"Stress scenarios must increase total ECL monotonically, got: {totals.to_dict()}"
    )


def test_ecl_pd_within_valid_range():
    path = _require(ROOT / "marts" / "mart_ecl.parquet")
    ecl = pd.read_parquet(path)
    assert ecl["pd_12m"].between(0, 1).all()
    assert ecl["pd_lifetime"].between(0, 1).all()
    assert (ecl["pd_lifetime"] >= ecl["pd_12m"] - 1e-9).all(), "lifetime PD must be >= 12-month PD"


def test_vintage_curve_is_monotonic_non_decreasing():
    db_path = ROOT / "warehouse" / "credit_risk.duckdb"
    if not db_path.exists():
        pytest.skip("warehouse not built yet")
    import duckdb

    from portfolio.vintage import build_vintage_curves

    con = duckdb.connect(str(db_path))
    df = build_vintage_curves(con)
    con.close()

    df = df.sort_values(["vintage", "months_on_book"])
    non_monotonic = (
        df.groupby("vintage")["cumulative_default_rate"]
        .apply(lambda s: (s.diff().dropna() < -1e-9).any())
    )
    assert not non_monotonic.any(), "cumulative default rate must never decrease within a vintage"


def test_hazard_model_out_of_time_validation():
    """A model trained only on 2005-2010 (crisis) vintages must still discriminate on
    2015-2019 (benign) vintages -- the actual out-of-time check per the project's own
    validation protocol (train early vintages, validate later ones), not just a random
    person-month holdout. See portfolio/survival.py's out_of_time_validation().
    """
    db_path = ROOT / "warehouse" / "credit_risk.duckdb"
    if not db_path.exists():
        pytest.skip("warehouse not built yet")
    import duckdb

    from portfolio.survival import build_person_month_panel, out_of_time_validation

    con = duckdb.connect(str(db_path))
    panel = build_person_month_panel(con)
    con.close()

    result = out_of_time_validation(panel)
    assert result["auc_on_oot_vintages"] > 0.55, (
        f"Out-of-time AUC too low to claim generalization: {result['auc_on_oot_vintages']:.4f}"
    )
    assert abs(result["predicted_monthly_hazard"] - result["actual_monthly_hazard"]) < 0.001, (
        "Predicted and actual monthly hazard on the out-of-time vintages should be close "
        "if the model is well calibrated across the regime shift, not just discriminating"
    )


def test_hazard_model_macro_coefficients_correctly_signed():
    import pickle

    path = _require(ROOT / "portfolio" / "artifacts" / "hazard_model.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    coefs = dict(zip(artifact["features"], artifact["model"].coef_[0]))
    assert coefs["unrate_at_performance_month"] > 0, "higher unemployment must raise hazard"
    assert coefs["hpi_decline_pct"] > 0, "larger home-price decline must raise hazard"
    assert coefs["fico_orig"] < 0, "higher FICO must lower hazard"
