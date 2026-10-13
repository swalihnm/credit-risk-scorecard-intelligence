"""Sanity checks on the synthetic Fannie Mae / FRED stand-in data (see docs/01-data-sources.md).
Not statistical validation of the model — just structural invariants that must hold for any
downstream vintage/roll-rate/ECL logic to be trustworthy.
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


def test_macro_panel_has_no_gaps():
    """Regression guard: ingest/fred_ingest.py once chained .resample("MS").ffill() directly
    on the concatenated raw series, which only keeps a value when the original observation
    lands exactly on day 1 of the month -- true for monthly series, almost never true for a
    daily one (DGS10), never true for a quarterly one (GDPC1). dropna() then discarded most
    months, collapsing ~470 expected months down to 81 scattered ones. This checks the
    month-to-month gaps are never more than a single month, however the panel was built.
    """
    path = _require(ROOT / "sources" / "fred" / "macro_monthly.parquet")
    macro = pd.read_parquet(path).sort_values("date")
    macro["date"] = pd.to_datetime(macro["date"])
    gaps = macro["date"].diff().dropna()
    max_gap_days = gaps.max().days
    assert max_gap_days <= 31, f"Macro panel has a gap of {max_gap_days} days -- expected continuous monthly data"
    assert len(macro) >= 200, f"Macro panel has only {len(macro)} rows -- suspiciously short for a monthly series"


def test_macro_panel_covers_crisis_and_benign_windows():
    path = _require(ROOT / "sources" / "fred" / "macro_monthly.parquet")
    macro = pd.read_parquet(path)
    macro["date"] = pd.to_datetime(macro["date"])
    crisis_peak = macro.loc[macro["date"].between("2009-06-01", "2010-06-01"), "unrate"].max()
    benign_avg = macro.loc[macro["date"].between("2015-01-01", "2019-12-01"), "unrate"].mean()
    assert crisis_peak > 8.0, "crisis window should show elevated unemployment"
    assert benign_avg < 6.0, "benign window should show materially lower unemployment"


def test_fannie_mae_bucket_values_valid():
    path = _require(ROOT / "sources" / "fannie_mae" / "loan_performance.parquet")
    perf = pd.read_parquet(path)
    valid = {"Current", "30", "60", "90", "120+", "Charge-off"}
    assert set(perf["bucket_curr"].unique()) <= valid
    assert set(perf["bucket_prev"].unique()) <= valid


def test_charge_off_is_terminal():
    path = _require(ROOT / "sources" / "fannie_mae" / "loan_performance.parquet")
    perf = pd.read_parquet(path).sort_values(["loan_id", "reporting_date"])
    last_row_per_loan = perf.groupby("loan_id").tail(1)
    charged_off_not_last = perf[
        (perf["bucket_curr"] == "Charge-off")
        & ~perf.set_index(["loan_id", "reporting_date"]).index.isin(
            last_row_per_loan.set_index(["loan_id", "reporting_date"]).index
        )
    ]
    assert charged_off_not_last.empty, "Charge-off must be the loan's final panel row"


def test_crisis_vintages_default_more_than_benign_vintages():
    loans = _require(ROOT / "sources" / "fannie_mae" / "loans_static.parquet")
    defaults = _require(ROOT / "sources" / "fannie_mae" / "loan_defaults.parquet")
    loans_df = pd.read_parquet(loans)
    defaults_df = pd.read_parquet(defaults)
    loans_df["origination_date"] = pd.to_datetime(loans_df["origination_date"])

    crisis = loans_df[loans_df["origination_date"].between("2006-01-01", "2008-12-01")]
    benign = loans_df[loans_df["origination_date"].between("2017-01-01", "2019-12-01")]

    crisis_rate = defaults_df["loan_id"].isin(crisis["loan_id"]).sum() / len(crisis)
    benign_rate = defaults_df["loan_id"].isin(benign["loan_id"]).sum() / len(benign)

    assert crisis_rate > benign_rate, "2006-2008 vintages must show higher default than 2017-2019"
