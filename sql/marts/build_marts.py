"""Builds the Fannie-Mae-side marts and dimension tables that don't depend on the Home
Credit scorecard (mart_application, mart_cutoff_frontier, mart_model_performance,
mart_psi_reference are built later, once the scorecard exists).

This is the Power BI contract: Python/SQL own modelling and aggregation, these parquet
files are the interface, Power BI owns presentation only.
"""
from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"
MARTS_DIR = ROOT / "marts"

BUCKETS = [
    ("Current", 0), ("30", 1), ("60", 2), ("90", 3), ("120+", 4), ("Charge-off", 5),
]
REGIONS = ["West", "South", "Northeast", "Midwest"]
SCENARIOS = [
    ("baseline", "Baseline", "Macro held at recent trailing-12-month average"),
    ("adverse", "Adverse", "Unemployment +3pp, HPI -15%, ramped over 8 quarters"),
    ("severely_adverse", "Severely Adverse", "Unemployment +6pp, HPI -30%, ramped over 8 quarters"),
]


def build_dim_bucket() -> pd.DataFrame:
    return pd.DataFrame(BUCKETS, columns=["bucket", "sort_order"])


def build_dim_bucket_from_to() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Two copies of the same bucket list for the roll-rate matrix visual (rows vs. columns).
    dim_from = pd.DataFrame(BUCKETS, columns=["bucket_from", "sort_order"])
    dim_to = pd.DataFrame(BUCKETS, columns=["bucket_to", "sort_order"])
    return dim_from, dim_to


def build_dim_geography(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT DISTINCT state, region FROM fm_loans ORDER BY region, state"
    ).fetchdf()


def build_dim_scenario() -> pd.DataFrame:
    return pd.DataFrame(SCENARIOS, columns=["scenario", "scenario_label", "scenario_description"])


def build_dim_date(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    bounds = con.execute(
        "SELECT MIN(reporting_date) AS lo, MAX(reporting_date) AS hi FROM fm_performance"
    ).fetchone()
    dates = pd.date_range(bounds[0], bounds[1], freq="D")
    dim = pd.DataFrame({"date": dates})
    dim["year"] = dim["date"].dt.year
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.strftime("%B")
    dim["quarter"] = dim["date"].dt.quarter
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    return dim


def build_mart_loan_performance(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        SELECT
            p.loan_id,
            l.origination_date,
            p.reporting_date,
            p.months_on_book,
            p.upb,
            p.bucket_prev,
            p.bucket_curr,
            p.is_default,
            l.state,
            l.region
        FROM fm_performance p
        JOIN fm_loans l USING (loan_id)
        """
    ).fetchdf()


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    MARTS_DIR.mkdir(parents=True, exist_ok=True)

    dim_bucket = build_dim_bucket()
    dim_bucket.to_parquet(MARTS_DIR / "dim_bucket.parquet", index=False)

    dim_bucket_from, dim_bucket_to = build_dim_bucket_from_to()
    dim_bucket_from.to_parquet(MARTS_DIR / "dim_bucket_from.parquet", index=False)
    dim_bucket_to.to_parquet(MARTS_DIR / "dim_bucket_to.parquet", index=False)

    dim_geography = build_dim_geography(con)
    dim_geography.to_parquet(MARTS_DIR / "dim_geography.parquet", index=False)

    dim_scenario = build_dim_scenario()
    dim_scenario.to_parquet(MARTS_DIR / "dim_scenario.parquet", index=False)

    dim_date = build_dim_date(con)
    dim_date.to_parquet(MARTS_DIR / "dim_date.parquet", index=False)

    mart_loan_performance = build_mart_loan_performance(con)
    mart_loan_performance.to_parquet(MARTS_DIR / "mart_loan_performance.parquet", index=False)

    con.close()

    for name, df in [
        ("dim_bucket", dim_bucket), ("dim_bucket_from", dim_bucket_from), ("dim_bucket_to", dim_bucket_to),
        ("dim_geography", dim_geography), ("dim_scenario", dim_scenario), ("dim_date", dim_date),
        ("mart_loan_performance", mart_loan_performance),
    ]:
        print(f"  {name:<24} {len(df):>10,} rows")

    print(f"\nWritten to {MARTS_DIR}")
    print("Still pending (require the Home Credit scorecard): "
          "mart_application, mart_cutoff_frontier, mart_model_performance, mart_psi_reference, dim_score_band")


if __name__ == "__main__":
    main()
