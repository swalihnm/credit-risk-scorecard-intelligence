"""Build the applicant-level feature table: hc_application + bureau/previous_application/
payment-behaviour aggregates. Writes `feature_table` into the DuckDB warehouse and a parquet
snapshot to marts/. Also builds `feature_table_rejects` (no TARGET column) from the
hc_reject_* tables, needed by scorecard/reject_inference.py.

Run after warehouse/build_warehouse.py.
"""
from __future__ import annotations

import pathlib
import sys

import duckdb
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features import bureau_aggregates, previous_app_aggregates, payment_behaviour  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"

# Raw application-level columns kept as candidate scorecard characteristics.
# All are known at application time by construction (they describe the applicant/loan
# terms as submitted, not the outcome) — see features/leakage_audit.md.
BASE_COLUMNS = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "CODE_GENDER",
]


def build_feature_table(
    con: duckdb.DuckDBPyConnection, table_prefix: str = "hc", include_target: bool = True
) -> pd.DataFrame:
    app_table = f"{table_prefix}_application"
    cols = BASE_COLUMNS + (["TARGET"] if include_target else [])
    cols_sql = ", ".join(cols)
    base = con.execute(f"SELECT {cols_sql} FROM {app_table}").fetchdf()

    bureau_feats = bureau_aggregates.build(con, table_prefix)
    prev_feats = previous_app_aggregates.build(con, table_prefix)
    pay_feats = payment_behaviour.build(con, table_prefix)

    df = (
        base.merge(bureau_feats, on="SK_ID_CURR", how="left")
        .merge(prev_feats, on="SK_ID_CURR", how="left")
        .merge(pay_feats, on="SK_ID_CURR", how="left")
    )

    df["bureau_debt_to_income"] = df["bureau_total_debt"] / df["AMT_INCOME_TOTAL"].replace(0, pd.NA)
    df = df.drop(columns=["bureau_total_debt"])
    return df


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit("Warehouse not found. Run warehouse/build_warehouse.py first.")

    con = duckdb.connect(str(DB_PATH))

    df = build_feature_table(con, table_prefix="hc", include_target=True)
    con.execute("CREATE OR REPLACE TABLE feature_table AS SELECT * FROM df")

    out_path = ROOT / "marts" / "feature_table.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"feature_table: {len(df):,} rows, {df.shape[1]} columns")
    print(f"Written to warehouse table 'feature_table' and {out_path}")
    print(f"Target rate: {df['TARGET'].mean():.4f}")
    print(f"Missingness (top 10):\n{df.isna().mean().sort_values(ascending=False).head(10)}")

    has_rejects = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'hc_reject_application'"
    ).fetchone()[0]
    if has_rejects:
        df_rejects = build_feature_table(con, table_prefix="hc_reject", include_target=False)
        con.execute("CREATE OR REPLACE TABLE feature_table_rejects AS SELECT * FROM df_rejects")
        rejects_path = ROOT / "marts" / "feature_table_rejects.parquet"
        df_rejects.to_parquet(rejects_path, index=False)
        print(f"\nfeature_table_rejects: {len(df_rejects):,} rows, {df_rejects.shape[1]} columns")
        print(f"Written to warehouse table 'feature_table_rejects' and {rejects_path}")
    else:
        print("\nNo hc_reject_* tables found -- skipping feature_table_rejects "
              "(run ingest/synthetic_home_credit.py, which generates the reject pool too).")

    con.close()


if __name__ == "__main__":
    main()
