"""Thin wrapper around sql/analytics/roll_rate_matrix.sql — see portfolio/vintage.py for why
this stays a wrapper rather than reimplementing the aggregation in pandas.
"""
from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"
SQL_PATH = ROOT / "sql" / "analytics" / "roll_rate_matrix.sql"

BUCKET_ORDER = ["Current", "30", "60", "90", "120+", "Charge-off"]


def build_roll_rate_matrix(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    owns_con = con is None
    con = con or duckdb.connect(str(DB_PATH))
    # roll_rate_matrix.sql has two statements (the query, then a commented sanity check) —
    # duckdb's Python API executes only the first.
    sql = SQL_PATH.read_text().split(";")[0]
    df = con.execute(sql).fetchdf()
    if owns_con:
        con.close()
    return df


def main() -> None:
    df = build_roll_rate_matrix()
    latest = df[df["reporting_date"] == df["reporting_date"].max()]
    matrix = (
        latest.pivot(index="bucket_prev", columns="bucket_curr", values="n_loans")
        .reindex(index=BUCKET_ORDER, columns=BUCKET_ORDER)
        .fillna(0)
        .astype(int)
    )
    print(f"Roll rate matrix (loan counts), latest reporting month {latest['reporting_date'].iloc[0]}:")
    print(matrix)


if __name__ == "__main__":
    main()
