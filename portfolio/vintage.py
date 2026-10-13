"""Thin wrapper around sql/analytics/vintage_curves.sql — the heavy lifting (cumulative
default rate by cohort x months-on-book) is done in SQL upstream, per the project's own
architecture note (SQL/Python does aggregation, Power BI does presentation).
"""
from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"
SQL_PATH = ROOT / "sql" / "analytics" / "vintage_curves.sql"


def build_vintage_curves(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    owns_con = con is None
    con = con or duckdb.connect(str(DB_PATH))
    df = con.execute(SQL_PATH.read_text()).fetchdf()
    if owns_con:
        con.close()
    return df


def main() -> None:
    df = build_vintage_curves()
    pivot = df.pivot(index="vintage", columns="months_on_book", values="cumulative_default_rate")
    print("Vintage curve matrix (cumulative default rate), months-on-book 1-12:")
    print(pivot.iloc[:, :12].round(4))


if __name__ == "__main__":
    main()
