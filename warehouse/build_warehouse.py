"""Load raw sources into a single DuckDB warehouse.

Home Credit CSVs -> 7 tables (application-scorecard portfolio).
Synthetic (or, once swapped in, real) Fannie Mae parquet + FRED macro parquet -> lifecycle
portfolio tables. Kept in clearly separate table name prefixes (hc_* / fm_*) since the two
portfolios are never merged — see docs/01-data-sources.md.
"""
from __future__ import annotations

import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"

HOME_CREDIT_DIR = ROOT / "sources" / "home_credit"
FANNIE_MAE_DIR = ROOT / "sources" / "fannie_mae"
FRED_DIR = ROOT / "sources" / "fred"

HOME_CREDIT_TABLES = {
    "hc_application": "application_train.csv",
    "hc_bureau": "bureau.csv",
    "hc_bureau_balance": "bureau_balance.csv",
    "hc_previous_application": "previous_application.csv",
    "hc_pos_cash_balance": "POS_CASH_balance.csv",
    "hc_installments_payments": "installments_payments.csv",
    "hc_credit_card_balance": "credit_card_balance.csv",
}

# Reject-pool tables (see ingest/synthetic_home_credit.py's build_reject_pool) -- needed for
# scorecard/reject_inference.py, since application_train (real or synthetic) only contains
# funded loans. Optional: absent for a real Kaggle pull until a reject population is sourced.
HOME_CREDIT_REJECT_TABLES = {
    "hc_reject_application": "application_rejects.csv",
    "hc_reject_bureau": "bureau_rejects.csv",
    "hc_reject_bureau_balance": "bureau_balance_rejects.csv",
    "hc_reject_previous_application": "previous_application_rejects.csv",
    "hc_reject_installments_payments": "installments_payments_rejects.csv",
    "hc_reject_credit_card_balance": "credit_card_balance_rejects.csv",
    "hc_reject_hidden_truth": "reject_hidden_truth.csv",
}


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    missing = [f for f in HOME_CREDIT_TABLES.values() if not (HOME_CREDIT_DIR / f).exists()]
    if missing:
        print(f"WARNING: Home Credit files missing, skipping hc_* tables: {missing}")
        print("Run ingest/kaggle_download.py once your Kaggle token is in place.")
    else:
        for table, fname in HOME_CREDIT_TABLES.items():
            csv_path = HOME_CREDIT_DIR / fname
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, sample_size=-1)", [str(csv_path)])
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<28} {n:>10,} rows")

    missing_rejects = [f for f in HOME_CREDIT_REJECT_TABLES.values() if not (HOME_CREDIT_DIR / f).exists()]
    if missing_rejects:
        print(f"WARNING: reject-pool files missing, skipping hc_reject_* tables: {missing_rejects}")
    else:
        for table, fname in HOME_CREDIT_REJECT_TABLES.items():
            csv_path = HOME_CREDIT_DIR / fname
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?, sample_size=-1)", [str(csv_path)])
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<28} {n:>10,} rows")

    fm_files = {
        "fm_loans": FANNIE_MAE_DIR / "loans_static.parquet",
        "fm_performance": FANNIE_MAE_DIR / "loan_performance.parquet",
        "fm_defaults": FANNIE_MAE_DIR / "loan_defaults.parquet",
        "macro_monthly": FRED_DIR / "macro_monthly.parquet",
    }
    for table, path in fm_files.items():
        if not path.exists():
            print(f"WARNING: {path} missing, skipping {table}. Run the synthetic generators first "
                  f"(ingest/synthetic_fannie_mae.py, ingest/synthetic_fred.py).")
            continue
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?)", [str(path)])
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<28} {n:>10,} rows")

    con.close()
    print(f"\nWarehouse built at {DB_PATH}")


if __name__ == "__main__":
    main()
