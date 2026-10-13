"""IFRS 9 style three-stage expected credit loss engine for the Fannie Mae portfolio.

    ECL = PD x LGD x EAD, discounted, by stage

- EAD = current UPB (Fannie Mae mortgages are fully drawn — no undrawn-commitment
  adjustment, unlike a revolving product).
- PD is term-structured from the hazard model (portfolio/survival.py), projected forward
  under a given macro scenario (portfolio/stress_scenarios.py).
- LGD from the credibility-weighted region x LTV-band table (portfolio/lgd.py).
- Stage assignment uses the standard IFRS 9 rebuttable-presumption backstops:
    Stage 1: current, no elevated risk (12-month ECL)
    Stage 2: 30-59 dpd (SICR backstop trigger) -> lifetime ECL
    Stage 3: 60+ dpd, i.e. 60/90/120+ (credit-impaired backstop) -> lifetime ECL
  A charged-off loan has already been written off and is excluded from the ECL calc
  (its loss is realized, not expected).
- Lifetime horizon is capped at 60 months because that is the simulation window of the
  synthetic loan-performance panel (see docs/06-assumptions-and-limitations.md) — a real
  Fannie Mae extract would support a full-term horizon.
"""
from __future__ import annotations

import pathlib
import pickle

import duckdb
import numpy as np
import pandas as pd

from portfolio import stress_scenarios

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"
HAZARD_MODEL_PATH = ROOT / "portfolio" / "artifacts" / "hazard_model.pkl"
LGD_TABLE_PATH = ROOT / "portfolio" / "artifacts" / "lgd_table.parquet"
OUT_PATH = ROOT / "marts" / "mart_ecl.parquet"

LIFETIME_HORIZON_MONTHS = 60
STAGE2_BUCKETS = {"30"}
STAGE3_BUCKETS = {"60", "90", "120+"}


def _stage(bucket_curr: str) -> int:
    if bucket_curr in STAGE3_BUCKETS:
        return 3
    if bucket_curr in STAGE2_BUCKETS:
        return 2
    return 1


def _ltv_band(ltv: float) -> str:
    if ltv < 60:
        return "<60"
    if ltv < 70:
        return "60-70"
    if ltv < 80:
        return "70-80"
    if ltv < 90:
        return "80-90"
    return "90+"


def load_active_loans(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Latest performance row per loan, excluding already-charged-off (realized-loss) loans."""
    return con.execute(
        """
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY reporting_date DESC) AS rn
            FROM fm_performance
        )
        SELECT p.loan_id, p.reporting_date, p.months_on_book, p.upb, p.bucket_curr,
               l.fico_orig, l.original_ltv, l.orig_rate, l.region
        FROM latest p
        JOIN fm_loans l USING (loan_id)
        WHERE p.rn = 1 AND p.bucket_curr != 'Charge-off'
        """
    ).fetchdf()


def compute_ecl(scenario: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    owns_con = con is None
    con = con or duckdb.connect(str(DB_PATH))

    loans = load_active_loans(con)
    if owns_con:
        con.close()

    with open(HAZARD_MODEL_PATH, "rb") as f:
        hazard = pickle.load(f)
    lgd_table = pd.read_parquet(LGD_TABLE_PATH)
    portfolio_mean_lgd = lgd_table["portfolio_mean_lgd"].iloc[0]

    loans["stage"] = loans["bucket_curr"].map(_stage)
    loans["ltv_band"] = loans["original_ltv"].apply(_ltv_band)
    loans = loans.merge(lgd_table[["region", "ltv_band", "lgd"]], on=["region", "ltv_band"], how="left")
    loans["lgd"] = loans["lgd"].fillna(portfolio_mean_lgd)

    macro_path = stress_scenarios.get_scenario_path(scenario, LIFETIME_HORIZON_MONTHS)

    from portfolio.survival import project_cumulative_pd_batch

    cum_pd = project_cumulative_pd_batch(
        hazard["model"], hazard["scaler"], loans, macro_path, LIFETIME_HORIZON_MONTHS, hazard["max_train_mob"]
    )
    loans["pd_12m"] = cum_pd[:, min(11, cum_pd.shape[1] - 1)]
    loans["pd_lifetime"] = cum_pd[:, -1]
    loans["pd_used"] = np.where(loans["stage"] == 1, loans["pd_12m"], loans["pd_lifetime"])
    loans["ead"] = loans["upb"]

    horizon_months = np.where(loans["stage"] == 1, 12, LIFETIME_HORIZON_MONTHS)
    discount_t = horizon_months / 2  # midpoint-of-window discounting
    loans["discount_factor"] = 1 / (1 + loans["orig_rate"] / 1200) ** discount_t

    loans["ecl"] = loans["pd_used"] * loans["lgd"] * loans["ead"] * loans["discount_factor"]
    loans["scenario"] = scenario

    return loans[
        [
            "loan_id", "reporting_date", "scenario", "stage", "bucket_curr",
            "pd_12m", "pd_lifetime", "pd_used", "lgd", "ead", "discount_factor", "ecl",
        ]
    ]


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    frames = [compute_ecl(s, con) for s in stress_scenarios.SCENARIOS]
    con.close()

    mart = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(OUT_PATH, index=False)

    print("ECL by scenario and stage:")
    print(mart.groupby(["scenario", "stage"])["ecl"].agg(["count", "sum"]).round(0))
    print("\nTotal ECL by scenario:")
    print(mart.groupby("scenario")["ecl"].sum().round(0))
    print(f"\nSaved {len(mart):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
