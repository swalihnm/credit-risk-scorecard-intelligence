"""Aggregate bureau + bureau_balance into applicant-level (SK_ID_CURR) risk signals.

Every feature here is built only from records with negative DAYS_* (i.e. dated before the
current application) or from bureau_balance MONTHS_BALANCE (always <= 0, relative to the
current application) — see features/leakage_audit.md for the per-feature justification.
"""
from __future__ import annotations

import duckdb
import pandas as pd

# bureau_balance STATUS -> numeric DPD bucket. 'C' (closed) and 'X' (unknown) excluded from
# the max-DPD calc; 'C'/'X' months still count in the denominator for pct_months_delinquent.
STATUS_TO_DPD = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def build(con: duckdb.DuckDBPyConnection, table_prefix: str = "hc") -> pd.DataFrame:
    bureau_table = f"{table_prefix}_bureau"
    bureau_balance_table = f"{table_prefix}_bureau_balance"
    active_closed = con.execute(
        f"""
        SELECT
            SK_ID_CURR,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS bureau_n_active_credit,
            SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS bureau_n_closed_credit,
            AVG(
                CASE WHEN CREDIT_ACTIVE = 'Active' AND AMT_CREDIT_SUM > 0
                     THEN LEAST(AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM, 1.0)
                END
            ) AS bureau_avg_utilization,
            SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_debt,
            AVG(
                CASE WHEN DAYS_CREDIT_UPDATE >= -180 AND CREDIT_ACTIVE = 'Active' AND AMT_CREDIT_SUM > 0
                     THEN LEAST(AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM, 1.0)
                END
            ) AS util_recent_6m,
            AVG(
                CASE WHEN DAYS_CREDIT_UPDATE >= -720 AND CREDIT_ACTIVE = 'Active' AND AMT_CREDIT_SUM > 0
                     THEN LEAST(AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM, 1.0)
                END
            ) AS util_recent_24m
        FROM {bureau_table}
        GROUP BY SK_ID_CURR
        """
    ).fetchdf()
    active_closed["bureau_utilization_trend_6m_minus_24m"] = (
        active_closed["util_recent_6m"] - active_closed["util_recent_24m"]
    )
    active_closed = active_closed.drop(columns=["util_recent_6m", "util_recent_24m"])

    balance_status = con.execute(
        f"""
        SELECT
            b.SK_ID_CURR,
            bb.STATUS,
            bb.MONTHS_BALANCE
        FROM {bureau_balance_table} bb
        JOIN {bureau_table} b ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
        """
    ).fetchdf()

    balance_status["dpd"] = balance_status["STATUS"].map(STATUS_TO_DPD)
    balance_status["is_delinquent"] = balance_status["dpd"].fillna(0) > 0

    dpd_agg = (
        balance_status.groupby("SK_ID_CURR")
        .agg(
            bureau_max_dpd_ever=("dpd", "max"),
            bureau_pct_months_delinquent=("is_delinquent", "mean"),
        )
        .reset_index()
    )
    dpd_agg["bureau_max_dpd_ever"] = dpd_agg["bureau_max_dpd_ever"].fillna(0)

    out = active_closed.merge(dpd_agg, on="SK_ID_CURR", how="outer")
    return out
