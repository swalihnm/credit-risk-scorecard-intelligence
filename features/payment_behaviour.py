"""Aggregate installments_payments and credit_card_balance into payment-behaviour signals.

Both source tables record history strictly prior to the current application decision
(DAYS_INSTALMENT / DAYS_ENTRY_PAYMENT / MONTHS_BALANCE are all <= 0) — see
features/leakage_audit.md.
"""
from __future__ import annotations

import duckdb
import pandas as pd


def build_installments(con: duckdb.DuckDBPyConnection, table_prefix: str = "hc") -> pd.DataFrame:
    table = f"{table_prefix}_installments_payments"
    return con.execute(
        f"""
        SELECT
            SK_ID_CURR,
            AVG(CASE WHEN AMT_INSTALMENT > 0 THEN AMT_PAYMENT / AMT_INSTALMENT END) AS instalment_avg_payment_ratio,
            AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT) AS instalment_avg_days_late,
            AVG(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END) AS instalment_pct_late,
            AVG(CASE WHEN DAYS_ENTRY_PAYMENT < DAYS_INSTALMENT THEN 1.0 ELSE 0.0 END) AS instalment_pct_early
        FROM {table}
        GROUP BY SK_ID_CURR
        """
    ).fetchdf()


def build_credit_card(con: duckdb.DuckDBPyConnection, table_prefix: str = "hc") -> pd.DataFrame:
    table = f"{table_prefix}_credit_card_balance"
    cc = con.execute(
        f"""
        SELECT
            SK_ID_CURR,
            MONTHS_BALANCE,
            CASE WHEN AMT_CREDIT_LIMIT_ACTUAL > 0 THEN AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL END AS utilization,
            CASE WHEN AMT_PAYMENT_TOTAL_CURRENT <= COALESCE(AMT_INST_MIN_REGULARITY, 0) * 1.05
                 THEN 1 ELSE 0 END AS min_payment_only
        FROM {table}
        """
    ).fetchdf()

    agg = (
        cc.groupby("SK_ID_CURR")
        .agg(
            cc_avg_utilization=("utilization", "mean"),
            cc_min_payment_only_freq=("min_payment_only", "mean"),
        )
        .reset_index()
    )

    recent = cc[cc["MONTHS_BALANCE"] >= -6].groupby("SK_ID_CURR")["utilization"].mean().rename("util_recent_6m")
    older = cc[cc["MONTHS_BALANCE"] < -6].groupby("SK_ID_CURR")["utilization"].mean().rename("util_older")
    trend = pd.concat([recent, older], axis=1).reset_index()
    trend["cc_utilization_trend"] = trend["util_recent_6m"] - trend["util_older"]
    trend = trend[["SK_ID_CURR", "cc_utilization_trend"]]

    return agg.merge(trend, on="SK_ID_CURR", how="left")


def build(con: duckdb.DuckDBPyConnection, table_prefix: str = "hc") -> pd.DataFrame:
    inst = build_installments(con, table_prefix)
    cc = build_credit_card(con, table_prefix)
    return inst.merge(cc, on="SK_ID_CURR", how="outer")
