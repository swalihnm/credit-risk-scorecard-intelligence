"""Aggregate previous_application into applicant-level historical approval-behaviour signals.

previous_application rows are, by definition, prior credit applications from the same
applicant (DAYS_DECISION < 0 relative to the current application) — see
features/leakage_audit.md.
"""
from __future__ import annotations

import duckdb
import pandas as pd


def build(con: duckdb.DuckDBPyConnection, table_prefix: str = "hc") -> pd.DataFrame:
    table = f"{table_prefix}_previous_application"
    return con.execute(
        f"""
        SELECT
            SK_ID_CURR,
            COUNT(*) AS prev_n_applications,
            AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0.0 END) AS prev_approval_rate,
            AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END) AS prev_refusal_rate,
            AVG(
                CASE WHEN NAME_CONTRACT_STATUS = 'Approved' AND AMT_APPLICATION > 0
                     THEN AMT_CREDIT / AMT_APPLICATION
                END
            ) AS prev_downsell_ratio
        FROM {table}
        GROUP BY SK_ID_CURR
        """
    ).fetchdf()
