"""LGD (loss given default) segmented by LTV band and region, from realized Fannie Mae
loss severities (net_loss / UPB at default).

Small segments (few observed defaults) are shrunk toward the portfolio-wide mean so a
segment with 3 defaults doesn't drive an extreme LGD estimate — a simple credibility-weighted
average, not a claim of statistical sophistication beyond what a handful of loss observations
can support.
"""
from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "warehouse" / "credit_risk.duckdb"
OUT_PATH = ROOT / "portfolio" / "artifacts" / "lgd_table.parquet"

MIN_CREDIBLE_N = 30  # below this, shrink hard toward the portfolio mean


def build_lgd_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
            l.loan_id,
            l.region,
            CASE
                WHEN l.original_ltv < 60 THEN '<60'
                WHEN l.original_ltv < 70 THEN '60-70'
                WHEN l.original_ltv < 80 THEN '70-80'
                WHEN l.original_ltv < 90 THEN '80-90'
                ELSE '90+'
            END AS ltv_band,
            d.severity
        FROM fm_defaults d
        JOIN fm_loans l USING (loan_id)
        """
    ).fetchdf()

    portfolio_mean = df["severity"].mean()

    seg = (
        df.groupby(["region", "ltv_band"])["severity"]
        .agg(n="count", raw_lgd="mean")
        .reset_index()
    )
    # credibility weight: n / (n + k) toward the observed segment mean, else the portfolio mean
    seg["credibility"] = seg["n"] / (seg["n"] + MIN_CREDIBLE_N)
    seg["lgd"] = seg["credibility"] * seg["raw_lgd"] + (1 - seg["credibility"]) * portfolio_mean
    seg["portfolio_mean_lgd"] = portfolio_mean

    return seg[["region", "ltv_band", "n", "raw_lgd", "lgd", "portfolio_mean_lgd"]]


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    lgd_table = build_lgd_table(con)
    con.close()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lgd_table.to_parquet(OUT_PATH, index=False)

    print(f"LGD table ({len(lgd_table)} region x LTV-band segments):")
    print(lgd_table.sort_values("lgd", ascending=False).round(3).to_string(index=False))
    print(f"\nPortfolio-wide mean LGD: {lgd_table['portfolio_mean_lgd'].iloc[0]:.3f}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
