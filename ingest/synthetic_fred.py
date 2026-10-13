"""Generate a synthetic macroeconomic panel matching the FRED API schema.

This is an explicitly labelled STAND-IN for real FRED data (see docs/01-data-sources.md).
Series shapes are hand-tuned to be directionally realistic (2008 crisis, benign 2015-2019)
so that vintage-curve / stress-testing logic downstream can be exercised end-to-end.

Swap-in path: ingest/fred_ingest.py reads the real FRED API with the same output schema
(date, unrate, hpi, dgs10, gdpc1) once FRED_API_KEY is set — nothing downstream changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT_PATH = "sources/fred/macro_monthly.parquet"

START = "2003-01-01"
END = "2021-12-01"


def _crisis_bump(dates: pd.DatetimeIndex, peak: str, width_months: int, height: float) -> np.ndarray:
    peak_idx = dates.get_indexer([pd.Timestamp(peak)], method="nearest")[0]
    t = np.arange(len(dates))
    return height * np.exp(-0.5 * ((t - peak_idx) / (width_months / 2.35)) ** 2)


def build_macro_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="MS")
    n = len(dates)

    # Unemployment rate (%): benign baseline ~5%, crisis peak ~10% around 2009-10,
    # mild elevation 2020Q2 (COVID-shaped notch) for extra realism.
    unrate = 5.0 + _crisis_bump(dates, "2009-10-01", 30, 5.0) + _crisis_bump(dates, "2020-04-01", 3, 6.0)
    unrate += rng.normal(0, 0.15, n).cumsum() * 0.02  # slow drift/noise
    unrate = np.clip(unrate, 3.4, 14.8)

    # Case-Shiller HPI (index, 2000=100-ish): rises into 2006, falls ~30pk-trough into 2011-12,
    # recovers and grows steadily 2015-2019.
    hpi = np.empty(n)
    hpi[0] = 140.0
    for i in range(1, n):
        d = dates[i]
        if d < pd.Timestamp("2006-07-01"):
            drift = 0.010
        elif d < pd.Timestamp("2012-01-01"):
            drift = -0.007
        else:
            drift = 0.0045
        hpi[i] = hpi[i - 1] * (1 + drift + rng.normal(0, 0.004))

    # 10Y Treasury yield (%): secular decline from ~4.5% to ~1.5-2%.
    dgs10 = 4.6 - 0.09 * (np.arange(n) / 12) + rng.normal(0, 0.08, n)
    dgs10 = np.clip(dgs10, 0.5, 5.2)

    # Real GDP (index, chained): steady growth with a 2008-09 contraction and 2020 shock.
    gdp = np.empty(n)
    gdp[0] = 100.0
    for i in range(1, n):
        d = dates[i]
        if pd.Timestamp("2008-01-01") <= d <= pd.Timestamp("2009-06-01"):
            g = -0.004
        elif pd.Timestamp("2020-02-01") <= d <= pd.Timestamp("2020-05-01"):
            g = -0.02
        else:
            g = 0.0021
        gdp[i] = gdp[i - 1] * (1 + g + rng.normal(0, 0.0015))

    return pd.DataFrame(
        {
            "date": dates,
            "unrate": unrate.round(2),
            "hpi": hpi.round(2),
            "dgs10": dgs10.round(2),
            "gdpc1": gdp.round(2),
            "source": "synthetic",
        }
    )


def main() -> None:
    import pathlib

    df = build_macro_panel()
    out = pathlib.Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} monthly macro rows to {out}")
    print(df.select_dtypes("number").describe().round(2))


if __name__ == "__main__":
    main()
