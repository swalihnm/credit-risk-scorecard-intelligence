"""Real FRED API ingestion. Not wired into the pipeline yet.

Set FRED_API_KEY (free, instant signup: https://fred.stlouis.gov/docs/api/api_key.html)
in a .env file, then run this to replace the synthetic macro panel with real series.
Output schema matches ingest/synthetic_fred.py exactly, so nothing downstream needs to change.
"""
from __future__ import annotations

import os
import pathlib

import pandas as pd
from dotenv import load_dotenv

OUT_PATH = "sources/fred/macro_monthly.parquet"
SERIES = {"UNRATE": "unrate", "CSUSHPINSA": "hpi", "DGS10": "dgs10", "GDPC1": "gdpc1"}


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise SystemExit(
            "FRED_API_KEY not set. Get a free key at "
            "https://fred.stlouis.gov/docs/api/api_key.html and put it in .env"
        )

    from fredapi import Fred

    fred = Fred(api_key=api_key)

    frames = []
    for series_id, col in SERIES.items():
        s = fred.get_series(series_id)
        frames.append(s.rename(col))

    # Each series is resampled to month-start *individually* first, taking the last
    # observation within each month (correct for DGS10's daily cadence -- there is no
    # guarantee day 1 of a month is a trading day). Only then are they combined and
    # forward-filled for GDPC1's quarterly gaps.
    #
    # Chaining `.resample("MS").ffill()` directly on the already-concatenated frame (the
    # first version of this function) is NOT equivalent to this: that shorthand only keeps
    # a value when the original data lands exactly on day 1 of the month, which is rare for
    # a daily series and never happens for most months of a quarterly one. It silently
    # dropped the dataset from the expected ~470 months (1975-2026) down to 81 scattered
    # months before `dropna()` even ran, because almost every bin was NaN for DGS10 or
    # GDPC1 and only "got lucky" occasionally when day 1 coincided with a reporting date.
    monthly = pd.concat([s.resample("MS").last() for s in frames], axis=1)
    df = monthly.ffill().dropna().reset_index()
    df = df.rename(columns={"index": "date"})
    df["source"] = "fred_real"

    out = pathlib.Path(OUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} monthly macro rows (real FRED data) to {out}")


if __name__ == "__main__":
    main()
