"""Real Fannie Mae Single-Family Loan Performance ingestion. Not wired into the pipeline yet.

Fannie Mae's Data Dynamics portal (https://datadynamics.fanniemae.com/data-dynamics/) requires
a human login and manual extract download — there is no API token flow, so this cannot be
automated the way ingest/kaggle_download.py is.

To swap real data in for the synthetic stand-in (ingest/synthetic_fannie_mae.py):
1. Register and download the "Single-Family Loan Performance" flat files for the vintages you
   want (recommended: 2005-2010 plus 2015-2019, per docs/01-data-sources.md).
2. Place the raw extract files under sources/fannie_mae/raw/.
3. Fill in `parse_acquisition_file()` and `parse_performance_file()` below using Fannie Mae's
   published file layout (column positions/names are documented in their file layout PDF that
   ships with the download).
4. Write to the SAME three output files as the synthetic generator:
   sources/fannie_mae/loans_static.parquet, loan_performance.parquet, loan_defaults.parquet
   with the same columns — nothing downstream (warehouse load, features, portfolio, marts)
   needs to change.
"""
from __future__ import annotations

RAW_DIR = "sources/fannie_mae/raw"


def parse_acquisition_file(path: str):
    raise NotImplementedError(
        "Fill in using Fannie Mae's acquisition file layout once real extracts are downloaded."
    )


def parse_performance_file(path: str):
    raise NotImplementedError(
        "Fill in using Fannie Mae's performance file layout once real extracts are downloaded."
    )


def main() -> None:
    raise SystemExit(
        "Real Fannie Mae ingestion is not implemented. Register at the Data Dynamics portal, "
        "download extracts to sources/fannie_mae/raw/, then implement the two parse_* functions "
        "above. Until then, run ingest/synthetic_fannie_mae.py."
    )


if __name__ == "__main__":
    main()
