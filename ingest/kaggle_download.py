"""Pull the Home Credit Default Risk competition tables via the Kaggle API.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (see docs/00-kaggle-setup.md)
and acceptance of the competition rules on kaggle.com.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "sources" / "home_credit"
COMPETITION = "home-credit-default-risk"

EXPECTED_FILES = [
    "application_train.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    DEST.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {COMPETITION} to {DEST} ...")
    api.competition_download_files(COMPETITION, path=str(DEST), quiet=False)

    zip_path = DEST / f"{COMPETITION}.zip"
    if zip_path.exists():
        print("Extracting archive ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DEST)
        zip_path.unlink()

    missing = [f for f in EXPECTED_FILES if not (DEST / f).exists()]
    if missing:
        raise SystemExit(f"Missing expected files after download: {missing}")

    print("All 7 Home Credit tables present:")
    for f in EXPECTED_FILES:
        size_mb = (DEST / f).stat().st_size / 1e6
        print(f"  {f:<32} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
