"""WOE transformation and IV screening.

Per the project's own binning-quality rule: drop characteristics with IV < 0.02 (too weak
to matter), flag IV > 0.5 as a possible-leakage smell worth scrutiny (not automatic exclusion
-- see features/leakage_audit.md for why EXT_SOURCE_1/2/3 are flagged rather than dropped
here: they are legitimate pre-decision external bureau scores, just unusually strong ones in
this synthetic dataset).
"""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py for why).
from optbinning import OptimalBinning

import pathlib

import pandas as pd

from scorecard import binning

IV_DROP_BELOW = 0.02
IV_FLAG_ABOVE = 0.5

WOE_TABLE_PATH = pathlib.Path("scorecard/artifacts/woe_table.parquet")
WOE_FEATURES_PATH = pathlib.Path("scorecard/artifacts/woe_features.parquet")


def summarize_iv(binning_objects: dict[str, OptimalBinning]) -> pd.DataFrame:
    rows = []
    for col, ob in binning_objects.items():
        table = ob.binning_table.build()
        iv = table["IV"].sum()
        rows.append({"characteristic": col, "iv": iv, "n_bins": len(ob.splits) + 1})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def apply_woe(df: pd.DataFrame, binning_objects: dict[str, OptimalBinning]) -> pd.DataFrame:
    woe_df = pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"], "TARGET": df["TARGET"]})
    for col, ob in binning_objects.items():
        woe_df[f"{col}_woe"] = ob.transform(df[col].values, metric="woe")
    return woe_df


def main() -> None:
    df = pd.read_parquet("marts/feature_table.parquet")
    binning_objects = binning.load()

    iv_summary = summarize_iv(binning_objects)
    print("IV summary (all fitted characteristics):")
    print(iv_summary.to_string(index=False))

    keep = iv_summary[iv_summary["iv"] >= IV_DROP_BELOW]["characteristic"].tolist()
    dropped = iv_summary[iv_summary["iv"] < IV_DROP_BELOW]["characteristic"].tolist()
    flagged = iv_summary[iv_summary["iv"] > IV_FLAG_ABOVE]["characteristic"].tolist()

    print(f"\nDropped (IV < {IV_DROP_BELOW}): {dropped}")
    print(f"Flagged for leakage scrutiny (IV > {IV_FLAG_ABOVE}): {flagged}")
    print(f"Retained for WOE transform: {len(keep)} characteristics")

    kept_binning = {c: binning_objects[c] for c in keep}
    woe_df = apply_woe(df, kept_binning)

    WOE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    iv_summary.to_parquet(WOE_TABLE_PATH, index=False)
    woe_df.to_parquet(WOE_FEATURES_PATH, index=False)
    print(f"\nSaved IV summary to {WOE_TABLE_PATH}")
    print(f"Saved WOE-transformed features to {WOE_FEATURES_PATH}")


if __name__ == "__main__":
    main()
