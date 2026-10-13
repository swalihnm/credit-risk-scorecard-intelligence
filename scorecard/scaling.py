"""Log-odds to points scaling. Standard scorecard formula:

    Score = Offset + Factor * ln(odds_good_to_bad)
    Factor = PDO / ln(2)
    Offset = Base - Factor * ln(Base_odds)

Base 600 at 50:1 (good:bad) odds, PDO (points to double the odds) = 20.

The fitted logistic regression predicts P(bad=1) from WOE, so
ln(odds_bad_to_good) = intercept + sum(coef_i * WOE_i), and
ln(odds_good_to_bad) = -(intercept + sum(coef_i * WOE_i)).

Points are then allocated per characteristic per bin:
    base_points          = Offset - Factor * intercept
    points_i(bin)        = -Factor * coef_i * WOE_i(bin)
    total score          = base_points + sum(points_i)
"""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py).
from optbinning import OptimalBinning

import pathlib
import pickle

import numpy as np
import pandas as pd

from scorecard import binning

BASE_SCORE = 600
BASE_ODDS = 50  # good:bad
PDO = 20

ARTIFACTS = pathlib.Path("scorecard/artifacts")


def compute_scaling_constants() -> tuple[float, float]:
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return factor, offset


def build_points_card(
    binning_objects: dict[str, OptimalBinning], coefs: dict[str, float], factor: float, base_points: float
) -> pd.DataFrame:
    rows = []
    for feature_woe, coef in coefs.items():
        char = feature_woe.removesuffix("_woe")
        ob = binning_objects[char]
        table = ob.binning_table.build()
        for _, row in table.iterrows():
            if row["Bin"] in ("", "Totals"):
                continue
            points = -factor * coef * row["WoE"]
            rows.append(
                {"characteristic": char, "bin": row["Bin"], "woe": row["WoE"], "points": round(points, 1)}
            )
    card = pd.DataFrame(rows)
    base_row = pd.DataFrame([{"characteristic": "BASE_POINTS", "bin": "", "woe": float("nan"), "points": round(base_points, 1)}])
    return pd.concat([base_row, card], ignore_index=True) if not card.empty else base_row


def score_applicants(
    df: pd.DataFrame, binning_objects: dict[str, OptimalBinning], coefs: dict[str, float],
    factor: float, base_points: float,
) -> np.ndarray:
    score = np.full(len(df), base_points)
    for feature_woe, coef in coefs.items():
        char = feature_woe.removesuffix("_woe")
        ob = binning_objects[char]
        woe_values = ob.transform(df[char].values, metric="woe")
        score = score - factor * coef * woe_values
    return score


def main() -> None:
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        model_artifact = pickle.load(f)
    binning_objects = binning.load()

    model = model_artifact["model"]
    features = model_artifact["features"]
    coefs = dict(zip(features, model.coef_[0]))
    intercept = model.intercept_[0]

    factor, offset = compute_scaling_constants()
    base_points = offset - factor * intercept

    print(f"Scaling constants: Factor={factor:.4f}, Offset={offset:.4f}, base_points={base_points:.2f}")

    points_card = build_points_card(binning_objects, coefs, factor, base_points)
    points_card.to_csv(ARTIFACTS / "points_card.csv", index=False)
    print(f"\nPoints card ({len(points_card)} rows) saved to {ARTIFACTS / 'points_card.csv'}")
    print(points_card.head(15).to_string(index=False))

    feature_table = pd.read_parquet("marts/feature_table.parquet")
    scores = score_applicants(feature_table, binning_objects, coefs, factor, base_points)

    print(f"\nScore distribution: min={scores.min():.0f}, max={scores.max():.0f}, "
          f"mean={scores.mean():.0f}, median={np.median(scores):.0f}")

    out = feature_table[["SK_ID_CURR", "TARGET"]].copy()
    out["score"] = scores.round(0).astype(int)
    out.to_parquet(ARTIFACTS / "applicant_scores.parquet", index=False)
    print(f"Saved per-applicant scores to {ARTIFACTS / 'applicant_scores.parquet'}")

    # sanity: score should be monotonically related to predicted PD (a pure rescaling)
    woe_df = pd.read_parquet(ARTIFACTS / "woe_features.parquet")
    proba = model.predict_proba(woe_df[features].values)[:, 1]
    corr = np.corrcoef(scores, proba)[0, 1]
    print(f"\nCorrelation(score, predicted PD) = {corr:.4f} (expected strongly negative)")


if __name__ == "__main__":
    main()
