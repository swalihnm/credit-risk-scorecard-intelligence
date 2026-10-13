"""Adverse-action reason codes: for a declined applicant, rank their characteristics by
points contribution and surface the lowest (worst) contributors as the reason codes.

This is straightforward for a points-based scorecard precisely because points ARE the
model's attribution mechanism -- a characteristic's points contribution is not an
approximation of "why," it IS why, by construction. See docs/07-results.md for the
discussion of why a GBM + SHAP is materially harder to defend on this specific point.
"""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py).
from optbinning import OptimalBinning

import pathlib
import pickle

import pandas as pd

from scorecard import binning

ARTIFACTS = pathlib.Path("scorecard/artifacts")

REASON_CODE_TEXT = {
    "EXT_SOURCE_1": "External credit bureau score 1 below approval threshold",
    "EXT_SOURCE_2": "External credit bureau score 2 below approval threshold",
    "EXT_SOURCE_3": "External credit bureau score 3 below approval threshold",
    "bureau_avg_utilization": "High utilization on existing external credit lines",
    "bureau_pct_months_delinquent": "History of delinquency on external credit lines",
    "bureau_max_dpd_ever": "History of severe delinquency on external credit lines",
    "bureau_n_active_credit": "Number of active external credit lines",
    "bureau_n_closed_credit": "Limited history of closed/repaid credit lines",
    "bureau_debt_to_income": "External debt relative to income",
    "AMT_INCOME_TOTAL": "Income relative to requested credit amount",
    "AMT_ANNUITY": "Requested annuity amount relative to profile",
    "AMT_CREDIT": "Requested credit amount relative to profile",
    "AMT_GOODS_PRICE": "Requested credit amount relative to the price of goods financed",
    "DAYS_EMPLOYED": "Length of employment history",
    "DAYS_BIRTH": "Applicant age relative to profile",
    "DAYS_ID_PUBLISH": "Time since identity document was issued",
    "CNT_FAM_MEMBERS": "Number of family members",
    "instalment_avg_days_late": "History of late payments on prior instalment loans",
    "instalment_avg_payment_ratio": "History of underpaying scheduled instalments on prior loans",
    "instalment_pct_late": "Frequency of late payments on prior instalment loans",
    "instalment_pct_early": "Frequency of early payments on prior instalment loans",
    "cc_avg_utilization": "High utilization on existing credit card balances",
    "cc_min_payment_only_freq": "Frequency of paying only the minimum due on credit cards",
    "cc_utilization_trend": "Trend in credit card utilization over recent months",
    "prev_n_applications": "Number of prior credit applications",
    "prev_approval_rate": "Approval rate on prior credit applications",
    "prev_refusal_rate": "Refusal rate on prior credit applications",
    "prev_downsell_ratio": "History of being granted less credit than requested",
}


def top_reason_codes(sk_id_curr: int, df: pd.DataFrame, n_reasons: int = 4) -> list[dict]:
    binning_objects = binning.load()
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        model_artifact = pickle.load(f)
    from scorecard.scaling import compute_scaling_constants

    model, features = model_artifact["model"], model_artifact["features"]
    factor, _ = compute_scaling_constants()

    row = df[df["SK_ID_CURR"] == sk_id_curr].iloc[0]
    contributions = []
    for feature_woe, coef in zip(features, model.coef_[0]):
        char = feature_woe.removesuffix("_woe")
        woe_value = binning_objects[char].transform([row[char]], metric="woe")[0]
        points = -factor * coef * woe_value
        contributions.append({"characteristic": char, "points": points, "reason": REASON_CODE_TEXT.get(char, char)})

    contributions.sort(key=lambda c: c["points"])  # most negative (worst) first
    return contributions[:n_reasons]


def main() -> None:
    df = pd.read_parquet("marts/feature_table.parquet")
    mart_application = pd.read_parquet("marts/mart_application.parquet")
    declined = mart_application[mart_application["decision"] == "Declined"]

    print(f"{len(declined):,} declined applicants. Example adverse-action reason codes:\n")
    for sk_id_curr in declined["SK_ID_CURR"].head(3):
        reasons = top_reason_codes(sk_id_curr, df)
        print(f"Applicant {sk_id_curr}:")
        for r in reasons:
            print(f"  [{r['points']:+.1f} pts] {r['characteristic']:<30} {r['reason']}")
        print()


if __name__ == "__main__":
    main()
