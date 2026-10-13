"""Reject inference via fuzzy augmentation.

Training data (application_train, real or this synthetic stand-in) contains only APPROVED
applicants -- fitting a scorecard on that alone means fitting on a sample biased by whatever
credit policy already existed (see ingest/synthetic_home_credit.py's build_reject_pool). This
implements fuzzy augmentation, the most common practical technique:

1. Score the reject pool with the accept-only model.
2. Inflate the accept-only model's predicted PD for rejects by REJECT_INFLATION -- the
   standard reject-inference assumption that a rejected applicant is systematically worse
   than an accepted applicant with the *same* observable score, because the population that
   got rejected differs from the accepted population in ways the accept-only model, having
   never seen a reject's outcome, cannot learn.
3. Duplicate every reject as both a weighted "good" and a weighted "bad" record, weighted by
   that inflated probability, and refit alongside the (unit-weighted, observed-outcome)
   accepted population.
4. Quantify the impact: how far the approval cutoff moves under a fixed bad-rate policy,
   comparing the accept-only vs. reject-inference-augmented model.

Discussion of the two alternatives named in the project spec:
- Parcelling would bin rejects into score bands and assign a single inferred bad rate per
  band (usually also inflated) rather than a continuous per-applicant probability -- coarser,
  but more transparent to a risk committee since the assumption is stated once per band
  instead of once per applicant. Fuzzy augmentation was chosen here because the continuous
  probability makes the *validation-against-hidden-truth* step below meaningful at the
  applicant level, not just at the band level.
- A Heckman two-stage model would explicitly model the selection process (why some applicants
  were rejected) jointly with the outcome model, correcting for selection bias statistically
  rather than by assumption. It is more defensible in principle but requires an instrument
  that shifts acceptance without directly affecting default risk, which is not available
  here -- exactly the practical reason most industry implementations use parcelling or fuzzy
  augmentation instead, despite the method being taught first in most textbooks.

IMPORTANT: this build can validate the inferred reject bad rate against a hidden ground
truth (reject_hidden_truth.csv) purely because the data is synthetic. A real bank can NEVER
do this -- a real reject was never funded, so its true outcome is never observed. That
validation step exists here only to make the technique's mechanics checkable, and is called
out explicitly wherever it's used.
"""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py).
from optbinning import OptimalBinning

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scorecard import binning
from scorecard.scaling import compute_scaling_constants

REJECT_INFLATION = 1.5
POLICY_MAX_BAD_RATE = 0.12  # the credit-policy criterion used for the cutoff-shift comparison

ARTIFACTS = pathlib.Path("scorecard/artifacts")


def score_with_binning(df: pd.DataFrame, binning_objects: dict[str, OptimalBinning], features: list[str]) -> pd.DataFrame:
    woe_df = pd.DataFrame(index=df.index)
    for feature_woe in features:
        char = feature_woe.removesuffix("_woe")
        woe_df[feature_woe] = binning_objects[char].transform(df[char].values, metric="woe")
    return woe_df




def main() -> None:
    accepts = pd.read_parquet("marts/feature_table.parquet")
    rejects = pd.read_parquet("marts/feature_table_rejects.parquet")
    hidden_truth = pd.read_csv("sources/home_credit/reject_hidden_truth.csv")

    binning_objects = binning.load()
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        accept_only_artifact = pickle.load(f)
    accept_only_model: LogisticRegression = accept_only_artifact["model"]
    features = accept_only_artifact["features"]

    woe_accepts = score_with_binning(accepts, binning_objects, features)
    woe_rejects = score_with_binning(rejects, binning_objects, features)

    pd_rejects_raw = accept_only_model.predict_proba(woe_rejects.values)[:, 1]
    pd_rejects_inflated = np.clip(pd_rejects_raw * REJECT_INFLATION, 0, 0.95)

    # --- validation against hidden truth (synthetic-only, see module docstring) ---
    merged = hidden_truth.merge(
        pd.DataFrame({"SK_ID_CURR": rejects["SK_ID_CURR"], "pd_raw": pd_rejects_raw, "pd_inflated": pd_rejects_inflated}),
        on="SK_ID_CURR",
    )
    print("=== Validation against hidden ground truth (synthetic-only; a real bank cannot do this) ===")
    print(f"True default rate among rejects:            {merged['true_target'].mean():.4f}")
    print(f"Accept-only model's raw predicted rate:      {merged['pd_raw'].mean():.4f}  "
          f"(understated -- fit on accepts only)")
    print(f"Reject-inference inflated predicted rate:    {merged['pd_inflated'].mean():.4f}")
    auc_raw = roc_auc_score(merged["true_target"], merged["pd_raw"])
    print(f"AUC of accept-only score ranking rejects by their true outcome: {auc_raw:.4f} "
          f"(rank-ordering is preserved even though the level is understated -- inflation "
          f"corrects the level, not the ranking)")

    # --- fuzzy augmentation ---
    accepts_aug = pd.DataFrame(woe_accepts)
    accepts_aug["target"] = accepts["TARGET"].values
    accepts_aug["weight"] = 1.0

    rejects_bad = pd.DataFrame(woe_rejects)
    rejects_bad["target"] = 1
    rejects_bad["weight"] = pd_rejects_inflated

    rejects_good = pd.DataFrame(woe_rejects)
    rejects_good["target"] = 0
    rejects_good["weight"] = 1 - pd_rejects_inflated

    augmented = pd.concat([accepts_aug, rejects_bad, rejects_good], ignore_index=True)

    augmented_model = LogisticRegression(penalty=None, max_iter=1000)
    augmented_model.fit(augmented[features].values, augmented["target"].values, sample_weight=augmented["weight"].values)

    print("\n=== Coefficient comparison (accept-only vs. reject-inference-augmented) ===")
    coef_compare = pd.DataFrame(
        {"accept_only": accept_only_model.coef_[0], "augmented": augmented_model.coef_[0]}, index=features
    )
    print(coef_compare.round(4))

    # --- quantify the cutoff shift under a fixed credit-policy bad-rate criterion ---
    # Policy: decline any individual applicant whose OWN predicted PD exceeds
    # POLICY_MAX_BAD_RATE -- the standard, individual-level scorecard decision rule (as
    # opposed to a rule on the *average* PD of the approved population, which barely binds
    # here since most applicants already sit well under the threshold and was tried first --
    # see git history for why that approach was replaced).
    through_the_door = pd.concat([woe_accepts, woe_rejects], ignore_index=True)
    pd_accept_only_everyone = accept_only_model.predict_proba(through_the_door.values)[:, 1]
    pd_augmented_everyone = augmented_model.predict_proba(through_the_door.values)[:, 1]

    approve_accept_only = pd_accept_only_everyone <= POLICY_MAX_BAD_RATE
    approve_augmented = pd_augmented_everyone <= POLICY_MAX_BAD_RATE

    true_target_everyone = np.concatenate([accepts["TARGET"].values, hidden_truth["true_target"].values])

    print(f"\n=== Approval impact under a fixed policy (decline if predicted PD > {POLICY_MAX_BAD_RATE:.0%}) ===")
    print(f"Accept-only model:  approval rate = {approve_accept_only.mean():.4f}, "
          f"TRUE bad rate of approved population = {true_target_everyone[approve_accept_only].mean():.4f}")
    print(f"Augmented model:    approval rate = {approve_augmented.mean():.4f}, "
          f"TRUE bad rate of approved population = {true_target_everyone[approve_augmented].mean():.4f}")

    # --- translate into a points-scale cutoff shift ---
    # With every WOE held at 0 (an "average-risk" applicant on every characteristic), score
    # reduces to base_points, so the base_points gap between the two models isolates exactly
    # how much the intercept-driven recalibration moved the whole scorecard, in points --
    # the direct answer to "how far does the cutoff move".
    factor, offset = compute_scaling_constants()
    base_points_accept_only = offset - factor * accept_only_model.intercept_[0]
    base_points_augmented = offset - factor * augmented_model.intercept_[0]
    points_shift = base_points_accept_only - base_points_augmented

    print(f"\n=== Points-scale cutoff shift ===")
    print(f"Base points (accept-only model):   {base_points_accept_only:.1f}")
    print(f"Base points (augmented model):     {base_points_augmented:.1f}")
    print(f"Shift: {points_shift:+.1f} points "
          f"({'the augmented scorecard requires more favorable characteristics to clear any given cutoff -- reject inference made the policy more conservative' if points_shift > 0 else 'the augmented scorecard is more permissive at any given cutoff'})")

    print(
        f"\nThis gap between the two models' view of the through-the-door population is only "
        f"checkable against a ground truth here because the data is synthetic; in production, "
        f"reject inference is validated by monitoring realized bad rates after deployment, not "
        f"against a ground truth."
    )

    with open(ARTIFACTS / "reject_inference.pkl", "wb") as f:
        pickle.dump(
            {
                "augmented_model": augmented_model,
                "features": features,
                "reject_inflation": REJECT_INFLATION,
                "points_shift": points_shift,
                "policy_max_bad_rate": POLICY_MAX_BAD_RATE,
            },
            f,
        )
    print(f"\nSaved augmented model and cutoff comparison to {ARTIFACTS / 'reject_inference.pkl'}")


if __name__ == "__main__":
    main()
