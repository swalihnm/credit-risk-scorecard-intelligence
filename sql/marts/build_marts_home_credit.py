"""Builds the Home-Credit-side marts that need the scorecard to already exist: mart_application,
mart_model_performance, mart_psi_reference, dim_score_band. See sql/marts/build_marts.py for
the Fannie-Mae-side marts (built independently, no scorecard dependency).
"""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py).
from optbinning import OptimalBinning

import json
import pathlib
import pickle

import numpy as np
import pandas as pd

from policy.cutoff_optimizer import ASSUMED_APR, CAPITAL_RATIO, LGD_ASSUMPTION, POLICY_CUTOFF_PATH, score_population
from scorecard import binning

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "scorecard" / "artifacts"
MARTS_DIR = ROOT / "marts"

SCORE_BAND_EDGES = [-np.inf, 500, 550, 600, 650, 700, np.inf]
SCORE_BAND_LABELS = ["<500", "500-550", "550-600", "600-650", "650-700", "700+"]


def load_policy_cutoff() -> int:
    """Read the cutoff policy/cutoff_optimizer.py actually computed, rather than a hardcoded
    number that silently goes stale whenever the data or constraints change -- this repo has
    already had to chase down a hardcoded cutoff value in multiple files twice during
    development (see README's bug list)."""
    if not POLICY_CUTOFF_PATH.exists():
        raise SystemExit(f"{POLICY_CUTOFF_PATH} not found. Run policy/cutoff_optimizer.py first.")
    with open(POLICY_CUTOFF_PATH) as f:
        return json.load(f)["policy_cutoff"]


def build_dim_score_band() -> pd.DataFrame:
    return pd.DataFrame({"band": SCORE_BAND_LABELS, "sort_order": range(len(SCORE_BAND_LABELS))})


def build_mart_application() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "marts" / "feature_table.parquet")
    scored = score_population(df)
    policy_cutoff = load_policy_cutoff()

    scored["lgd"] = LGD_ASSUMPTION
    scored["expected_revenue"] = scored["ead"] * ASSUMED_APR * scored["expected_tenure_years"]
    scored["capital_required"] = CAPITAL_RATIO * scored["ead"]
    scored["decision"] = np.where(scored["score"] >= policy_cutoff, "Approved", "Declined")
    scored["score_band"] = pd.cut(scored["score"], SCORE_BAND_EDGES, labels=SCORE_BAND_LABELS, right=False)

    target_lookup = df.set_index("SK_ID_CURR")["TARGET"]
    scored["target"] = scored["SK_ID_CURR"].map(target_lookup)

    return scored[
        ["SK_ID_CURR", "score", "score_band", "pd", "lgd", "ead", "expected_revenue",
         "capital_required", "decision", "income_tercile", "target"]
    ]


def build_mart_psi_reference() -> pd.DataFrame:
    """Development-sample (train) expected bin distribution for each final scorecard
    characteristic, plus the score itself -- reused directly from the already-fitted
    optbinning binning tables rather than recomputed.
    """
    binning_objects = binning.load()
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        model_artifact = pickle.load(f)
    features = model_artifact["features"]

    rows = []
    for feature_woe in features:
        char = feature_woe.removesuffix("_woe")
        table = binning_objects[char].binning_table.build()
        for _, row in table.iterrows():
            if row["Bin"] in ("", "Totals"):
                continue
            rows.append({"characteristic": char, "bin": str(row["Bin"]), "expected_pct": row["Count (%)"]})

    train = pd.read_parquet(ARTIFACTS / "scorecard_train.parquet")
    scored_train = score_population(pd.read_parquet(ROOT / "marts" / "feature_table.parquet"))
    scored_train = scored_train[scored_train["SK_ID_CURR"].isin(train["SK_ID_CURR"])]
    score_band = pd.cut(scored_train["score"], SCORE_BAND_EDGES, labels=SCORE_BAND_LABELS, right=False)
    band_pct = (score_band.value_counts(normalize=True) * 100).reindex(SCORE_BAND_LABELS)
    for band, pct in band_pct.items():
        rows.append({"characteristic": "score", "bin": band, "expected_pct": float(pct) if not pd.isna(pct) else 0.0})

    return pd.DataFrame(rows)


def build_mart_model_performance(n_months: int = 12, seed: int = 7) -> pd.DataFrame:
    """Simulated 12-month monitoring history via bootstrap resampling of the held-out test
    set. A model this recently built has no real monitoring history yet -- this is what the
    monitoring page will look like once one accumulates, disclosed explicitly rather than
    presented as real production monitoring data (see docs/06-assumptions-and-limitations.md).
    No drift is injected: this shows a currently-healthy model, consistent with a scorecard
    that was just validated in-sample and out-of-sample above.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        model_artifact = pickle.load(f)
    model, features = model_artifact["model"], model_artifact["features"]
    test_df = pd.read_parquet(ARTIFACTS / "scorecard_test.parquet")

    rng = np.random.default_rng(seed)
    months = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1), periods=n_months, freq="MS")

    rows = []
    for month in months:
        sample = test_df.sample(n=min(3000, len(test_df)), replace=True, random_state=rng.integers(0, 1_000_000))
        proba = model.predict_proba(sample[features].values)[:, 1]
        y = sample["TARGET"].values
        auc = roc_auc_score(y, proba)
        rows.append(
            {
                "reporting_month": month,
                "segment": "All",
                "gini": 2 * auc - 1,
                "ks": _ks_statistic(y, proba),
                "auc": auc,
                "brier": brier_score_loss(y, proba),
            }
        )
    return pd.DataFrame(rows)


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    cum_bad = np.cumsum(y_sorted) / max(n_bad, 1)
    cum_good = np.cumsum(1 - y_sorted) / max(n_good, 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def main() -> None:
    MARTS_DIR.mkdir(parents=True, exist_ok=True)

    dim_score_band = build_dim_score_band()
    dim_score_band.to_parquet(MARTS_DIR / "dim_score_band.parquet", index=False)

    mart_application = build_mart_application()
    mart_application.to_parquet(MARTS_DIR / "mart_application.parquet", index=False)

    mart_psi_reference = build_mart_psi_reference()
    mart_psi_reference.to_parquet(MARTS_DIR / "mart_psi_reference.parquet", index=False)

    mart_model_performance = build_mart_model_performance()
    mart_model_performance.to_parquet(MARTS_DIR / "mart_model_performance.parquet", index=False)

    for name, df in [
        ("dim_score_band", dim_score_band), ("mart_application", mart_application),
        ("mart_psi_reference", mart_psi_reference), ("mart_model_performance", mart_model_performance),
    ]:
        print(f"  {name:<24} {len(df):>10,} rows")
    print(f"\nWritten to {MARTS_DIR}")
    print(f"Approval rate at policy cutoff {load_policy_cutoff()}: {(mart_application['decision'] == 'Approved').mean():.4f}")


if __name__ == "__main__":
    main()
