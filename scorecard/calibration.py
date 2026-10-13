"""Calibration: Platt scaling vs. isotonic regression, reliability diagram, Brier score,
expected calibration error (ECE), and Hosmer-Lemeshow (with the standard caveat that HL is
sensitive to sample size / bin count and easily rejects on large samples even when
calibration is practically fine).
"""
from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ARTIFACTS = pathlib.Path("scorecard/artifacts")
N_HL_GROUPS = 10


def platt_scaling(raw_proba: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling = a 1-D logistic regression of y on the logit of the raw probability."""
    eps = 1e-6
    logit = np.log(np.clip(raw_proba, eps, 1 - eps) / np.clip(1 - raw_proba, eps, 1 - eps))
    platt = LogisticRegression()
    platt.fit(logit.reshape(-1, 1), y)
    return platt


def apply_platt(platt: LogisticRegression, raw_proba: np.ndarray) -> np.ndarray:
    eps = 1e-6
    logit = np.log(np.clip(raw_proba, eps, 1 - eps) / np.clip(1 - raw_proba, eps, 1 - eps))
    return platt.predict_proba(logit.reshape(-1, 1))[:, 1]


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(proba, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        avg_pred = proba[mask].mean()
        avg_actual = y[mask].mean()
        ece += (mask.sum() / len(y)) * abs(avg_pred - avg_actual)
    return ece


def reliability_table(y: np.ndarray, proba: np.ndarray, n_bins: int = N_HL_GROUPS) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "proba": proba})
    df["decile"] = pd.qcut(df["proba"], n_bins, duplicates="drop")
    return (
        df.groupby("decile", observed=True)
        .agg(n=("y", "size"), predicted_pd=("proba", "mean"), actual_default_rate=("y", "mean"))
        .reset_index()
    )


def hosmer_lemeshow(y: np.ndarray, proba: np.ndarray, n_groups: int = N_HL_GROUPS) -> tuple[float, float]:
    df = pd.DataFrame({"y": y, "proba": proba})
    df["group"] = pd.qcut(df["proba"], n_groups, duplicates="drop")
    stat = 0.0
    for _, g in df.groupby("group", observed=True):
        n = len(g)
        observed = g["y"].sum()
        expected = g["proba"].sum()
        var = (g["proba"] * (1 - g["proba"])).sum()
        if var > 0:
            stat += (observed - expected) ** 2 / var
    dof = df["group"].nunique() - 2
    p_value = 1 - chi2.cdf(stat, dof) if dof > 0 else float("nan")
    return stat, p_value


def main() -> None:
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        model_artifact = pickle.load(f)
    train_df = pd.read_parquet(ARTIFACTS / "scorecard_train.parquet")
    test_df = pd.read_parquet(ARTIFACTS / "scorecard_test.parquet")
    model, features = model_artifact["model"], model_artifact["features"]

    raw_train = model.predict_proba(train_df[features].values)[:, 1]
    raw_test = model.predict_proba(test_df[features].values)[:, 1]
    y_train, y_test = train_df["TARGET"].values, test_df["TARGET"].values

    platt = platt_scaling(raw_train, y_train)
    platt_test = apply_platt(platt, raw_test)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_train, y_train)
    iso_test = iso.predict(raw_test)

    print("Calibration comparison on the held-out test set:")
    for name, proba in [("raw logistic", raw_test), ("Platt-scaled", platt_test), ("isotonic", iso_test)]:
        brier = brier_score_loss(y_test, proba)
        ece = expected_calibration_error(y_test, proba)
        print(f"  {name:<14} Brier={brier:.5f}  ECE={ece:.5f}")

    print("\nReliability table (raw logistic, deciles):")
    print(reliability_table(y_test, raw_test).round(4).to_string(index=False))

    hl_stat, hl_p = hosmer_lemeshow(y_test, raw_test)
    print(
        f"\nHosmer-Lemeshow: chi2={hl_stat:.2f}, p={hl_p:.4f} "
        f"(n={len(y_test):,} -- HL is well known to reject on large samples even when "
        f"calibration is practically adequate; read this alongside the reliability table "
        f"and ECE above, not in isolation)"
    )

    # Platt scaling is chosen as the deployed calibrator: this dataset's default rate is low
    # (~8%) and isotonic regression can overfit / produce a jagged step function in sparse
    # probability regions with this few observed events, whereas Platt's 1-parameter logistic
    # form is stable and, per the Brier/ECE comparison above, performs comparably here.
    with open(ARTIFACTS / "calibrator.pkl", "wb") as f:
        pickle.dump({"platt": platt, "isotonic": iso, "chosen": "platt"}, f)
    print(f"\nSaved calibrators to {ARTIFACTS / 'calibrator.pkl'} (chosen: Platt)")


if __name__ == "__main__":
    main()
