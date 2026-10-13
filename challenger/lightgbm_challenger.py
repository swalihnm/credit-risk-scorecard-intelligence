"""LightGBM challenger model, compared against the logistic scorecard on Gini/KS/calibration,
plus SHAP for explainability. Uses the raw (pre-WOE) engineered features -- the point of a
challenger is to see what's left on the table when the monotonic-binning constraint is
relaxed, not to re-run the same WOE pipeline through a different estimator.
"""
from __future__ import annotations

# Import order: optbinning before pandas -- this module later calls scorecard.binning.load(),
# which imports optbinning. See scorecard/binning.py for why the order matters on this
# Windows/Python 3.12 setup (a native DLL-loading-order conflict, not a code bug).
from optbinning import OptimalBinning  # noqa: F401

import pathlib
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ARTIFACTS = pathlib.Path("challenger/artifacts")

CATEGORICAL_COLS = ["NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "CODE_GENDER"]


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    cum_bad = np.cumsum(y_sorted) / max(n_bad, 1)
    cum_good = np.cumsum(1 - y_sorted) / max(n_good, 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def main() -> None:
    df = pd.read_parquet("marts/feature_table.parquet")
    feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]

    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    y = df["TARGET"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.3, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLS)

    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    print("=== LightGBM challenger ===")
    for name, y_split, proba in [("train", y_train, proba_train), ("test", y_test, proba_test)]:
        auc = roc_auc_score(y_split, proba)
        gini = 2 * auc - 1
        ks = _ks_statistic(y_split, proba)
        brier = brier_score_loss(y_split, proba)
        print(f"  {name:<6} AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}  Brier={brier:.5f}")

    # --- compare against the logistic scorecard on the SAME test split ---
    with open("scorecard/artifacts/logistic_model.pkl", "rb") as f:
        scorecard_artifact = pickle.load(f)
    scorecard_model = scorecard_artifact["model"]
    scorecard_features = scorecard_artifact["features"]

    from scorecard import binning as scorecard_binning  # optbinning-before-pandas already satisfied

    binning_objects = scorecard_binning.load()
    woe_test = pd.DataFrame(index=X_test.index)
    for feature_woe in scorecard_features:
        char = feature_woe.removesuffix("_woe")
        woe_test[feature_woe] = binning_objects[char].transform(df.loc[X_test.index, char].values, metric="woe")
    scorecard_proba_test = scorecard_model.predict_proba(woe_test.values)[:, 1]
    scorecard_auc = roc_auc_score(y_test, scorecard_proba_test)

    print(f"\n=== Head-to-head on the same test split ===")
    print(f"  Logistic scorecard (12 char.):  Gini={2*scorecard_auc-1:.4f}")
    print(f"  LightGBM challenger:             Gini={2*roc_auc_score(y_test, proba_test)-1:.4f}")

    # --- SHAP ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP value|:")
    print(mean_abs_shap.head(10).round(4))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "lightgbm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feature_cols, "categorical_cols": CATEGORICAL_COLS}, f)
    mean_abs_shap.to_frame("mean_abs_shap").to_csv(ARTIFACTS / "shap_importance.csv")
    print(f"\nSaved model and SHAP importance to {ARTIFACTS}")


if __name__ == "__main__":
    main()
