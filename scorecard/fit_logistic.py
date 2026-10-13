"""Correlation/VIF filtering, then L1-guided variable selection down to 10-15 final
characteristics, then a plain (unpenalized) refit on just those for the final scorecard --
standard practice: use penalized/stepwise selection to choose variables, refit ordinary
logistic regression on the selected set for clean, interpretable coefficients.

Validation note: the (synthetic, like the real) Home Credit application table has no
application date, so a true out-of-time split isn't possible here -- this is a genuine
limitation of the real dataset too, not just the synthetic stand-in (see
docs/06-assumptions-and-limitations.md). A stratified random train/test split is used
instead; true out-of-time validation is demonstrated on the Fannie Mae side
(portfolio/survival.py), which does have a genuine time dimension.
"""
from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

CORR_THRESHOLD = 0.75
VIF_THRESHOLD = 5.0
TARGET_N_FEATURES = (10, 15)

ARTIFACTS = pathlib.Path("scorecard/artifacts")


def correlation_filter(woe_df: pd.DataFrame, feature_cols: list[str], iv_lookup: pd.Series) -> list[str]:
    corr = woe_df[feature_cols].corr().abs()
    to_drop: set[str] = set()
    cols = list(feature_cols)
    for i, a in enumerate(cols):
        if a in to_drop:
            continue
        for b in cols[i + 1:]:
            if b in to_drop:
                continue
            if corr.loc[a, b] > CORR_THRESHOLD:
                # drop whichever has lower IV
                loser = a if iv_lookup.get(a, 0) < iv_lookup.get(b, 0) else b
                to_drop.add(loser)
    kept = [c for c in feature_cols if c not in to_drop]
    return kept, sorted(to_drop)


def vif_filter(woe_df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    cols = list(feature_cols)
    dropped = []
    while len(cols) > 1:
        X = woe_df[cols].values
        vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_vif = max(vifs)
        if max_vif <= VIF_THRESHOLD:
            break
        worst = cols[int(np.argmax(vifs))]
        cols.remove(worst)
        dropped.append(worst)
    return cols, dropped


def select_via_l1(woe_df: pd.DataFrame, feature_cols: list[str], target: np.ndarray) -> list[str]:
    X = woe_df[feature_cols].values
    for C in [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01]:
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000)
        model.fit(X, target)
        n_nonzero = int((model.coef_[0] != 0).sum())
        if n_nonzero <= TARGET_N_FEATURES[1]:
            selected = [f for f, coef in zip(feature_cols, model.coef_[0]) if coef != 0]
            if len(selected) >= TARGET_N_FEATURES[0] or C == 0.01:
                return selected
    return feature_cols[: TARGET_N_FEATURES[1]]


def enforce_sign_consistency(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], LogisticRegression, list[str]]:
    """Refit repeatedly, dropping any feature whose coefficient sign disagrees with the
    business-logic direction, until every remaining coefficient is negative.

    WOE is coded consistently (via MONOTONIC_TREND in scorecard/binning.py) so that a higher
    WOE always means a safer bin, for every characteristic. A well-specified logistic
    regression predicting TARGET (bad=1) from WOE must therefore have every coefficient
    negative; a positive one signals the multivariate fit reversed sign relative to that
    feature's own univariate WOE/event-rate relationship (residual collinearity with other
    selected features), which is not defensible to a risk committee even though it is not,
    on its own, a modelling error.
    """
    cols = list(feature_cols)
    dropped = []
    while True:
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(train_df[cols].values, train_df["TARGET"].values)
        bad_signs = [c for c, coef in zip(cols, model.coef_[0]) if coef >= 0]
        if not bad_signs:
            return cols, model, dropped
        # drop the worst offender (largest positive coefficient) one at a time
        worst = cols[int(np.argmax(model.coef_[0]))]
        cols.remove(worst)
        dropped.append(worst)
        if len(cols) < 5:
            raise RuntimeError("Sign-consistency pruning removed too many characteristics")


def main() -> None:
    woe_df = pd.read_parquet(ARTIFACTS / "woe_features.parquet")
    iv_summary = pd.read_parquet(ARTIFACTS / "woe_table.parquet")
    iv_lookup = iv_summary.set_index("characteristic")["iv"]

    feature_cols = [c for c in woe_df.columns if c.endswith("_woe")]
    iv_lookup_woe = pd.Series(
        {f"{c}_woe": iv for c, iv in iv_lookup.items()}
    )

    train_df, test_df = train_test_split(
        woe_df, test_size=0.3, stratify=woe_df["TARGET"], random_state=42
    )

    print(f"Starting from {len(feature_cols)} WOE characteristics")

    kept, corr_dropped = correlation_filter(train_df, feature_cols, iv_lookup_woe)
    print(f"After correlation filter (>{CORR_THRESHOLD}): {len(kept)} kept, dropped {corr_dropped}")

    kept, vif_dropped = vif_filter(train_df, kept)
    print(f"After VIF filter (>{VIF_THRESHOLD}): {len(kept)} kept, dropped {vif_dropped}")

    selected = select_via_l1(train_df, kept, train_df["TARGET"].values)
    print(f"After L1-guided selection: {len(selected)} characteristics -> {selected}")

    selected, final_model, sign_dropped = enforce_sign_consistency(train_df, selected)
    if sign_dropped:
        print(
            f"Dropped for counterintuitive sign (WOE is coded so higher = safer everywhere, "
            f"so every coefficient must be negative -- a positive one here means the "
            f"multivariate fit reversed its sign relative to its own univariate WOE/event-rate "
            f"relationship, a classic residual-collinearity effect not defensible to a risk "
            f"committee): {sign_dropped}"
        )
    print(f"Final characteristic count after sign-consistency pruning: {len(selected)}")

    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        proba = final_model.predict_proba(split_df[selected].values)[:, 1]
        auc = roc_auc_score(split_df["TARGET"].values, proba)
        gini = 2 * auc - 1
        ks = _ks_statistic(split_df["TARGET"].values, proba)
        print(f"  {split_name:<6} AUC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}  n={len(split_df):,}")

    coef_report = pd.Series(final_model.coef_[0], index=selected).sort_values(key=abs, ascending=False)
    print("\nFinal logistic regression coefficients (on WOE scale):")
    print(coef_report.round(4))

    with open(ARTIFACTS / "logistic_model.pkl", "wb") as f:
        pickle.dump(
            {"model": final_model, "features": selected, "intercept": final_model.intercept_[0]}, f
        )
    train_df.to_parquet(ARTIFACTS / "scorecard_train.parquet", index=False)
    test_df.to_parquet(ARTIFACTS / "scorecard_test.parquet", index=False)
    print(f"\nSaved final model and train/test splits to {ARTIFACTS}")


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n_bad = y_sorted.sum()
    n_good = len(y_sorted) - n_bad
    cum_bad = np.cumsum(y_sorted) / max(n_bad, 1)
    cum_good = np.cumsum(1 - y_sorted) / max(n_good, 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


if __name__ == "__main__":
    main()
