"""Regression guards on the scorecard/ layer."""
from __future__ import annotations

import pathlib
import pickle

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "scorecard" / "artifacts"


def _require(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    return path


def test_final_model_characteristic_count_in_target_range():
    path = _require(ARTIFACTS / "logistic_model.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    assert 8 <= len(artifact["features"]) <= 16, (
        f"Expected roughly 10-15 final characteristics, got {len(artifact['features'])}"
    )


def test_all_coefficients_negative():
    """WOE is coded so higher = safer for every characteristic (see scorecard/binning.py's
    MONOTONIC_TREND); a well-specified model predicting TARGET=1 (bad) must have every
    coefficient negative. A positive one means multivariate fitting reversed a feature's
    sign relative to its own univariate relationship -- not defensible to a risk committee.
    """
    path = _require(ARTIFACTS / "logistic_model.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    coefs = artifact["model"].coef_[0]
    assert (coefs < 0).all(), f"All scorecard coefficients must be negative, got {dict(zip(artifact['features'], coefs))}"


def test_reject_inference_makes_policy_more_conservative():
    """Reject inference should reveal that the accept-only model understates risk in some
    segment of the through-the-door population (see docs/03-reject-inference.md) -- the
    augmented model should require MORE points to clear the same bar, not fewer.
    """
    path = _require(ARTIFACTS / "reject_inference.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    assert artifact["points_shift"] > 0, (
        f"Expected reject inference to shift the scorecard stricter (positive points_shift), "
        f"got {artifact['points_shift']}"
    )


def test_adverse_action_reason_text_covers_all_selected_characteristics():
    """REASON_CODE_TEXT is a hardcoded lookup, and it has already silently gone stale once:
    it was written against the 12 characteristics an earlier synthetic-data fit selected,
    and 8 of the real-data fit's 14 characteristics had no entry (degrading gracefully to
    showing the raw column name via .get(char, char), rather than crashing -- which is
    exactly why it wasn't caught until someone read the actual output). This guards against
    it happening again for whatever characteristic set the model currently uses.
    """
    from scorecard.adverse_action import REASON_CODE_TEXT

    path = _require(ARTIFACTS / "logistic_model.pkl")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    characteristics = {f.removesuffix("_woe") for f in artifact["features"]}
    missing = characteristics - set(REASON_CODE_TEXT)
    assert not missing, f"REASON_CODE_TEXT has no human-readable text for: {missing}"


def test_gini_generalizes_train_to_test():
    from sklearn.metrics import roc_auc_score

    train_path = _require(ARTIFACTS / "scorecard_train.parquet")
    test_path = _require(ARTIFACTS / "scorecard_test.parquet")
    model_path = _require(ARTIFACTS / "logistic_model.pkl")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model, features = artifact["model"], artifact["features"]
    train_gini = 2 * roc_auc_score(train_df["TARGET"], model.predict_proba(train_df[features])[:, 1]) - 1
    test_gini = 2 * roc_auc_score(test_df["TARGET"], model.predict_proba(test_df[features])[:, 1]) - 1

    assert train_gini > 0.3, f"Train Gini too low to be a usable scorecard: {train_gini:.3f}"
    assert abs(train_gini - test_gini) < 0.08, (
        f"Train/test Gini gap too large ({train_gini:.3f} vs {test_gini:.3f}) -- likely overfitting"
    )
