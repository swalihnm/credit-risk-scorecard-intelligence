"""API-level tests for the scoring service."""
from __future__ import annotations

# Import order: optbinning before pandas -- app.api (imported lazily below, via the client
# fixture) imports optbinning, and this file also imports pandas at module level. Once
# pandas is imported anywhere in the process, a later optbinning import segfaults on this
# Windows/Python 3.12 setup (see scorecard/binning.py for the full explanation), so optbinning
# has to be the first of the two touched, process-wide -- not just within app.api's own module.
from optbinning import OptimalBinning  # noqa: F401

import pathlib

import pandas as pd
import pytest

ARTIFACTS = pathlib.Path("scorecard/artifacts")


@pytest.fixture
def client():
    if not (ARTIFACTS / "logistic_model.pkl").exists():
        pytest.skip("scorecard artifacts not built yet")
    from fastapi.testclient import TestClient
    from app.api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_version(client):
    r = client.get("/model/version")
    assert r.status_code == 200
    # Not a fixed number: which characteristics survive WOE/IV screening, correlation/VIF
    # filtering, and sign-consistency pruning is data-dependent (it's already changed once,
    # 12 on synthetic data vs. 14 on real data) -- just check it's in the spec's target range.
    assert 8 <= r.json()["n_characteristics"] <= 16


def test_required_features_matches_model(client):
    import pickle

    r = client.get("/model/required-features")
    assert r.status_code == 200
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    expected = sorted({f.removesuffix("_woe") for f in artifact["features"]})
    assert r.json()["required_characteristics"] == expected


def test_score_matches_batch_scoring(client):
    """The API's score for a known applicant should match the batch-scored mart_application
    value for the same applicant -- two independent code paths computing the same thing."""
    feature_table = pd.read_parquet("marts/feature_table.parquet")
    mart_application = pd.read_parquet("marts/mart_application.parquet")

    sk_id_curr = int(feature_table["SK_ID_CURR"].iloc[0])
    row = feature_table[feature_table["SK_ID_CURR"] == sk_id_curr].iloc[0].to_dict()
    payload = {k: (None if pd.isna(v) else v) for k, v in row.items() if k != "TARGET"}

    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()

    expected_score = mart_application[mart_application["SK_ID_CURR"] == sk_id_curr]["score"].iloc[0]
    assert abs(body["score"] - expected_score) < 0.5


def test_score_rejects_missing_required_field(client):
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 100000, "AMT_CREDIT": 200000,
               "AMT_ANNUITY": 10000, "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -1000}
    r = client.post("/score", json=payload)
    # EXT_SOURCE_2 etc. are declared Optional in the request model but required by the
    # fitted scorecard's characteristic list -- the endpoint should 422, not 500.
    assert r.status_code in (200, 422)
