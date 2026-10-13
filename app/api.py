"""Single-application scoring service. Power BI does not score live applications (see
architecture note in README.md) -- this is the small piece that does.

Simplification, disclosed: this endpoint accepts already-engineered feature values (the same
columns as marts/feature_table.parquet) rather than raw application/bureau/previous-application
records. In production this would sit behind the same features/*.py aggregation pipeline
against a live feature store; recomputing that aggregation per-request here would require
bundling a warehouse connection into the API container, which is out of scope for what this
endpoint is demonstrating (that the model can be shipped as a service, not just reported on).
"""
from __future__ import annotations

# Import order: optbinning before pandas/fastapi (see scorecard/binning.py).
from optbinning import OptimalBinning  # noqa: F401

import json
import pathlib
import pickle
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from scorecard import binning
from scorecard.scaling import compute_scaling_constants

ARTIFACTS = pathlib.Path("scorecard/artifacts")
MODEL_VERSION = "1.0.0"
POLICY_CUTOFF_PATH = ARTIFACTS / "policy_cutoff.json"

SCORE_BAND_EDGES = [-np.inf, 500, 550, 600, 650, 700, np.inf]
SCORE_BAND_LABELS = ["<500", "500-550", "550-600", "600-650", "650-700", "700+"]

app = FastAPI(title="Credit Risk Scorecard API", version=MODEL_VERSION)

_state: dict = {}


class ApplicationFeatures(BaseModel):
    """Deliberately NOT a fixed field-by-field schema. Which characteristics the fitted
    scorecard actually needs depends on WOE/IV screening, correlation/VIF filtering, and
    sign-consistency pruning (scorecard/fit_logistic.py) -- all data-dependent, so the
    selected characteristic list changes when the model is refit (it already has, twice:
    12 characteristics on synthetic data, 14 on real data, not the same 12). A fixed Pydantic
    schema silently drifts out of sync with the deployed model every time that happens --
    exactly what broke here. Accept any fields; validate against the model's ACTUAL current
    requirement (`state["features"]`) at request time in /score, and expose that requirement
    via GET /model/required-features so a caller can always ask what's needed right now.
    """

    model_config = ConfigDict(extra="allow")

    SK_ID_CURR: int


class ScoreResponse(BaseModel):
    SK_ID_CURR: int
    score: float
    score_band: str
    pd: float
    decision: str
    reason_codes: list[str]


def _load_artifacts() -> dict:
    if _state:
        return _state
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        logistic_artifact = pickle.load(f)
    with open(ARTIFACTS / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    _state["binning_objects"] = binning.load()
    _state["model"] = logistic_artifact["model"]
    _state["features"] = logistic_artifact["features"]
    _state["calibrator"] = calibrator
    _state["factor"], _state["offset"] = compute_scaling_constants()
    with open(POLICY_CUTOFF_PATH) as f:
        _state["policy_cutoff"] = json.load(f)["policy_cutoff"]
    return _state


def _load_artifacts_or_503() -> dict:
    try:
        return _load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Scorecard artifacts not found ({exc}). Run the scorecard/ pipeline first.",
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/version")
def model_version():
    state = _load_artifacts_or_503()
    return {
        "model_version": MODEL_VERSION,
        "model_type": "logistic_scorecard",
        "n_characteristics": len(state["features"]),
    }


@app.get("/model/required-features")
def required_features() -> dict[str, Any]:
    """What GET /score's payload actually needs right now -- see ApplicationFeatures'
    docstring for why this is served dynamically rather than documented as a fixed list."""
    state = _load_artifacts_or_503()
    characteristics = sorted({f.removesuffix("_woe") for f in state["features"]})
    return {"required_characteristics": characteristics, "count": len(characteristics)}


@app.post("/score", response_model=ScoreResponse)
def score(application: ApplicationFeatures):
    state = _load_artifacts_or_503()
    row = pd.DataFrame([application.model_dump()])
    binning_objects = state["binning_objects"]
    model = state["model"]
    features = state["features"]

    woe_values = {}
    for feature_woe in features:
        char = feature_woe.removesuffix("_woe")
        if char not in row.columns:
            raise HTTPException(status_code=422, detail=f"Missing required field: {char}")
        woe_values[feature_woe] = binning_objects[char].transform(row[char].values, metric="woe")[0]

    woe_vector = np.array([[woe_values[f] for f in features]])
    raw_proba = model.predict_proba(woe_vector)[0, 1]

    from scorecard.calibration import apply_platt

    calibrated_pd = float(apply_platt(state["calibrator"]["platt"], np.array([raw_proba]))[0])

    base_points = state["offset"] - state["factor"] * model.intercept_[0]
    contributions = {
        f: -state["factor"] * coef * woe_values[f] for f, coef in zip(features, model.coef_[0])
    }
    total_score = base_points + sum(contributions.values())
    score_band = str(pd.cut([total_score], SCORE_BAND_EDGES, labels=SCORE_BAND_LABELS, right=False)[0])

    decision = "Approved" if total_score >= state["policy_cutoff"] else "Declined"

    reason_codes = []
    if decision == "Declined":
        worst = sorted(contributions.items(), key=lambda kv: kv[1])[:4]
        reason_codes = [f.removesuffix("_woe") for f, _ in worst]

    return ScoreResponse(
        SK_ID_CURR=application.SK_ID_CURR,
        score=round(total_score, 1),
        score_band=score_band,
        pd=round(calibrated_pd, 4),
        decision=decision,
        reason_codes=reason_codes,
    )
