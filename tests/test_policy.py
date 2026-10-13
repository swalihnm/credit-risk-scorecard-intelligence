"""Regression guards on the policy/ (cutoff optimizer) layer."""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py).
from optbinning import OptimalBinning  # noqa: F401

import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _require(path: pathlib.Path):
    if not path.exists():
        pytest.skip(f"{path} not generated yet")
    return path


def test_frontier_has_concentration_columns():
    path = _require(ROOT / "marts" / "mart_cutoff_frontier.parquet")
    frontier = pd.read_parquet(path)
    all_segment = frontier[frontier["segment"] == "All"]
    assert "max_segment_concentration" in all_segment.columns
    assert all_segment["max_segment_concentration"].dropna().between(0, 1).all()


def test_unconstrained_optimum_breaches_concentration_limit():
    """Without a concentration constraint, the risk-adjusted-return-maximizing cutoff lets
    one income tercile dominate the approved book -- this is precisely why the constraint
    belongs in solve_constrained() (see docs/07-results.md), not an incidental detail.
    """
    path = _require(ROOT / "marts" / "mart_cutoff_frontier.parquet")
    frontier = pd.read_parquet(path)
    base_all = frontier[(frontier["scenario"] == "base") & (frontier["segment"] == "All")]
    best_unconstrained = base_all.loc[base_all["risk_adj_return"].idxmax()]
    assert best_unconstrained["max_segment_concentration"] > 0.45, (
        "Expected the unconstrained optimum to breach a 45% concentration limit -- if it "
        "no longer does, the concentration constraint may no longer be the one that binds "
        "and docs/07-results.md's narrative needs revisiting"
    )


def test_constrained_optimum_respects_concentration_limit():
    from policy.cutoff_optimizer import MAX_SEGMENT_CONCENTRATION, solve_constrained

    path = _require(ROOT / "marts" / "mart_cutoff_frontier.parquet")
    frontier = pd.read_parquet(path)
    solved = solve_constrained(frontier)
    assert solved["feasible"]
    assert solved["max_segment_concentration"] <= MAX_SEGMENT_CONCENTRATION + 1e-9
