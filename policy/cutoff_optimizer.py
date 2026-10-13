"""Sweep candidate score cutoffs 400-800 (step 5) and compute portfolio economics at each,
for the Home Credit application scorecard. Also solves the genuinely constrained version and
reports the shadow price of the loss constraint.

Economic assumptions (Home Credit is unsecured consumer lending, so the Fannie Mae LGD table
does NOT transfer -- see docs/06-assumptions-and-limitations.md):
- LGD = 0.70 flat, a standard industry benchmark for unsecured retail lending recoveries
  (~25-35%), not re-derived from data (Home Credit has no realized loss-given-default field).
- EAD = AMT_CREDIT (fully disbursed instalment loan, no undrawn-commitment adjustment).
- expected_tenure_years = (AMT_CREDIT / AMT_ANNUITY) / 12, back-derived from the loan's own
  payment schedule.
- interest_revenue = EAD * ASSUMED_APR * expected_tenure_years, ASSUMED_APR = 15% (typical
  unsecured consumer instalment lending).
- capital_required = 8% of EAD (Basel standardized-approach minimum, flat risk weight) --
  a simplification; a real bank would use an IRB risk weight that scales with PD/LGD.
- opex = 2% of EAD (origination + servicing cost simplification).

Two macro "scenarios" are swept for the frontier mart (mart_cutoff_frontier's grain is
cutoff x scenario x segment): "base" uses the calibrated PD as scored; "stress" multiplies
PD by 1.5x as a simple downturn overlay -- distinct from, and simpler than, the Fannie Mae
mortgage-specific macro scenarios in portfolio/stress_scenarios.py, since Home Credit has no
loan-performance panel to fit a real hazard model against.
"""
from __future__ import annotations

# Import order: optbinning before pandas (see scorecard/binning.py).
from optbinning import OptimalBinning

import json
import pathlib
import pickle

import numpy as np
import pandas as pd

from scorecard import binning

ARTIFACTS = pathlib.Path("scorecard/artifacts")
POLICY_CUTOFF_PATH = ARTIFACTS / "policy_cutoff.json"

LGD_ASSUMPTION = 0.70
ASSUMED_APR = 0.15
CAPITAL_RATIO = 0.08
OPEX_RATIO = 0.02

CUTOFF_RANGE = range(400, 805, 5)
SCENARIOS = {"base": 1.0, "stress": 1.5}
MAX_SEGMENT_CONCENTRATION = 0.45  # no single income tercile may exceed 45% of approved volume


def score_population(df: pd.DataFrame) -> pd.DataFrame:
    binning_objects = binning.load()
    with open(ARTIFACTS / "logistic_model.pkl", "rb") as f:
        model_artifact = pickle.load(f)
    with open(ARTIFACTS / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)

    model = model_artifact["model"]
    features = model_artifact["features"]

    woe = pd.DataFrame(index=df.index)
    for feature_woe in features:
        char = feature_woe.removesuffix("_woe")
        woe[feature_woe] = binning_objects[char].transform(df[char].values, metric="woe")

    raw_proba = model.predict_proba(woe.values)[:, 1]
    from scorecard.calibration import apply_platt

    calibrated_pd = apply_platt(calibrator["platt"], raw_proba)

    from scorecard.scaling import compute_scaling_constants

    factor, offset = compute_scaling_constants()
    base_points = offset - factor * model.intercept_[0]
    score = base_points - factor * (woe.values * model.coef_[0]).sum(axis=1)

    out = df[["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY", "NAME_INCOME_TYPE"]].copy()
    out["pd"] = calibrated_pd
    out["score"] = score
    out["ead"] = out["AMT_CREDIT"]
    out["expected_tenure_years"] = (out["AMT_CREDIT"] / out["AMT_ANNUITY"]) / 12
    out["income_tercile"] = pd.qcut(df["AMT_INCOME_TOTAL"], 3, labels=["Low income", "Mid income", "High income"])
    return out


def sweep_frontier(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    terciles = list(scored["income_tercile"].cat.categories)
    for scenario, pd_multiplier in SCENARIOS.items():
        pd_scenario = np.clip(scored["pd"] * pd_multiplier, 0, 0.99)
        for segment_name, segment_mask in [("All", pd.Series(True, index=scored.index))] + [
            (seg, scored["income_tercile"] == seg) for seg in terciles
        ]:
            seg_df = scored[segment_mask]
            seg_pd = pd_scenario[segment_mask]
            for cutoff in CUTOFF_RANGE:
                approved = seg_df["score"] >= cutoff
                n_total = len(seg_df)
                n_approved = int(approved.sum())
                if n_total == 0:
                    continue
                ead = seg_df.loc[approved, "ead"]
                pd_app = seg_pd[approved]
                tenure = seg_df.loc[approved, "expected_tenure_years"]

                expected_loss = float((pd_app * LGD_ASSUMPTION * ead).sum())
                interest_revenue = float((ead * ASSUMED_APR * tenure).sum())
                capital_required = float((CAPITAL_RATIO * ead).sum())
                opex = float((OPEX_RATIO * ead).sum())
                risk_adj_return = (
                    (interest_revenue - expected_loss - opex) / capital_required if capital_required > 0 else np.nan
                )

                row = {
                    "cutoff": cutoff,
                    "scenario": scenario,
                    "segment": segment_name,
                    "approval_rate": n_approved / n_total,
                    "n_approved": n_approved,
                    "expected_loss": expected_loss,
                    "expected_loss_rate": expected_loss / ead.sum() if n_approved > 0 else np.nan,
                    "interest_revenue": interest_revenue,
                    "capital_required": capital_required,
                    "opex": opex,
                    "risk_adj_return": risk_adj_return,
                }

                # Concentration limits (spec section 7.3): only meaningful on the "All"
                # segment's approved population -- what share of THAT population falls into
                # each income tercile. Segment-specific rows (segment == a tercile) don't
                # need this, since they're already 100% one tercile by construction.
                if segment_name == "All" and n_approved > 0:
                    approved_terciles = seg_df.loc[approved, "income_tercile"]
                    for tercile in terciles:
                        row[f"conc_{tercile.lower().replace(' ', '_')}"] = (
                            approved_terciles.eq(tercile).sum() / n_approved
                        )
                    row["max_segment_concentration"] = max(
                        row[f"conc_{t.lower().replace(' ', '_')}"] for t in terciles
                    )

                rows.append(row)
    return pd.DataFrame(rows)


def solve_constrained(
    frontier: pd.DataFrame,
    max_loss_rate: float = 0.03,
    min_approval_rate: float = 0.50,
    max_segment_concentration: float = MAX_SEGMENT_CONCENTRATION,
) -> dict:
    """Best cutoff (by risk-adjusted return) on the base scenario, 'All' segment, subject to
    a maximum portfolio expected-loss rate, a minimum origination-volume floor, AND a
    segment concentration limit (no income tercile over max_segment_concentration of the
    approved book) -- all four constraints named in the project spec's section 7.3, not just
    the first two.
    """
    base_all = frontier[(frontier["scenario"] == "base") & (frontier["segment"] == "All")]
    feasible = base_all[
        (base_all["expected_loss_rate"] <= max_loss_rate)
        & (base_all["approval_rate"] >= min_approval_rate)
        & (base_all["max_segment_concentration"] <= max_segment_concentration)
    ]
    if feasible.empty:
        return {"feasible": False}
    best = feasible.loc[feasible["risk_adj_return"].idxmax()]
    return {"feasible": True, **best.to_dict()}


def shadow_price_of_loss_constraint(frontier: pd.DataFrame, base_cap: float = 0.03, bump: float = 0.01) -> float:
    """Finite-difference shadow price: extra (revenue - expected loss - opex) available from
    relaxing the max-loss-rate constraint by `bump` (default 1 percentage point), holding the
    volume floor fixed.
    """
    solved_base = solve_constrained(frontier, max_loss_rate=base_cap)
    solved_bumped = solve_constrained(frontier, max_loss_rate=base_cap + bump)
    if not (solved_base["feasible"] and solved_bumped["feasible"]):
        return float("nan")
    revenue_base = solved_base["interest_revenue"] - solved_base["expected_loss"] - solved_base["opex"]
    revenue_bumped = solved_bumped["interest_revenue"] - solved_bumped["expected_loss"] - solved_bumped["opex"]
    return revenue_bumped - revenue_base


def shadow_price_of_volume_floor(frontier: pd.DataFrame, base_floor: float = 0.50, bump: float = -0.10) -> float:
    """Same finite-difference idea, but for the volume floor: how much (revenue - expected
    loss - opex) is gained by relaxing (lowering) the minimum-approval-rate requirement.
    """
    solved_base = solve_constrained(frontier, min_approval_rate=base_floor)
    solved_bumped = solve_constrained(frontier, min_approval_rate=base_floor + bump)
    if not (solved_base["feasible"] and solved_bumped["feasible"]):
        return float("nan")
    revenue_base = solved_base["interest_revenue"] - solved_base["expected_loss"] - solved_base["opex"]
    revenue_bumped = solved_bumped["interest_revenue"] - solved_bumped["expected_loss"] - solved_bumped["opex"]
    return revenue_bumped - revenue_base


def shadow_price_of_concentration_limit(
    frontier: pd.DataFrame, base_cap: float = MAX_SEGMENT_CONCENTRATION, bump: float = 0.05
) -> float:
    """Same finite-difference idea, for the segment concentration limit: how much
    (revenue - expected loss - opex) is gained by relaxing the max-tercile-share cap.
    """
    solved_base = solve_constrained(frontier, max_segment_concentration=base_cap)
    solved_bumped = solve_constrained(frontier, max_segment_concentration=base_cap + bump)
    if not (solved_base["feasible"] and solved_bumped["feasible"]):
        return float("nan")
    revenue_base = solved_base["interest_revenue"] - solved_base["expected_loss"] - solved_base["opex"]
    revenue_bumped = solved_bumped["interest_revenue"] - solved_bumped["expected_loss"] - solved_bumped["opex"]
    return revenue_bumped - revenue_base


def main() -> None:
    df = pd.read_parquet("marts/feature_table.parquet")
    scored = score_population(df)
    frontier = sweep_frontier(scored)

    out_path = pathlib.Path("marts/mart_cutoff_frontier.parquet")
    frontier.to_parquet(out_path, index=False)
    print(f"Saved {len(frontier):,} rows to {out_path}")

    print("\n=== Unconstrained optimum (base scenario, All segment, by risk-adjusted return) ===")
    print("(Maximizing a RATIO with no volume floor trivially shrinks to the single safest "
          "loan -- this is exactly why the constrained problem below, not this one, is the "
          "one worth solving for a bank.)")
    base_all = frontier[(frontier["scenario"] == "base") & (frontier["segment"] == "All")]
    best_unconstrained = base_all.loc[base_all["risk_adj_return"].idxmax()]
    print(best_unconstrained.round(4))

    print("\n=== Constrained optimum (max loss rate 3%, min approval rate 50%, "
          f"max segment concentration {MAX_SEGMENT_CONCENTRATION:.0%}) ===")
    solved = solve_constrained(frontier)
    print(solved)

    loss_cap, vol_floor, conc_cap = 0.03, 0.50, MAX_SEGMENT_CONCENTRATION
    # Grid-aware bindingness: true if the NEXT cutoff step up (which the optimizer would
    # prefer, since risk_adj_return increases with cutoff over this range) would violate the
    # constraint -- not whether the current value sits near the boundary in absolute terms,
    # which is misleading on a coarse 5-point grid where a single step can move approval_rate
    # by several points at once.
    base_all_sorted = base_all.sort_values("cutoff").reset_index(drop=True)
    next_row = base_all_sorted[base_all_sorted["cutoff"] == solved["cutoff"] + 5]
    loss_binding = (not next_row.empty) and next_row["expected_loss_rate"].iloc[0] > loss_cap
    vol_binding = (not next_row.empty) and next_row["approval_rate"].iloc[0] < vol_floor
    conc_binding = (not next_row.empty) and next_row["max_segment_concentration"].iloc[0] > conc_cap
    print(f"\nBinding constraint check (would the next cutoff step up violate it?): "
          f"loss cap {'BINDING' if loss_binding else 'slack'}; "
          f"volume floor {'BINDING' if vol_binding else 'slack'}; "
          f"segment concentration {'BINDING' if conc_binding else 'slack'} "
          f"(max tercile share at the optimum: {solved['max_segment_concentration']:.1%})")

    shadow_price_loss = shadow_price_of_loss_constraint(frontier, base_cap=loss_cap)
    shadow_price_volume = shadow_price_of_volume_floor(frontier, base_floor=vol_floor)
    shadow_price_conc = shadow_price_of_concentration_limit(frontier, base_cap=conc_cap)

    def _slack_note(binding: bool, other_bindings: dict[str, bool]) -> str:
        if binding:
            return ""
        bound_elsewhere = [name for name, b in other_bindings.items() if b]
        if bound_elsewhere:
            return f" (slack -- {' and '.join(bound_elsewhere)} binds instead)"
        return " (slack)"

    bindings = {"the loss cap": loss_binding, "the volume floor": vol_binding, "the concentration limit": conc_binding}
    print(f"\nShadow price of the loss-rate cap: ${shadow_price_loss:,.0f} per +1pp of relaxed loss appetite"
          f"{_slack_note(loss_binding, {k: v for k, v in bindings.items() if k != 'the loss cap'})}")
    print(f"Shadow price of the volume floor: ${shadow_price_volume:,.0f} per -10pp of relaxed volume requirement"
          f"{_slack_note(vol_binding, {k: v for k, v in bindings.items() if k != 'the volume floor'})}")
    print(f"Shadow price of the concentration limit: ${shadow_price_conc:,.0f} per +5pp of relaxed concentration cap"
          f"{_slack_note(conc_binding, {k: v for k, v in bindings.items() if k != 'the concentration limit'})}"
          f"{' -- THIS is the constraint actually limiting risk-adjusted return here' if conc_binding else ''}")

    # Persisted so downstream consumers (marts, the API's decline/approve policy) read the
    # cutoff this optimizer actually computed instead of a hardcoded number that silently
    # goes stale whenever the data or constraints change (as happened twice during
    # development -- see git history / README's bug list).
    with open(POLICY_CUTOFF_PATH, "w") as f:
        json.dump(
            {
                "policy_cutoff": int(solved["cutoff"]),
                "max_loss_rate": loss_cap,
                "min_approval_rate": vol_floor,
                "max_segment_concentration": conc_cap,
                "approval_rate_at_cutoff": solved["approval_rate"],
            },
            f,
            indent=2,
        )
    print(f"\nSaved policy cutoff ({solved['cutoff']}) to {POLICY_CUTOFF_PATH}")


if __name__ == "__main__":
    main()
