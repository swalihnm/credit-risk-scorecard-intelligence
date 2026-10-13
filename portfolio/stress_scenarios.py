"""Three macro scenarios fed into the hazard model's forward PD projection (portfolio/survival.py)
and, from there, the ECL engine. Not regulator-prescribed CCAR/DFAST paths — illustrative,
per docs/13/06-assumptions-and-limitations.md.

- baseline: last 12 observed months of unrate/hpi_yoy held flat forward.
- adverse: unemployment +3pp, HPI -15% (linear), sustained over 8 quarters (24 months),
  then flat.
- severely_adverse: unemployment +6pp, HPI -30% (linear), sustained over 8 quarters (24
  months), then flat — magnitudes matched to the synthetic macro panel's actual 2008 peak
  (~11% unemployment, per ingest/synthetic_fred.py) rather than replaying the raw historical
  slice, so the scenario is guaranteed to be uniformly harsher than "adverse" on both drivers.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
MACRO_PATH = ROOT / "sources" / "fred" / "macro_monthly.parquet"

SCENARIOS = ["baseline", "adverse", "severely_adverse"]


def _load_macro() -> pd.DataFrame:
    macro = pd.read_parquet(MACRO_PATH).sort_values("date").reset_index(drop=True)
    macro["hpi_yoy_pct"] = macro["hpi"].pct_change(12) * 100
    return macro


def baseline_path(macro: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    last12 = macro.tail(12)
    unrate = last12["unrate"].mean()
    hpi_yoy = last12["hpi_yoy_pct"].mean()
    return pd.DataFrame({"unrate": [unrate] * horizon_months, "hpi_yoy_pct": [hpi_yoy] * horizon_months})


def adverse_path(macro: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    start_unrate = macro["unrate"].iloc[-1]
    ramp = min(24, horizon_months)
    unrate = []
    hpi_yoy = []
    for m in range(horizon_months):
        if m < ramp:
            frac = (m + 1) / ramp
        else:
            frac = 1.0
        unrate.append(start_unrate + 3.0 * frac)
        hpi_yoy.append(-15.0 * frac)
    return pd.DataFrame({"unrate": unrate, "hpi_yoy_pct": hpi_yoy})


def severely_adverse_path(macro: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    start_unrate = macro["unrate"].iloc[-1]
    ramp = min(24, horizon_months)
    unrate = []
    hpi_yoy = []
    for m in range(horizon_months):
        frac = (m + 1) / ramp if m < ramp else 1.0
        unrate.append(start_unrate + 6.0 * frac)
        hpi_yoy.append(-30.0 * frac)
    return pd.DataFrame({"unrate": unrate, "hpi_yoy_pct": hpi_yoy})


def get_scenario_path(scenario: str, horizon_months: int) -> pd.DataFrame:
    macro = _load_macro()
    if scenario == "baseline":
        return baseline_path(macro, horizon_months)
    if scenario == "adverse":
        return adverse_path(macro, horizon_months)
    if scenario == "severely_adverse":
        return severely_adverse_path(macro, horizon_months)
    raise ValueError(f"Unknown scenario: {scenario}. Expected one of {SCENARIOS}")


def main() -> None:
    for scenario in SCENARIOS:
        path = get_scenario_path(scenario, 24)
        print(f"\n=== {scenario} (first 8 months) ===")
        print(path.head(8).round(2).to_string(index=False))
        print(f"  peak unrate: {path['unrate'].max():.2f}, trough hpi_yoy: {path['hpi_yoy_pct'].min():.2f}")


if __name__ == "__main__":
    main()
