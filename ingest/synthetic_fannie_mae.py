"""Generate a synthetic loan-level monthly performance panel matching the shape of a
Fannie Mae Single-Family Loan Performance extract.

Explicitly labelled STAND-IN data (see docs/01-data-sources.md) — real Fannie Mae extracts
require a human to register at the Data Dynamics portal and are not scriptable. This generator
produces the same grain and columns a real extract would, driven by a macro-modulated Markov
delinquency model, so vintage curves / roll rates / survival / LGD / ECL / stress testing can
be built and proven end-to-end now, then re-pointed at real extracts later without changing
any downstream code (see ingest/fannie_mae_ingest.py for the swap-in stub).

Buckets (dim_bucket, sort order): 0 Current, 1 30dpd, 2 60dpd, 3 90dpd, 4 120+dpd, 5 Charge-off.
Charge-off is absorbing; the loan exits the panel the month it is first reached.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

BUCKET_NAMES = {0: "Current", 1: "30", 2: "60", 3: "90", 4: "120+", 5: "Charge-off"}

STATES = ["CA", "TX", "FL", "NY", "GA", "IL", "OH", "AZ", "NC", "PA", "MI", "NJ", "WA", "CO", "NV"]
STATE_WEIGHTS = np.array([0.16, 0.10, 0.09, 0.08, 0.06, 0.06, 0.05, 0.06, 0.05, 0.05,
                           0.04, 0.04, 0.05, 0.04, 0.07])
STATE_WEIGHTS = STATE_WEIGHTS / STATE_WEIGHTS.sum()
REGION_MAP = {
    "CA": "West", "AZ": "West", "NV": "West", "WA": "West", "CO": "West",
    "TX": "South", "FL": "South", "GA": "South", "NC": "South",
    "NY": "Northeast", "PA": "Northeast", "NJ": "Northeast",
    "IL": "Midwest", "OH": "Midwest", "MI": "Midwest",
}

OBS_END = pd.Timestamp("2021-12-01")
MAX_MOB = 60
TERM_MONTHS = 360

OUT_DIR = pathlib.Path("sources/fannie_mae")


def _amort_balance(principal: np.ndarray, annual_rate: np.ndarray, t: np.ndarray, term: int = TERM_MONTHS) -> np.ndarray:
    r = annual_rate / 100 / 12
    r = np.where(r <= 0, 1e-6, r)
    factor = ((1 + r) ** term - (1 + r) ** t) / ((1 + r) ** term - 1)
    return np.clip(principal * factor, 0, None)


def _load_macro() -> pd.DataFrame:
    macro = pd.read_parquet(OUT_DIR.parent / "fred" / "macro_monthly.parquet")
    macro = macro.sort_values("date").reset_index(drop=True)
    macro["hpi_yoy_pct"] = macro["hpi"].pct_change(12) * 100
    return macro.set_index("date")


def build_loans(n_loans: int, vintage_start: str, vintage_end: str, rng: np.random.Generator, id_start: int) -> pd.DataFrame:
    vintages = pd.date_range(vintage_start, vintage_end, freq="MS")
    origination_date = rng.choice(vintages, size=n_loans)
    state = rng.choice(STATES, size=n_loans, p=STATE_WEIGHTS)
    fico = np.clip(rng.normal(725, 55, n_loans), 580, 830).round(0)
    ltv = np.clip(rng.normal(78, 12, n_loans), 50, 100).round(1)
    upb = np.clip(rng.lognormal(np.log(220_000), 0.35, n_loans), 60_000, 900_000).round(0)
    rate = np.clip(rng.normal(5.2, 1.1, n_loans) - 0.01 * (fico - 700), 2.5, 9.5).round(3)

    return pd.DataFrame(
        {
            "loan_id": np.arange(id_start, id_start + n_loans),
            "origination_date": origination_date,
            "state": state,
            "region": [REGION_MAP[s] for s in state],
            "original_upb": upb,
            "original_ltv": ltv,
            "fico_orig": fico,
            "orig_rate": rate,
            "term_months": TERM_MONTHS,
        }
    )


# base monthly transition rates, before macro/risk adjustment
BASE = {
    0: {"worsen": 0.0060},
    1: {"cure": 0.38, "worsen": 0.30},
    2: {"cure": 0.15, "worsen": 0.40},
    3: {"cure": 0.08, "worsen": 0.45},
    4: {"cure": 0.05, "worsen": 0.30},
}


def simulate(loans: pd.DataFrame, macro: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(loans)
    bucket = np.zeros(n, dtype=np.int8)
    mob = np.zeros(n, dtype=np.int32)
    active = np.ones(n, dtype=bool)
    origination = loans["origination_date"].values
    fico = loans["fico_orig"].values
    ltv = loans["original_ltv"].values
    upb0 = loans["original_upb"].values
    rate = loans["orig_rate"].values

    risk_mult = np.clip(1 + 0.02 * (700 - fico) / 10 + 0.03 * (ltv - 80) / 10, 0.3, 3.0)

    perf_rows = []
    default_rows = []

    timeline = pd.date_range(macro.index.min(), OBS_END, freq="MS")
    for t in timeline:
        started = (origination <= t) & active
        if not started.any():
            continue
        within_window = mob < MAX_MOB
        live = started & within_window & active
        if not live.any():
            continue

        macro_row = macro.loc[t] if t in macro.index else macro.iloc[-1]
        unrate = macro_row["unrate"]
        hpi_yoy = macro_row["hpi_yoy_pct"] if not np.isnan(macro_row["hpi_yoy_pct"]) else 0.0
        stress = 0.15 * (unrate - 5.0) - 0.02 * hpi_yoy
        mult_roll = np.clip(1 + stress, 0.4, 3.0)
        mult_cure = np.clip(1 - 0.5 * stress, 0.3, 1.5)

        bucket_prev_snapshot = bucket.copy()
        idx_live = np.where(live)[0]

        for b in range(5):
            idx_b = idx_live[bucket[idx_live] == b]
            if idx_b.size == 0:
                continue
            rm = risk_mult[idx_b]
            u = rng.random(idx_b.size)

            if b == 0:
                p_worsen = np.clip(BASE[0]["worsen"] * mult_roll * rm, 0, 0.9)
                to_30 = u < p_worsen
                bucket[idx_b[to_30]] = 1
            else:
                p_cure = np.clip(BASE[b]["cure"] * mult_cure / np.maximum(rm, 1.0), 0, 0.9)
                p_worsen = np.clip(BASE[b]["worsen"] * mult_roll * rm, 0, 0.9 - p_cure)
                cure_mask = u < p_cure
                worsen_mask = (~cure_mask) & (u < p_cure + p_worsen)
                bucket[idx_b[cure_mask]] -= 1
                bucket[idx_b[worsen_mask]] += 1

        mob[live] += 1
        t_upb = _amort_balance(upb0[live], rate[live], mob[live])

        new_defaults = live & (bucket == 5) & (bucket_prev_snapshot != 5)

        perf_rows.append(
            pd.DataFrame(
                {
                    "loan_id": loans["loan_id"].values[live],
                    "reporting_date": t,
                    "months_on_book": mob[live],
                    "upb": t_upb.round(0),
                    "bucket_prev": [BUCKET_NAMES[c] for c in bucket_prev_snapshot[live]],
                    "bucket_curr": [BUCKET_NAMES[c] for c in bucket[live]],
                    "is_default": new_defaults[live].astype(int),
                }
            )
        )

        if new_defaults.any():
            idx_def = np.where(new_defaults)[0]
            severity = np.clip(
                0.35 - 0.004 * hpi_yoy + 0.0075 * (ltv[idx_def] - 80) + rng.normal(0, 0.05, idx_def.size),
                0.05,
                0.95,
            )
            upb_at_default = _amort_balance(upb0[idx_def], rate[idx_def], mob[idx_def])
            default_rows.append(
                pd.DataFrame(
                    {
                        "loan_id": loans["loan_id"].values[idx_def],
                        "default_date": t,
                        "upb_at_default": upb_at_default.round(0),
                        "severity": severity.round(4),
                        "net_loss": (upb_at_default * severity).round(0),
                    }
                )
            )
            active[idx_def] = False  # charge-off is terminal; loan exits panel

    performance = pd.concat(perf_rows, ignore_index=True)
    defaults = pd.concat(default_rows, ignore_index=True) if default_rows else pd.DataFrame(
        columns=["loan_id", "default_date", "upb_at_default", "severity", "net_loss"]
    )
    return performance, defaults


def main(seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loans_crisis = build_loans(12_000, "2005-01-01", "2010-12-01", rng, id_start=1)
    loans_benign = build_loans(10_000, "2015-01-01", "2019-12-01", rng, id_start=100_001)
    loans = pd.concat([loans_crisis, loans_benign], ignore_index=True)

    macro = _load_macro()
    performance, defaults = simulate(loans, macro, rng)

    loans.to_parquet(OUT_DIR / "loans_static.parquet", index=False)
    performance.to_parquet(OUT_DIR / "loan_performance.parquet", index=False)
    defaults.to_parquet(OUT_DIR / "loan_defaults.parquet", index=False)

    print(f"Loans: {len(loans):,}")
    print(f"Performance rows: {len(performance):,}")
    print(f"Defaults: {len(defaults):,}  (cumulative default rate: {len(defaults) / len(loans):.2%})")


if __name__ == "__main__":
    main()
