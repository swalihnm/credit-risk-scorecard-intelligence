# Independent Validation Report — Home Credit Application Scorecard v1.0

Written as a second-line-of-defence review of the model documented in `MODEL_CARD.md`. The
same person built and validated this model in this build (a genuine limitation, disclosed
below), but the review follows the structure a real independent validation function would use.

## 1. Data quality

- **Resolved finding:** development data was initially synthetic; real Kaggle data
  (307,511 applications, `docs/01-data-sources.md`) has since been downloaded and the full
  pipeline re-run against it. Every number in this report now reflects the real dataset
  unless stated otherwise. This is left in the report, marked resolved, rather than deleted,
  so the "before/after" comparison (e.g. the challenger's conclusion flipping — see section 4)
  stays visible.
- **Finding:** the reject pool used for reject inference (`docs/03-reject-inference.md`) is
  itself synthetic and generated from the *same* risk model as the accepted population
  (shifted, not independently specified) — meaning the reject-inference validation numbers
  in that document are somewhat optimistic evidence of the *mechanics* working correctly,
  not independent evidence that the specific 1.5x inflation assumption is well-calibrated
  for a real reject population.
- **No missing-completely-at-random assumption tested** for `EXT_SOURCE_1`/`EXT_SOURCE_3`
  missingness (40%/20% respectively) — WOE binning's "Missing" bin treats missingness as
  informative, which is appropriate if missingness itself correlates with risk (plausible for
  a real bureau score not being available), but this hasn't been separately tested here.

## 2. Methodology soundness

- **Binning and WOE:** sound; monotonicity enforced by business logic where required
  (`scorecard/binning.py`'s `MONOTONIC_TREND`), not left to the optimizer to discover
  incidentally.
- **Variable selection:** correlation + VIF filtering, L1-guided reduction to 15
  characteristics, then a sign-consistency pruning pass caught and removed 1 characteristic
  (`bureau_debt_to_income`) with multivariate sign reversal, landing at 14
  (`docs/02-scorecard-methodology.md`) — this is a genuinely good practice a validator would
  want to see, not something to flag.
- **Validation split:** a stratified random 70/30 split, not out-of-time, because the
  real dataset has no application-date field (confirmed now that real data is in use, not
  just a synthetic-stand-in artifact). **This is a finding**, not
  fully mitigated: a random split cannot detect a model that would degrade over time due to
  macro or population drift, only whether it generalizes to a randomly-held-out slice of the
  same population at the same (undated) point.
- **Calibration:** appropriately assessed via three methods (raw, Platt, isotonic) plus
  Hosmer-Lemeshow with the standard large-sample caveat applied correctly (not over-read as a
  hard pass/fail).

## 3. Assumption testing

- **LGD = 70% flat** (unsecured lending, `policy/cutoff_optimizer.py`): a standard industry
  benchmark, not derived from this portfolio's own loss data (none exists — Home Credit has
  no realized-loss field). **Finding:** sensitivity of the cutoff recommendation to this
  assumption has not been tested; recommend a sensitivity run at LGD 60%/80% before any real
  deployment decision leans on the specific cutoff=565 recommendation.
- **15% APR, 8% capital ratio, 2% opex ratio:** all flat, undifferentiated-by-risk
  assumptions. A real deployment would risk-differentiate at minimum the capital ratio (IRB
  risk weight scaling with PD/LGD) — flagged as a simplification in
  `policy/cutoff_optimizer.py`'s own docstring, not hidden.

## 4. Benchmark comparison

LightGBM challenger built and compared honestly (`docs/07-results.md`): on real data, it
**does** out-perform the scorecard out-of-sample (test Gini 0.538 vs. 0.497) — the opposite
of what an earlier run on the synthetic stand-in showed (challenger overfit on 30K rows,
scorecard won there). **Finding:** this project's own earlier conclusion ("no accuracy gain,
clean no on deployment") no longer holds under real data — a validator reviewing only the
synthetic-stand-in-era writeup would have signed off on a conclusion that reverses once real
data is used. This is exactly the kind of thing an independent review is supposed to catch,
and exactly why re-running validation after a data source changes (not just after a
methodology changes) matters.

## 5. Findings summary and recommendations

| # | Finding | Severity | Recommendation |
|---|---|---|---|
| 1 | ~~Development data is synthetic~~ — **resolved**: real Kaggle data now in use | Resolved | None — re-validate again if the reject pool or Fannie Mae/FRED sources are ever swapped to real data too |
| 2 | No out-of-time validation possible (Home Credit side) | Medium | Accept as a dataset limitation; monitor Gini/PSI in production closely post-deployment as compensation |
| 3 | LGD/APR/capital assumptions are flat, untested for sensitivity | Medium | Run a sensitivity grid before finalizing the cutoff recommendation |
| 4 | Reject-inference validation is against a synthetic reject pool not matched to the real accepted population's distributions | Low-Medium | Treat the "+0.5 points, -1pp approval" result as illustrative of the *mechanism*, not a production-ready calibration of the inflation factor |
| 5 | **The challenger now beats the scorecard on real data (Gini 0.538 vs. 0.497)** | High | This is a genuine committee-level tradeoff now, not a settled "no" — see `docs/07-results.md` section on deploy-or-not. Do not defer this decision by re-citing the earlier synthetic-data conclusion. |
| 6 | Validator and developer are the same party in this build | Process | A real deployment requires genuine independent review before go-live |
