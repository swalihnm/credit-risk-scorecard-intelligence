# Scorecard methodology

Fit on real Home Credit data (307,511 applications, downloaded from Kaggle — see
[`01-data-sources.md`](01-data-sources.md)). Numbers below are from that run; an earlier
pass on the synthetic stand-in (used before the real dataset was downloaded) produced
directionally identical results at different specific values, referenced where useful for
contrast.

## Windows and definitions

- Default definition: `TARGET = 1` (Home Credit's own labeling; conceptually a
  90+ DPD-equivalent bad definition).
- Observation/performance window: implicit in the source data —
  `application_train` has no explicit performance window column — a genuine limitation of
  this dataset (real, not just the earlier synthetic stand-in), disclosed in
  [`06-assumptions-and-limitations.md`](06-assumptions-and-limitations.md).

## Sequence

1. **Coarse classification** — `optbinning` monotonic optimal binning
   (`scorecard/binning.py`), with monotonicity enforced by business logic on characteristics
   where it's non-negotiable (higher DTI/utilization must not reduce risk) via
   `MONOTONIC_TREND`.

   *Windows/Python 3.12 note:* `optbinning` (via its ortools/CP-SAT dependency) segfaults at
   import time if `pandas` is imported first in the same process — a native
   DLL-loading-order conflict, not a bug in either library's Python code. Every module using
   both must `import optbinning` before `import pandas`. On the real dataset, 5 of 30
   candidate characteristics also fail during CP-SAT model construction with an ortools
   API-version incompatibility for their specific split configuration (more than the 2 that
   failed on the smaller synthetic dataset — the failure is sensitive to how many distinct
   split combinations the solver has to consider); all are gracefully skipped rather than
   chased further, per the same fallback logic.

2. **WOE transformation and IV screening** (`scorecard/woe_iv.py`) — drop IV < 0.02, flag
   IV > 0.5. Only **2** characteristics flag on real data (`EXT_SOURCE_2` at 0.643,
   `EXT_SOURCE_3` at 0.671) — right in the range real Home Credit `EXT_SOURCE` IVs are known
   to run (0.3-0.7), a useful confirmation that the earlier synthetic generator had made
   these signals implausibly strong (IVs up to ~2.0 there, vs. a real ceiling around 0.67
   here).

3. **Correlation and VIF filtering** (`scorecard/fit_logistic.py`) — drop one of any pair
   with |correlation| > 0.75 (keeping the higher-IV member; only `AMT_CREDIT` dropped this
   way on real data), then iteratively drop the highest-VIF characteristic until all
   remaining are < 5 (none needed dropping on real data).

4. **L1-guided selection down to 10-15 characteristics**, then a plain (unpenalized) refit on
   just the selected set — standard practice: use penalized/stepwise selection to choose
   variables, refit ordinary logistic regression for clean, interpretable coefficients.
   15 characteristics survived L1 selection on real data.

5. **Sign-consistency pruning.** WOE is coded so higher = safer for every characteristic (by
   construction of `MONOTONIC_TREND`), so a properly specified model must have every
   coefficient negative. On real data, 1 of the 15 L1-selected characteristics
   (`bureau_debt_to_income`) came out positive in the initial refit — a residual-collinearity
   sign reversal (individually negatively related to the target via its own WOE/event-rate
   table, but partially redundant with other bureau characteristics in the multivariate fit).
   Dropped and refit. **Final: 14 characteristics** — `EXT_SOURCE_2/3/1`, `prev_downsell_ratio`,
   `AMT_GOODS_PRICE`, `bureau_avg_utilization`, `instalment_avg_payment_ratio`,
   `cc_avg_utilization`, `bureau_n_closed_credit`, `instalment_pct_early`,
   `DAYS_ID_PUBLISH`, `bureau_n_active_credit`, `prev_refusal_rate`, `prev_approval_rate`.

6. **Score scaling** (`scorecard/scaling.py`) — Base 600 at 50:1 odds, PDO 20:
   `Factor = PDO / ln(2) ≈ 28.85`, `Offset = Base - Factor * ln(BaseOdds) ≈ 487.12`.
   `base_points = Offset - Factor * intercept`; each characteristic bin contributes
   `-Factor * coef * WOE(bin)`. Real-data score distribution: min 442, max 656, mean/median
   ≈ 567.

## Validation

- **Train/test split, not out-of-time.** `application_train` (real) has no application date
  column, so a genuine out-of-time split isn't possible — a real, structural limitation of
  this dataset, confirmed now that real data is in use (not an artifact of the earlier
  synthetic stand-in). A stratified 70/30 random split is used instead. True out-of-time
  validation *is* demonstrated on the Fannie Mae side (`portfolio/survival.py`), which has a
  genuine time dimension — see [`04-ecl-and-stress-testing.md`](04-ecl-and-stress-testing.md).
- **Result on real data:** AUC 0.748 train / 0.749 test, Gini **0.496 / 0.497**,
  KS 0.370 / 0.374 — stable across the split, no overfitting signal, and consistent with
  published benchmarks for a plain WOE/logistic scorecard on this dataset (top Kaggle
  solutions with heavy feature engineering and gradient boosting reach 0.75-0.80 AUC; a
  12-15 characteristic linear scorecard landing at 0.748 AUC is exactly the kind of gap the
  challenger-model comparison in `docs/07-results.md` is meant to quantify).

## Calibration

Platt scaling and isotonic regression compared against the raw logistic output
(`scorecard/calibration.py`). Brier and ECE are nearly identical across all three on real
data (Brier ≈ 0.0685, ECE 0.0011-0.0027); the reliability table shows predicted vs. actual
default rate tracking closely across deciles (e.g. the top decile: 25.85% predicted vs.
25.98% actual). Hosmer-Lemeshow (chi2=14.21, p=0.076 on n=92,254) fails to reject at the 5%
level — read alongside the reliability table and ECE, not in isolation, per the standard HL
caveat about sample-size sensitivity.

**Platt scaling is the deployed calibrator**: at this dataset's ~8% base rate, isotonic
regression's step function can get jagged in sparse probability regions, while Platt's
1-parameter logistic form is stable and performs comparably here on Brier/ECE (isotonic's ECE
is actually lower on real data, 0.0011 vs. Platt's 0.0026 — but Platt remains the deployed
choice for its stability guarantee, a legitimate judgment call to revisit if ECE is weighted
more heavily than robustness in a real deployment decision).

## Challenger model

See [`docs/07-results.md`](07-results.md) for the LightGBM/SHAP comparison and the
deploy-or-not discussion — **the conclusion changed on real data**: the challenger now beats
the scorecard out-of-sample (unlike on the synthetic stand-in, where it didn't), making this
a genuine accuracy-vs-interpretability tradeoff rather than a clean "no."
