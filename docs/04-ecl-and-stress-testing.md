# Expected credit loss engine and stress testing

Applies to the Fannie Mae (synthetic stand-in) portfolio only — see
[`01-data-sources.md`](01-data-sources.md) for why this is never combined with the Home
Credit application scorecard. The loan-performance panel itself is still synthetic, but the
macro covariates driving it (`unrate_at_performance_month`, `hpi_decline_pct`) are now **real
FRED data** (1987-2026) — see `01-data-sources.md` for a resampling bug that was caught and
fixed while wiring this in.

## PD: discrete-time hazard model (`portfolio/survival.py`)

Fit as a logistic regression on the loan-month panel (one row per loan per reporting month),
predicting the monthly hazard of first default. Covariates: `months_on_book` (tenure/seasoning),
static risk factors (`fico_orig`, `original_ltv`, `orig_rate`), and macro conditions **as of
that month** (`unrate_at_performance_month`, `hpi_decline_pct`) — genuinely time-varying
covariates, not a single snapshot.

`hpi_decline_pct = max(0, -HPI YoY%)` rather than the raw signed YoY change: only price
*declines* are expected to raise default risk (negative equity), while further appreciation
beyond baseline shouldn't keep lowering it — an asymmetric, more identifiable feature than
the raw series.

Cumulative PD over any horizon = `1 - product(1 - monthly hazard)` across the horizon,
projected forward month-by-month under a given macro path
(`project_cumulative_pd_batch`). Two model-development issues surfaced and were fixed during
development (see the docstrings in `portfolio/survival.py` and `tests/test_portfolio.py`,
which now guard against both regressing):

1. **Fitting with `class_weight="balanced"`** inflated predicted probabilities far above the
   true ~0.3%-per-month base rate. A PD model must be calibrated, not balanced-accuracy
   optimized, so this was removed.
2. **Unclipped `months_on_book` extrapolation.** Surviving loans in the synthetic panel sit at
   `months_on_book` at or near the panel's simulation ceiling (60). Projecting forward without
   bound pushed the tenure feature far outside its training range, and the (correctly-signed)
   positive seasoning coefficient blew the extrapolated hazard up toward certainty. The fix
   clips the projected `months_on_book` at the training-set maximum — a real Fannie Mae extract
   spanning the full loan term wouldn't need this, so it's disclosed as a synthetic-panel
   limitation in [`06-assumptions-and-limitations.md`](06-assumptions-and-limitations.md).

## Out-of-time validation (`portfolio/survival.py`'s `out_of_time_validation`)

Per the project's validation protocol (train early vintages, validate later ones — not just
a random holdout): the hazard model is fit on person-months from **2005-2010 (crisis)
vintages only** and validated on **2015-2019 (benign) vintages** — a genuinely harder test
than a random split, since the two windows sit in materially different macro regimes. Result:
**AUC 0.629** on the out-of-time vintages (real FRED macro data; 0.660 on the earlier
synthetic macro panel), with predicted vs. actual monthly hazard of 0.00193 vs. 0.00149 —
close enough to call well calibrated across the regime shift, not just discriminating. This
is only possible because the model leans on its macro covariates
(unemployment, HPI decline) rather than memorizing vintage-specific patterns — exactly the
point of including them as time-varying covariates in the first place.

The **deployed** model (used by the ECL engine and stress scenarios below) is refit on the
full panel once this out-of-time check confirms the methodology generalizes — standard
practice: validate on a genuine holdout, then use all available data for the model that
actually goes into production.

## LGD (`portfolio/lgd.py`)

Realized loss severity (`net_loss / UPB at default`) segmented by region x LTV band, with
credibility-weighted shrinkage toward the portfolio mean for thin segments
(`n / (n + 30)` credibility). Result is cleanly monotonic in LTV band — mean LGD rises from
~24-28% at LTV <60 to ~45-48% at LTV 90+, consistent with collateral-value intuition.

## EAD

`EAD = current UPB`. Fannie Mae mortgages are fully drawn at origination, so there is no
undrawn-commitment adjustment (unlike a revolving product).

## Stage assignment (IFRS 9 rebuttable-presumption backstops)

| Stage | Trigger | ECL window |
|---|---|---|
| 1 | Current, no elevated risk | 12-month |
| 2 | 30-59 dpd (SICR backstop) | Lifetime (capped at 60 months — see below) |
| 3 | 60+ dpd (credit-impaired backstop) | Lifetime |

Charged-off loans are excluded from the ECL calculation — their loss is realized, not
expected.

## Discounting

`ECL = PD x LGD x EAD x discount_factor`, with `discount_factor = 1 / (1 + orig_rate/1200)^t`
where `t` is the midpoint of the ECL window (6 months for Stage 1, 30 months for Stage 2/3).
The loan's own contractual rate is used as the discount rate — a simplification defensible
for an illustrative build but not what a bank would use (effective interest rate net of fees).

## Lifetime horizon cap

Lifetime ECL is capped at a 60-month horizon because that is the simulation window of the
synthetic loan-performance panel (`MAX_MOB` in `ingest/synthetic_fannie_mae.py`). A real
Fannie Mae extract spanning the full loan term would support projecting to contractual
maturity.

## Stress scenarios (`portfolio/stress_scenarios.py`)

| Scenario | Unemployment | HPI | Ramp |
|---|---|---|---|
| Baseline | Last 12 observed months, held flat | Last 12 observed months, held flat | n/a |
| Adverse | +3pp | -15% | Linear over 8 quarters, then flat |
| Severely adverse | +6pp | -30% | Linear over 8 quarters, then flat |

Magnitudes (+6pp unemployment, -30% HPI for severely adverse) were originally matched to the
synthetic macro panel's own 2008 peak (~11% unemployment); now that the macro panel is real
FRED data, the baseline scenario reflects actual current macro conditions (the real trailing
12-month average as of ingestion) rather than a synthetic endpoint, while the stress
magnitudes remain fixed, explicit assumptions on top of that real baseline. (Replaying the
literal 2008 historical slice for severely adverse was tried first and produced a
non-monotonic scenario ordering purely from that window's boundary effects — see git
history.)

## Results (illustrative — see limitations)

Total portfolio ECL across active (non-charged-off) loans as of the last simulated
reporting month, using real FRED macro data:

| Scenario | Total ECL | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| Baseline | ~$27.8M | ~$26.0M | ~$0.5M | ~$1.3M |
| Adverse | ~$51.0M | ~$43.5M | ~$2.1M | ~$5.4M |
| Severely adverse | ~$96.8M | ~$78.3M | ~$5.3M | ~$13.2M |

**What drives ECL most:** unemployment has the largest standardized effect on monthly hazard,
narrowly ahead of HPI decline; both dominate the static risk factors (FICO, LTV) at the
portfolio level because the macro shock hits every active loan simultaneously, while FICO/LTV
only differentiate risk *within* a given macro state.
