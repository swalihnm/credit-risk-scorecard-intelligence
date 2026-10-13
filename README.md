# Credit Risk Scorecard & Portfolio Decision Intelligence

A credit policy recommendation, not a classifier: an application scorecard producing
calibrated probability of default, a portfolio expected-credit-loss engine with a
macroeconomic overlay, an approval-cutoff optimizer, and a Power BI reporting/monitoring
layer.

## Division of labour

- **Python** — modelling, reject inference, calibration, ECL simulation, cutoff optimization.
- **Power BI** — portfolio reporting, monitoring, interactive cutoff exploration.
- **FastAPI** — single-application scoring endpoint.

## Data sources

Three sources, deliberately kept separate (never merged into one portfolio) — see
[`docs/01-data-sources.md`](docs/01-data-sources.md) for the full rationale and swap-in path
for each:

| Source | Role | Status |
|---|---|---|
| Home Credit Default Risk (Kaggle) | Application scorecard | **Real data** — 307,511 applications, downloaded from Kaggle and verified against the known dataset (row count, ~8.07% base default rate). Still synthetic: the reject-inference pool (`docs/03-reject-inference.md`), since no version of this dataset includes one. |
| Fannie Mae Loan Performance | Vintage/roll-rate/survival/ECL | **Synthetic stand-in** — real extracts require manual portal registration. |
| FRED macro series | Stress testing overlay | **Real data** — 478 months (1987-2026) pulled via `fredapi`, feeding the Fannie Mae hazard model and stress scenarios. |

Every generator matches the real schema exactly and every real-data reader is already
written, so swapping in the remaining one (Fannie Mae) means re-running ingestion and the
warehouse build — nothing in `features/`, `scorecard/`, `challenger/`, `portfolio/`, or the
marts changes. See [`docs/06-assumptions-and-limitations.md`](docs/06-assumptions-and-limitations.md)
for the full disclosure.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Windows/Python 3.12 note:** `optbinning` (via ortools/CP-SAT) segfaults at import time if
`pandas` is imported first in the same process — a native DLL-loading-order conflict, not a
bug. Every module that touches both imports `optbinning` first; see `scorecard/binning.py`'s
header comment and `tests/conftest.py`.

### Getting the real Home Credit data

1. Log into [kaggle.com](https://www.kaggle.com), go to
   [kaggle.com/c/home-credit-default-risk/data](https://www.kaggle.com/c/home-credit-default-risk/data),
   accept the competition rules, and click **Download All** (~700MB zip).
2. Extract the 7 needed CSVs into `sources/home_credit/` (same filenames as in the zip):
   `application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`,
   `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`.
   This overwrites the synthetic stand-in versions; the `*_rejects.csv` files (a separate,
   still-synthetic reject pool — see `docs/03-reject-inference.md`) are untouched.
3. Re-run the pipeline from `warehouse/build_warehouse.py` onward (below).

Alternatively, `ingest/kaggle_download.py` does steps 1-2 via the Kaggle API once you have a
token (`docs/00-kaggle-setup.md`).

### Full pipeline, in order

```
python ingest/synthetic_fred.py
python ingest/synthetic_fannie_mae.py
python ingest/synthetic_home_credit.py        # only if not using real Kaggle data -- generates accepts + reject pool either way (reject pool is always needed)
python warehouse/build_warehouse.py
python -m features.build_features             # feature_table + feature_table_rejects

python scorecard/binning.py
python -m scorecard.woe_iv
python -m scorecard.fit_logistic
python -m scorecard.scaling
python -m scorecard.calibration
python -m scorecard.reject_inference
python -m scorecard.adverse_action

python -m challenger.lightgbm_challenger

python -m portfolio.survival
python -m portfolio.lgd
python -m portfolio.ecl_engine

python sql/marts/build_marts.py                # Fannie Mae side
python -m policy.cutoff_optimizer              # writes mart_cutoff_frontier + scorecard/artifacts/policy_cutoff.json
python -m sql.marts.build_marts_home_credit    # needs cutoff_optimizer's output

pytest tests/ -v

uvicorn app.api:app --reload                   # scoring service
```

**Note if using real Kaggle data:** still run `ingest/synthetic_home_credit.py` once for its
reject-pool output (`*_rejects.csv`, `reject_hidden_truth.csv`) — real Kaggle data has no
reject pool either, so this file is needed regardless of which `application_train.csv` you
use. Just don't let it overwrite real CSVs you've already placed in `sources/home_credit/`
(it writes `application_train.csv` etc. too; extract the real zip *after* running it, or move
the reject-pool files elsewhere first).

**Swapping in real Fannie Mae data:** register at the Fannie Mae Data Dynamics portal, fill
in `ingest/fannie_mae_ingest.py`'s two parser stubs against the real file layout, then re-run
from `warehouse/build_warehouse.py` onward. (FRED is already real data — see below.)

## What's built

- **Data & features** — DuckDB warehouse, 7-table Home Credit relational feature engineering
  (bureau, previous-application, payment behaviour) on real Kaggle data, point-in-time
  leakage audit, 5 SQL analytics files (vintage curves, roll rates, concentration/HHI,
  segment performance, macro alignment).
- **Scorecard** — monotonic optimal binning → WOE/IV screening → correlation/VIF filtering →
  L1-guided selection → sign-consistency pruning → 14-characteristic logistic scorecard
  (Gini **0.497** test, on real data) → points scaling (Base 600 @ 50:1, PDO 20) →
  Platt/isotonic calibration → fuzzy-augmentation reject inference (+0.5 points, quantified
  against a hidden-truth reject pool) → LightGBM + SHAP challenger (**beats** the scorecard
  out-of-sample on real data, test Gini 0.538 vs. 0.497 — a genuine tradeoff now, see
  `docs/07-results.md` for the deploy-or-not discussion) → adverse-action reason codes.
- **Portfolio (Fannie Mae side, synthetic loan panel + real FRED macro data)** —
  discrete-time hazard/survival model with time-varying macro covariates, out-of-time
  validated (trained on 2005-2010 vintages only, AUC 0.629 on 2015-2019 vintages),
  credibility-weighted LGD by region×LTV, 3 macro stress scenarios, IFRS 9 three-stage ECL
  engine (baseline ~$27.8M → severely adverse ~$96.8M, correctly monotonic).
- **Decision** — cutoff sweep + constrained optimizer (loss-rate cap, volume floor, AND a
  segment concentration limit) with a genuine binding-constraint finding on real data: the
  **volume floor** binds (cutoff 565, 55.0% approval) — the loss cap and concentration limit
  are both slack, the opposite of what the smaller synthetic-data run showed (there, the
  concentration limit bound instead). Shadow prices for all three constraints
  (see `docs/07-results.md`).
- **Power BI contract** — every mart in `marts/`, all DAX measures as reviewable `.txt` in
  `powerbi/measures/`, star-schema design doc (`docs/05-powerbi-data-model.md`), RLS design.
  The `.pbix` itself is not built — Power BI Desktop is a Windows GUI app with no scripting
  interface for model/report authoring, so this is the exact blueprint to wire up by hand.
- **Governance** — model card, monitoring plan (PSI/Gini/calibration thresholds and
  retraining triggers), adverse-action logic, an independent-style validation report,
  RLS design — all updated to reflect real-data numbers and the challenger's flipped
  conclusion.
- **Serving** — FastAPI scoring endpoint (`/score`, `/health`, `/model/version`,
  `/model/required-features`). Dockerfile built and verified end-to-end: `docker build`,
  ran the container, and hit `/health`, `/model/version`, `/model/required-features`, and a
  real `/score` request — the containerized response matched the local pipeline's output
  exactly (same score, decision, and reason codes) for a real applicant.
- **Tests** — 23 tests across data integrity, portfolio economics, scorecard correctness,
  policy optimization, and the API, several of which exist specifically because they caught
  real bugs during development (see below).

## Bugs caught and fixed during development (kept as regression tests)

1. **Miscalibrated hazard model** — `class_weight="balanced"` inflated predicted PD to ~100%
   for nearly every loan. Removed; PD models need calibration, not balanced accuracy.
2. **Unclipped tenure extrapolation** — projecting `months_on_book` beyond the training
   panel's ceiling blew up the hazard via the (correctly-signed) seasoning coefficient.
   Fixed by clipping at the training maximum.
3. **Wrong-signed HPI feature** — collinearity between unemployment and HPI YoY gave the raw
   HPI coefficient the wrong sign. Fixed by reframing as an asymmetric `hpi_decline_pct`
   feature (only price *declines* should raise risk).
4. **Undercounted vintage curve** — the SQL joined through `fm_performance`, where a
   charged-off loan has no further rows, silently dropping it from every later
   months-on-book bucket. Fixed by computing cumulative defaults directly against a
   month-offset grid.
5. **Non-monotonic stress scenarios** — replaying the literal 2008 macro slice for "severely
   adverse" produced a milder HPI trough than "adverse" at that window's boundary. Fixed with
   explicit, guaranteed-ordered magnitudes.
6. **Sign-reversed scorecard coefficients** — L1-selected characteristics flipped sign in the
   multivariate fit (residual collinearity) despite being correctly signed individually.
   Fixed with iterative sign-consistency pruning (this recurred once on real data too — 1 of
   15 characteristics, `bureau_debt_to_income` — confirming the fix generalizes, not a
   one-off patch for the synthetic dataset's specific quirks).
7. **A pandas/optbinning import-order segfault** — see the Windows note above.
8. **No real out-of-time validation** — an early version of the docs claimed the Fannie Mae
   hazard model demonstrated out-of-time validation, but the code just fit on the full panel.
   Fixed by actually training on 2005-2010 vintages and validating on 2015-2019 vintages
   (AUC 0.629 on real FRED macro data, well-calibrated hazard, across a materially different
   macro regime).
9. **Missing segment concentration constraint** — the cutoff optimizer enforced a loss-rate
   cap and a volume floor but not the concentration limit the spec also asks for. Added it.
10. **Hardcoded policy cutoff drifting stale in 3 files** — the optimizer's computed cutoff
    (570 → 560 → 565 as the model and data changed) was hand-copied into `app/api.py` and
    `sql/marts/build_marts_home_credit.py`, and had to be manually re-chased each time it
    changed. Fixed by persisting it to `scorecard/artifacts/policy_cutoff.json` and having
    both consumers read it instead of hardcoding a number.
11. **API request schema hardcoded to a stale characteristic list** — `app/api.py`'s Pydantic
    model declared fields for the 12 characteristics a synthetic-data fit had selected; the
    real-data fit selected a different 14, several missing from the schema, and every score
    request for those returned 422. Fixed by making the request model accept arbitrary
    fields and validating against the model's *actual current* requirement at request time,
    exposed via a new `GET /model/required-features` endpoint — the same class of "hardcoded
    against one fit's output" bug as #10, fixed the same way.
12. **Adverse-action reason text missing for 8 of 14 real characteristics** — same root cause
    as #11 (a lookup hardcoded against the synthetic fit's characteristic set); degraded
    gracefully to showing the raw column name rather than crashing, which is why it wasn't
    caught until someone read the actual output. Fixed by covering every characteristic seen
    across both fits.
13. **FRED ingestion silently dropped from ~470 months to 81 scattered ones** —
    `ingest/fred_ingest.py` chained `.resample("MS").ffill()` directly on the concatenated raw
    series. That shorthand only keeps a value when the original observation lands exactly on
    day 1 of the month — true for monthly series reported on the 1st, almost never true for
    `DGS10` (daily; day 1 is frequently a weekend/holiday), and never true for `GDPC1`
    (quarterly). `dropna()` then discarded most months as incomplete. Fixed by resampling
    each series to month-start *individually* with `.resample("MS").last()` (correctly
    aggregating "the last observation within the month") before combining.

## Repository layout

See [`docs/`](docs/) for full methodology write-ups (start with `07-results.md` for the
headline numbers), [`governance/`](governance/) for the model card / monitoring plan /
adverse-action logic / validation report, and [`powerbi/`](powerbi/) for the DAX measures and
data-model design.
