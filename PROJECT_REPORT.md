# Credit Risk Scorecard & Portfolio Decision Intelligence — Project Report

**Status as of this report:** Python/SQL/API layer complete and verified. Power BI report in progress.

---

## 1. Executive Summary

This project builds a full credit risk decisioning system, not a classifier: an application
scorecard producing a calibrated probability of default (PD), a portfolio expected-credit-loss
(ECL) engine with a macroeconomic stress-testing overlay, a constrained approval-cutoff
optimizer, a governance layer, a scoring API, and a Power BI reporting/monitoring layer.

**Business questions answered:**
- At what score do we approve, and what does that cost in expected loss versus earn in volume and yield?
- How much do we provision under baseline and stressed macro scenarios?
- Is the model still working, and how would we know before it costs us money?

**Division of labour:** Python does modelling, reject inference, calibration, ECL simulation,
and the constrained cutoff optimizer. Power BI does portfolio reporting, monitoring, and
interactive cutoff exploration. FastAPI exposes single-application scoring.

**Headline results:**
- 14-characteristic logistic scorecard on **real** Home Credit data (307,511 applications):
  Gini 0.497 (test), well calibrated.
- A LightGBM challenger **beats** the scorecard out-of-sample (Gini 0.538) — a genuine
  accuracy-vs-adverse-action-defensibility tradeoff, not a clean deploy/don't-deploy answer.
- Reject inference (fuzzy augmentation) quantifiably tightens the scorecard by +0.5 points.
- A constrained cutoff optimizer settles on cutoff 565 (55.0% approval), with the **volume
  floor**, not the loss cap or concentration limit, as the actual binding constraint.
- An IFRS 9 three-stage ECL engine on the (synthetic, disclosed) Fannie Mae mortgage
  portfolio produces baseline ECL ~$28.7M rising to ~$104.1M under a severely adverse
  scenario, out-of-time validated (trained on 2005-2010 vintages, AUC 0.660 on 2015-2019
  vintages).
- 23 automated tests pass; a FastAPI scoring service is built, containerized, and verified
  end-to-end in Docker.
- 12 real bugs were caught and fixed during development (full list in Section 9) — several
  specifically because real data was swapped in for an earlier synthetic stand-in.

---

## 2. Data Sources

Three sources, deliberately kept separate — never merged into one portfolio, since they are
different products, markets, and regulatory regimes.

| Source | Role | Status |
|---|---|---|
| **Home Credit Default Risk** (Kaggle) | Application scorecard | **Real data.** 307,511 applications downloaded from Kaggle, verified against the known dataset (row count, ~8.07% base default rate). |
| **Fannie Mae Single-Family Loan Performance** | Vintage curves, roll rates, survival/hazard model, LGD, ECL, stress testing | **Synthetic stand-in.** Real extracts require manual portal registration (no scriptable API). The generator (`ingest/synthetic_fannie_mae.py`) is a macro-modulated Markov delinquency model matching the real schema exactly, tuned to be directionally realistic (crisis vintages default more than benign vintages). |
| **FRED macro series** (`UNRATE`, `CSUSHPINSA`, `DGS10`, `GDPC1`) | Stress-testing overlay | **Synthetic stand-in.** A free API key was attempted but not obtained during this build; the real-data reader (`ingest/fred_ingest.py`) is written and ready. |
| **Home Credit reject pool** | Reject inference | **Synthetic, and stays synthetic regardless of the accepts data.** No version of this dataset — real or synthetic — includes a population of declined applicants with known features but no observed outcome. This is a structural gap in the dataset itself, not something downloading real data fixes. |

Every synthetic generator produces the exact real-world schema, and every corresponding
real-data reader is already written — swapping in the remaining real sources requires only
running those readers and rebuilding the warehouse; nothing in `features/`, `scorecard/`,
`challenger/`, `portfolio/`, or the marts needs to change.

---

## 3. Architecture

```
sources/          home_credit/ (real), fannie_mae/ (synthetic), fred/ (synthetic)
   |
ingest/           real + synthetic readers, identical output schemas
   |
warehouse/        DuckDB (hc_*, hc_reject_*, fm_*, macro_monthly tables)
   |
features/         relational aggregation (bureau, previous_application, payment behaviour)
   |
scorecard/        binning -> WOE/IV -> logistic -> scaling -> calibration -> reject inference
challenger/       LightGBM + SHAP
portfolio/        survival/hazard, LGD, ECL engine, stress scenarios     (Fannie Mae)
policy/           constrained cutoff optimizer -> mart_cutoff_frontier   (Home Credit)
   |
marts/            <-- the Power BI contract (parquet)
   |
   +--> Power BI    portfolio MIS, monitoring, cutoff explorer   [IN PROGRESS]
   +--> FastAPI     single-application scoring endpoint          [DONE, containerized]
```

**Stack:** Python 3.12, DuckDB, pandas/numpy, `optbinning`, scikit-learn, LightGBM, SHAP,
statsmodels, FastAPI, Docker, Power BI Desktop.

---

## 4. Data & Feature Engineering

Relational feature engineering across all 7 Home Credit tables (`application_train`,
`bureau`, `bureau_balance`, `previous_application`, `POS_CASH_balance`,
`installments_payments`, `credit_card_balance`), producing engineered signals such as:

- Bureau: active/closed credit line counts, utilization and its trend, max DPD ever, %
  months delinquent, debt-to-income.
- Previous applications: approval rate, refusal rate, downsell ratio (requested vs. granted).
- Instalments: payment ratio, days late/early, % late/early.
- Credit cards: utilization and its trend, minimum-payment-only frequency.

**Leakage audit** (`features/leakage_audit.md`): every feature is justified as constructible
at application time — bureau/previous-application/payment tables are historical by
construction (all `DAYS_*` / `MONTHS_BALANCE` fields are `<= 0` relative to the current
application).

5 SQL analytics files (`sql/analytics/`): vintage curves, roll rate matrix, portfolio
concentration/HHI, segment performance with confidence intervals, macro alignment.

---

## 5. Scorecard Development

**Sequence** (`scorecard/binning.py` → `woe_iv.py` → `fit_logistic.py` → `scaling.py` → `calibration.py`):

1. **Monotonic optimal binning** (`optbinning`), with monotonicity enforced by business logic
   where non-negotiable (e.g. higher DTI/utilization must not reduce risk).
2. **WOE transformation and IV screening** — drop IV < 0.02, flag IV > 0.5. On real data, only
   `EXT_SOURCE_2` (0.643) and `EXT_SOURCE_3` (0.671) flag — right in the range real Home
   Credit `EXT_SOURCE` IVs are known to run, a useful confirmation that an earlier synthetic
   version of this build had made these signals implausibly strong (IV up to ~2.0 there).
3. **Correlation and VIF filtering** — drop one of any pair with |correlation| > 0.75 (keeping
   the higher-IV member), then iteratively drop the highest-VIF characteristic until all
   remaining are < 5.
4. **L1-guided selection** down to 10-15 characteristics (15 survived on real data), then a
   plain (unpenalized) refit for clean, interpretable coefficients.
5. **Sign-consistency pruning** — WOE is coded so higher = safer for every characteristic; a
   properly specified model must have every coefficient negative. On real data, 1 of 15
   L1-selected characteristics (`bureau_debt_to_income`) came out positive in the initial
   refit — a residual-collinearity sign reversal, not a coding bug, but not defensible to a
   risk committee. Dropped and refit. **Final: 14 characteristics.**
6. **Score scaling** — Base 600 at 50:1 odds, PDO 20 (`Factor ≈ 28.85`, `Offset ≈ 487.12`).

**Final 14 characteristics:** `EXT_SOURCE_1/2/3`, `AMT_GOODS_PRICE`, `bureau_avg_utilization`,
`prev_refusal_rate`, `prev_approval_rate`, `cc_avg_utilization`, `bureau_n_active_credit`,
`instalment_avg_payment_ratio`, `prev_downsell_ratio`, `instalment_pct_early`,
`bureau_n_closed_credit`, `DAYS_ID_PUBLISH`.

**Validation:** stratified 70/30 random split (a true out-of-time split is impossible — the
real dataset has no application-date field, a genuine structural limitation, not a synthetic
artifact). Result: **AUC 0.748 train / 0.749 test, Gini 0.496 / 0.497, KS 0.370 / 0.374** —
stable, no overfitting, consistent with published benchmarks for a plain WOE/logistic
scorecard on this dataset.

**Calibration:** Platt scaling, isotonic regression, and the raw logistic output compared.
Brier ≈ 0.0685 and ECE 0.001-0.003 across all three; the reliability table shows predicted vs.
actual default rate tracking closely across deciles (top decile: 25.85% predicted vs. 25.98%
actual). Hosmer-Lemeshow (chi²=14.21, p=0.076, n=92,254) fails to reject at 5%. **Platt
scaling is the deployed calibrator** — stable at this dataset's ~8% base rate, where isotonic
regression's step function can get jagged in sparse regions.

---

## 6. Reject Inference

Training data (real or synthetic) contains only funded loans — fitting a scorecard on that
alone means fitting on a sample already filtered by whatever credit policy existed before,
the single most common fatal error in this discipline.

**Method: fuzzy augmentation.** A synthetic reject pool (8,000 applicants, `build_reject_pool`
in `ingest/synthetic_home_credit.py`) is scored with the accept-only model, its predicted PD
inflated 1.5x (the standard reject-inference assumption that a rejected applicant is
systematically worse than an accepted applicant at the same score), then duplicated as
weighted "good" and "bad" records and blended with the accepted population for a refit.

**Validation against hidden ground truth** (possible only because the reject pool is
synthetic — a real bank can never observe a real reject's true outcome):

| | Value |
|---|---|
| True default rate among rejects | 18.16% |
| Accept-only model's raw predicted rate | 17.00% (understated) |
| Reject-inference-inflated predicted rate | 25.48% |
| AUC of accept-only score ranking rejects by true outcome | 0.7375 |

**Quantified impact:** under a fixed policy (decline if predicted PD > 12%), approval rate
moves from 79.61% to 78.71%, and base points tighten from 556.5 to 556.0 (**+0.5 points**) —
smaller than an earlier development-stage run on fully synthetic data (+3.1 points), because
the real accept-only model was already better specified and needed less correction. Alternatives
considered and why they weren't used: parcelling (coarser, more transparent to a risk
committee, but the continuous probability from fuzzy augmentation is what makes the
hidden-truth validation meaningful) and Heckman two-stage (more principled but needs an
instrument unavailable here — the practical reason most industry implementations use fuzzy
augmentation or parcelling instead).

---

## 7. Challenger Model (LightGBM + SHAP)

| | Test Gini |
|---|---|
| Logistic scorecard (14 characteristics) | 0.497 |
| LightGBM (300 trees, depth 4, regularized) | **0.538** |

On real data, the challenger **beats** the scorecard out-of-sample by ~0.04 Gini — the
opposite of an earlier development run on the smaller synthetic dataset, where it overfit and
lost. SHAP confirms the same top drivers as the scorecard (`EXT_SOURCE_2/3/1`,
`bureau_avg_utilization`, `prev_downsell_ratio`, `DAYS_EMPLOYED`).

**Deploy-or-not — now a genuine tradeoff:**
1. The accuracy gain is real (~0.04 Gini would show up in a real book's loss rate).
2. The adverse-action cost is unchanged: a scorecard's reason is exactly "your points
   contribution from characteristic X was lowest" — an exact decomposition. SHAP values are a
   valid *local* attribution but are interaction-dependent and model-specific, not a stable
   causal statement about the applicant. Whether SHAP-based reasons satisfy a regulator's
   adverse-action requirement is a live legal question, not a solved one.
3. **Practical conclusion:** the gain is large enough that a risk committee shouldn't dismiss
   the GBM out of hand, but shouldn't approve it either without resolving the adverse-action
   question (e.g. a supplementary rule-based reason layer, or accepting the scorecard's
   accuracy cost as the price of unambiguous defensibility). A genuine committee-level
   judgment call.

---

## 8. Portfolio, ECL & Stress Testing (Fannie Mae side)

**Discrete-time hazard model** (`portfolio/survival.py`): logistic regression of monthly
default hazard on tenure, static risk factors (FICO, LTV, rate), and **time-varying macro
covariates** (unemployment, an asymmetric `hpi_decline_pct` feature — only price declines
should raise risk). Gives a PD **term structure** rather than a single 12-month PD.

**Out-of-time validation:** trained only on 2005-2010 (crisis) vintages, validated on
2015-2019 (benign) vintages — a genuinely harder test than a random split, since the two
windows sit in materially different macro regimes. Result: **AUC 0.660**, with predicted vs.
actual monthly hazard of 0.00154 vs. 0.00149 — well calibrated across the regime shift,
because the model leans on macro covariates rather than memorizing vintage-specific patterns.

**LGD** (`portfolio/lgd.py`): realized loss severity segmented by region × LTV band, with
credibility-weighted shrinkage (`n/(n+30)`) toward the portfolio mean for thin segments.
Cleanly monotonic: ~24-28% at LTV <60 rising to ~45-48% at LTV 90+.

**IFRS 9 three-stage ECL engine** (`portfolio/ecl_engine.py`): `ECL = PD × LGD × EAD ×
discount factor`. Stage assignment via standard rebuttable-presumption backstops (Stage 1:
current; Stage 2: 30-59 dpd; Stage 3: 60+ dpd).

**Stress scenarios** (`portfolio/stress_scenarios.py`): baseline (trailing macro, held flat),
adverse (+3pp unemployment, -15% HPI, ramped over 8 quarters), severely adverse (+6pp, -30%,
matched to the synthetic panel's own 2008 peak).

| Scenario | Total ECL |
|---|---|
| Baseline | ~$28.7M |
| Adverse | ~$54.4M |
| Severely adverse | ~$104.1M |

Correctly monotonic after fixing two model-development bugs caught during this build (see
Section 9).

---

## 9. Cutoff Optimization

**Economic assumptions** (Home Credit is unsecured consumer lending — the Fannie Mae LGD
table does not transfer): LGD 70% flat (industry benchmark for unsecured recoveries), EAD =
`AMT_CREDIT`, 15% assumed APR, 8% flat capital ratio, 2% opex ratio.

**The unconstrained risk-adjusted-return optimum is a trap:** maximizing a *ratio* with no
volume floor shrinks to 15 loans (cutoff 650, 0.005% approval, RAR 4.68) — technically
optimal, useless as policy.

**Constrained optimum** (max loss rate 3%, min approval rate 50%, max segment concentration
45% of approved volume in any one income tercile): **cutoff 565, approval rate 55.0%,
expected loss rate 2.27%, risk-adjusted return 3.55.**

**Binding-constraint finding:** on real data, both the loss cap (2.27% << 3%) and the
concentration limit (35.6% << 45%) are slack — **the volume floor actually binds.** Its
shadow price is **-$6.8B per 10pp of relaxed floor**: relaxing the volume requirement lets
the optimizer shrink toward a smaller, safer, higher-*ratio* book that earns *less* absolute
profit — exactly why a volume floor belongs in the constraint set, not as a
business-development nicety but as a check against a pure ratio-maximizer shrinking the
balance sheet. (On the earlier synthetic-data run, the concentration limit bound instead,
at a different cutoff — which constraint binds is a property of the data, and checking it
explicitly rather than assuming it's always the loss cap is the actual point of this
analysis.)

---

## 10. Power BI Layer

**Contract fully built (Python/SQL side):**
- All 15 marts in `marts/` (7 dimensions, 6 facts, 2 intermediate feature tables not loaded
  into the model): `dim_date`, `dim_score_band`, `dim_bucket`/`_from`/`_to`, `dim_geography`,
  `dim_scenario`, `mart_application`, `mart_loan_performance`, `mart_ecl`,
  `mart_cutoff_frontier`, `mart_model_performance`, `mart_psi_reference`.
- All DAX measures written and reviewable as `.txt` in `powerbi/measures/` (core metrics,
  vintage curve, roll rate, PSI monitoring, concentration/HHI, cutoff what-if parameter, ECL,
  time intelligence).
- Star-schema design doc (`docs/05-powerbi-data-model.md`): role-playing date dimension
  (active relationship on reporting date, inactive on origination date via
  `USERELATIONSHIP`), roll-rate matrix modelling (`dim_bucket_from`/`dim_bucket_to`),
  incremental refresh design, Import vs. DirectQuery rationale.
- RLS design (`governance/RLS_DESIGN.md`): Regional Risk Manager vs. Portfolio Head roles.

**Build status (Power BI Desktop, in progress):**
- All 13 required tables loaded via the native Parquet connector.
- Power BI's autodetect created 5 relationships on load; 3 were spurious (matched on a
  coincidentally-shared `sort_order` column between unrelated dimension tables) and were
  identified and deleted; 2 were correct and kept (`mart_ecl→dim_scenario`,
  `mart_loan_performance→dim_geography`).
- Remaining work: 5 more relationships (role-playing date dimension × 2, roll-rate
  `bucket_prev`/`bucket_curr` joins × 2, `mart_application→dim_score_band`,
  `mart_cutoff_frontier→dim_scenario`), the disconnected `Cutoff` what-if parameter, all DAX
  measures, 5 report pages, and RLS role testing. TMDL (text-based model editing) is
  available in this Power BI Desktop installation, which should speed up entering
  relationships and measures versus one-at-a-time dialogs.

---

## 11. Governance

- **`MODEL_CARD.md`** — purpose, intended use, out-of-scope uses, training data (real,
  307,511 applications), performance table, limitations, owner/review cadence.
- **`MONITORING_PLAN.md`** — PSI thresholds (<0.10 stable, 0.10-0.25 investigate, >0.25
  rebuild), Gini decay tolerance, calibration drift thresholds, retraining triggers with
  named owners and timelines, explicit cross-reference to the Power BI monitoring page as
  this plan's operational implementation.
- **`ADVERSE_ACTION.md`** — reason-code logic (exact points decomposition for the scorecard),
  worked real example, and the full scorecard-vs-GBM defensibility argument.
- **`VALIDATION_REPORT.md`** — independent-style second-line review: data quality, methodology
  soundness, assumption testing, benchmark comparison, findings/recommendations table. Notably
  documents that the project's own earlier conclusion ("no accuracy gain from the challenger")
  reversed once real data was used — exactly the kind of thing an independent review exists
  to catch.
- **`RLS_DESIGN.md`** — roles, filter logic, testing plan.

---

## 12. Serving Layer

FastAPI service (`app/api.py`): `POST /score` (score, calibrated PD, band, decision, reason
codes), `GET /health`, `GET /model/version`, `GET /model/required-features` (dynamically
reports whatever characteristics the *currently* fitted model needs — added after a real bug,
see Section 13 #11).

**Docker:** built (`credit-risk-api:latest`, 497MB) and verified end-to-end — container run,
all four endpoints hit, `/score` response for a real applicant matched the local pipeline's
output exactly (same score, decision, reason codes).

---

## 13. Testing & Bugs Caught During Development

**23 automated tests** across data integrity, portfolio economics, scorecard correctness,
policy optimization, and the API (`tests/`). Several exist specifically because they caught
real bugs — kept as permanent regression guards:

1. **Miscalibrated hazard model** — `class_weight="balanced"` inflated predicted PD to ~100%
   for nearly every loan. Fixed: removed; PD models need calibration, not balanced accuracy.
2. **Unclipped tenure extrapolation** — projecting `months_on_book` beyond the training
   panel's ceiling blew up the hazard via the (correctly-signed) seasoning coefficient. Fixed:
   clipped at the training maximum.
3. **Wrong-signed HPI feature** — collinearity between unemployment and HPI YoY gave the raw
   HPI coefficient the wrong sign. Fixed: reframed as asymmetric `hpi_decline_pct`.
4. **Undercounted vintage curve** — SQL joined through a fact table where charged-off loans
   have no further rows, silently dropping them from later buckets. Fixed: computed
   cumulative defaults directly against a month-offset grid.
5. **Non-monotonic stress scenarios** — replaying the literal 2008 macro slice for "severely
   adverse" produced a milder HPI trough than "adverse" at that window's boundary. Fixed:
   explicit, guaranteed-ordered magnitudes.
6. **Sign-reversed scorecard coefficients** — L1-selected characteristics flipped sign in the
   multivariate fit despite being correctly signed individually. Fixed: iterative
   sign-consistency pruning (recurred once on real data too, confirming the fix generalizes).
7. **A pandas/optbinning import-order segfault** — native DLL-loading-order conflict on this
   Windows/Python 3.12 setup. Fixed: `optbinning` imported before `pandas` everywhere both
   are touched, enforced via `tests/conftest.py`.
8. **No real out-of-time validation** — an early doc claimed the Fannie Mae hazard model
   demonstrated out-of-time validation; the code just fit on the full panel. Fixed: actually
   implemented (Section 8).
9. **Missing segment concentration constraint** — the cutoff optimizer enforced a loss cap
   and volume floor but not the concentration limit the project also requires. Added.
10. **Hardcoded policy cutoff drifting stale in 3 files** — the optimizer's computed cutoff
    changed twice as data/constraints changed, requiring manual re-chasing in `app/api.py`
    and the marts builder each time. Fixed: persisted to
    `scorecard/artifacts/policy_cutoff.json`; both consumers now read it instead of
    hardcoding a number.
11. **API request schema hardcoded to a stale characteristic list** — swapping to real data
    changed which 14 characteristics the model selected; the API's fixed Pydantic schema was
    missing several, so requests 422'd. Fixed: request model now accepts arbitrary fields,
    validated dynamically against the model's actual current requirement, exposed via
    `GET /model/required-features`.
12. **Adverse-action reason text missing for 8 of 14 real characteristics** — same root cause
    as #11, degraded gracefully to raw column names rather than crashing. Fixed: covered every
    characteristic seen across both the synthetic and real fits.

---

## 14. Assumptions & Limitations

- Home Credit and Fannie Mae are different portfolios, markets, and regulatory regimes —
  used to demonstrate complementary techniques, never merged into one book.
- Fannie Mae and FRED remain synthetic stand-ins; the Home Credit reject pool is synthetic
  regardless of which accepts data is used (no real reject population exists in any version
  of this dataset).
- No true out-of-time validation is possible for the Home Credit scorecard (no
  application-date field in the real dataset) — a structural dataset limitation, mitigated
  by demonstrating true out-of-time validation on the Fannie Mae side instead.
- LGD (70% flat, unsecured), APR (15%), capital ratio (8%), and opex ratio (2%) are flat
  industry-benchmark assumptions, not derived from this portfolio's own realized-loss data
  (none exists) — sensitivity testing recommended before any real deployment decision leans
  on the specific cutoff recommendation.
- Stress scenarios are illustrative, not regulator-prescribed CCAR/DFAST paths.
- **No fair lending or disparate impact testing has been performed.** In a real deployment
  this would be mandatory — named explicitly as an omission, not left implicit.
- The reject-inference validation is checkable against ground truth only because the reject
  pool is synthetic; a real deployment can never do this and must instead monitor realized
  bad rates after deployment.

---

## 15. Implementation Plan — Remaining Work

### Completed
- [x] Data ingestion (real Home Credit, synthetic Fannie Mae/FRED, synthetic reject pool)
- [x] Warehouse build (DuckDB)
- [x] Relational feature engineering + leakage audit
- [x] SQL analytics (vintage, roll rate, concentration, segment performance, macro alignment)
- [x] Scorecard: binning → WOE/IV → correlation/VIF → L1 selection → sign pruning → scaling → calibration
- [x] Reject inference (fuzzy augmentation, hidden-truth validation)
- [x] Challenger model (LightGBM + SHAP) + deploy-or-not analysis
- [x] Adverse action reason codes
- [x] Fannie Mae survival/hazard model + out-of-time validation
- [x] LGD estimation
- [x] IFRS 9 ECL engine + 3 stress scenarios
- [x] Constrained cutoff optimizer (loss cap, volume floor, concentration limit) + shadow prices
- [x] All Power BI marts built
- [x] All DAX measures written (as reviewable text)
- [x] Star-schema, RLS, and incremental-refresh design docs
- [x] Governance suite (model card, monitoring plan, adverse action, validation report, RLS design)
- [x] FastAPI scoring service
- [x] Dockerfile, built and verified end-to-end
- [x] 23 automated tests, all passing

### Remaining (Power BI Desktop build)
- [ ] Finish star-schema relationships in Power BI Desktop:
  - [ ] `mart_loan_performance[reporting_date]` → `dim_date[date]` (active)
  - [ ] `mart_loan_performance[origination_date]` → `dim_date[date]` (inactive — role-playing dimension)
  - [ ] `mart_loan_performance[bucket_prev]` → `dim_bucket_from[bucket_from]`
  - [ ] `mart_loan_performance[bucket_curr]` → `dim_bucket_to[bucket_to]`
  - [ ] `mart_application[score_band]` → `dim_score_band[band]`
  - [ ] `mart_cutoff_frontier[scenario]` → `dim_scenario[scenario]`
  - [ ] `mart_ecl[reporting_date]` → `dim_date[date]`
- [ ] Mark `dim_date` as the official Date Table
- [ ] Create the disconnected `Cutoff` what-if parameter (400-800, step 5, default 565)
- [ ] Enter all DAX measures from `powerbi/measures/*.txt` (organized into matching display folders)
- [ ] Build the 5 report pages: Portfolio Overview, Vintage & Roll Rates, Credit Policy, Stress Testing, Model Monitoring
- [ ] Configure and test RLS roles (Regional Risk Manager, Portfolio Head) via "View as Role"
- [ ] Optional: configure incremental refresh parameters on `mart_loan_performance` (design already specified in `docs/05-powerbi-data-model.md`)
- [ ] Save the `.pbix`, export a PDF, capture screenshots for the README

### Optional / explicitly deferred
- [ ] Real Fannie Mae data (manual portal registration + approval; parser stubs in `ingest/fannie_mae_ingest.py` ready to fill in)
- [ ] Real FRED data (attempted, not completed this session — reader in `ingest/fred_ingest.py` is ready)
- [ ] Git commit (held per explicit request; nothing committed yet)
- [ ] Fair lending / disparate impact testing (deliberately out of scope, named as a limitation per governance best practice)

---

## 16. Repository Layout

```
credit-risk-intelligence/
├── README.md, PROJECT_REPORT.md (this file)
├── docs/                  01-07: data sources, scorecard methodology, reject inference,
│                          ECL/stress testing, Power BI data model, assumptions, results
├── governance/            MODEL_CARD, MONITORING_PLAN, ADVERSE_ACTION, VALIDATION_REPORT, RLS_DESIGN
├── ingest/                real + synthetic data readers (Home Credit, Fannie Mae, FRED, reject pool)
├── warehouse/             DuckDB build script
├── sql/                   analytics/ (5 SQL files), marts/ (2 mart-building Python scripts)
├── features/              bureau/previous-application/payment-behaviour aggregation, leakage audit
├── scorecard/             binning, WOE/IV, logistic fit, scaling, calibration, reject inference, adverse action
├── challenger/            LightGBM + SHAP
├── portfolio/             vintage, roll rates, survival/hazard, LGD, ECL engine, stress scenarios
├── policy/                cutoff optimizer
├── marts/                 the Power BI contract (parquet)
├── powerbi/               measures/ (DAX as .txt), screenshots/ (pending)
├── app/                   FastAPI scoring service
├── tests/                 23 tests + conftest
└── Dockerfile             built and verified
```
