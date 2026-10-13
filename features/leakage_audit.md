# Leakage audit

Hard rule (per project spec): every feature must be constructible at the moment of
application. This document justifies that for every feature in `feature_table`.

## Why the source tables are safe by construction

Home Credit's supplementary tables are historical by design relative to the current
application (`SK_ID_CURR`):

- `bureau` / `bureau_balance`: external credit bureau records. `DAYS_CREDIT_UPDATE` and all
  `DAYS_*` columns are negative (days before the current application). `bureau_balance.MONTHS_BALANCE`
  is always `<= 0` for the same reason.
- `previous_application`: the same applicant's *prior* credit applications. `DAYS_DECISION < 0`.
- `installments_payments`: scheduled/actual payments on *previous* credits. `DAYS_INSTALMENT`
  and `DAYS_ENTRY_PAYMENT` are both `<= 0`.
- `credit_card_balance`: monthly snapshots of *previous* credit card products. `MONTHS_BALANCE <= 0`.

No feature below joins on anything from `application_train` other than the join key and
`AMT_INCOME_TOTAL` (needed for a ratio), so none can leak the current application's own
outcome or credit terms.

## Bureau features (`features/bureau_aggregates.py`)

| Feature | Source | Why pre-decision |
|---|---|---|
| `bureau_n_active_credit`, `bureau_n_closed_credit` | `bureau.CREDIT_ACTIVE` | Static count of prior external credit lines |
| `bureau_avg_utilization` | `bureau.AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM` | Snapshot utilization of active prior lines |
| `bureau_utilization_trend_6m_minus_24m` | Same, filtered by `DAYS_CREDIT_UPDATE` window | Compares recency-filtered utilization; both windows end at application date |
| `bureau_max_dpd_ever` | `bureau_balance.STATUS` mapped to DPD bucket | Worst delinquency ever recorded on external bureau lines, all dated `MONTHS_BALANCE <= 0` |
| `bureau_pct_months_delinquent` | `bureau_balance.STATUS` | Share of historical bureau-line months in any delinquent status |
| `bureau_debt_to_income` | `bureau.AMT_CREDIT_SUM_DEBT` / `application.AMT_INCOME_TOTAL` | Numerator is prior debt; denominator (stated income) is submitted *as part of* the current application, not an outcome of it |

## Previous-application features (`features/previous_app_aggregates.py`)

| Feature | Why pre-decision |
|---|---|
| `prev_n_applications`, `prev_approval_rate`, `prev_refusal_rate` | Counts/rates over the applicant's *earlier* applications (`DAYS_DECISION < 0`) |
| `prev_downsell_ratio` | Ratio of granted to requested amount on *earlier, already-decided* applications |

## Payment-behaviour features (`features/payment_behaviour.py`)

| Feature | Why pre-decision |
|---|---|
| `instalment_avg_payment_ratio`, `instalment_avg_days_late`, `instalment_pct_late`, `instalment_pct_early` | Built from `installments_payments`, which only contains dates `<= 0` relative to the current application |
| `cc_avg_utilization`, `cc_utilization_trend`, `cc_min_payment_only_freq` | Built from `credit_card_balance.MONTHS_BALANCE <= 0` |

## Excluded / flagged during IV screening (documented in `scorecard/woe_iv.py` output, not here)

`EXT_SOURCE_1/2/3` are included as raw application-level characteristics. They are external
scoring-bureau outputs already known at application time, but because they are themselves
compact summaries of applicant risk, IV screening should be watched for `IV > 0.5` on these —
per project spec, a very high IV is a leakage smell even when the raw availability argument
holds, so they are flagged for scrutiny rather than excluded outright.
