# Power BI data model

This documents the model to build in Power BI Desktop from the marts in `marts/` — see
`docs/00-kaggle-setup.md`-style handoff: Python/SQL produced every mart below; Power BI's job
is the model, DAX, and report pages, not more computation. DAX text (organized by the same
folders as section 8.3 of the original spec) lives in `powerbi/measures/*.txt` — a recruiter
can read those without opening the `.pbix`.

## Marts produced (contract with Power BI)

| Mart / dim | Grain | Built by |
|---|---|---|
| `mart_application` | one row per Home Credit application | `sql/marts/build_marts_home_credit.py` |
| `mart_loan_performance` | Fannie Mae loan x reporting month | `sql/marts/build_marts.py` |
| `mart_ecl` | Fannie Mae loan x scenario x reporting month | `portfolio/ecl_engine.py` |
| `mart_cutoff_frontier` | cutoff x scenario x segment (Home Credit) | `policy/cutoff_optimizer.py` |
| `mart_model_performance` | reporting month x segment (Home Credit) | `sql/marts/build_marts_home_credit.py` |
| `mart_psi_reference` | characteristic x bin (Home Credit) | `sql/marts/build_marts_home_credit.py` |
| `dim_date`, `dim_score_band`, `dim_bucket`, `dim_bucket_from`, `dim_bucket_to`, `dim_geography`, `dim_scenario` | — | both build scripts |

Run order: `warehouse/build_warehouse.py` → `features/build_features.py` →
`scorecard/*.py` (binning → woe_iv → fit_logistic → scaling → calibration →
reject_inference) → `sql/marts/build_marts.py` and `sql/marts/build_marts_home_credit.py` →
`portfolio/*.py` (survival → lgd → ecl_engine) → `policy/cutoff_optimizer.py`.

## Star schema

All facts (`mart_loan_performance`, `mart_ecl`, `mart_application`, `mart_cutoff_frontier`,
`mart_model_performance`) connect only to dimensions — **no fact-to-fact relationships**.
Every relationship is single-direction (dim → fact); bidirectional filtering is off
everywhere. `dim_date` is marked as the model's official date table.

### Role-playing date dimension

`mart_loan_performance` has both `origination_date` and `reporting_date`. Build:
- **Active** relationship: `mart_loan_performance[reporting_date]` → `dim_date[date]`
  (everything time-based — Default Rate trend, Exposure Rolling 12M — filters through this).
- **Inactive** relationship: `mart_loan_performance[origination_date]` → `dim_date[date]`,
  activated only inside `Originations by Vintage` via `USERELATIONSHIP`.

This is the standard pattern for vintage analysis: a loan's *origination* cohort and its
*current reporting* position are different time axes on the same fact row, and only one can
be the model's default active filter path.

### Roll-rate matrix modelling

`dim_bucket_from` and `dim_bucket_to` are two copies of the same 6-row bucket list
(`Current, 30, 60, 90, 120+, Charge-off`, sort order 0-5), joined to
`mart_loan_performance[bucket_prev]` and `[bucket_curr]` respectively. Put
`dim_bucket_from[bucket_from]` on matrix rows and `dim_bucket_to[bucket_to]` on columns — the
transition pairs are already landed as columns on the fact table (see
`sql/analytics/roll_rate_matrix.sql`), so the matrix visual is a plain `COUNTROWS`, no DAX
window-function gymnastics.

### Scenario join (a disclosed simplification)

`mart_ecl[scenario]` and `mart_cutoff_frontier[scenario]` join to `dim_scenario[scenario]` on
the scenario **name** (`"baseline"` / `"adverse"` / `"severely_adverse"`), not an integer
surrogate key. Section 8.5's performance guidance calls for integer surrogate keys on
relationships; a text join was accepted here because there are only 3 scenario values and the
model is small enough that it doesn't matter in practice. In a production model with a larger
scenario or bucket dimension, add a surrogate key column at the mart-build stage rather than
in DAX.

## Import vs. DirectQuery

**Import**, for every fact table. Rationale: the largest fact
(`mart_loan_performance`, ~1.1M rows) is small enough that Import's full DAX support and
in-memory performance outweigh DirectQuery's freshness advantage — nothing here needs
sub-hour data freshness, since the marts are batch-rebuilt by the Python pipeline, not
streamed. DirectQuery would make sense if this connected straight to a transactional
database still being written to; it doesn't.

## Incremental refresh (design, not yet deployed to a Premium workspace)

On `mart_loan_performance`: define `RangeStart`/`RangeEnd` parameters, filter the Power
Query source on `reporting_date` between them, then configure an incremental refresh policy
of *archive 5 years, refresh last 12 months*. The partitioning logic is fully specified here
even though this build never publishes to a Premium workspace to actually exercise it —
being able to explain the partition boundaries is the point section 8.5 asks for.

## Performance notes

- Remove unused columns at the Power Query stage (e.g. `mart_loan_performance[state]` is
  needed for `dim_geography`'s relationship but the raw `region` column on the fact table
  can be dropped once RLS filters through `dim_geography` instead).
- Disable auto date/time globally (File > Options > Data Load) — it silently creates a
  hidden date table per date column otherwise, defeating the single `dim_date` design above.
- Integer surrogate keys everywhere except the disclosed scenario-name join above.
