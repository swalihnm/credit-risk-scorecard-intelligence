# Monitoring Plan — Home Credit Application Scorecard v1.0

## PSI thresholds

Applied to the score distribution and to every one of the 12 final characteristics,
against the development-sample (train) expected distribution in `mart_psi_reference`:

| PSI | Status | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 - 0.25 | Investigate | Root-cause the shift (population mix change vs. genuine risk drift); no automatic action |
| > 0.25 | Rebuild | Trigger scorecard redevelopment |

**Operational implementation:** the Model Monitoring Power BI page (`mart_psi_reference` +
`dim_score_band`, `PSI` / `PSI Status` measures in `powerbi/measures/monitoring_psi.txt`) *is*
this plan's day-to-day execution — this document is the policy, that page is the control.

## Gini decay tolerance

Alert if rolling 3-month average Gini drops more than 0.05 absolute below the development
Gini (0.542 on the held-out test set) — e.g. below ~0.49. Investigate if it drops below 0.45;
treat as a rebuild trigger below 0.40.

## Calibration drift

Monthly Brier score and ECE, tracked in `mart_model_performance`. Alert if ECE exceeds 0.02
(roughly 3x the ~0.0034-0.0057 observed at development) for two consecutive months.

## Actual vs. expected default rate by score band

Tracked on the Model Monitoring page (score distribution, development vs. current). A
systematic gap in one direction across multiple bands (not just noise in one band) is a
calibration-drift signal even before PSI or Gini move.

## Retraining triggers, owners, timelines

| Trigger | Owner | Timeline |
|---|---|---|
| PSI (score or any characteristic) > 0.25 | Credit Risk Modelling | Redevelopment kicked off within 2 weeks |
| Gini < 0.40 (rolling 3-month) | Credit Risk Modelling | Redevelopment kicked off within 1 week |
| ECE > 0.02 for 2 consecutive months | Credit Risk Modelling | Recalibration (Platt refit) within 2 weeks; full rebuild if recalibration doesn't resolve it |
| Any fair-lending finding (once such testing exists — see limitations) | Model Risk / Compliance | Immediate escalation, decisioning paused on affected segment |

## Override monitoring

Not implemented in this build (there is no production decisioning system generating
overrides to monitor). Documented here as a placeholder: a real deployment would track the
rate and direction of manual overrides to the model's decision by underwriter/segment, since
a high override rate is itself a signal the model isn't trusted or isn't fit for a segment.
