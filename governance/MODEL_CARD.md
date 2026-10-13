# Model Card — Home Credit Application Scorecard v1.0

## Purpose

Produces a calibrated probability of default (PD) and a points-based score at application
time, for an unsecured consumer instalment lending product, to support an approve/decline
decision and downstream portfolio economics (cutoff optimization, expected loss).

## Intended use

- Application-time credit decisioning for new unsecured instalment loan applications from
  the same population the model was developed on.
- Input to the cutoff optimizer (`policy/cutoff_optimizer.py`) and to adverse-action reason
  generation (`ADVERSE_ACTION.md`).

## Out-of-scope uses

- **Not** for the Fannie Mae mortgage portfolio — a different product, market, and
  regulatory regime (see `docs/01-data-sources.md`). That portfolio has its own hazard model
  (`portfolio/survival.py`).
- **Not** validated for any population materially different from the development sample
  (see "Training data" below) — in particular, not validated across time, since the dataset
  has no application-date field to test that.
- **Not** a fair-lending or disparate-impact-tested model. No such testing has been
  performed. See `docs/06-assumptions-and-limitations.md`.

## Training data and period

307,511 real applications, downloaded from Kaggle (`kaggle.com/c/home-credit-default-risk` —
see `docs/01-data-sources.md`; row count and ~8.07% base default rate verified against the
known real dataset). The reject-inference pool remains synthetic (no version of this dataset
includes a real one — see `docs/03-reject-inference.md`). No application-date field exists
in `application_train`, so there is no development "period" in the usual sense — a genuine
limitation of this dataset, confirmed now that real data is in use.

## Methodology

14-characteristic WOE/logistic scorecard with reject inference (fuzzy augmentation, against
a synthetic reject pool) and Platt calibration. Full methodology:
`docs/02-scorecard-methodology.md`, `docs/03-reject-inference.md`.

## Performance

| Split | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Train | 0.748 | 0.496 | 0.370 | ~0.0685 |
| Test | 0.749 | 0.497 | 0.374 | ~0.0685 |

A LightGBM challenger beats this scorecard out-of-sample (test Gini 0.538 vs. 0.497) — see
`docs/07-results.md` for the full accuracy-vs-adverse-action-defensibility discussion. This
scorecard remains the deployed model in this build; the challenger is not currently deployed.

Segment-level performance (default rate by FICO-equivalent band, region) is in
`sql/analytics/segment_performance.sql`'s output — run against the Home Credit population by
substituting the equivalent join (not yet wired up as a standalone report; see "Limitations").

## Limitations

See `docs/06-assumptions-and-limitations.md` in full. Most material to this model
specifically: no out-of-time validation is possible with this dataset; LGD (70% flat) is an
industry-benchmark assumption, not derived from realized loss data (Home Credit has none);
no fair-lending testing has been performed.

## Owner and review

Owner: Credit Risk Modelling (placeholder — assign a named owner before any production use).
Review date: 12 months from deployment, or immediately upon any PSI "Rebuild" trigger per
`MONITORING_PLAN.md`.
