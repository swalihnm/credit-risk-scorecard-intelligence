# Adverse Action Reason Codes

## Logic

For a declined applicant, `scorecard/adverse_action.py` computes each of the 14 final
characteristics' points contribution (`-Factor * coefficient * WOE(applicant's bin)`) and
surfaces the lowest (most negative) contributors as the reason codes. This is not an
approximation of "why the model declined this applicant" — for a linear, points-based
scorecard, the points contribution literally **is** the decomposition of the score into
per-characteristic effects. Score = base points + sum of characteristic points, by
construction (`scorecard/scaling.py`), so ranking characteristics by their points
contribution is an exact, not approximate, attribution.

## Example output (real applicant, real data)

```
Applicant 100002:
  [-22.1 pts] EXT_SOURCE_3       External credit bureau score 3 below approval threshold
  [-13.3 pts] EXT_SOURCE_1       External credit bureau score 1 below approval threshold
  [-9.0 pts]  EXT_SOURCE_2       External credit bureau score 2 below approval threshold
  [-4.3 pts]  AMT_GOODS_PRICE    Requested credit amount relative to the price of goods financed
```

**A gap caught while wiring this up on real data:** the reason-code text lookup
(`REASON_CODE_TEXT` in `scorecard/adverse_action.py`) was hardcoded against the 12
characteristics the earlier synthetic-data scorecard had selected. Real data selected a
different 14-characteristic set, and 8 of them (including `AMT_GOODS_PRICE` above) had no
entry — the code has a `.get(char, char)` fallback so it didn't crash, but it silently
degraded to showing the raw column name instead of a human-readable reason, which is exactly
the kind of thing that's easy to miss in review. Fixed by adding text for every characteristic
that has ever been selected across both the synthetic and real runs, not just the current
set — so the same class of gap can't recur next time the selected set changes again.

## Why a points-based scorecard makes this straightforward, and a GBM makes it contestable

A logistic-regression-on-WOE scorecard is a **sum of independent, monotonic, per-characteristic
effects**. There is no interaction term: characteristic X's points contribution is the same
regardless of what characteristic Y's value is. That's exactly what a defensible adverse-action
reason needs — a claim that holds for that applicant on its own terms, not one contingent on
an opaque interaction with other fields.

A GBM (`challenger/lightgbm_challenger.py`) has neither property. SHAP values are a valid
*local* attribution under a specific, principled game-theoretic definition (Shapley values
from cooperative game theory) — but:
- They are **interaction-dependent**: two applicants with an identical value on characteristic
  X can get materially different SHAP attributions for X, because SHAP distributes credit for
  interaction effects between the interacting features, and *which* features an applicant
  happens to interact with varies applicant to applicant.
- They are **model-specific**, not a causal statement about the applicant's characteristics —
  retraining the same GBM on a resampled dataset can shift SHAP rankings for borderline cases
  even when the model's overall accuracy is unchanged.

Whether SHAP-based reasons satisfy a regulator's adverse-action reason-code requirement
(e.g., the substance of what Regulation B / ECOA-style adverse action notices require) is a
live, unsettled question in the industry, not a solved one. A points-based scorecard's reasons
were never in doubt to begin with — this is the concrete cost side of the
"should we deploy the challenger" discussion in `docs/07-results.md`. On real data the
challenger *does* win on accuracy (test Gini 0.538 vs. 0.497), which makes this cost side of
the argument matter more, not less: it's now the deciding factor in a genuine tradeoff,
rather than a second independent reason stacked on top of "no accuracy gain anyway."
