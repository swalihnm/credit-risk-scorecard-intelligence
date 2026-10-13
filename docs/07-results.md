# Results

All numbers below are from the real Home Credit dataset (307,511 applications, downloaded
from Kaggle — see [`01-data-sources.md`](01-data-sources.md)), except the Fannie Mae ECL
section, which remains the disclosed synthetic stand-in.

## Scorecard

14-characteristic logistic scorecard (see [`02-scorecard-methodology.md`](02-scorecard-methodology.md)):
AUC 0.748 train / 0.749 test, Gini **0.496 / 0.497**, KS 0.370 / 0.374. Well calibrated
(Brier ≈ 0.0685, ECE ≈ 0.001-0.003, HL fails to reject at 5%). Consistent with published
benchmarks for a plain WOE/logistic scorecard on this dataset.

## Reject inference

See [`03-reject-inference.md`](03-reject-inference.md) in full. Headline number: fuzzy
augmentation (using the still-synthetic reject pool — no version of this dataset includes a
real one) shifted the scorecard's base points by **+0.5** (tightening) and cut approval
volume from 79.6% to 78.7% under a fixed 12%-PD policy — smaller than the +3.1 point shift
seen when the accept-only model was itself fit on synthetic data, consistent with the real
model already being better specified (higher AUC ranking the reject pool) and needing less
correction.

## Challenger model (LightGBM + SHAP) — conclusion changed on real data

| | Train Gini | Test Gini |
|---|---|---|
| Logistic scorecard (14 characteristics) | 0.571 | 0.497 |
| LightGBM (300 trees, depth 4, regularized) | 0.571 | **0.538** |

*(Table reads train/test for each model on its own fit; the head-to-head that matters is test
Gini: scorecard 0.497 vs. LightGBM 0.538.)*

**The challenger now beats the scorecard out-of-sample** — a genuine ~0.04 Gini improvement,
the opposite of what the earlier development run on the synthetic stand-in showed (there, the
challenger overfit on the smaller 30K-row synthetic sample and lost to the scorecard). More
data changes the answer to "should we deploy the challenger" from a clean no to a real
tradeoff. SHAP confirms the same top drivers as the scorecard (`EXT_SOURCE_2/3/1`,
`bureau_avg_utilization`, `prev_downsell_ratio`, `DAYS_EMPLOYED`), so the two models agree on
*what* matters; the GBM extracts more discrimination from it, and 307K rows is enough data
for that extra capacity to generalize rather than overfit.

**Should the bank deploy it? Now a genuine tradeoff, not a clean answer:**
1. **Accuracy gain is real.** ~0.04 Gini is not noise — it would show up in a real book's
   loss rate at a fixed approval volume.
2. **Adverse action cost is unchanged and still weighs against it.** A GBM's SHAP-based
   reason codes are contestable in a way a points-based scorecard's aren't: a scorecard's
   adverse-action reasons are literally "your points contribution from characteristic X was
   the lowest," traceable to a single monotonic bin. A SHAP value is a valid *local*
   attribution under a specific game-theoretic definition, but it is a model-specific
   artifact, not a stable causal statement about the applicant's characteristics — two
   similar applicants can get noticeably different SHAP attributions for the same
   characteristic depending on interactions elsewhere in the model. Whether that satisfies a
   regulator's adverse-action reason-code requirement is a live legal question, not a solved
   one. See [`governance/ADVERSE_ACTION.md`](../governance/ADVERSE_ACTION.md).
3. **What this means in practice:** the accuracy gain is large enough that a risk committee
   should not dismiss the GBM out of hand (as they reasonably could on the earlier synthetic
   result), but should not approve it either without first resolving #2 — e.g. via a
   supplementary rule-based adverse-action layer on top of the GBM, or accepting the
   scorecard's ~0.04 Gini cost as the price of unambiguous adverse-action defensibility. This
   is now a genuine committee-level judgment call, which is a more realistic and more
   interesting place for this discussion to land than a clean "no."

## Cutoff optimization (policy/cutoff_optimizer.py)

Economic assumptions: LGD 70% flat (unsecured consumer lending — the Fannie Mae LGD table
does not transfer), EAD = AMT_CREDIT, 15% APR, 8% flat capital ratio, 2% opex ratio. Full
sweep (400-800 points, base/stress scenario x 4 segments) in `marts/mart_cutoff_frontier.parquet`.

**The unconstrained risk-adjusted-return optimum is a trap**: maximizing a *ratio* with no
volume floor trivially shrinks to 15 loans (cutoff 650, approval rate 0.005%, RAR 4.68) —
technically optimal, useless as a policy.

**Constrained optimum** (max loss rate 3%, min approval rate 50%, max segment concentration
45% of approved volume in any one income tercile): cutoff **565**, approval rate 55.0%,
expected loss rate 2.27%, risk-adjusted return 3.55.

**Binding-constraint finding:** on real data, both the loss cap (2.27% << 3%) and the
concentration limit (35.6% << 45%) are *slack* — **the volume floor is what actually binds**
(55.0% approval, just above the 50% floor). Its shadow price is **-$6.8B per 10pp of relaxed
floor**: the familiar pattern — relaxing the volume requirement lets the
risk-adjusted-return-maximizing optimizer shrink toward a smaller, safer, higher-*ratio* book
that earns *less* absolute profit, which is exactly why the volume floor belongs in the
constraint set rather than being treated as a business-development nicety.

*(This finding is data-dependent, which is itself worth stating plainly: on the earlier
30K-row synthetic stand-in, the concentration limit bound instead, because that generator's
income/risk relationship happened to be strong enough to make one income tercile dominate
the unconstrained optimum. On the real 307K-row dataset, income concentration among high
scorers is milder, so the concentration constraint has slack and the volume floor binds
again — the same shape of result seen in an even earlier iteration of this optimizer, before
the concentration constraint existed at all, at a 570 cutoff with a -$271M shadow price.
Which constraint binds is a property of the data, not of the optimizer's structure, and
checking it explicitly — rather than assuming it's always the loss cap, or always the same
constraint run to run — is the actual point of this section.)*

## ECL and stress testing (Fannie Mae portfolio — still synthetic, see docs/01)

See [`04-ecl-and-stress-testing.md`](04-ecl-and-stress-testing.md) in full. Total portfolio
ECL: baseline ~$28.7M, adverse ~$54.4M, severely adverse ~$104.1M — correctly monotonic after
fixing two model-development bugs (miscalibrated hazard model, unclipped tenure extrapolation)
caught during development. Out-of-time validated: a hazard model trained only on 2005-2010
vintages still discriminates (AUC 0.660) and stays well-calibrated on 2015-2019 vintages.
