"""Generate a synthetic Home Credit Default Risk dataset matching the real Kaggle schema.

Explicitly labelled STAND-IN data (see docs/01-data-sources.md) — real Kaggle credentials
were not available when this build ran. Real ingestion (ingest/kaggle_download.py) is
already written and produces the identical file layout, so swapping in real data later is a
matter of running that script and rerunning warehouse/build_warehouse.py — nothing in
features/, scorecard/, challenger/, or the marts changes.

A single per-applicant latent risk factor drives TARGET *and* the realized behavior in every
supplementary table (bureau delinquency, previous-application refusal, installment lateness,
credit-card utilization), so the engineered features in features/*.py carry genuine signal —
this is what makes WOE/IV screening, monotonic binning, and calibration meaningful to run
against it, not just structurally present.

Scale: 30,000 applicants (vs. the real dataset's 307,511) — enough to fit a scorecard and
challenger convincingly, kept smaller so the full pipeline runs quickly end-to-end.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT_DIR = pathlib.Path("sources/home_credit")

N_APPLICANTS = 30_000
TARGET_DEFAULT_RATE = 0.081  # matches the real Home Credit dataset's ~8.07% base rate


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _repeat_index(counts: np.ndarray) -> np.ndarray:
    """Row index into the parent array for each exploded child row."""
    return np.repeat(np.arange(len(counts)), counts)


def build_applicants(
    rng: np.random.Generator,
    n: int = N_APPLICANTS,
    id_start: int = 100_000,
    risk_shift: float = 0.0,
    intercept_override: float | None = None,
) -> pd.DataFrame:
    """risk_shift: additive shift to the standardized latent risk factor for the whole
    population (used to generate a riskier reject pool in build_reject_pool below).
    intercept_override: reuse a previously-calibrated intercept rather than recalibrating --
    so a shifted-risk population's TARGET reflects the SAME underlying risk-to-outcome
    relationship as the accepted population, just applied to a riskier mix of applicants,
    rather than being independently re-tuned back to the same overall rate.
    """
    sk_id_curr = np.arange(id_start, id_start + n)

    gender = rng.choice(["M", "F"], size=n, p=[0.34, 0.66])
    age_years = np.clip(rng.normal(43, 11.5, n), 21, 69)
    days_birth = (-age_years * 365 - rng.integers(0, 365, n)).astype(int)

    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
        size=n, p=[0.51, 0.23, 0.18, 0.07, 0.01],
    )
    education_type = rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher",
         "Lower secondary", "Academic degree"],
        size=n, p=[0.71, 0.24, 0.03, 0.015, 0.005],
    )
    family_status = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n, p=[0.64, 0.15, 0.10, 0.07, 0.04],
    )
    occupation_type = rng.choice(
        ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff",
         "Accountants", "Medicine staff", "Security staff", "Cooking staff", "Cleaning staff"],
        size=n,
    )
    occupation_type = occupation_type.astype(object)
    occupation_type[income_type == "Pensioner"] = None
    occupation_type[income_type == "Unemployed"] = None

    employed_years = np.clip(rng.normal(7, 6.5, n), 0, 45)
    days_employed = (-employed_years * 365).astype(int)
    days_employed[income_type == "Pensioner"] = 365_243  # real dataset's own sentinel quirk
    days_employed[income_type == "Unemployed"] = 365_243

    income_type_income_mult = pd.Series(income_type).map(
        {"Working": 1.0, "Commercial associate": 1.35, "Pensioner": 0.55,
         "State servant": 1.05, "Unemployed": 0.35}
    ).values
    income = np.clip(rng.lognormal(np.log(165_000), 0.5, n) * income_type_income_mult, 30_000, 3_000_000)

    cnt_fam_members = np.clip(1 + rng.poisson(1.1, n), 1, 8)

    credit_multiple = np.clip(rng.lognormal(np.log(4.2), 0.4, n), 1.0, 12.0)
    amt_credit = np.round(income * credit_multiple, -2)
    term_months = np.clip(rng.normal(18, 6, n), 6, 36)
    amt_annuity = np.round(amt_credit / term_months, 2)
    amt_goods_price = np.round(amt_credit * rng.uniform(0.85, 1.0, n), -2)

    days_registration = -rng.integers(30, 8000, n)
    days_id_publish = -rng.integers(30, 5000, n)

    education_risk = pd.Series(education_type).map(
        {"Academic degree": -0.6, "Higher education": -0.35, "Incomplete higher": 0.05,
         "Secondary / secondary special": 0.15, "Lower secondary": 0.55}
    ).values
    income_type_risk = pd.Series(income_type).map(
        {"State servant": -0.3, "Pensioner": -0.2, "Commercial associate": -0.1,
         "Working": 0.15, "Unemployed": 1.1}
    ).values
    family_risk = pd.Series(family_status).map(
        {"Married": -0.1, "Civil marriage": 0.05, "Single / not married": 0.15,
         "Separated": 0.2, "Widow": 0.0}
    ).values

    noise = rng.normal(0, 0.9, n)
    risk_z = (
        -0.45 * _zscore(np.log(income))
        - 0.35 * _zscore(np.clip(employed_years, 0, 40))
        + 0.30 * _zscore(-age_years)
        + education_risk
        + income_type_risk
        + family_risk
        + noise
    )
    risk_z = _zscore(risk_z) + risk_shift  # renormalize the composite, then shift for reject pool

    # behavior_z drives supplementary-table realizations: correlated with, not identical to,
    # the application-level latent risk (a real applicant's bureau behavior isn't a perfect
    # mirror of their application-level risk factors either).
    behavior_z = _zscore(0.75 * risk_z + rng.normal(0, 0.7, n))

    ext_source_1 = np.clip(_sigmoid(-0.9 * risk_z + rng.normal(0, 0.6, n)), 0.001, 0.999)
    ext_source_2 = np.clip(_sigmoid(-1.1 * risk_z + rng.normal(0, 0.45, n)), 0.001, 0.999)
    ext_source_3 = np.clip(_sigmoid(-0.8 * risk_z + rng.normal(0, 0.65, n)), 0.001, 0.999)
    ext_source_1[rng.random(n) < 0.40] = np.nan
    ext_source_3[rng.random(n) < 0.20] = np.nan

    if intercept_override is None:
        intercept = _calibrate_intercept(risk_z, scale=1.15, target_rate=TARGET_DEFAULT_RATE)
    else:
        intercept = intercept_override
    true_default_prob = _sigmoid(1.15 * risk_z + intercept)
    target = (rng.random(n) < true_default_prob).astype(int)

    applicants = pd.DataFrame(
        {
            "SK_ID_CURR": sk_id_curr,
            "TARGET": target,
            "CODE_GENDER": gender,
            "AMT_INCOME_TOTAL": income.round(2),
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods_price,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": days_registration,
            "DAYS_ID_PUBLISH": days_id_publish,
            "CNT_FAM_MEMBERS": cnt_fam_members,
            "EXT_SOURCE_1": ext_source_1.round(4),
            "EXT_SOURCE_2": ext_source_2.round(4),
            "EXT_SOURCE_3": ext_source_3.round(4),
            "NAME_EDUCATION_TYPE": education_type,
            "NAME_FAMILY_STATUS": family_status,
            "NAME_INCOME_TYPE": income_type,
            "OCCUPATION_TYPE": occupation_type,
        }
    )
    # kept out-of-frame for downstream generators / reject-inference validation,
    # never written to the applicant CSV itself
    applicants.attrs["behavior_z"] = behavior_z
    applicants.attrs["risk_z"] = risk_z
    applicants.attrs["intercept"] = intercept
    applicants.attrs["true_default_prob"] = true_default_prob
    return applicants


def build_reject_pool(
    accepted_applicants: pd.DataFrame, rng: np.random.Generator, n: int = 8_000, id_start: int = 900_000
) -> pd.DataFrame:
    """A synthetic 'through-the-door' population that an existing (unknown-to-us) credit
    policy declined -- needed because Home Credit's application_train (real or synthetic)
    contains only funded loans, by construction. Real reject inference needs exactly this
    kind of population: applicant-level features, no observed outcome.

    Generated with the SAME risk model as the accepted population but shifted riskier
    (risk_shift=+0.9, reusing the accepted pool's calibrated intercept) -- representing that
    an existing credit policy already screens out the worse tail, so among applicants it
    actually declines, risk is shifted right relative to the funded population.

    TARGET is generated here too, but only as a HIDDEN ground truth used later purely to
    validate the reject-inference technique itself (scorecard/reject_inference.py) -- something
    only possible because this is simulated data. It is written to a separate file
    (reject_hidden_truth.csv) and must never be used as a model input.
    """
    rejects = build_applicants(
        rng, n=n, id_start=id_start, risk_shift=0.9, intercept_override=accepted_applicants.attrs["intercept"]
    )
    return rejects


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-9)


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _calibrate_intercept(z: np.ndarray, scale: float, target_rate: float, tol: float = 1e-5) -> float:
    """Bisection search for the intercept b such that mean(sigmoid(scale*z + b)) == target_rate.

    A naive intercept = logit(target_rate) undershoots here: sigmoid is convex in the region
    around a rare-event intercept, so averaging it over a spread-out z pushes the realized
    mean rate above sigmoid(intercept) (Jensen's inequality) — this search corrects for that
    directly against the actual distribution of z instead of assuming z is degenerate.
    """
    lo, hi = -15.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        rate = _sigmoid(scale * z + mid).mean()
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_bureau(applicants: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = applicants.attrs["behavior_z"]
    sk_id_curr = applicants["SK_ID_CURR"].values
    n_lines = rng.poisson(np.clip(2.5 + 0.4 * z, 0.2, None)).astype(int)
    n_lines = np.clip(n_lines, 0, 12)

    idx = _repeat_index(n_lines)
    n_rows = len(idx)
    sk_id_bureau = np.arange(1, n_rows + 1) + 500_000
    z_rep = z[idx]

    p_active = np.clip(_sigmoid(0.5 * z_rep), 0.1, 0.9)
    credit_active = np.where(rng.random(n_rows) < p_active, "Active", "Closed")

    amt_credit_sum = np.round(rng.lognormal(np.log(150_000), 0.6, n_rows), -2)
    utilization = np.clip(_sigmoid(0.8 * z_rep + rng.normal(0, 0.5, n_rows)), 0, 1) * 0.9
    amt_credit_sum_debt = np.where(
        credit_active == "Active",
        amt_credit_sum * utilization,
        amt_credit_sum * 0.02 * rng.random(n_rows),
    ).round(2)
    days_credit_update = -rng.integers(1, 900, n_rows)

    bureau = pd.DataFrame(
        {
            "SK_ID_BUREAU": sk_id_bureau,
            "SK_ID_CURR": sk_id_curr[idx],
            "CREDIT_ACTIVE": credit_active,
            "AMT_CREDIT_SUM": amt_credit_sum,
            "AMT_CREDIT_SUM_DEBT": amt_credit_sum_debt,
            "DAYS_CREDIT_UPDATE": days_credit_update,
        }
    )

    # bureau_balance: up to 24 months of history per bureau line
    n_months = rng.integers(3, 25, n_rows)
    bb_idx = _repeat_index(n_months)
    bb_n = len(bb_idx)
    months_balance = np.concatenate([np.arange(0, -m, -1) for m in n_months])
    bb_z = z_rep[bb_idx]
    bb_active = (credit_active[bb_idx] == "Active")

    p_delinquent = np.clip(_sigmoid(1.1 * bb_z - 1.6), 0.01, 0.5)
    is_delinquent = rng.random(bb_n) < p_delinquent
    depth = rng.choice([1, 2, 3, 4, 5], size=bb_n, p=[0.5, 0.25, 0.13, 0.07, 0.05])
    status = np.where(~bb_active, "C", np.where(is_delinquent, depth.astype(str), "0"))

    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": sk_id_bureau[bb_idx],
            "MONTHS_BALANCE": months_balance,
            "STATUS": status,
        }
    )
    return bureau, bureau_balance


def build_previous_application(applicants: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    z = applicants.attrs["behavior_z"]
    sk_id_curr = applicants["SK_ID_CURR"].values
    income = applicants["AMT_INCOME_TOTAL"].values

    n_prev = np.clip(rng.poisson(np.clip(1.8 + 0.2 * z, 0.1, None)), 0, 8).astype(int)
    idx = _repeat_index(n_prev)
    n_rows = len(idx)
    sk_id_prev = np.arange(1, n_rows + 1) + 700_000
    z_rep = z[idx]

    p_refused = np.clip(_sigmoid(1.0 * z_rep - 1.2), 0.03, 0.7)
    p_cancelled = 0.08
    status_draw = rng.random(n_rows)
    contract_status = np.where(
        status_draw < p_refused, "Refused",
        np.where(status_draw < p_refused + p_cancelled, "Cancelled", "Approved"),
    )

    amt_application = np.round(income[idx] * rng.lognormal(np.log(1.1), 0.5, n_rows), -2)
    downsell = rng.uniform(0.75, 1.0, n_rows)
    amt_credit = np.where(contract_status == "Approved", amt_application * downsell, 0.0).round(2)
    days_decision = -rng.integers(30, 1800, n_rows)

    return pd.DataFrame(
        {
            "SK_ID_PREV": sk_id_prev,
            "SK_ID_CURR": sk_id_curr[idx],
            "NAME_CONTRACT_STATUS": contract_status,
            "AMT_APPLICATION": amt_application,
            "AMT_CREDIT": amt_credit,
            "DAYS_DECISION": days_decision,
        }
    )


def build_installments_and_pos(
    previous_application: pd.DataFrame, applicants: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    approved = previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Approved"].reset_index(drop=True)
    z_by_curr = pd.Series(applicants.attrs["behavior_z"], index=applicants["SK_ID_CURR"].values)
    z_rep = approved["SK_ID_CURR"].map(z_by_curr).values

    n_inst = rng.integers(6, 37, len(approved))
    idx = _repeat_index(n_inst)
    n_rows = len(idx)
    z_row = z_rep[idx]

    days_decision_rep = approved["DAYS_DECISION"].values[idx]
    offsets = np.concatenate([np.arange(m) for m in n_inst])  # 0..n_inst-1 per group, ~30 days apart
    days_instalment = days_decision_rep + offsets * 30

    lateness = rng.normal(2.5 * np.clip(z_row, -2, 3), 4, n_rows)
    days_entry_payment = np.minimum(days_instalment + lateness, -1)

    amt_instalment = np.round(rng.lognormal(np.log(14_000), 0.5, n_rows), 2)
    payment_ratio = np.clip(rng.normal(1 - 0.04 * np.clip(z_row, 0, None), 0.06, n_rows), 0.4, 1.05)
    amt_payment = np.round(amt_instalment * payment_ratio, 2)

    installments = pd.DataFrame(
        {
            "SK_ID_PREV": approved["SK_ID_PREV"].values[idx],
            "SK_ID_CURR": approved["SK_ID_CURR"].values[idx],
            "DAYS_INSTALMENT": days_instalment.astype(int),
            "DAYS_ENTRY_PAYMENT": days_entry_payment.astype(int),
            "AMT_INSTALMENT": amt_instalment,
            "AMT_PAYMENT": amt_payment,
        }
    )

    # lightweight POS_CASH_balance for schema completeness (not consumed by feature scripts)
    n_pos = rng.integers(1, 8, len(approved))
    pos_idx = _repeat_index(n_pos)
    pos_offsets = np.concatenate([np.arange(m) for m in n_pos])
    pos_cash = pd.DataFrame(
        {
            "SK_ID_PREV": approved["SK_ID_PREV"].values[pos_idx],
            "SK_ID_CURR": approved["SK_ID_CURR"].values[pos_idx],
            "MONTHS_BALANCE": -pos_offsets,
            "CNT_INSTALMENT": rng.integers(6, 36, len(pos_idx)),
            "CNT_INSTALMENT_FUTURE": rng.integers(0, 24, len(pos_idx)),
            "NAME_CONTRACT_STATUS": "Active",
        }
    )
    return installments, pos_cash


def build_credit_card_balance(
    previous_application: pd.DataFrame, applicants: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    approved = previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Approved"].reset_index(drop=True)
    has_card = rng.random(len(approved)) < 0.35
    cards = approved[has_card].reset_index(drop=True)

    z_by_curr = pd.Series(applicants.attrs["behavior_z"], index=applicants["SK_ID_CURR"].values)
    z_cards = cards["SK_ID_CURR"].map(z_by_curr).values

    n_months = rng.integers(3, 25, len(cards))
    idx = _repeat_index(n_months)
    n_rows = len(idx)
    z_row = z_cards[idx]
    month_idx = np.concatenate([np.arange(m) for m in n_months])

    credit_limit = np.round(rng.lognormal(np.log(80_000), 0.5, len(cards)), -2)
    credit_limit_rep = credit_limit[idx]

    recency_drift = 0.01 * month_idx * np.clip(z_row, 0, None)  # riskier accounts drift up over time
    utilization = np.clip(_sigmoid(0.7 * z_row + recency_drift + rng.normal(0, 0.4, n_rows)), 0, 1) * 0.95
    amt_balance = (utilization * credit_limit_rep).round(2)
    amt_inst_min = (0.02 * credit_limit_rep).round(2)

    p_min_only = np.clip(_sigmoid(0.9 * z_row - 0.5), 0.02, 0.8)
    min_only = rng.random(n_rows) < p_min_only
    amt_payment_total = np.where(
        min_only,
        amt_inst_min * rng.uniform(0.95, 1.05, n_rows),
        amt_balance * rng.uniform(0.3, 1.0, n_rows),
    ).round(2)

    return pd.DataFrame(
        {
            "SK_ID_PREV": cards["SK_ID_PREV"].values[idx],
            "SK_ID_CURR": cards["SK_ID_CURR"].values[idx],
            "MONTHS_BALANCE": -month_idx,
            "AMT_BALANCE": amt_balance,
            "AMT_CREDIT_LIMIT_ACTUAL": credit_limit_rep,
            "AMT_PAYMENT_TOTAL_CURRENT": amt_payment_total,
            "AMT_INST_MIN_REGULARITY": amt_inst_min,
        }
    )


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    applicants = build_applicants(rng)
    bureau, bureau_balance = build_bureau(applicants, rng)
    previous_application = build_previous_application(applicants, rng)
    installments, pos_cash = build_installments_and_pos(previous_application, applicants, rng)
    credit_card_balance = build_credit_card_balance(previous_application, applicants, rng)

    applicants.drop(columns=[], errors="ignore").to_csv(OUT_DIR / "application_train.csv", index=False)
    bureau.to_csv(OUT_DIR / "bureau.csv", index=False)
    bureau_balance.to_csv(OUT_DIR / "bureau_balance.csv", index=False)
    previous_application.to_csv(OUT_DIR / "previous_application.csv", index=False)
    pos_cash.to_csv(OUT_DIR / "POS_CASH_balance.csv", index=False)
    installments.to_csv(OUT_DIR / "installments_payments.csv", index=False)
    credit_card_balance.to_csv(OUT_DIR / "credit_card_balance.csv", index=False)

    print(f"application_train:      {len(applicants):>10,} rows, TARGET rate = {applicants['TARGET'].mean():.4f}")
    print(f"bureau:                  {len(bureau):>10,} rows")
    print(f"bureau_balance:          {len(bureau_balance):>10,} rows")
    print(f"previous_application:    {len(previous_application):>10,} rows")
    print(f"POS_CASH_balance:        {len(pos_cash):>10,} rows")
    print(f"installments_payments:   {len(installments):>10,} rows")
    print(f"credit_card_balance:     {len(credit_card_balance):>10,} rows")
    print(f"Written to {OUT_DIR}")

    # Reject pool -- needed for scorecard/reject_inference.py. Real application_train (or
    # this synthetic stand-in) only contains funded loans, so a separate through-the-door
    # population must be generated. See build_reject_pool's docstring.
    rejects = build_reject_pool(applicants, rng)
    r_bureau, r_bureau_balance = build_bureau(rejects, rng)
    r_previous_application = build_previous_application(rejects, rng)
    r_installments, r_pos_cash = build_installments_and_pos(r_previous_application, rejects, rng)
    r_credit_card_balance = build_credit_card_balance(r_previous_application, rejects, rng)

    hidden_truth = pd.DataFrame(
        {"SK_ID_CURR": rejects["SK_ID_CURR"], "true_target": rejects["TARGET"],
         "true_default_prob": rejects.attrs["true_default_prob"]}
    )
    rejects_features_only = rejects.drop(columns=["TARGET"])

    rejects_features_only.to_csv(OUT_DIR / "application_rejects.csv", index=False)
    r_bureau.to_csv(OUT_DIR / "bureau_rejects.csv", index=False)
    r_bureau_balance.to_csv(OUT_DIR / "bureau_balance_rejects.csv", index=False)
    r_previous_application.to_csv(OUT_DIR / "previous_application_rejects.csv", index=False)
    r_installments.to_csv(OUT_DIR / "installments_payments_rejects.csv", index=False)
    r_credit_card_balance.to_csv(OUT_DIR / "credit_card_balance_rejects.csv", index=False)
    hidden_truth.to_csv(OUT_DIR / "reject_hidden_truth.csv", index=False)

    print(f"\napplication_rejects:     {len(rejects_features_only):>10,} rows (no TARGET column -- "
          f"unobserved by construction)")
    print(f"  hidden true default rate (validation-only, never a model input): "
          f"{hidden_truth['true_target'].mean():.4f}")
    print(f"Written reject-pool tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
