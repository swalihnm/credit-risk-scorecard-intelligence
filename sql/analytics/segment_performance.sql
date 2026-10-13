-- Default rate and (proxy) yield by segment with a Wald confidence interval on the default
-- rate, so small segments don't get over-interpreted on the monitoring page.

WITH loan_outcome AS (
    SELECT
        l.loan_id,
        l.state,
        l.region,
        CASE
            WHEN l.fico_orig < 620 THEN '<620'
            WHEN l.fico_orig < 660 THEN '620-660'
            WHEN l.fico_orig < 700 THEN '660-700'
            WHEN l.fico_orig < 740 THEN '700-740'
            ELSE '740+'
        END AS fico_band,
        l.orig_rate,
        CASE WHEN d.loan_id IS NOT NULL THEN 1 ELSE 0 END AS defaulted
    FROM fm_loans l
    LEFT JOIN fm_defaults d USING (loan_id)
)
SELECT
    region,
    fico_band,
    COUNT(*) AS n_loans,
    AVG(defaulted) AS default_rate,
    AVG(orig_rate) AS avg_rate,
    -- 95% Wald CI on the default rate proportion
    AVG(defaulted) - 1.96 * SQRT(AVG(defaulted) * (1 - AVG(defaulted)) / COUNT(*)) AS ci_lower,
    AVG(defaulted) + 1.96 * SQRT(AVG(defaulted) * (1 - AVG(defaulted)) / COUNT(*)) AS ci_upper
FROM loan_outcome
GROUP BY region, fico_band
ORDER BY region, fico_band;
