-- Exposure concentration by geography, LTV band, and FICO band, with Herfindahl-Hirschman
-- Index (HHI) per dimension. Snapshot as of the latest reporting month in the panel.

WITH latest_month AS (
    SELECT MAX(reporting_date) AS d FROM fm_performance
),
snapshot AS (
    SELECT p.loan_id, p.upb, l.state, l.original_ltv, l.fico_orig
    FROM fm_performance p
    JOIN fm_loans l USING (loan_id)
    JOIN latest_month m ON p.reporting_date = m.d
    WHERE p.bucket_curr != 'Charge-off'
),
banded AS (
    SELECT
        *,
        CASE
            WHEN original_ltv < 60 THEN '<60'
            WHEN original_ltv < 70 THEN '60-70'
            WHEN original_ltv < 80 THEN '70-80'
            WHEN original_ltv < 90 THEN '80-90'
            ELSE '90+'
        END AS ltv_band,
        CASE
            WHEN fico_orig < 620 THEN '<620'
            WHEN fico_orig < 660 THEN '620-660'
            WHEN fico_orig < 700 THEN '660-700'
            WHEN fico_orig < 740 THEN '700-740'
            ELSE '740+'
        END AS fico_band
    FROM snapshot
),
totals AS (SELECT SUM(upb) AS total_upb FROM banded)
SELECT 'state' AS dimension, state AS segment, SUM(upb) AS exposure,
       SUM(upb) / (SELECT total_upb FROM totals) AS share
FROM banded GROUP BY state
UNION ALL
SELECT 'ltv_band', ltv_band, SUM(upb), SUM(upb) / (SELECT total_upb FROM totals)
FROM banded GROUP BY ltv_band
UNION ALL
SELECT 'fico_band', fico_band, SUM(upb), SUM(upb) / (SELECT total_upb FROM totals)
FROM banded GROUP BY fico_band
ORDER BY dimension, share DESC;

-- HHI per dimension = SUM(share^2), computed the same way the Concentration HHI DAX measure
-- computes it over dim_geography — this query is the reference value to validate that measure.
