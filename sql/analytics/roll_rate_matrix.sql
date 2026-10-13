-- Bucket transition counts (roll rate matrix), computed upstream so Power BI's matrix visual
-- is a plain COUNT — no DAX gymnastics. bucket_prev/bucket_curr are already landed as columns
-- on fm_performance by the loan-performance simulation/ingestion, so this is a pure aggregation
-- (LAG is shown here for documentation of how they'd be derived from a flat monthly extract
-- that only has bucket_curr per row, e.g. once real Fannie Mae files are swapped in).

WITH ordered AS (
    SELECT
        loan_id,
        reporting_date,
        bucket_curr,
        LAG(bucket_curr) OVER (PARTITION BY loan_id ORDER BY reporting_date) AS bucket_prev_derived
    FROM fm_performance
)
SELECT
    reporting_date,
    bucket_prev,
    bucket_curr,
    COUNT(*) AS n_loans
FROM fm_performance
GROUP BY reporting_date, bucket_prev, bucket_curr
ORDER BY reporting_date, bucket_prev, bucket_curr;

-- Sanity check: bucket_prev on fm_performance should equal the LAG-derived value
-- (validates that the upstream simulation/ingestion computed transitions correctly).
-- SELECT COUNT(*) FROM ordered o JOIN fm_performance p USING (loan_id, reporting_date)
-- WHERE o.bucket_prev_derived IS NOT NULL AND o.bucket_prev_derived != p.bucket_prev;
