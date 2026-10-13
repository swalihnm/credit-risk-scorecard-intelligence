-- Aligns each loan's vintage to macro conditions AT ORIGINATION, and each performance row
-- to macro conditions AT THE PERFORMANCE MONTH. This is the SQL-side counterpart of the
-- Power BI role-playing date dimension (active relationship on reporting date, inactive on
-- origination date activated via USERELATIONSHIP) — both should agree if you cross-check them.

SELECT
    p.loan_id,
    l.origination_date,
    p.reporting_date,
    p.months_on_book,
    p.bucket_curr,
    p.upb,
    m_orig.unrate  AS unrate_at_origination,
    m_orig.hpi     AS hpi_at_origination,
    m_perf.unrate  AS unrate_at_performance_month,
    m_perf.hpi     AS hpi_at_performance_month,
    (m_perf.hpi / m_orig.hpi - 1) AS hpi_change_since_origination
FROM fm_performance p
JOIN fm_loans l USING (loan_id)
JOIN macro_monthly m_orig ON m_orig.date = date_trunc('month', l.origination_date)
JOIN macro_monthly m_perf ON m_perf.date = date_trunc('month', p.reporting_date)
ORDER BY p.loan_id, p.reporting_date;
