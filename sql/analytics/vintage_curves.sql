-- Cumulative default rate by months-on-book, by origination cohort (vintage).
-- Grain: one row per (origination cohort month, months_on_book).
-- Feeds the vintage curve matrix (cohort on rows, MOB on columns) directly —
-- Power BI does no recomputation here, only pivot/format.
--
-- NOTE: a charged-off loan has no further fm_performance rows after its default month
-- (charge-off is terminal), so cumulative default counts are computed directly from
-- fm_defaults against a month-offset grid, NOT by joining through fm_performance's
-- months_on_book column for later months — a defaulted loan would silently disappear
-- from every later MOB bucket if we did that, undercounting the cumulative rate.

WITH loan_vintage AS (
    SELECT loan_id, date_trunc('month', origination_date) AS vintage
    FROM fm_loans
),
loan_default_mob AS (
    SELECT
        lv.loan_id,
        lv.vintage,
        DATEDIFF('month', lv.vintage, date_trunc('month', d.default_date)) + 1 AS default_mob
    FROM loan_vintage lv
    LEFT JOIN fm_defaults d USING (loan_id)
),
cohort_size AS (
    SELECT vintage, COUNT(*) AS cohort_loans FROM loan_vintage GROUP BY 1
),
mob_grid AS (
    SELECT DISTINCT months_on_book FROM fm_performance
)
SELECT
    ldm.vintage,
    g.months_on_book,
    cs.cohort_loans,
    COUNT(CASE WHEN ldm.default_mob IS NOT NULL AND ldm.default_mob <= g.months_on_book THEN 1 END) AS cum_defaulted_loans,
    COUNT(CASE WHEN ldm.default_mob IS NOT NULL AND ldm.default_mob <= g.months_on_book THEN 1 END)::DOUBLE
        / cs.cohort_loans AS cumulative_default_rate
FROM loan_default_mob ldm
JOIN cohort_size cs ON cs.vintage = ldm.vintage
CROSS JOIN mob_grid g
GROUP BY ldm.vintage, g.months_on_book, cs.cohort_loans
ORDER BY ldm.vintage, g.months_on_book;
