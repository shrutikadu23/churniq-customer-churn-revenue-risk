-- ============================================================
-- ChurnIQ — 01 Data Validation
-- ============================================================
-- Purpose:
--   Validate the structural integrity, target quality, temporal
--   consistency, and core business-data validity of the raw
--   training dataset before downstream SQL analytics and ML.
--
-- Source:
--   staging.train_raw
--
-- Important:
--   This script is READ-ONLY.
--   No source data is modified.
--
-- Dataset:
--   Telecom Churn Case Study Hackathon
--
-- Prediction framework:
--   Prediction point : End of August 2014
--   Prediction horizon: September 2014 / subsequent churn outcome
--
-- Validation philosophy:
--   PASS  = condition is valid
--   REVIEW = anomaly exists and requires investigation
--
-- Known source characteristics:
--   - Month-end date columns are TEXT in M/D/YYYY format.
--   - July and August contain legitimate missing month-end values.
--   - Some ARPU values are negative and are therefore investigated,
--     not automatically classified as invalid.
--   - Recharge count/amount relationships are investigated rather
--     than automatically rejected because source semantics matter.
-- ============================================================


-- ============================================================
-- 0. DATASET CONTEXT
-- ============================================================

SELECT
    'ChurnIQ Raw Training Dataset' AS dataset,
    'staging.train_raw' AS source_table,
    COUNT(*) AS row_count,
    COUNT(*) FILTER (WHERE id IS NOT NULL) AS non_null_customer_ids,
    COUNT(*) FILTER (WHERE churn_probability IS NOT NULL) AS non_null_targets
FROM staging.train_raw;


-- ============================================================
-- 1. DATASET STRUCTURE AND ID INTEGRITY
-- ============================================================
-- Expected:
--   69,999 rows
--   69,999 non-null IDs
--   69,999 unique IDs

SELECT
    COUNT(*) AS row_count,
    COUNT(id) AS non_null_id_count,
    COUNT(DISTINCT id) AS distinct_id_count,
    COUNT(*) - COUNT(id) AS null_id_count,
    COUNT(*) - COUNT(DISTINCT id) AS duplicate_id_rows
FROM staging.train_raw;


-- ============================================================
-- 2. DUPLICATE CUSTOMER IDs
-- ============================================================

SELECT
    id,
    COUNT(*) AS occurrence_count
FROM staging.train_raw
GROUP BY id
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC, id;


-- ============================================================
-- 3. TARGET VALIDATION
-- ============================================================
-- Target:
--   churn_probability
--
-- Expected values:
--   0 = retained
--   1 = churned

SELECT
    COUNT(*) AS total_rows,

    COUNT(*) FILTER (
        WHERE churn_probability IS NULL
    ) AS null_target_count,

    COUNT(*) FILTER (
        WHERE churn_probability NOT IN (0, 1)
    ) AS invalid_target_count,

    COUNT(*) FILTER (
        WHERE churn_probability = 0
    ) AS retained_count,

    COUNT(*) FILTER (
        WHERE churn_probability = 1
    ) AS churned_count,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_percent,

    CASE
        WHEN COUNT(*) FILTER (
                 WHERE churn_probability IS NULL
             ) = 0
         AND COUNT(*) FILTER (
                 WHERE churn_probability NOT IN (0, 1)
             ) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 4. TARGET DISTRIBUTION
-- ============================================================

SELECT
    churn_probability AS target,
    COUNT(*) AS customer_count,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS percentage_of_dataset
FROM staging.train_raw
GROUP BY churn_probability
ORDER BY churn_probability;


-- ============================================================
-- 5. CONSTANT / NON-INFORMATIVE COLUMNS
-- ============================================================
-- These columns contain only one distinct value and therefore
-- provide no predictive variation.

SELECT
    column_name,
    COUNT(*) AS row_count,
    COUNT(DISTINCT
        CASE
            WHEN column_name = 'circle_id'
            THEN circle_id::text
            WHEN column_name = 'last_date_of_month_6'
            THEN last_date_of_month_6
        END
    ) AS distinct_value_count
FROM staging.train_raw
CROSS JOIN (
    VALUES
        ('circle_id'),
        ('last_date_of_month_6')
) AS c(column_name)
GROUP BY column_name
ORDER BY column_name;


-- ============================================================
-- 6. CRITICAL NULL VALIDATION
-- ============================================================
-- Core fields required for downstream analysis.

SELECT
    COUNT(*) FILTER (WHERE id IS NULL) AS missing_customer_id,
    COUNT(*) FILTER (WHERE churn_probability IS NULL) AS missing_target,
    COUNT(*) FILTER (WHERE arpu_8 IS NULL) AS missing_arpu_8,
    COUNT(*) FILTER (WHERE total_rech_amt_8 IS NULL) AS missing_recharge_amount_8,
    COUNT(*) FILTER (WHERE total_og_mou_8 IS NULL) AS missing_outgoing_usage_8,
    COUNT(*) FILTER (WHERE total_ic_mou_8 IS NULL) AS missing_incoming_usage_8,

    CASE
        WHEN COUNT(*) FILTER (WHERE id IS NULL) = 0
         AND COUNT(*) FILTER (WHERE churn_probability IS NULL) = 0
         AND COUNT(*) FILTER (WHERE arpu_8 IS NULL) = 0
         AND COUNT(*) FILTER (WHERE total_rech_amt_8 IS NULL) = 0
         AND COUNT(*) FILTER (WHERE total_og_mou_8 IS NULL) = 0
         AND COUNT(*) FILTER (WHERE total_ic_mou_8 IS NULL) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 7. AUGUST MISSINGNESS PROFILE
-- ============================================================
-- August is the action phase immediately before the prediction
-- horizon, so its data availability is especially important.

SELECT
    COUNT(*) FILTER (WHERE arpu_8 IS NULL) AS arpu_8_missing,
    COUNT(*) FILTER (WHERE total_rech_amt_8 IS NULL) AS recharge_amt_8_missing,
    COUNT(*) FILTER (WHERE total_og_mou_8 IS NULL) AS outgoing_mou_8_missing,
    COUNT(*) FILTER (WHERE total_ic_mou_8 IS NULL) AS incoming_mou_8_missing,
    COUNT(*) FILTER (WHERE date_of_last_rech_8 IS NULL) AS last_rech_8_missing,
    COUNT(*) FILTER (WHERE date_of_last_rech_data_8 IS NULL) AS last_data_rech_8_missing,
    COUNT(*) FILTER (WHERE total_rech_data_8 IS NULL) AS total_data_rech_8_missing
FROM staging.train_raw;


-- ============================================================
-- 8. TENURE / AON VALIDITY
-- ============================================================
-- AON = Age On Network.
--
-- Expected:
--   AON > 0

SELECT
    MIN(aon) AS minimum_aon,
    MAX(aon) AS maximum_aon,
    ROUND(AVG(aon)::numeric, 2) AS average_aon,

    COUNT(*) FILTER (
        WHERE aon IS NULL
    ) AS null_aon_count,

    COUNT(*) FILTER (
        WHERE aon <= 0
    ) AS invalid_aon_count,

    CASE
        WHEN COUNT(*) FILTER (WHERE aon IS NULL) = 0
         AND COUNT(*) FILTER (WHERE aon <= 0) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 9. RECHARGE VALIDITY
-- ============================================================
-- Recharge amount and count fields should not contain negative
-- values.
--
-- This is a validity check, not a semantic assumption about
-- whether zero recharge amount alongside positive recharge count
-- is possible.

SELECT
    COUNT(*) FILTER (
        WHERE total_rech_amt_6 < 0
           OR total_rech_amt_7 < 0
           OR total_rech_amt_8 < 0
    ) AS negative_recharge_amount_rows,

    COUNT(*) FILTER (
        WHERE total_rech_num_6 < 0
           OR total_rech_num_7 < 0
           OR total_rech_num_8 < 0
    ) AS negative_recharge_count_rows,

    COUNT(*) FILTER (
        WHERE max_rech_amt_6 < 0
           OR max_rech_amt_7 < 0
           OR max_rech_amt_8 < 0
    ) AS negative_max_recharge_rows,

    CASE
        WHEN COUNT(*) FILTER (
                 WHERE total_rech_amt_6 < 0
                    OR total_rech_amt_7 < 0
                    OR total_rech_amt_8 < 0
                    OR total_rech_num_6 < 0
                    OR total_rech_num_7 < 0
                    OR total_rech_num_8 < 0
                    OR max_rech_amt_6 < 0
                    OR max_rech_amt_7 < 0
                    OR max_rech_amt_8 < 0
             ) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 10. USAGE VALIDITY
-- ============================================================
-- Core voice/data usage measures should not be negative.

SELECT
    COUNT(*) FILTER (
        WHERE total_og_mou_6 < 0
           OR total_og_mou_7 < 0
           OR total_og_mou_8 < 0
           OR total_ic_mou_6 < 0
           OR total_ic_mou_7 < 0
           OR total_ic_mou_8 < 0
    ) AS negative_voice_usage_rows,

    COUNT(*) FILTER (
        WHERE vol_2g_mb_6 < 0
           OR vol_2g_mb_7 < 0
           OR vol_2g_mb_8 < 0
           OR vol_3g_mb_6 < 0
           OR vol_3g_mb_7 < 0
           OR vol_3g_mb_8 < 0
    ) AS negative_data_usage_rows,

    CASE
        WHEN COUNT(*) FILTER (
                 WHERE total_og_mou_6 < 0
                    OR total_og_mou_7 < 0
                    OR total_og_mou_8 < 0
                    OR total_ic_mou_6 < 0
                    OR total_ic_mou_7 < 0
                    OR total_ic_mou_8 < 0
                    OR vol_2g_mb_6 < 0
                    OR vol_2g_mb_7 < 0
                    OR vol_2g_mb_8 < 0
                    OR vol_3g_mb_6 < 0
                    OR vol_3g_mb_7 < 0
                    OR vol_3g_mb_8 < 0
             ) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 11. NEGATIVE ARPU INVESTIGATION
-- ============================================================
-- Negative ARPU values were observed in the source dataset.
--
-- They are NOT automatically removed because their business
-- meaning must be understood before transformation.
--
-- This section records the anomaly for downstream treatment.

SELECT
    COUNT(*) FILTER (WHERE arpu_6 < 0) AS negative_arpu_6_count,
    COUNT(*) FILTER (WHERE arpu_7 < 0) AS negative_arpu_7_count,
    COUNT(*) FILTER (WHERE arpu_8 < 0) AS negative_arpu_8_count,

    MIN(arpu_6) AS minimum_arpu_6,
    MIN(arpu_7) AS minimum_arpu_7,
    MIN(arpu_8) AS minimum_arpu_8,

    MAX(arpu_6) AS maximum_arpu_6,
    MAX(arpu_7) AS maximum_arpu_7,
    MAX(arpu_8) AS maximum_arpu_8,

    'INVESTIGATE — source values retained; no automatic deletion' AS treatment_status

FROM staging.train_raw;


-- ============================================================
-- 12. RECHARGE COUNT / AMOUNT INVESTIGATION
-- ============================================================
-- Investigate customers where recharge count > 0 but recharge
-- amount = 0.
--
-- This is recorded as an investigation flag rather than an
-- automatic data-quality failure.

SELECT
    COUNT(*) FILTER (
        WHERE total_rech_num_6 > 0
          AND total_rech_amt_6 = 0
    ) AS month_6_count,

    COUNT(*) FILTER (
        WHERE total_rech_num_7 > 0
          AND total_rech_amt_7 = 0
    ) AS month_7_count,

    COUNT(*) FILTER (
        WHERE total_rech_num_8 > 0
          AND total_rech_amt_8 = 0
    ) AS month_8_count,

    'INVESTIGATE — semantic/source behavior; not treated as automatic failure'
        AS treatment_status

FROM staging.train_raw;


-- ============================================================
-- 13. MONTH-END DATE VALIDATION
-- ============================================================
-- Source columns are TEXT in M/D/YYYY format.
--
-- Expected:
--   June   = 6/30/2014
--   July   = 7/31/2014
--   August = 8/31/2014
--
-- July and August contain legitimate missing values:
--   July   = 399
--   August = 733
--
-- NULLs are therefore reported separately and are NOT treated
-- as malformed dates.

SELECT
    COUNT(*) FILTER (
        WHERE last_date_of_month_6 IS NOT NULL
          AND last_date_of_month_6 <> '6/30/2014'
    ) AS invalid_month_6_dates,

    COUNT(*) FILTER (
        WHERE last_date_of_month_7 IS NOT NULL
          AND last_date_of_month_7 <> '7/31/2014'
    ) AS invalid_month_7_dates,

    COUNT(*) FILTER (
        WHERE last_date_of_month_8 IS NOT NULL
          AND last_date_of_month_8 <> '8/31/2014'
    ) AS invalid_month_8_dates,

    COUNT(*) FILTER (
        WHERE last_date_of_month_6 IS NULL
    ) AS missing_month_6_dates,

    COUNT(*) FILTER (
        WHERE last_date_of_month_7 IS NULL
    ) AS missing_month_7_dates,

    COUNT(*) FILTER (
        WHERE last_date_of_month_8 IS NULL
    ) AS missing_month_8_dates,

    CASE
        WHEN COUNT(*) FILTER (
                 WHERE last_date_of_month_6 IS NOT NULL
                   AND last_date_of_month_6 <> '6/30/2014'
             ) = 0
         AND COUNT(*) FILTER (
                 WHERE last_date_of_month_7 IS NOT NULL
                   AND last_date_of_month_7 <> '7/31/2014'
             ) = 0
         AND COUNT(*) FILTER (
                 WHERE last_date_of_month_8 IS NOT NULL
                   AND last_date_of_month_8 <> '8/31/2014'
             ) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 14. RECHARGE DATE AVAILABILITY
-- ============================================================
-- Date availability is profiled separately from date validity.

SELECT
    COUNT(*) FILTER (
        WHERE date_of_last_rech_6 IS NOT NULL
    ) AS available_recharge_date_6,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_7 IS NOT NULL
    ) AS available_recharge_date_7,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_8 IS NOT NULL
    ) AS available_recharge_date_8,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_6 IS NULL
    ) AS missing_recharge_date_6,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_7 IS NULL
    ) AS missing_recharge_date_7,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_8 IS NULL
    ) AS missing_recharge_date_8

FROM staging.train_raw;


-- ============================================================
-- 15. DATA RECHARGE DATE AVAILABILITY
-- ============================================================

SELECT
    COUNT(*) FILTER (
        WHERE date_of_last_rech_data_6 IS NOT NULL
    ) AS available_data_recharge_date_6,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_data_7 IS NOT NULL
    ) AS available_data_recharge_date_7,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_data_8 IS NOT NULL
    ) AS available_data_recharge_date_8,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_data_6 IS NULL
    ) AS missing_data_recharge_date_6,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_data_7 IS NULL
    ) AS missing_data_recharge_date_7,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_data_8 IS NULL
    ) AS missing_data_recharge_date_8

FROM staging.train_raw;


-- ============================================================
-- 16. RECHARGE DATE MONTH CONSISTENCY
-- ============================================================
-- Recharge dates are TEXT and may contain values such as
-- 6/1/2014 through 6/9/2014 for the June observation window.
--
-- The check converts only non-null values to DATE.
--
-- Expected:
--   *_6 -> June 2014
--   *_7 -> July 2014
--   *_8 -> August 2014

SELECT
    COUNT(*) FILTER (
        WHERE date_of_last_rech_6 IS NOT NULL
          AND TO_DATE(date_of_last_rech_6, 'MM/DD/YYYY')
              NOT BETWEEN DATE '2014-06-01' AND DATE '2014-06-30'
    ) AS invalid_recharge_month_6,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_7 IS NOT NULL
          AND TO_DATE(date_of_last_rech_7, 'MM/DD/YYYY')
              NOT BETWEEN DATE '2014-07-01' AND DATE '2014-07-31'
    ) AS invalid_recharge_month_7,

    COUNT(*) FILTER (
        WHERE date_of_last_rech_8 IS NOT NULL
          AND TO_DATE(date_of_last_rech_8, 'MM/DD/YYYY')
              NOT BETWEEN DATE '2014-08-01' AND DATE '2014-08-31'
    ) AS invalid_recharge_month_8,

    CASE
        WHEN COUNT(*) FILTER (
                 WHERE date_of_last_rech_6 IS NOT NULL
                   AND TO_DATE(date_of_last_rech_6, 'MM/DD/YYYY')
                       NOT BETWEEN DATE '2014-06-01'
                       AND DATE '2014-06-30'
             ) = 0
         AND COUNT(*) FILTER (
                 WHERE date_of_last_rech_7 IS NOT NULL
                   AND TO_DATE(date_of_last_rech_7, 'MM/DD/YYYY')
                       NOT BETWEEN DATE '2014-07-01'
                       AND DATE '2014-07-31'
             ) = 0
         AND COUNT(*) FILTER (
                 WHERE date_of_last_rech_8 IS NOT NULL
                   AND TO_DATE(date_of_last_rech_8, 'MM/DD/YYYY')
                       NOT BETWEEN DATE '2014-08-01'
                       AND DATE '2014-08-31'
             ) = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS validation_status

FROM staging.train_raw;


-- ============================================================
-- 17. PREDICTION TIMELINE VALIDATION
-- ============================================================
-- The modeling framework uses:
--
--   June + July  -> historical / good phase
--   August       -> action phase
--   September    -> future outcome / prediction horizon
--
-- September predictor columns must NOT be present in the raw
-- training feature set.

SELECT
    'June' AS period,
    'Historical / good phase' AS role,
    'Eligible as predictor' AS feature_status

UNION ALL

SELECT
    'July',
    'Historical / good phase',
    'Eligible as predictor'

UNION ALL

SELECT
    'August',
    'Action phase / prediction point',
    'Eligible as predictor'

UNION ALL

SELECT
    'September',
    'Future outcome phase',
    'Excluded from predictor set';


-- ============================================================
-- 18. FINAL SQL DATA-QUALITY GATE
-- ============================================================
-- Seven core checks:
--   1. Row count
--   2. Customer ID integrity
--   3. Target integrity
--   4. AON validity
--   5. Recharge validity
--   6. Usage validity
--   7. Month-end dates
--
-- A check is PASS only when the underlying validity condition
-- is satisfied.
--
-- Investigative anomalies such as negative ARPU are deliberately
-- excluded from the hard gate because they require semantic
-- treatment rather than automatic rejection.

WITH checks AS (

    -- --------------------------------------------------------
    -- 1. Row count
    -- --------------------------------------------------------
    SELECT
        'Row count' AS check_name,
        CASE
            WHEN COUNT(*) = 69999
            THEN 1
            ELSE 0
        END AS check_pass
    FROM staging.train_raw


    UNION ALL


    -- --------------------------------------------------------
    -- 2. Customer ID integrity
    -- --------------------------------------------------------
    SELECT
        'Customer ID integrity',
        CASE
            WHEN COUNT(*) = COUNT(id)
             AND COUNT(*) = COUNT(DISTINCT id)
            THEN 1
            ELSE 0
        END
    FROM staging.train_raw


    UNION ALL


    -- --------------------------------------------------------
    -- 3. Target integrity
    -- --------------------------------------------------------
    SELECT
        'Target integrity',
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE churn_probability IS NULL
                        OR churn_probability NOT IN (0, 1)
                 ) = 0
            THEN 1
            ELSE 0
        END
    FROM staging.train_raw


    UNION ALL


    -- --------------------------------------------------------
    -- 4. AON validity
    -- --------------------------------------------------------
    SELECT
        'AON validity',
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE aon IS NULL
                        OR aon <= 0
                 ) = 0
            THEN 1
            ELSE 0
        END
    FROM staging.train_raw


    UNION ALL


    -- --------------------------------------------------------
    -- 5. Recharge validity
    -- --------------------------------------------------------
    SELECT
        'Recharge validity',
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE total_rech_amt_6 < 0
                        OR total_rech_amt_7 < 0
                        OR total_rech_amt_8 < 0
                        OR total_rech_num_6 < 0
                        OR total_rech_num_7 < 0
                        OR total_rech_num_8 < 0
                        OR max_rech_amt_6 < 0
                        OR max_rech_amt_7 < 0
                        OR max_rech_amt_8 < 0
                 ) = 0
            THEN 1
            ELSE 0
        END
    FROM staging.train_raw


    UNION ALL


    -- --------------------------------------------------------
    -- 6. Usage validity
    -- --------------------------------------------------------
    SELECT
        'Usage validity',
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE total_og_mou_6 < 0
                        OR total_og_mou_7 < 0
                        OR total_og_mou_8 < 0
                        OR total_ic_mou_6 < 0
                        OR total_ic_mou_7 < 0
                        OR total_ic_mou_8 < 0
                        OR vol_2g_mb_6 < 0
                        OR vol_2g_mb_7 < 0
                        OR vol_2g_mb_8 < 0
                        OR vol_3g_mb_6 < 0
                        OR vol_3g_mb_7 < 0
                        OR vol_3g_mb_8 < 0
                 ) = 0
            THEN 1
            ELSE 0
        END
    FROM staging.train_raw


    UNION ALL


    -- --------------------------------------------------------
    -- 7. Month-end dates
    -- --------------------------------------------------------
    SELECT
        'Month-end dates',
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE
                         (
                             last_date_of_month_6 IS NOT NULL
                             AND last_date_of_month_6 <> '6/30/2014'
                         )
                         OR
                         (
                             last_date_of_month_7 IS NOT NULL
                             AND last_date_of_month_7 <> '7/31/2014'
                         )
                         OR
                         (
                             last_date_of_month_8 IS NOT NULL
                             AND last_date_of_month_8 <> '8/31/2014'
                         )
                 ) = 0
            THEN 1
            ELSE 0
        END
    FROM staging.train_raw
)

SELECT
    check_name,
    check_pass,
    CASE
        WHEN check_pass = 1
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS status
FROM checks
ORDER BY check_name;


-- ============================================================
-- FINAL SUMMARY
-- ============================================================

WITH checks AS (

    SELECT
        CASE
            WHEN COUNT(*) = 69999
            THEN 1 ELSE 0
        END AS pass
    FROM staging.train_raw

    UNION ALL

    SELECT
        CASE
            WHEN COUNT(*) = COUNT(id)
             AND COUNT(*) = COUNT(DISTINCT id)
            THEN 1 ELSE 0
        END
    FROM staging.train_raw

    UNION ALL

    SELECT
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE churn_probability IS NULL
                        OR churn_probability NOT IN (0, 1)
                 ) = 0
            THEN 1 ELSE 0
        END
    FROM staging.train_raw

    UNION ALL

    SELECT
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE aon IS NULL
                        OR aon <= 0
                 ) = 0
            THEN 1 ELSE 0
        END
    FROM staging.train_raw

    UNION ALL

    SELECT
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE total_rech_amt_6 < 0
                        OR total_rech_amt_7 < 0
                        OR total_rech_amt_8 < 0
                        OR total_rech_num_6 < 0
                        OR total_rech_num_7 < 0
                        OR total_rech_num_8 < 0
                        OR max_rech_amt_6 < 0
                        OR max_rech_amt_7 < 0
                        OR max_rech_amt_8 < 0
                 ) = 0
            THEN 1 ELSE 0
        END
    FROM staging.train_raw

    UNION ALL

    SELECT
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE total_og_mou_6 < 0
                        OR total_og_mou_7 < 0
                        OR total_og_mou_8 < 0
                        OR total_ic_mou_6 < 0
                        OR total_ic_mou_7 < 0
                        OR total_ic_mou_8 < 0
                        OR vol_2g_mb_6 < 0
                        OR vol_2g_mb_7 < 0
                        OR vol_2g_mb_8 < 0
                        OR vol_3g_mb_6 < 0
                        OR vol_3g_mb_7 < 0
                        OR vol_3g_mb_8 < 0
                 ) = 0
            THEN 1 ELSE 0
        END
    FROM staging.train_raw

    UNION ALL

    SELECT
        CASE
            WHEN COUNT(*) FILTER (
                     WHERE
                         (
                             last_date_of_month_6 IS NOT NULL
                             AND last_date_of_month_6 <> '6/30/2014'
                         )
                         OR
                         (
                             last_date_of_month_7 IS NOT NULL
                             AND last_date_of_month_7 <> '7/31/2014'
                         )
                         OR
                         (
                             last_date_of_month_8 IS NOT NULL
                             AND last_date_of_month_8 <> '8/31/2014'
                         )
                 ) = 0
            THEN 1 ELSE 0
        END
    FROM staging.train_raw
)

SELECT
    COUNT(*) AS validation_checks,
    SUM(pass) AS passed_checks,
    COUNT(*) - SUM(pass) AS failed_checks,
    CASE
        WHEN SUM(pass) = COUNT(*)
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS overall_validation_status
FROM checks;


-- ============================================================
-- END OF 01 DATA VALIDATION
-- ============================================================