-- ============================================================
-- CHURNIQ — CUSTOMER PROFILE ANALYSIS
-- ============================================================
-- Purpose:
-- Descriptive customer profiling to understand how tenure,
-- customer value, recharge behavior, activity, and recency
-- differ across the customer base and observed churn outcomes.
--
-- Prediction point:
-- End of August 2014
--
-- Data source:
-- staging.train_raw
--
-- Important:
-- This script is descriptive/business analysis only.
-- It does NOT use model predictions, risk scores, thresholds,
-- or retention prioritization outputs.
--
-- Customer value proxy:
-- August ARPU (arpu_8)
-- ============================================================


-- ============================================================
-- 01. CUSTOMER PROFILE SNAPSHOT
-- ============================================================

SELECT
    COUNT(*) AS total_customers,

    ROUND(AVG(aon)::numeric, 2) AS avg_tenure_days,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY aon)::numeric,
        2
    ) AS median_tenure_days,

    ROUND(AVG(arpu_8)::numeric, 2) AS avg_august_arpu,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY arpu_8)::numeric,
        2
    ) AS median_august_arpu,

    ROUND(AVG(total_rech_amt_8)::numeric, 2)
        AS avg_august_recharge_amount,

    ROUND(AVG(total_rech_num_8)::numeric, 2)
        AS avg_august_recharge_count,

    ROUND(AVG(total_og_mou_8)::numeric, 2)
        AS avg_august_outgoing_mou,

    ROUND(AVG(total_ic_mou_8)::numeric, 2)
        AS avg_august_incoming_mou

FROM staging.train_raw;


-- ============================================================
-- 02. TENURE PROFILE
-- ============================================================

WITH tenure_profile AS (
    SELECT
        CASE
            WHEN aon < 365 THEN 'Under 1 year'
            WHEN aon < 730 THEN '1-2 years'
            WHEN aon < 1095 THEN '2-3 years'
            WHEN aon < 1825 THEN '3-5 years'
            ELSE '5+ years'
        END AS tenure_band,

        aon,
        churn_probability

    FROM staging.train_raw
)

SELECT
    tenure_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(AVG(aon)::numeric, 2)
        AS avg_tenure_days,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM tenure_profile

GROUP BY tenure_band

ORDER BY
    CASE tenure_band
        WHEN 'Under 1 year' THEN 1
        WHEN '1-2 years' THEN 2
        WHEN '2-3 years' THEN 3
        WHEN '3-5 years' THEN 4
        WHEN '5+ years' THEN 5
        ELSE 6
    END;


-- ============================================================
-- 03. AUGUST ARPU PROFILE
-- ============================================================

WITH arpu_profile AS (
    SELECT
        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 100 THEN 'Under 100'
            WHEN arpu_8 < 200 THEN '100-199'
            WHEN arpu_8 < 300 THEN '200-299'
            WHEN arpu_8 < 500 THEN '300-499'
            ELSE '500+'
        END AS arpu_band,

        arpu_8,
        churn_probability

    FROM staging.train_raw
)

SELECT
    arpu_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(AVG(arpu_8)::numeric, 2)
        AS avg_august_arpu,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM arpu_profile

GROUP BY arpu_band

ORDER BY
    CASE arpu_band
        WHEN 'ARPU missing' THEN 1
        WHEN 'Under 100' THEN 2
        WHEN '100-199' THEN 3
        WHEN '200-299' THEN 4
        WHEN '300-499' THEN 5
        WHEN '500+' THEN 6
        ELSE 7
    END;


-- ============================================================
-- 04. AUGUST RECHARGE VALUE PROFILE
-- ============================================================

WITH recharge_value_profile AS (
    SELECT
        CASE
            WHEN total_rech_amt_8 IS NULL
                THEN 'Recharge value missing'
            WHEN total_rech_amt_8 = 0
                THEN 'Zero recharge'
            WHEN total_rech_amt_8 < 100
                THEN 'Under 100'
            WHEN total_rech_amt_8 < 250
                THEN '100-249'
            WHEN total_rech_amt_8 < 500
                THEN '250-499'
            WHEN total_rech_amt_8 < 1000
                THEN '500-999'
            ELSE '1000+'
        END AS recharge_value_band,

        total_rech_amt_8,
        churn_probability

    FROM staging.train_raw
)

SELECT
    recharge_value_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_rech_amt_8)::numeric,
        2
    ) AS avg_august_recharge_amount,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM recharge_value_profile

GROUP BY recharge_value_band

ORDER BY
    CASE recharge_value_band
        WHEN 'Recharge value missing' THEN 1
        WHEN 'Zero recharge' THEN 2
        WHEN 'Under 100' THEN 3
        WHEN '100-249' THEN 4
        WHEN '250-499' THEN 5
        WHEN '500-999' THEN 6
        WHEN '1000+' THEN 7
        ELSE 8
    END;


-- ============================================================
-- 05. AUGUST RECHARGE FREQUENCY PROFILE
-- ============================================================

WITH recharge_frequency_profile AS (
    SELECT
        CASE
            WHEN total_rech_num_8 IS NULL
                THEN 'Recharge frequency missing'
            WHEN total_rech_num_8 = 0
                THEN 'Zero recharge'
            WHEN total_rech_num_8 = 1
                THEN '1 recharge'
            WHEN total_rech_num_8 <= 3
                THEN '2-3 recharges'
            WHEN total_rech_num_8 <= 5
                THEN '4-5 recharges'
            ELSE '6+ recharges'
        END AS recharge_frequency_band,

        total_rech_num_8,
        churn_probability

    FROM staging.train_raw
)

SELECT
    recharge_frequency_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_rech_num_8)::numeric,
        2
    ) AS avg_august_recharge_count,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM recharge_frequency_profile

GROUP BY recharge_frequency_band

ORDER BY
    CASE recharge_frequency_band
        WHEN 'Recharge frequency missing' THEN 1
        WHEN 'Zero recharge' THEN 2
        WHEN '1 recharge' THEN 3
        WHEN '2-3 recharges' THEN 4
        WHEN '4-5 recharges' THEN 5
        WHEN '6+ recharges' THEN 6
        ELSE 7
    END;


-- ============================================================
-- 06. CUSTOMER VALUE PROFILE
-- ============================================================
-- August ARPU is used as a descriptive customer-value proxy.
-- Missing values remain a separate category.
-- ============================================================

WITH value_profile AS (
    SELECT
        CASE
            WHEN arpu_8 IS NULL
                THEN 'ARPU missing'
            WHEN arpu_8 < 200
                THEN 'Lower value'
            WHEN arpu_8 < 500
                THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_profile,

        arpu_8,
        churn_probability

    FROM staging.train_raw
)

SELECT
    value_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(arpu_8)::numeric,
        2
    ) AS avg_august_arpu,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM value_profile

GROUP BY value_profile

ORDER BY
    CASE value_profile
        WHEN 'ARPU missing' THEN 1
        WHEN 'Lower value' THEN 2
        WHEN 'Mid value' THEN 3
        WHEN 'Higher value' THEN 4
        ELSE 5
    END;


-- ============================================================
-- 07. CUSTOMER ACTIVITY PROFILE
-- ============================================================

WITH activity_profile AS (
    SELECT
        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                 AND total_rech_amt_8 IS NULL
                THEN 'Activity information missing'

            WHEN COALESCE(total_og_mou_8, 0) = 0
                 AND COALESCE(total_ic_mou_8, 0) = 0
                 AND COALESCE(total_rech_amt_8, 0) = 0
                THEN 'Very low activity'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
                + COALESCE(total_rech_amt_8, 0)
            ) < 200
                THEN 'Low activity'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
                + COALESCE(total_rech_amt_8, 0)
            ) < 700
                THEN 'Moderate activity'

            ELSE 'High activity'
        END AS activity_profile,

        churn_probability

    FROM staging.train_raw
)

SELECT
    activity_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM activity_profile

GROUP BY activity_profile

ORDER BY
    CASE activity_profile
        WHEN 'Activity information missing' THEN 1
        WHEN 'Very low activity' THEN 2
        WHEN 'Low activity' THEN 3
        WHEN 'Moderate activity' THEN 4
        WHEN 'High activity' THEN 5
        ELSE 6
    END;


-- ============================================================
-- 08. RECHARGE RECENCY PROFILE
-- ============================================================
-- Prediction point: August 31, 2014
--
-- Source dates are stored as text in MM/DD/YYYY format.
-- ============================================================

WITH recharge_recency AS (
    SELECT
        CASE
            WHEN date_of_last_rech_8 IS NULL
                THEN 'Recharge date missing'

            WHEN DATE '2014-08-31'
                 - TO_DATE(
                     date_of_last_rech_8,
                     'MM/DD/YYYY'
                   ) <= 2
                THEN '0-2 days'

            WHEN DATE '2014-08-31'
                 - TO_DATE(
                     date_of_last_rech_8,
                     'MM/DD/YYYY'
                   ) <= 5
                THEN '3-5 days'

            WHEN DATE '2014-08-31'
                 - TO_DATE(
                     date_of_last_rech_8,
                     'MM/DD/YYYY'
                   ) <= 10
                THEN '6-10 days'

            ELSE '11+ days'
        END AS recharge_recency_band,

        churn_probability

    FROM staging.train_raw
)

SELECT
    recharge_recency_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM recharge_recency

GROUP BY recharge_recency_band

ORDER BY
    CASE recharge_recency_band
        WHEN '0-2 days' THEN 1
        WHEN '3-5 days' THEN 2
        WHEN '6-10 days' THEN 3
        WHEN '11+ days' THEN 4
        WHEN 'Recharge date missing' THEN 5
        ELSE 6
    END;


-- ============================================================
-- 09. TENURE × CUSTOMER VALUE PROFILE
-- ============================================================

WITH profile AS (
    SELECT
        CASE
            WHEN aon < 365 THEN 'Under 1 year'
            WHEN aon < 730 THEN '1-2 years'
            WHEN aon < 1095 THEN '2-3 years'
            WHEN aon < 1825 THEN '3-5 years'
            ELSE '5+ years'
        END AS tenure_band,

        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        churn_probability

    FROM staging.train_raw
)

SELECT
    tenure_band,
    value_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM profile

GROUP BY
    tenure_band,
    value_band

HAVING COUNT(*) >= 100

ORDER BY
    CASE tenure_band
        WHEN 'Under 1 year' THEN 1
        WHEN '1-2 years' THEN 2
        WHEN '2-3 years' THEN 3
        WHEN '3-5 years' THEN 4
        WHEN '5+ years' THEN 5
        ELSE 6
    END,

    CASE value_band
        WHEN 'ARPU missing' THEN 1
        WHEN 'Lower value' THEN 2
        WHEN 'Mid value' THEN 3
        WHEN 'Higher value' THEN 4
        ELSE 5
    END;


-- ============================================================
-- 10. CUSTOMER VALUE × RECHARGE BEHAVIOR
-- ============================================================

WITH profile AS (
    SELECT
        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        CASE
            WHEN total_rech_num_8 IS NULL
                THEN 'Recharge frequency missing'
            WHEN total_rech_num_8 <= 1
                THEN 'Low recharge frequency'
            WHEN total_rech_num_8 <= 3
                THEN 'Moderate recharge frequency'
            ELSE 'High recharge frequency'
        END AS recharge_behavior,

        churn_probability

    FROM staging.train_raw
)

SELECT
    value_band,
    recharge_behavior,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM profile

GROUP BY
    value_band,
    recharge_behavior

HAVING COUNT(*) >= 100

ORDER BY
    CASE value_band
        WHEN 'ARPU missing' THEN 1
        WHEN 'Lower value' THEN 2
        WHEN 'Mid value' THEN 3
        WHEN 'Higher value' THEN 4
        ELSE 5
    END,

    CASE recharge_behavior
        WHEN 'Recharge frequency missing' THEN 1
        WHEN 'Low recharge frequency' THEN 2
        WHEN 'Moderate recharge frequency' THEN 3
        WHEN 'High recharge frequency' THEN 4
        ELSE 5
    END;


-- ============================================================
-- 11. CUSTOMER PROFILE ARCHETYPES
-- ============================================================
-- Descriptive archetypes based on August value and activity.
-- These are analytical profiles, not model risk tiers.
-- ============================================================

WITH customer_profiles AS (
    SELECT
        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        CASE
            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
                + COALESCE(total_rech_amt_8, 0)
            ) >= 700
                THEN 'Active'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
                + COALESCE(total_rech_amt_8, 0)
            ) < 200
                THEN 'Low engagement'

            ELSE 'Moderate engagement'
        END AS engagement_band,

        churn_probability

    FROM staging.train_raw
)

SELECT
    CASE
        WHEN value_band = 'Higher value'
             AND engagement_band = 'Active'
            THEN 'High-value active'

        WHEN value_band = 'Higher value'
             AND engagement_band = 'Low engagement'
            THEN 'High-value low-engagement'

        WHEN value_band = 'Mid value'
             AND engagement_band = 'Active'
            THEN 'Mid-value active'

        WHEN value_band = 'Mid value'
             AND engagement_band = 'Low engagement'
            THEN 'Mid-value low-engagement'

        WHEN value_band = 'Lower value'
             AND engagement_band = 'Active'
            THEN 'Lower-value active'

        WHEN value_band = 'Lower value'
             AND engagement_band = 'Low engagement'
            THEN 'Lower-value low-engagement'

        ELSE 'Other / insufficient information'
    END AS customer_archetype,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM customer_profiles

GROUP BY customer_archetype

ORDER BY
    customers DESC;


-- ============================================================
-- 12. HIGH-VALUE CUSTOMER CONCENTRATION
-- ============================================================
-- High-value definition:
-- August ARPU >= 500
--
-- This is descriptive concentration analysis only.
-- ============================================================

WITH value_groups AS (
    SELECT
        CASE
            WHEN arpu_8 >= 500
                THEN 'Higher-value customers'
            WHEN arpu_8 IS NULL
                THEN 'ARPU missing'
            ELSE 'Other customers'
        END AS value_group,

        churn_probability

    FROM staging.train_raw
)

SELECT
    value_group,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM value_groups

GROUP BY value_group

ORDER BY
    CASE value_group
        WHEN 'Higher-value customers' THEN 1
        WHEN 'Other customers' THEN 2
        WHEN 'ARPU missing' THEN 3
        ELSE 4
    END;


-- ============================================================
-- 13. PROFILE CONCENTRATION
-- ============================================================
-- Minimum-volume guardrail:
-- Only profile combinations representing at least 1%
-- of the customer base are displayed.
-- ============================================================

WITH profile AS (
    SELECT
        CASE
            WHEN aon < 365 THEN 'Under 1 year'
            WHEN aon < 730 THEN '1-2 years'
            WHEN aon < 1095 THEN '2-3 years'
            WHEN aon < 1825 THEN '3-5 years'
            ELSE '5+ years'
        END AS tenure_band,

        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        churn_probability

    FROM staging.train_raw
),

profile_summary AS (
    SELECT
        tenure_band,
        value_band,

        COUNT(*) AS customers,

        ROUND(
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
            2
        ) AS customer_share_pct,

        ROUND(
            100.0 * AVG(churn_probability::numeric),
            2
        ) AS observed_churn_rate_pct

    FROM profile

    GROUP BY
        tenure_band,
        value_band
)

SELECT
    tenure_band,
    value_band,
    customers,
    customer_share_pct,
    observed_churn_rate_pct

FROM profile_summary

WHERE customer_share_pct >= 1

ORDER BY
    observed_churn_rate_pct DESC;


-- ============================================================
-- 14. EXECUTIVE CUSTOMER PROFILE SIGNALS
-- ============================================================
-- Compact descriptive signals for stakeholder interpretation.
-- These are associations in the observed data and should not
-- be interpreted as causal effects.
-- ============================================================

WITH profile_signals AS (

    SELECT
        'Tenure' AS profile_dimension,
        CASE
            WHEN aon < 365 THEN 'Under 1 year'
            WHEN aon < 730 THEN '1-2 years'
            WHEN aon < 1095 THEN '2-3 years'
            WHEN aon < 1825 THEN '3-5 years'
            ELSE '5+ years'
        END AS profile_group,
        churn_probability
    FROM staging.train_raw

    UNION ALL

    SELECT
        'August ARPU',
        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END,
        churn_probability
    FROM staging.train_raw

    UNION ALL

    SELECT
        'August recharge frequency',
        CASE
            WHEN total_rech_num_8 IS NULL
                THEN 'Recharge frequency missing'
            WHEN total_rech_num_8 <= 1
                THEN 'Low recharge frequency'
            WHEN total_rech_num_8 <= 3
                THEN 'Moderate recharge frequency'
            ELSE 'High recharge frequency'
        END,
        churn_probability
    FROM staging.train_raw
),

profile_summary AS (
    SELECT
        profile_dimension,
        profile_group,

        COUNT(*) AS customers,

        ROUND(
            100.0 * COUNT(*) / SUM(COUNT(*))
            OVER (PARTITION BY profile_dimension),
            2
        ) AS customer_share_pct,

        ROUND(
            100.0 * AVG(churn_probability::numeric),
            2
        ) AS observed_churn_rate_pct

    FROM profile_signals

    GROUP BY
        profile_dimension,
        profile_group
)

SELECT
    profile_dimension,
    profile_group,
    customers,
    customer_share_pct,
    observed_churn_rate_pct

FROM profile_summary

WHERE customer_share_pct >= 1

ORDER BY
    profile_dimension,
    observed_churn_rate_pct DESC;


-- ============================================================
-- 15. ANALYTICAL QUALITY GATE
-- ============================================================

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT id) AS distinct_customer_ids,

    COUNT(*) - COUNT(DISTINCT id)
        AS duplicate_customer_ids,

    COUNT(*) FILTER (
        WHERE churn_probability IS NULL
    ) AS missing_target_values,

    CASE
        WHEN COUNT(*) = 69999
         AND COUNT(DISTINCT id) = 69999
         AND COUNT(*) - COUNT(DISTINCT id) = 0
         AND COUNT(*) FILTER (
             WHERE churn_probability IS NULL
         ) = 0
        THEN 'PASS'

        ELSE 'REVIEW'
    END AS quality_gate_status

FROM staging.train_raw;


-- ============================================================
-- END OF CUSTOMER PROFILE ANALYSIS
-- ============================================================