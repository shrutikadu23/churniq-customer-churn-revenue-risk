/*==============================================================================
    CHURNIQ — 10. RISK & PRIORITY ANALYSIS

    Purpose:
        Identify where historical churn, customer value, and behavioral
        deterioration overlap.

    Business questions:
        1. Where is observed churn concentrated?
        2. Where is customer value concentrated?
        3. Are high-value customers experiencing greater historical churn?
        4. Which behavioral deterioration patterns are associated with churn?
        5. Which customer cohorts combine meaningful value with deterioration?
        6. Which descriptive cohorts should receive greater business attention?

    IMPORTANT:
        - DESCRIPTIVE BUSINESS ANALYSIS ONLY.
        - churn_probability is the observed historical churn outcome.
        - It is NOT an ML prediction.
        - No calibrated probability, ML risk tier, or retention action is
          created here.
        - Predictive customer prioritization is handled later by the
          Customer Risk Intelligence layer.
        - September data is excluded from predictor logic.
        - arpu_8 is treated as a customer-value proxy, not booked revenue.
        - Negative ARPU is retained in raw analysis but clipped to zero for
          value-exposure calculations.
==============================================================================*/


/*==============================================================================
    01. OVERALL RISK & VALUE SNAPSHOT
==============================================================================*/

SELECT
    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS observed_churned_customers,

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
            / NULLIF(COUNT(*), 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        AVG(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS avg_positive_value_proxy,

    ROUND(
        SUM(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS total_positive_value_proxy,

    ROUND(
        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        )::numeric,
        2
    ) AS observed_churned_value_proxy

FROM staging.train_raw;


/*==============================================================================
    02. OBSERVED CHURN BY CUSTOMER VALUE QUARTILE

    Quartiles create equally sized descriptive value groups.
==============================================================================*/

WITH value_ranked AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        NTILE(4) OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_quartile

    FROM staging.train_raw
),

classified AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        CASE
            WHEN value_quartile = 1
                THEN 'Q1 — Lowest Value'

            WHEN value_quartile = 2
                THEN 'Q2 — Lower-Mid Value'

            WHEN value_quartile = 3
                THEN 'Q3 — Upper-Mid Value'

            WHEN value_quartile = 4
                THEN 'Q4 — Highest Value'
        END AS value_group

    FROM value_ranked
),

summary AS (
    SELECT
        value_group,

        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        ) AS observed_churned_value_proxy

    FROM classified

    GROUP BY value_group
)

SELECT
    value_group,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        value_proxy::numeric,
        2
    ) AS value_proxy,

    ROUND(
        (
            100.0 * value_proxy /
            NULLIF(SUM(value_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(SUM(observed_churned_customers) OVER (), 0)
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        observed_churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy

FROM summary

ORDER BY
    value_proxy DESC;


/*==============================================================================
    03. HIGH-VALUE CUSTOMER CHURN EXPOSURE

    High Value = top 25% of customers by positive ARPU proxy.
==============================================================================*/

WITH ranked AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

classified AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group

    FROM ranked
),

summary AS (
    SELECT
        value_group,

        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        ) AS observed_churned_value_proxy

    FROM classified

    GROUP BY value_group
)

SELECT
    value_group,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        value_proxy::numeric,
        2
    ) AS value_proxy,

    ROUND(
        (
            100.0 * value_proxy /
            NULLIF(SUM(value_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(SUM(observed_churned_customers) OVER (), 0)
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        observed_churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy

FROM summary

ORDER BY
    value_proxy DESC;


/*==============================================================================
    04. ARPU DETERIORATION × OBSERVED CHURN
==============================================================================*/

WITH behavior AS (
    SELECT
        id,
        arpu_6,
        arpu_7,
        arpu_8,
        churn_probability,

        CASE
            WHEN arpu_6 IS NULL
              OR arpu_7 IS NULL
              OR arpu_8 IS NULL
                THEN 'Incomplete Trend'

            WHEN arpu_8 < arpu_7
             AND arpu_7 < arpu_6
                THEN 'Consistent Decline'

            WHEN arpu_8 < arpu_7
                THEN 'Recent Decline'

            WHEN arpu_8 >= arpu_7
             AND arpu_7 >= arpu_6
                THEN 'Stable / Improving'

            ELSE 'Mixed Trend'
        END AS arpu_trend

    FROM staging.train_raw
),

summary AS (
    SELECT
        arpu_trend,
        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers

    FROM behavior

    GROUP BY arpu_trend
)

SELECT
    arpu_trend,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(
                SUM(observed_churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary

ORDER BY
    observed_churn_rate_pct DESC;


/*==============================================================================
    05. RECHARGE DETERIORATION × OBSERVED CHURN
==============================================================================*/

WITH behavior AS (
    SELECT
        id,
        total_rech_amt_6,
        total_rech_amt_7,
        total_rech_amt_8,
        churn_probability,

        CASE
            WHEN total_rech_amt_6 IS NULL
              OR total_rech_amt_7 IS NULL
              OR total_rech_amt_8 IS NULL
                THEN 'Incomplete Trend'

            WHEN total_rech_amt_8 < total_rech_amt_7
             AND total_rech_amt_7 < total_rech_amt_6
                THEN 'Consistent Decline'

            WHEN total_rech_amt_8 < total_rech_amt_7
                THEN 'Recent Decline'

            WHEN total_rech_amt_8 >= total_rech_amt_7
             AND total_rech_amt_7 >= total_rech_amt_6
                THEN 'Stable / Improving'

            ELSE 'Mixed Trend'
        END AS recharge_trend

    FROM staging.train_raw
),

summary AS (
    SELECT
        recharge_trend,
        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers

    FROM behavior

    GROUP BY recharge_trend
)

SELECT
    recharge_trend,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(
                SUM(observed_churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary

ORDER BY
    observed_churn_rate_pct DESC;


/*==============================================================================
    06. ENGAGEMENT DETERIORATION × OBSERVED CHURN
==============================================================================*/

WITH behavior AS (
    SELECT
        id,
        total_og_mou_6,
        total_og_mou_7,
        total_og_mou_8,
        total_ic_mou_6,
        total_ic_mou_7,
        total_ic_mou_8,
        churn_probability,

        CASE
            WHEN total_og_mou_6 IS NULL
              OR total_og_mou_7 IS NULL
              OR total_og_mou_8 IS NULL
              OR total_ic_mou_6 IS NULL
              OR total_ic_mou_7 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Incomplete Trend'

            WHEN total_og_mou_8 < total_og_mou_7
             AND total_og_mou_7 < total_og_mou_6
             AND total_ic_mou_8 < total_ic_mou_7
             AND total_ic_mou_7 < total_ic_mou_6
                THEN 'Consistent Decline'

            WHEN total_og_mou_8 < total_og_mou_7
             AND total_ic_mou_8 < total_ic_mou_7
                THEN 'Recent Decline'

            WHEN total_og_mou_8 >= total_og_mou_7
             AND total_ic_mou_8 >= total_ic_mou_7
                THEN 'Stable / Improving'

            ELSE 'Mixed Trend'
        END AS engagement_trend

    FROM staging.train_raw
),

summary AS (
    SELECT
        engagement_trend,
        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers

    FROM behavior

    GROUP BY engagement_trend
)

SELECT
    engagement_trend,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(
                SUM(observed_churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary

ORDER BY
    observed_churn_rate_pct DESC;


/*==============================================================================
    07. HIGH VALUE × BEHAVIORAL DETERIORATION

    Stronger deterioration evidence:
        - ARPU decline
        - Recharge decline
        - Engagement decline

    Priority analysis requires multiple signals rather than a single
    arbitrary weighted score.
==============================================================================*/

WITH base AS (
    SELECT
        id,
        arpu_7,
        arpu_8,
        total_rech_amt_7,
        total_rech_amt_8,
        total_og_mou_7,
        total_og_mou_8,
        total_ic_mou_7,
        total_ic_mou_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

signals AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        CASE
            WHEN arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND arpu_8 < arpu_7
                THEN 1
            ELSE 0
        END AS arpu_decline,

        CASE
            WHEN total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_rech_amt_8 < total_rech_amt_7
                THEN 1
            ELSE 0
        END AS recharge_decline,

        CASE
            WHEN total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND total_og_mou_8 < total_og_mou_7
                THEN 1
            ELSE 0
        END AS outgoing_decline,

        CASE
            WHEN total_ic_mou_7 IS NOT NULL
             AND total_ic_mou_8 IS NOT NULL
             AND total_ic_mou_8 < total_ic_mou_7
                THEN 1
            ELSE 0
        END AS incoming_decline

    FROM base
),

classified AS (
    SELECT
        *,
        (
            arpu_decline
            + recharge_decline
            + outgoing_decline
            + incoming_decline
        ) AS deterioration_signal_count

    FROM signals
),

summary AS (
    SELECT
        value_group,

        CASE
            WHEN deterioration_signal_count >= 2
                THEN 'Multiple Deterioration Signals'

            WHEN deterioration_signal_count = 1
                THEN 'Single Deterioration Signal'

            ELSE 'Stable / No Detected Decline'
        END AS deterioration_group,

        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        ) AS observed_churned_value_proxy

    FROM classified

    GROUP BY
        value_group,
        CASE
            WHEN deterioration_signal_count >= 2
                THEN 'Multiple Deterioration Signals'

            WHEN deterioration_signal_count = 1
                THEN 'Single Deterioration Signal'

            ELSE 'Stable / No Detected Decline'
        END
)

SELECT
    value_group,
    deterioration_group,
    customers,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        value_proxy::numeric,
        2
    ) AS value_proxy,

    ROUND(
        (
            100.0 * value_proxy /
            NULLIF(SUM(value_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(
                SUM(observed_churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        observed_churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy

FROM summary

ORDER BY
    observed_churned_value_proxy DESC;


/*==============================================================================
    08. DESCRIPTIVE PRIORITY COHORT MATRIX

    Priority conditions:

        Priority Cohort:
            High Value + Multiple Deterioration Signals

        Elevated Cohort:
            High Value + Single Deterioration Signal

        Monitoring Cohort:
            Other Value + Multiple Deterioration Signals

        Lower Priority:
            Remaining customers

    These labels are descriptive only and are NOT predictive risk tiers.
==============================================================================*/

WITH base AS (
    SELECT
        id,
        arpu_7,
        arpu_8,
        total_rech_amt_7,
        total_rech_amt_8,
        total_og_mou_7,
        total_og_mou_8,
        total_ic_mou_7,
        total_ic_mou_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

signals AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        (
            CASE
                WHEN arpu_7 IS NOT NULL
                 AND arpu_8 IS NOT NULL
                 AND arpu_8 < arpu_7
                    THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN total_rech_amt_7 IS NOT NULL
                 AND total_rech_amt_8 IS NOT NULL
                 AND total_rech_amt_8 < total_rech_amt_7
                    THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN total_og_mou_7 IS NOT NULL
                 AND total_og_mou_8 IS NOT NULL
                 AND total_og_mou_8 < total_og_mou_7
                    THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN total_ic_mou_7 IS NOT NULL
                 AND total_ic_mou_8 IS NOT NULL
                 AND total_ic_mou_8 < total_ic_mou_7
                    THEN 1
                ELSE 0
            END
        ) AS deterioration_signal_count

    FROM base
),

classified AS (
    SELECT
        *,
        
        CASE
            WHEN value_group = 'High Value'
             AND deterioration_signal_count >= 2
                THEN 'Priority Cohort'

            WHEN value_group = 'High Value'
             AND deterioration_signal_count = 1
                THEN 'Elevated Cohort'

            WHEN value_group = 'Other Value'
             AND deterioration_signal_count >= 2
                THEN 'Monitoring Cohort'

            ELSE 'Lower Priority'
        END AS descriptive_priority

    FROM signals
),

summary AS (
    SELECT
        descriptive_priority,

        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        ) AS observed_churned_value_proxy

    FROM classified

    GROUP BY descriptive_priority
)

SELECT
    descriptive_priority,
    customers,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        value_proxy::numeric,
        2
    ) AS value_proxy,

    ROUND(
        (
            100.0 * value_proxy /
            NULLIF(SUM(value_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(
                SUM(observed_churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        observed_churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy

FROM summary

ORDER BY
    CASE descriptive_priority
        WHEN 'Priority Cohort' THEN 1
        WHEN 'Elevated Cohort' THEN 2
        WHEN 'Monitoring Cohort' THEN 3
        ELSE 4
    END;


/*==============================================================================
    09. EXECUTIVE PRIORITY RANKING

    Ranks descriptive cohorts by:
        1. Observed churned value exposure
        2. Value contribution
        3. Historical churn contribution

    This ranking is NOT an ML risk ranking.
==============================================================================*/

WITH base AS (
    SELECT
        id,
        arpu_7,
        arpu_8,
        total_rech_amt_7,
        total_rech_amt_8,
        total_og_mou_7,
        total_og_mou_8,
        total_ic_mou_7,
        total_ic_mou_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

classified AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        (
            CASE
                WHEN arpu_7 IS NOT NULL
                 AND arpu_8 IS NOT NULL
                 AND arpu_8 < arpu_7
                    THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN total_rech_amt_7 IS NOT NULL
                 AND total_rech_amt_8 IS NOT NULL
                 AND total_rech_amt_8 < total_rech_amt_7
                    THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN total_og_mou_7 IS NOT NULL
                 AND total_og_mou_8 IS NOT NULL
                 AND total_og_mou_8 < total_og_mou_7
                    THEN 1
                ELSE 0
            END
            +
            CASE
                WHEN total_ic_mou_7 IS NOT NULL
                 AND total_ic_mou_8 IS NOT NULL
                 AND total_ic_mou_8 < total_ic_mou_7
                    THEN 1
                ELSE 0
            END
        ) AS deterioration_signal_count

    FROM base
),

cohorts AS (
    SELECT
        CASE
            WHEN value_group = 'High Value'
             AND deterioration_signal_count >= 2
                THEN 'Priority Cohort'

            WHEN value_group = 'High Value'
             AND deterioration_signal_count = 1
                THEN 'Elevated Cohort'

            WHEN value_group = 'Other Value'
             AND deterioration_signal_count >= 2
                THEN 'Monitoring Cohort'

            ELSE 'Lower Priority'
        END AS descriptive_priority,

        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS observed_churned_customers,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        ) AS observed_churned_value_proxy

    FROM classified

    GROUP BY
        CASE
            WHEN value_group = 'High Value'
             AND deterioration_signal_count >= 2
                THEN 'Priority Cohort'

            WHEN value_group = 'High Value'
             AND deterioration_signal_count = 1
                THEN 'Elevated Cohort'

            WHEN value_group = 'Other Value'
             AND deterioration_signal_count >= 2
                THEN 'Monitoring Cohort'

            ELSE 'Lower Priority'
        END
)

SELECT
    ROW_NUMBER() OVER (
        ORDER BY observed_churned_value_proxy DESC
    ) AS exposure_rank,

    descriptive_priority,
    customers,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        value_proxy::numeric,
        2
    ) AS value_proxy,

    ROUND(
        (
            100.0 * value_proxy /
            NULLIF(SUM(value_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct,

    observed_churned_customers,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(
                SUM(observed_churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * observed_churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        observed_churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy

FROM cohorts

ORDER BY
    observed_churned_value_proxy DESC;


/*==============================================================================
    10. QUALITY GATE
==============================================================================*/

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT id) AS distinct_customer_ids,

    COUNT(*) - COUNT(DISTINCT id) AS duplicate_id_count,

    COUNT(*) FILTER (
        WHERE churn_probability IS NULL
    ) AS missing_target_count,

    COUNT(*) FILTER (
        WHERE arpu_8 IS NULL
    ) AS missing_arpu_8_count,

    COUNT(*) FILTER (
        WHERE total_rech_amt_8 IS NULL
    ) AS missing_recharge_8_count,

    CASE
        WHEN COUNT(*) = 69999
         AND COUNT(DISTINCT id) = 69999
         AND COUNT(*) - COUNT(DISTINCT id) = 0
         AND COUNT(*) FILTER (
                WHERE churn_probability IS NULL
             ) = 0
         AND COUNT(*) FILTER (
                WHERE arpu_8 IS NULL
             ) = 0
         AND COUNT(*) FILTER (
                WHERE total_rech_amt_8 IS NULL
             ) = 0
            THEN 'PASS'

        ELSE 'REVIEW'
    END AS quality_gate

FROM staging.train_raw;


/*==============================================================================
    END OF 10
==============================================================================*/