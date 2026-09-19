/*==============================================================================
    CHURNIQ — CUSTOMER SEGMENTATION ANALYSIS
    Script: 09_customer_segmentation_analysis.sql

    Purpose:
    Segment customers using observable customer characteristics and behavior
    available at the August 2014 prediction point.

    Business Lens:
    Understand customer groups, their value, engagement, recharge behavior,
    lifecycle characteristics, behavioral deterioration, and observed churn
    concentration.

    Analytical Role:
    Descriptive customer segmentation only.

    Important:
    - August customer state is the primary segmentation point.
    - July-to-August changes are used only for descriptive deterioration analysis.
    - August ARPU is a customer-value proxy, not booked revenue.
    - Observed churn uses the actual churn_probability label.
    - No model predictions are used.
    - No ML risk tiers are used.
    - No threshold logic is used.
    - No retention actions are assigned.
    - September data is not used as a predictor.
    - Missing values are not automatically interpreted as zero.
    - Small segments are flagged for interpretation.
==============================================================================*/


/*==============================================================================
    01. CUSTOMER SEGMENTATION UNIVERSE CHECK
==============================================================================*/

SELECT
    COUNT(*) AS total_customers,
    COUNT(DISTINCT id) AS distinct_customers,

    COUNT(*) - COUNT(DISTINCT id)
        AS duplicate_customer_ids,

    COUNT(*) FILTER (
        WHERE churn_probability IS NULL
    ) AS missing_churn_labels,

    COUNT(*) FILTER (
        WHERE arpu_8 IS NULL
    ) AS missing_arpu_8,

    COUNT(*) FILTER (
        WHERE total_rech_amt_8 IS NULL
    ) AS missing_recharge_8,

    COUNT(*) FILTER (
        WHERE total_og_mou_8 IS NULL
    ) AS missing_outgoing_voice_8,

    COUNT(*) FILTER (
        WHERE total_ic_mou_8 IS NULL
    ) AS missing_incoming_voice_8

FROM staging.train_raw;


/*==============================================================================
    02. LIFECYCLE / TENURE SEGMENTATION
==============================================================================*/

WITH classified AS (
    SELECT
        id,
        aon,
        churn_probability,

        CASE
            WHEN aon IS NULL
                THEN 'Tenure Unavailable'
            WHEN aon < 365
                THEN 'Under 1 Year'
            WHEN aon < 730
                THEN '1–2 Years'
            WHEN aon < 1095
                THEN '2–3 Years'
            WHEN aon < 1825
                THEN '3–5 Years'
            ELSE '5+ Years'
        END AS tenure_segment

    FROM staging.train_raw
),

summary AS (
    SELECT
        tenure_segment,
        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers,

        AVG(aon) AS avg_tenure_days

    FROM classified

    GROUP BY tenure_segment
)

SELECT
    tenure_segment,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        avg_tenure_days::numeric,
        0
    ) AS avg_tenure_days,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN tenure_segment = 'Under 1 Year' THEN 1
        WHEN tenure_segment = '1–2 Years' THEN 2
        WHEN tenure_segment = '2–3 Years' THEN 3
        WHEN tenure_segment = '3–5 Years' THEN 4
        WHEN tenure_segment = '5+ Years' THEN 5
        ELSE 6
    END;


/*==============================================================================
    03. CUSTOMER VALUE SEGMENTATION
==============================================================================*/

WITH classified AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        CASE
            WHEN arpu_8 IS NULL
                THEN 'Value Unavailable'
            WHEN arpu_8 < 0
                THEN 'Negative'
            WHEN arpu_8 = 0
                THEN 'Zero'
            WHEN arpu_8 < 100
                THEN 'Low'
            WHEN arpu_8 < 250
                THEN 'Medium'
            WHEN arpu_8 < 500
                THEN 'High'
            ELSE 'Very High'
        END AS value_segment

    FROM staging.train_raw
),

summary AS (
    SELECT
        value_segment,
        COUNT(*) AS customers,

        AVG(arpu_8) AS avg_arpu,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY value_segment
)

SELECT
    value_segment,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        avg_arpu::numeric,
        2
    ) AS avg_arpu,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN value_segment = 'Negative' THEN 1
        WHEN value_segment = 'Zero' THEN 2
        WHEN value_segment = 'Low' THEN 3
        WHEN value_segment = 'Medium' THEN 4
        WHEN value_segment = 'High' THEN 5
        WHEN value_segment = 'Very High' THEN 6
        ELSE 7
    END;


/*==============================================================================
    04. HIGH-VALUE CUSTOMER DEFINITION
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_8,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
)

SELECT
    COUNT(*) FILTER (
        WHERE value_percentile >= 0.75
    ) AS high_value_customers,

    ROUND(
        (
            100.0 *
            COUNT(*) FILTER (
                WHERE value_percentile >= 0.75
            ) / COUNT(*)
        )::numeric,
        2
    ) AS high_value_customer_share_pct,

    ROUND(
        SUM(
            CASE
                WHEN value_percentile >= 0.75
                    THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        )::numeric,
        2
    ) AS high_value_proxy,

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN value_percentile >= 0.75
                        THEN GREATEST(arpu_8, 0)
                    ELSE 0
                END
            )
            /
            NULLIF(
                SUM(GREATEST(arpu_8, 0)),
                0
            )
        )::numeric,
        2
    ) AS high_value_contribution_pct

FROM ranked_customers;


/*==============================================================================
    05. VOICE ENGAGEMENT SEGMENTATION
==============================================================================*/

WITH classified AS (
    SELECT
        id,
        total_og_mou_8,
        total_ic_mou_8,
        churn_probability,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Usage Unavailable'

            WHEN total_og_mou_8 < 10
             AND total_ic_mou_8 < 10
                THEN 'Very Low Engagement'

            WHEN total_og_mou_8 < 50
             AND total_ic_mou_8 < 50
                THEN 'Low Engagement'

            ELSE 'Active Engagement'
        END AS engagement_segment

    FROM staging.train_raw
),

summary AS (
    SELECT
        engagement_segment,
        COUNT(*) AS customers,

        AVG(total_og_mou_8) AS avg_outgoing_mou,
        AVG(total_ic_mou_8) AS avg_incoming_mou,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY engagement_segment
)

SELECT
    engagement_segment,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        avg_outgoing_mou::numeric,
        2
    ) AS avg_outgoing_mou,

    ROUND(
        avg_incoming_mou::numeric,
        2
    ) AS avg_incoming_mou,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN engagement_segment = 'Very Low Engagement' THEN 1
        WHEN engagement_segment = 'Low Engagement' THEN 2
        WHEN engagement_segment = 'Active Engagement' THEN 3
        ELSE 4
    END;


/*==============================================================================
    06. RECHARGE BEHAVIOR SEGMENTATION
==============================================================================*/

WITH classified AS (
    SELECT
        id,
        total_rech_amt_8,
        total_rech_num_8,
        churn_probability,

        CASE
            WHEN total_rech_amt_8 IS NULL
              OR total_rech_num_8 IS NULL
                THEN 'Recharge Unavailable'

            WHEN total_rech_amt_8 = 0
             AND total_rech_num_8 = 0
                THEN 'No Recharge'

            WHEN total_rech_amt_8 < 100
              OR total_rech_num_8 <= 2
                THEN 'Low Recharge'

            WHEN total_rech_amt_8 < 300
              OR total_rech_num_8 <= 5
                THEN 'Moderate Recharge'

            ELSE 'High Recharge'
        END AS recharge_segment

    FROM staging.train_raw
),

summary AS (
    SELECT
        recharge_segment,
        COUNT(*) AS customers,

        AVG(total_rech_amt_8) AS avg_recharge_amount,
        AVG(total_rech_num_8) AS avg_recharge_frequency,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY recharge_segment
)

SELECT
    recharge_segment,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        avg_recharge_amount::numeric,
        2
    ) AS avg_recharge_amount,

    ROUND(
        avg_recharge_frequency::numeric,
        2
    ) AS avg_recharge_frequency,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN recharge_segment = 'No Recharge' THEN 1
        WHEN recharge_segment = 'Low Recharge' THEN 2
        WHEN recharge_segment = 'Moderate Recharge' THEN 3
        WHEN recharge_segment = 'High Recharge' THEN 4
        ELSE 5
    END;


/*==============================================================================
    07. DATA ENGAGEMENT SEGMENTATION
==============================================================================*/

WITH classified AS (
    SELECT
        id,
        vol_2g_mb_8,
        vol_3g_mb_8,
        churn_probability,

        CASE
            WHEN vol_2g_mb_8 IS NULL
              AND vol_3g_mb_8 IS NULL
                THEN 'Data Usage Unavailable'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) = 0
                THEN 'No Data Engagement'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 100
                THEN 'Low Data Engagement'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 500
                THEN 'Moderate Data Engagement'

            ELSE 'High Data Engagement'
        END AS data_segment

    FROM staging.train_raw
),

summary AS (
    SELECT
        data_segment,
        COUNT(*) AS customers,

        AVG(
            COALESCE(vol_2g_mb_8, 0)
            + COALESCE(vol_3g_mb_8, 0)
        ) AS avg_total_data_mb,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY data_segment
)

SELECT
    data_segment,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        avg_total_data_mb::numeric,
        2
    ) AS avg_total_data_mb,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN data_segment = 'No Data Engagement' THEN 1
        WHEN data_segment = 'Low Data Engagement' THEN 2
        WHEN data_segment = 'Moderate Data Engagement' THEN 3
        WHEN data_segment = 'High Data Engagement' THEN 4
        ELSE 5
    END;


/*==============================================================================
    08. TENURE × VALUE SEGMENTATION
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        aon,
        arpu_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

classified AS (
    SELECT
        *,

        CASE
            WHEN aon IS NULL
                THEN 'Tenure Unavailable'
            WHEN aon < 365
                THEN 'Under 1 Year'
            WHEN aon < 730
                THEN '1–2 Years'
            WHEN aon < 1095
                THEN '2–3 Years'
            WHEN aon < 1825
                THEN '3–5 Years'
            ELSE '5+ Years'
        END AS tenure_segment,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group

    FROM ranked_customers
),

summary AS (
    SELECT
        tenure_segment,
        value_group,

        COUNT(*) AS customers,

        AVG(arpu_8) AS avg_arpu,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY
        tenure_segment,
        value_group
)

SELECT
    tenure_segment,
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
        avg_arpu::numeric,
        2
    ) AS avg_arpu,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN tenure_segment = 'Under 1 Year' THEN 1
        WHEN tenure_segment = '1–2 Years' THEN 2
        WHEN tenure_segment = '2–3 Years' THEN 3
        WHEN tenure_segment = '3–5 Years' THEN 4
        WHEN tenure_segment = '5+ Years' THEN 5
        ELSE 6
    END,
    CASE
        WHEN value_group = 'High Value' THEN 1
        ELSE 2
    END;


/*==============================================================================
    09. VALUE × ENGAGEMENT SEGMENTATION
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_8,
        total_og_mou_8,
        total_ic_mou_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

classified AS (
    SELECT
        *,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Engagement Unavailable'

            WHEN total_og_mou_8 < 10
             AND total_ic_mou_8 < 10
                THEN 'Very Low Engagement'

            WHEN total_og_mou_8 < 50
             AND total_ic_mou_8 < 50
                THEN 'Low Engagement'

            ELSE 'Active Engagement'
        END AS engagement_group

    FROM ranked_customers
),

summary AS (
    SELECT
        value_group,
        engagement_group,

        COUNT(*) AS customers,

        AVG(arpu_8) AS avg_arpu,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY
        value_group,
        engagement_group
)

SELECT
    value_group,
    engagement_group,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        avg_arpu::numeric,
        2
    ) AS avg_arpu,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN value_group = 'High Value' THEN 1
        ELSE 2
    END,
    CASE
        WHEN engagement_group = 'Very Low Engagement' THEN 1
        WHEN engagement_group = 'Low Engagement' THEN 2
        WHEN engagement_group = 'Active Engagement' THEN 3
        ELSE 4
    END;


/*==============================================================================
    10. RECHARGE × ENGAGEMENT SEGMENTATION
==============================================================================*/

WITH classified AS (
    SELECT
        id,
        total_rech_amt_8,
        total_rech_num_8,
        total_og_mou_8,
        total_ic_mou_8,
        churn_probability,

        CASE
            WHEN total_rech_amt_8 IS NULL
              OR total_rech_num_8 IS NULL
                THEN 'Recharge Unavailable'

            WHEN total_rech_amt_8 = 0
             AND total_rech_num_8 = 0
                THEN 'No Recharge'

            WHEN total_rech_amt_8 < 100
              OR total_rech_num_8 <= 2
                THEN 'Low Recharge'

            WHEN total_rech_amt_8 < 300
              OR total_rech_num_8 <= 5
                THEN 'Moderate Recharge'

            ELSE 'High Recharge'
        END AS recharge_group,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Engagement Unavailable'

            WHEN total_og_mou_8 < 10
             AND total_ic_mou_8 < 10
                THEN 'Very Low Engagement'

            WHEN total_og_mou_8 < 50
             AND total_ic_mou_8 < 50
                THEN 'Low Engagement'

            ELSE 'Active Engagement'
        END AS engagement_group

    FROM staging.train_raw
),

summary AS (
    SELECT
        recharge_group,
        engagement_group,

        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM classified

    GROUP BY
        recharge_group,
        engagement_group
)

SELECT
    recharge_group,
    engagement_group,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    churned_customers DESC;


/*==============================================================================
    11. VALUE × TEMPORAL DETERIORATION
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_7,
        arpu_8,
        total_rech_amt_7,
        total_rech_amt_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

classified AS (
    SELECT
        *,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        CASE
            WHEN arpu_7 IS NULL
              OR arpu_8 IS NULL
              OR total_rech_amt_7 IS NULL
              OR total_rech_amt_8 IS NULL
                THEN 'Incomplete Trend'

            WHEN arpu_8 < arpu_7
             AND total_rech_amt_8 < total_rech_amt_7
                THEN 'ARPU + Recharge Decline'

            WHEN arpu_8 < arpu_7
                THEN 'ARPU Decline'

            WHEN total_rech_amt_8 < total_rech_amt_7
                THEN 'Recharge Decline'

            ELSE 'Stable / Improving'
        END AS temporal_status

    FROM ranked_customers
),

summary AS (
    SELECT
        value_group,
        temporal_status,

        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy

    FROM classified

    GROUP BY
        value_group,
        temporal_status
)

SELECT
    value_group,
    temporal_status,
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
            NULLIF(
                SUM(value_proxy) OVER (),
                0
            )
        )::numeric,
        2
    ) AS value_contribution_pct,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    CASE
        WHEN value_group = 'High Value' THEN 1
        ELSE 2
    END,
    churned_customers DESC;


/*==============================================================================
    12. HIGH-VALUE DETERIORATION SUMMARY
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_7,
        arpu_8,
        total_rech_amt_7,
        total_rech_amt_8,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

high_value AS (
    SELECT
        *,
        
        CASE
            WHEN arpu_7 IS NULL
              OR arpu_8 IS NULL
              OR total_rech_amt_7 IS NULL
              OR total_rech_amt_8 IS NULL
                THEN 'Incomplete Trend'

            WHEN arpu_8 < arpu_7
             AND total_rech_amt_8 < total_rech_amt_7
                THEN 'ARPU + Recharge Decline'

            WHEN arpu_8 < arpu_7
                THEN 'ARPU Decline'

            WHEN total_rech_amt_8 < total_rech_amt_7
                THEN 'Recharge Decline'

            ELSE 'Stable / Improving'
        END AS deterioration_status

    FROM ranked_customers

    WHERE value_percentile >= 0.75
),

summary AS (
    SELECT
        deterioration_status,
        COUNT(*) AS customers,

        AVG(
            CASE
                WHEN arpu_7 <> 0
                THEN ((arpu_8 - arpu_7) / ABS(arpu_7)) * 100
            END
        ) AS avg_arpu_change_pct,

        AVG(
            CASE
                WHEN total_rech_amt_7 <> 0
                THEN (
                    (total_rech_amt_8 - total_rech_amt_7)
                    / ABS(total_rech_amt_7)
                ) * 100
            END
        ) AS avg_recharge_change_pct

    FROM high_value

    GROUP BY deterioration_status
)

SELECT
    deterioration_status,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS high_value_customer_share_pct,

    ROUND(
        avg_arpu_change_pct::numeric,
        2
    ) AS avg_arpu_change_pct,

    ROUND(
        avg_recharge_change_pct::numeric,
        2
    ) AS avg_recharge_change_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY customers DESC;


/*==============================================================================
    13. EXECUTIVE CUSTOMER SEGMENT VIEW
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_8,
        total_rech_amt_8,
        total_rech_num_8,
        total_og_mou_8,
        total_ic_mou_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

classified AS (
    SELECT
        *,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Engagement Unavailable'

            WHEN total_og_mou_8 < 10
             AND total_ic_mou_8 < 10
                THEN 'Very Low Engagement'

            WHEN total_og_mou_8 < 50
             AND total_ic_mou_8 < 50
                THEN 'Low Engagement'

            ELSE 'Active Engagement'
        END AS engagement_group,

        CASE
            WHEN total_rech_amt_8 IS NULL
              OR total_rech_num_8 IS NULL
                THEN 'Recharge Unavailable'

            WHEN total_rech_amt_8 = 0
             AND total_rech_num_8 = 0
                THEN 'No Recharge'

            WHEN total_rech_amt_8 < 100
              OR total_rech_num_8 <= 2
                THEN 'Low Recharge'

            WHEN total_rech_amt_8 < 300
              OR total_rech_num_8 <= 5
                THEN 'Moderate Recharge'

            ELSE 'High Recharge'
        END AS recharge_group

    FROM ranked_customers
),

executive_segments AS (
    SELECT
        *,

        CASE
            WHEN value_group = 'High Value'
             AND engagement_group IN (
                    'Very Low Engagement',
                    'Low Engagement'
                )
                THEN 'High Value / Low Engagement'

            WHEN value_group = 'High Value'
             AND recharge_group IN (
                    'No Recharge',
                    'Low Recharge'
                )
                THEN 'High Value / Low Recharge'

            WHEN value_group = 'High Value'
             AND engagement_group = 'Active Engagement'
             AND recharge_group IN (
                    'Moderate Recharge',
                    'High Recharge'
                )
                THEN 'High Value / Active'

            WHEN value_group = 'Other Value'
             AND engagement_group IN (
                    'Very Low Engagement',
                    'Low Engagement'
                )
                THEN 'Other Value / Low Engagement'

            WHEN value_group = 'Other Value'
             AND recharge_group IN (
                    'No Recharge',
                    'Low Recharge'
                )
                THEN 'Other Value / Low Recharge'

            ELSE 'Other / Mixed'
        END AS executive_segment

    FROM classified
),

summary AS (
    SELECT
        executive_segment,

        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM executive_segments

    GROUP BY executive_segment
)

SELECT
    executive_segment,
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
            NULLIF(
                SUM(value_proxy) OVER (),
                0
            )
        )::numeric,
        2
    ) AS value_contribution_pct,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(
                SUM(churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    value_proxy DESC;


/*==============================================================================
    14. SEGMENT PRIORITIZATION — DESCRIPTIVE ONLY
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_8,
        total_og_mou_8,
        total_ic_mou_8,
        total_rech_amt_8,
        total_rech_num_8,
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
        value_percentile,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Unavailable'

            WHEN total_og_mou_8 < 50
             AND total_ic_mou_8 < 50
                THEN 'Low Engagement'

            ELSE 'Active Engagement'
        END AS engagement_group,

        CASE
            WHEN total_rech_amt_8 IS NULL
              OR total_rech_num_8 IS NULL
                THEN 'Unavailable'

            WHEN total_rech_amt_8 < 100
              OR total_rech_num_8 <= 2
                THEN 'Low Recharge'

            ELSE 'Active Recharge'
        END AS recharge_group

    FROM ranked_customers
),

segments AS (
    SELECT
        CASE
            WHEN value_group = 'High Value'
             AND engagement_group = 'Low Engagement'
             AND recharge_group = 'Low Recharge'
                THEN 'High Value / Low Engagement + Low Recharge'

            WHEN value_group = 'High Value'
             AND engagement_group = 'Low Engagement'
                THEN 'High Value / Low Engagement'

            WHEN value_group = 'High Value'
             AND recharge_group = 'Low Recharge'
                THEN 'High Value / Low Recharge'

            WHEN value_group = 'High Value'
                THEN 'High Value / Active'

            WHEN engagement_group = 'Low Engagement'
             AND recharge_group = 'Low Recharge'
                THEN 'Other Value / Low Engagement + Low Recharge'

            WHEN engagement_group = 'Low Engagement'
                THEN 'Other Value / Low Engagement'

            WHEN recharge_group = 'Low Recharge'
                THEN 'Other Value / Low Recharge'

            ELSE 'Other / Active'
        END AS segment_name,

        arpu_8,
        churn_probability

    FROM classified
),

summary AS (
    SELECT
        segment_name,
        COUNT(*) AS customers,

        SUM(
            GREATEST(arpu_8, 0)
        ) AS value_proxy,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers

    FROM segments

    GROUP BY segment_name
)

SELECT
    segment_name,
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
            NULLIF(
                SUM(value_proxy) OVER (),
                0
            )
        )::numeric,
        2
    ) AS value_contribution_pct,

    churned_customers,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(
                SUM(churned_customers) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    CASE
        WHEN customers < 100
            THEN 'Small Segment — Interpret Carefully'
        ELSE 'Sufficient Segment Size'
    END AS segment_size_flag

FROM summary

ORDER BY
    value_proxy DESC;


/*==============================================================================
    15. FINAL ANALYTICAL QUALITY GATE
==============================================================================*/

WITH checks AS (
    SELECT
        COUNT(*) AS total_rows,

        COUNT(DISTINCT id) AS distinct_ids,

        COUNT(*) - COUNT(DISTINCT id)
            AS duplicate_id_count,

        COUNT(*) FILTER (
            WHERE churn_probability IS NULL
        ) AS missing_target_count,

        COUNT(*) FILTER (
            WHERE arpu_8 IS NULL
        ) AS missing_arpu_8_count,

        COUNT(*) FILTER (
            WHERE total_rech_amt_8 IS NULL
        ) AS missing_recharge_8_count

    FROM staging.train_raw
)

SELECT
    total_rows,
    distinct_ids,
    duplicate_id_count,
    missing_target_count,
    missing_arpu_8_count,
    missing_recharge_8_count,

    CASE
        WHEN total_rows = 69999
         AND distinct_ids = 69999
         AND duplicate_id_count = 0
         AND missing_target_count = 0
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS segmentation_quality_gate

FROM checks;


/*==============================================================================
    END OF 09 — CUSTOMER SEGMENTATION ANALYSIS
==============================================================================*/