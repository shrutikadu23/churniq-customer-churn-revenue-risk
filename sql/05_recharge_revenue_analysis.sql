 /*==============================================================================
    CHURNIQ — SQL BUSINESS ANALYSIS
    05_RECHARGE_REVENUE_ANALYSIS.SQL

    Purpose:
    Analyze customer monetization, recharge behavior, ARPU patterns,
    customer-value concentration, and recent commercial deterioration.

    Primary business question:
    How is customer value distributed, how are customers monetizing/recharging,
    and where are signs of commercial deterioration concentrated?

    Analytical boundaries:
    - ARPU is treated as a customer-value / revenue proxy.
    - Recharge amount represents recharge behavior, not booked revenue.
    - Observed churn is used for descriptive comparison only.
    - Missing values are not automatically treated as zero.
    - Negative ARPU observations are retained for investigation.
    - No model predictions, risk scores, thresholds, or retention tiers
      are used in this analysis.
==============================================================================*/


/*==============================================================================
    01. OVERALL REVENUE & RECHARGE SNAPSHOT

    Provides the overall monetization baseline before segmentation.
==============================================================================*/

SELECT
    COUNT(*) AS customers,

    ROUND(AVG(aon)::numeric, 2)
        AS avg_tenure_days,

    ROUND(AVG(arpu_8)::numeric, 2)
        AS avg_august_arpu,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY arpu_8)::numeric,
        2
    ) AS median_august_arpu,

    ROUND(AVG(total_rech_amt_8)::numeric, 2)
        AS avg_august_recharge_amount,

    ROUND(
        PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY total_rech_amt_8)::numeric,
        2
    ) AS median_august_recharge_amount,

    ROUND(AVG(total_rech_num_8)::numeric, 2)
        AS avg_august_recharge_frequency,

    ROUND(AVG(total_og_mou_8)::numeric, 2)
        AS avg_august_outgoing_mou,

    ROUND(AVG(total_ic_mou_8)::numeric, 2)
        AS avg_august_incoming_mou,

    COUNT(*) FILTER (
        WHERE arpu_8 < 0
    ) AS negative_arpu_customers,

    COUNT(*) FILTER (
        WHERE arpu_8 = 0
    ) AS zero_arpu_customers,

    COUNT(*) FILTER (
        WHERE total_rech_amt_8 = 0
    ) AS zero_recharge_amount_customers

FROM staging.train_raw;


/*==============================================================================
    02. AUGUST ARPU PROFILE

    ARPU bands provide a descriptive customer-value segmentation.

    Negative ARPU is kept separate rather than being automatically treated
    as invalid or removed.
==============================================================================*/

WITH arpu_profile AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative ARPU'
            WHEN arpu_8 = 0 THEN 'Zero ARPU'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS arpu_band,

        arpu_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        arpu_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(arpu_8) AS avg_arpu
    FROM arpu_profile
    GROUP BY arpu_band
)
SELECT
    arpu_band,

    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_arpu::numeric, 2)
        AS avg_arpu,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        (
            100.0 * churners
            / NULLIF(SUM(churners) OVER (), 0)
        )::numeric,
        2
    ) AS churn_contribution_pct

FROM summary
ORDER BY
    CASE
        WHEN arpu_band = 'Negative ARPU' THEN 0
        WHEN arpu_band = 'Zero ARPU' THEN 1
        WHEN arpu_band = 'Lower value' THEN 2
        WHEN arpu_band = 'Mid value' THEN 3
        WHEN arpu_band = 'Higher value' THEN 4
        ELSE 5
    END;


/*==============================================================================
    03. AUGUST RECHARGE VALUE PROFILE

    Recharge amount is analyzed as customer recharge behavior rather than
    direct revenue.
==============================================================================*/

WITH recharge_profile AS (
    SELECT
        CASE
            WHEN total_rech_amt_8 = 0 THEN 'Zero recharge amount'
            WHEN total_rech_amt_8 < 100 THEN 'Low recharge'
            WHEN total_rech_amt_8 < 300 THEN 'Moderate recharge'
            WHEN total_rech_amt_8 < 500 THEN 'High recharge'
            ELSE 'Very high recharge'
        END AS recharge_band,

        total_rech_amt_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        recharge_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(total_rech_amt_8) AS avg_recharge_amount
    FROM recharge_profile
    GROUP BY recharge_band
)
SELECT
    recharge_band,

    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_recharge_amount::numeric, 2)
        AS avg_recharge_amount,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        (
            100.0 * churners
            / NULLIF(SUM(churners) OVER (), 0)
        )::numeric,
        2
    ) AS churn_contribution_pct

FROM summary
ORDER BY
    CASE
        WHEN recharge_band = 'Zero recharge amount' THEN 0
        WHEN recharge_band = 'Low recharge' THEN 1
        WHEN recharge_band = 'Moderate recharge' THEN 2
        WHEN recharge_band = 'High recharge' THEN 3
        WHEN recharge_band = 'Very high recharge' THEN 4
        ELSE 5
    END;


/*==============================================================================
    04. AUGUST RECHARGE FREQUENCY PROFILE

    Measures how frequently customers recharged during August.
==============================================================================*/

WITH frequency_profile AS (
    SELECT
        CASE
            WHEN total_rech_num_8 <= 1 THEN '1 recharge'
            WHEN total_rech_num_8 <= 2 THEN '2 recharges'
            WHEN total_rech_num_8 <= 5 THEN '3-5 recharges'
            ELSE '6+ recharges'
        END AS frequency_band,

        total_rech_num_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        frequency_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(total_rech_num_8) AS avg_frequency
    FROM frequency_profile
    GROUP BY frequency_band
)
SELECT
    frequency_band,

    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_frequency::numeric, 2)
        AS avg_recharge_frequency,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        (
            100.0 * churners
            / NULLIF(SUM(churners) OVER (), 0)
        )::numeric,
        2
    ) AS churn_contribution_pct

FROM summary
ORDER BY
    CASE
        WHEN frequency_band = '1 recharge' THEN 0
        WHEN frequency_band = '2 recharges' THEN 1
        WHEN frequency_band = '3-5 recharges' THEN 2
        WHEN frequency_band = '6+ recharges' THEN 3
        ELSE 4
    END;


/*==============================================================================
    05. RECHARGE AMOUNT × FREQUENCY

    Connects recharge intensity with recharge frequency.

    Minimum-volume guardrail:
    Segments with fewer than 100 customers are excluded.
==============================================================================*/

WITH recharge_behavior AS (
    SELECT
        CASE
            WHEN total_rech_amt_8 = 0 THEN 'Zero amount'
            WHEN total_rech_amt_8 < 200 THEN 'Low amount'
            WHEN total_rech_amt_8 < 500 THEN 'Mid amount'
            ELSE 'High amount'
        END AS amount_band,

        CASE
            WHEN total_rech_num_8 <= 1 THEN '1 recharge'
            WHEN total_rech_num_8 <= 2 THEN '2 recharges'
            WHEN total_rech_num_8 <= 5 THEN '3-5 recharges'
            ELSE '6+ recharges'
        END AS frequency_band,

        total_rech_amt_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        amount_band,
        frequency_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(total_rech_amt_8) AS avg_recharge_amount
    FROM recharge_behavior
    GROUP BY amount_band, frequency_band
    HAVING COUNT(*) >= 100
)
SELECT
    amount_band,
    frequency_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_recharge_amount::numeric, 2)
        AS avg_recharge_amount,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    06. ARPU × RECHARGE VALUE

    Shows whether customer-value bands are associated with different
    recharge-value patterns.

    Minimum-volume guardrail:
    Segments with fewer than 100 customers are excluded.
==============================================================================*/

WITH combined_profile AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative ARPU'
            WHEN arpu_8 = 0 THEN 'Zero ARPU'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS arpu_band,

        CASE
            WHEN total_rech_amt_8 = 0 THEN 'Zero recharge'
            WHEN total_rech_amt_8 < 200 THEN 'Low recharge'
            WHEN total_rech_amt_8 < 500 THEN 'Mid recharge'
            ELSE 'High recharge'
        END AS recharge_band,

        arpu_8,
        total_rech_amt_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        arpu_band,
        recharge_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(arpu_8) AS avg_arpu,
        AVG(total_rech_amt_8) AS avg_recharge
    FROM combined_profile
    GROUP BY arpu_band, recharge_band
    HAVING COUNT(*) >= 100
)
SELECT
    arpu_band,
    recharge_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_arpu::numeric, 2)
        AS avg_arpu,

    ROUND(avg_recharge::numeric, 2)
        AS avg_recharge,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    07. ARPU × ENGAGEMENT

    Connects customer value with August voice engagement.

    Detailed engagement analysis is covered in SQL analysis #04.
    This section focuses on the commercial-value relationship.

    Minimum-volume guardrail:
    Segments with fewer than 100 customers are excluded.
==============================================================================*/

WITH customer_profile AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative value'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_ic_mou_8 IS NULL
                THEN 'Insufficient information'

            WHEN total_og_mou_8 = 0
             AND total_ic_mou_8 = 0
                THEN 'No voice activity'

            WHEN total_og_mou_8 + total_ic_mou_8 < 100
                THEN 'Low voice engagement'

            WHEN total_og_mou_8 + total_ic_mou_8 < 500
                THEN 'Moderate voice engagement'

            ELSE 'Higher voice engagement'
        END AS engagement_band,

        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        value_band,
        engagement_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners
    FROM customer_profile
    GROUP BY value_band, engagement_band
    HAVING COUNT(*) >= 100
)
SELECT
    value_band,
    engagement_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    08. JULY → AUGUST ARPU CHANGE

    Identifies recent customer-value deterioration.

    Percentage change is intentionally not calculated for customers whose
    July ARPU is non-positive because percentage change would be misleading.
==============================================================================*/

WITH arpu_change AS (
    SELECT
        CASE
            WHEN arpu_7 <= 0
             AND arpu_8 > 0
                THEN 'Moved from non-positive to positive'

            WHEN arpu_7 > 0
             AND arpu_8 <= 0
                THEN 'Dropped to non-positive'

            WHEN arpu_7 > 0
             AND arpu_8 < arpu_7 * 0.50
                THEN 'Severe decline'

            WHEN arpu_7 > 0
             AND arpu_8 < arpu_7 * 0.80
                THEN 'Moderate decline'

            WHEN arpu_7 > 0
             AND arpu_8 < arpu_7
                THEN 'Mild decline'

            WHEN arpu_7 > 0
             AND arpu_8 >= arpu_7 * 1.20
                THEN 'Strong increase'

            WHEN arpu_7 > 0
             AND arpu_8 > arpu_7
                THEN 'Increase'

            WHEN arpu_7 = arpu_8
                THEN 'No change'

            ELSE 'Insufficient information'
        END AS change_band,

        arpu_7,
        arpu_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        change_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(arpu_8 - arpu_7) AS avg_arpu_change
    FROM arpu_change
    GROUP BY change_band
)
SELECT
    change_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_arpu_change::numeric, 2)
        AS avg_arpu_change,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY
    CASE
        WHEN change_band = 'Severe decline' THEN 0
        WHEN change_band = 'Dropped to non-positive' THEN 1
        WHEN change_band = 'Moderate decline' THEN 2
        WHEN change_band = 'Mild decline' THEN 3
        WHEN change_band = 'No change' THEN 4
        WHEN change_band = 'Increase' THEN 5
        WHEN change_band = 'Strong increase' THEN 6
        WHEN change_band = 'Moved from non-positive to positive' THEN 7
        ELSE 8
    END;


/*==============================================================================
    09. JULY → AUGUST RECHARGE CHANGE

    Measures recent deterioration or improvement in recharge behavior.
==============================================================================*/

WITH recharge_change AS (
    SELECT
        CASE
            WHEN total_rech_amt_7 = 0
             AND total_rech_amt_8 > 0
                THEN 'Started recharging'

            WHEN total_rech_amt_7 > 0
             AND total_rech_amt_8 = 0
                THEN 'Dropped to zero'

            WHEN total_rech_amt_7 > 0
             AND total_rech_amt_8 < total_rech_amt_7 * 0.50
                THEN 'Severe decline'

            WHEN total_rech_amt_7 > 0
             AND total_rech_amt_8 < total_rech_amt_7 * 0.80
                THEN 'Moderate decline'

            WHEN total_rech_amt_7 > 0
             AND total_rech_amt_8 < total_rech_amt_7
                THEN 'Mild decline'

            WHEN total_rech_amt_8 > total_rech_amt_7
                THEN 'Increase'

            WHEN total_rech_amt_8 = total_rech_amt_7
                THEN 'No change'

            ELSE 'Insufficient information'
        END AS change_band,

        total_rech_amt_7,
        total_rech_amt_8,
        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        change_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        AVG(total_rech_amt_8 - total_rech_amt_7)
            AS avg_recharge_change
    FROM recharge_change
    GROUP BY change_band
)
SELECT
    change_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(avg_recharge_change::numeric, 2)
        AS avg_recharge_change,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY
    CASE
        WHEN change_band = 'Severe decline' THEN 0
        WHEN change_band = 'Dropped to zero' THEN 1
        WHEN change_band = 'Moderate decline' THEN 2
        WHEN change_band = 'Mild decline' THEN 3
        WHEN change_band = 'No change' THEN 4
        WHEN change_band = 'Increase' THEN 5
        WHEN change_band = 'Started recharging' THEN 6
        ELSE 7
    END;


/*==============================================================================
    10. RECHARGE CONSISTENCY — JUNE TO AUGUST

    Separates consistently positive, consistently zero, increasing,
    declining, and variable recharge behavior.
==============================================================================*/

WITH recharge_consistency AS (
    SELECT
        CASE
            WHEN total_rech_amt_6 > 0
             AND total_rech_amt_7 > 0
             AND total_rech_amt_8 > 0
                THEN 'Consistently positive'

            WHEN total_rech_amt_6 = 0
             AND total_rech_amt_7 = 0
             AND total_rech_amt_8 = 0
                THEN 'Consistently zero'

            WHEN total_rech_amt_6 <= total_rech_amt_7
             AND total_rech_amt_7 <= total_rech_amt_8
             AND total_rech_amt_6 < total_rech_amt_8
                THEN 'Increasing'

            WHEN total_rech_amt_6 >= total_rech_amt_7
             AND total_rech_amt_7 >= total_rech_amt_8
             AND total_rech_amt_6 > total_rech_amt_8
                THEN 'Declining'

            ELSE 'Variable'
        END AS consistency_band,

        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        consistency_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners
    FROM recharge_consistency
    GROUP BY consistency_band
)
SELECT
    consistency_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY
    CASE
        WHEN consistency_band = 'Declining' THEN 0
        WHEN consistency_band = 'Consistently zero' THEN 1
        WHEN consistency_band = 'Variable' THEN 2
        WHEN consistency_band = 'Consistently positive' THEN 3
        WHEN consistency_band = 'Increasing' THEN 4
        ELSE 5
    END;


/*==============================================================================
    11. ARPU-BASED VALUE CONCENTRATION

    Uses non-negative ARPU as a concentration proxy.

    This prevents negative ARPU observations from distorting value-share
    calculations.
==============================================================================*/

WITH value_bands AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative ARPU'
            WHEN arpu_8 = 0 THEN 'Zero ARPU'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        GREATEST(arpu_8, 0) AS nonnegative_arpu_proxy,

        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        value_band,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        SUM(nonnegative_arpu_proxy) AS total_arpu_proxy
    FROM value_bands
    GROUP BY value_band
)
SELECT
    value_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(total_arpu_proxy::numeric, 2)
        AS total_arpu_value_proxy,

    ROUND(
        (
            100.0 * total_arpu_proxy
            / NULLIF(SUM(total_arpu_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS arpu_value_contribution_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY total_arpu_proxy DESC;


/*==============================================================================
    12. HIGH-VALUE CUSTOMER CONCENTRATION

    Definition:
    Higher-value customer = August ARPU >= 500

    This is an analytical segmentation threshold, not an official
    customer-management tier.
==============================================================================*/

WITH high_value AS (
    SELECT
        CASE
            WHEN arpu_8 >= 500
                THEN 'Higher-value customer'
            ELSE 'Below higher-value threshold'
        END AS value_segment,

        GREATEST(arpu_8, 0) AS nonnegative_arpu_proxy,

        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        value_segment,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners,
        SUM(nonnegative_arpu_proxy) AS total_arpu_proxy
    FROM high_value
    GROUP BY value_segment
)
SELECT
    value_segment,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(total_arpu_proxy::numeric, 2)
        AS total_arpu_value_proxy,

    ROUND(
        (
            100.0 * total_arpu_proxy
            / NULLIF(SUM(total_arpu_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY
    CASE
        WHEN value_segment = 'Higher-value customer' THEN 0
        ELSE 1
    END;


/*==============================================================================
    13. HIGH-VALUE + DECLINING RECHARGE

    Identifies a commercially important descriptive segment:
    higher-value customers whose recharge amount declined from July to August.

    This is a behavioral signal, NOT a predictive risk tier.
==============================================================================*/

WITH customer_signal AS (
    SELECT
        CASE
            WHEN arpu_8 >= 500
             AND total_rech_amt_7 > 0
             AND total_rech_amt_8 < total_rech_amt_7
                THEN 'Higher-value + declining recharge'

            WHEN arpu_8 >= 500
             AND total_rech_amt_7 > 0
             AND total_rech_amt_8 >= total_rech_amt_7
                THEN 'Higher-value + stable/increasing recharge'

            WHEN arpu_8 < 500
             AND total_rech_amt_7 > 0
             AND total_rech_amt_8 < total_rech_amt_7
                THEN 'Below higher-value + declining recharge'

            ELSE 'Other'
        END AS customer_signal,

        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        customer_signal,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners
    FROM customer_signal
    GROUP BY customer_signal
    HAVING COUNT(*) >= 100
)
SELECT
    customer_signal,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY
    CASE
        WHEN customer_signal = 'Higher-value + declining recharge'
            THEN 0
        WHEN customer_signal = 'Higher-value + stable/increasing recharge'
            THEN 1
        WHEN customer_signal = 'Below higher-value + declining recharge'
            THEN 2
        ELSE 3
    END;


/*==============================================================================
    14. RECHARGE DATA-SEMANTICS CHECK

    Important dataset-specific validation:

    A customer may have a positive recharge frequency while showing
    zero total recharge amount.

    We do NOT automatically classify this as an error. The purpose of this
    section is to quantify the pattern so its underlying data semantics can
    be documented and investigated appropriately.
==============================================================================*/

WITH recharge_semantics AS (
    SELECT
        CASE
            WHEN total_rech_num_8 > 0
             AND total_rech_amt_8 = 0
                THEN 'Positive frequency + zero amount'

            WHEN total_rech_num_8 > 0
             AND total_rech_amt_8 > 0
                THEN 'Positive frequency + positive amount'

            WHEN total_rech_num_8 = 0
             AND total_rech_amt_8 = 0
                THEN 'Zero frequency + zero amount'

            WHEN total_rech_num_8 = 0
             AND total_rech_amt_8 > 0
                THEN 'Zero frequency + positive amount'

            ELSE 'Other / insufficient information'
        END AS recharge_semantic_group,

        churn_probability
    FROM staging.train_raw
),
summary AS (
    SELECT
        recharge_semantic_group,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners
    FROM recharge_semantics
    GROUP BY recharge_semantic_group
)
SELECT
    recharge_semantic_group,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    15. REVENUE / CUSTOMER-VALUE CONTRIBUTION BY VALUE BAND

    Executive-oriented view.

    ARPU is used as a value proxy; these figures should not be interpreted
    as booked revenue.
==============================================================================*/

WITH value_bands AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative ARPU'
            WHEN arpu_8 = 0 THEN 'Zero ARPU'
            WHEN arpu_8 < 200 THEN 'Lower value'
            WHEN arpu_8 < 500 THEN 'Mid value'
            ELSE 'Higher value'
        END AS value_band,

        GREATEST(arpu_8, 0) AS arpu_proxy
    FROM staging.train_raw
),
summary AS (
    SELECT
        value_band,
        COUNT(*) AS customers,
        SUM(arpu_proxy) AS arpu_value
    FROM value_bands
    GROUP BY value_band
)
SELECT
    value_band,
    customers,

    ROUND(
        100.0 * customers / SUM(customers) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(arpu_value::numeric, 2)
        AS arpu_value_proxy,

    ROUND(
        (
            100.0 * arpu_value
            / NULLIF(SUM(arpu_value) OVER (), 0)
        )::numeric,
        2
    ) AS value_contribution_pct

FROM summary
ORDER BY arpu_value DESC;


/*==============================================================================
    16. EXECUTIVE REVENUE SIGNALS

    Minimum-volume guardrail:
    Only signals affecting at least 100 customers are included.

    These are descriptive business signals for management review.
    They are NOT causal conclusions and NOT model predictions.
==============================================================================*/

WITH signal_base AS (

    /* Higher-value customer segment */
    SELECT
        'Customer value' AS signal_dimension,
        'Higher-value customers' AS signal,
        COUNT(*) AS customers,
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners
    FROM staging.train_raw
    WHERE arpu_8 >= 500

    UNION ALL

    /* Zero recharge amount */
    SELECT
        'Recharge behavior',
        'Zero August recharge amount',
        COUNT(*),
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )
    FROM staging.train_raw
    WHERE total_rech_amt_8 = 0

    UNION ALL

    /* Higher-value + declining recharge */
    SELECT
        'Combined value signal',
        'Higher-value + declining recharge',
        COUNT(*),
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )
    FROM staging.train_raw
    WHERE arpu_8 >= 500
      AND total_rech_amt_7 > 0
      AND total_rech_amt_8 < total_rech_amt_7

    UNION ALL

    /* Severe ARPU deterioration */
    SELECT
        'ARPU trend',
        'ARPU declined by more than 50%',
        COUNT(*),
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )
    FROM staging.train_raw
    WHERE arpu_7 > 0
      AND arpu_8 < arpu_7 * 0.50

    UNION ALL

    /* Severe recharge deterioration */
    SELECT
        'Recharge trend',
        'Recharge declined by more than 50%',
        COUNT(*),
        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )
    FROM staging.train_raw
    WHERE total_rech_amt_7 > 0
      AND total_rech_amt_8 < total_rech_amt_7 * 0.50
),
filtered AS (
    SELECT *
    FROM signal_base
    WHERE customers >= 100
)
SELECT
    signal_dimension,
    signal,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER (
                PARTITION BY signal_dimension
            )
        )::numeric,
        2
    ) AS customer_share_within_dimension_pct,

    churners,

    ROUND(
        100.0 * churners / NULLIF(customers, 0),
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        (
            100.0 * churners
            / NULLIF(
                SUM(churners) OVER (
                    PARTITION BY signal_dimension
                ),
                0
            )
        )::numeric,
        2
    ) AS churn_contribution_within_dimension_pct

FROM filtered
ORDER BY
    signal_dimension,
    observed_churn_rate_pct DESC;


/*==============================================================================
    17. ANALYTICAL QUALITY GATE

    Validates that the revenue/recharge analysis is operating on the full
    training population and that key August monetization fields are usable.
==============================================================================*/

SELECT
    COUNT(*) AS row_count,

    COUNT(DISTINCT id) AS distinct_customer_ids,

    COUNT(*) - COUNT(DISTINCT id)
        AS duplicate_id_count,

    COUNT(*) FILTER (
        WHERE arpu_8 IS NULL
    ) AS missing_august_arpu,

    COUNT(*) FILTER (
        WHERE total_rech_amt_8 IS NULL
    ) AS missing_august_recharge_amount,

    COUNT(*) FILTER (
        WHERE total_rech_num_8 IS NULL
    ) AS missing_august_recharge_frequency,

    COUNT(*) FILTER (
        WHERE total_rech_amt_8 < 0
    ) AS negative_recharge_amounts,

    CASE
        WHEN COUNT(*) = 69999
         AND COUNT(DISTINCT id) = 69999

         AND COUNT(*) - COUNT(DISTINCT id) = 0

         AND COUNT(*) FILTER (
                WHERE arpu_8 IS NULL
             ) = 0

         AND COUNT(*) FILTER (
                WHERE total_rech_amt_8 IS NULL
             ) = 0

         AND COUNT(*) FILTER (
                WHERE total_rech_num_8 IS NULL
             ) = 0

         AND COUNT(*) FILTER (
                WHERE total_rech_amt_8 < 0
             ) = 0

        THEN 'PASS'

        ELSE 'REVIEW'
    END AS analytical_quality_gate

FROM staging.train_raw;


/*==============================================================================
    END OF RECHARGE & REVENUE ANALYSIS
==============================================================================*/