/*==============================================================================
    CHURNIQ — REVENUE & CUSTOMER-VALUE EXPOSURE ANALYSIS
    Script: 08_revenue_exposure_analysis.sql

    Purpose:
    Analyze customer-value concentration, observed churn exposure, and
    scenario-based potential value exposure using August customer economics.

    Business Lens:
    Identify where customer value is concentrated, where observed churn is
    concentrated, and which customer groups may represent greater potential
    value exposure.

    Important:
    - August ARPU is used as a customer-value proxy, not booked revenue.
    - Observed churn exposure uses the actual churn_probability label.
    - Scenario-based exposure is hypothetical and not guaranteed revenue loss.
    - Negative ARPU values are retained for analytical transparency.
    - Negative ARPU is clipped to zero ONLY for exposure calculations.
    - Missing values are not automatically interpreted as zero.
    - September data is not used as a predictor.
    - This script is descriptive business analysis only.
    - No model predictions, thresholds, risk tiers, or retention actions are used.
==============================================================================*/


/*==============================================================================
    01. OVERALL REVENUE & CUSTOMER-VALUE SNAPSHOT
==============================================================================*/

SELECT
    COUNT(*) AS total_customers,
    COUNT(DISTINCT id) AS distinct_customers,

    ROUND(AVG(arpu_8)::numeric, 2) AS avg_august_arpu,
    ROUND(MIN(arpu_8)::numeric, 2) AS min_august_arpu,
    ROUND(MAX(arpu_8)::numeric, 2) AS max_august_arpu,

    ROUND(
        SUM(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS total_positive_august_value_proxy,

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
    02. AUGUST ARPU DISTRIBUTION
==============================================================================*/

WITH value_distribution AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative'
            WHEN arpu_8 = 0 THEN 'Zero'
            WHEN arpu_8 < 100 THEN 'Low'
            WHEN arpu_8 < 250 THEN 'Medium'
            WHEN arpu_8 < 500 THEN 'High'
            ELSE 'Very High'
        END AS value_band,

        arpu_8,
        churn_probability

    FROM staging.train_raw
),

value_summary AS (
    SELECT
        value_band,
        COUNT(*) AS customers,

        AVG(arpu_8) AS avg_arpu,

        AVG(
            CASE
                WHEN churn_probability = 1
                THEN 1.0
                ELSE 0.0
            END
        ) AS churn_rate

    FROM value_distribution

    GROUP BY value_band
)

SELECT
    value_band,
    customers,

    ROUND(
        (
            100.0 * customers /
            SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(avg_arpu::numeric, 2) AS avg_arpu,

    ROUND(
        (100.0 * churn_rate)::numeric,
        2
    ) AS observed_churn_rate_pct

FROM value_summary

ORDER BY
    CASE
        WHEN value_band = 'Negative' THEN 1
        WHEN value_band = 'Zero' THEN 2
        WHEN value_band = 'Low' THEN 3
        WHEN value_band = 'Medium' THEN 4
        WHEN value_band = 'High' THEN 5
        WHEN value_band = 'Very High' THEN 6
    END;


/*==============================================================================
    03. CUSTOMER VALUE BANDS & OBSERVED CHURN
==============================================================================*/

WITH value_bands AS (
    SELECT
        arpu_8,

        CASE
            WHEN arpu_8 < 0 THEN 'Negative'
            WHEN arpu_8 = 0 THEN 'Zero'
            WHEN arpu_8 < 100 THEN 'Low'
            WHEN arpu_8 < 250 THEN 'Medium'
            WHEN arpu_8 < 500 THEN 'High'
            ELSE 'Very High'
        END AS value_band,

        churn_probability

    FROM staging.train_raw
)

SELECT
    value_band,
    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        AVG(arpu_8)::numeric,
        2
    ) AS avg_arpu,

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            ) / COUNT(*)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM value_bands

GROUP BY value_band

ORDER BY
    CASE
        WHEN value_band = 'Negative' THEN 1
        WHEN value_band = 'Zero' THEN 2
        WHEN value_band = 'Low' THEN 3
        WHEN value_band = 'Medium' THEN 4
        WHEN value_band = 'High' THEN 5
        WHEN value_band = 'Very High' THEN 6
    END;


/*==============================================================================
    04. CUSTOMER VALUE CONTRIBUTION BY BAND
==============================================================================*/

WITH value_bands AS (
    SELECT
        CASE
            WHEN arpu_8 < 0 THEN 'Negative'
            WHEN arpu_8 = 0 THEN 'Zero'
            WHEN arpu_8 < 100 THEN 'Low'
            WHEN arpu_8 < 250 THEN 'Medium'
            WHEN arpu_8 < 500 THEN 'High'
            ELSE 'Very High'
        END AS value_band,

        GREATEST(arpu_8, 0) AS value_proxy

    FROM staging.train_raw
),

band_values AS (
    SELECT
        value_band,
        COUNT(*) AS customers,
        SUM(value_proxy) AS value_proxy

    FROM value_bands

    GROUP BY value_band
)

SELECT
    value_band,
    customers,

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
    ) AS value_contribution_pct

FROM band_values

ORDER BY value_proxy DESC;


/*==============================================================================
    05. OBSERVED CHURN VALUE EXPOSURE
==============================================================================*/

SELECT
    COUNT(*) AS churned_customers,

    ROUND(
        SUM(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS observed_churned_value_proxy,

    ROUND(
        AVG(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS avg_churned_value_proxy,

    ROUND(
        MIN(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS min_churned_value_proxy,

    ROUND(
        MAX(GREATEST(arpu_8, 0))::numeric,
        2
    ) AS max_churned_value_proxy

FROM staging.train_raw

WHERE churn_probability = 1;


/*==============================================================================
    06. CHURN CONTRIBUTION BY VALUE BAND
==============================================================================*/

WITH value_bands AS (
    SELECT
        arpu_8,
        churn_probability,

        CASE
            WHEN arpu_8 < 0 THEN 'Negative'
            WHEN arpu_8 = 0 THEN 'Zero'
            WHEN arpu_8 < 100 THEN 'Low'
            WHEN arpu_8 < 250 THEN 'Medium'
            WHEN arpu_8 < 500 THEN 'High'
            ELSE 'Very High'
        END AS value_band

    FROM staging.train_raw
),

band_churn AS (
    SELECT
        value_band,

        COUNT(*) AS customers,

        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        ) AS churned_customers,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN GREATEST(arpu_8, 0)
                ELSE 0
            END
        ) AS churned_value_proxy

    FROM value_bands

    GROUP BY value_band
)

SELECT
    value_band,
    customers,
    churned_customers,

    ROUND(
        churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy,

    ROUND(
        (
            100.0 * churned_customers /
            NULLIF(SUM(churned_customers) OVER (), 0)
        )::numeric,
        2
    ) AS churn_customer_contribution_pct,

    ROUND(
        (
            100.0 * churned_value_proxy /
            NULLIF(SUM(churned_value_proxy) OVER (), 0)
        )::numeric,
        2
    ) AS churn_value_contribution_pct

FROM band_churn

ORDER BY churned_value_proxy DESC;


/*==============================================================================
    07. VALUE CONCENTRATION / PARETO VIEW
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        GREATEST(arpu_8, 0) AS value_proxy,

        NTILE(10) OVER (
            ORDER BY GREATEST(arpu_8, 0) DESC, id
        ) AS value_decile

    FROM staging.train_raw
),

decile_totals AS (
    SELECT
        value_decile,
        COUNT(*) AS customers,
        SUM(value_proxy) AS value_proxy

    FROM ranked_customers

    GROUP BY value_decile
),

decile_cumulative AS (
    SELECT
        value_decile,
        customers,
        value_proxy,

        SUM(value_proxy) OVER (
            ORDER BY value_decile
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_value_proxy,

        SUM(value_proxy) OVER () AS total_value_proxy

    FROM decile_totals
)

SELECT
    value_decile,
    customers,

    ROUND(
        value_proxy::numeric,
        2
    ) AS value_proxy,

    ROUND(
        (
            100.0 * value_proxy /
            NULLIF(total_value_proxy, 0)
        )::numeric,
        2
    ) AS value_share_pct,

    ROUND(
        (
            100.0 * cumulative_value_proxy /
            NULLIF(total_value_proxy, 0)
        )::numeric,
        2
    ) AS cumulative_value_share_pct

FROM decile_cumulative

ORDER BY value_decile;


/*==============================================================================
    08. HIGH-VALUE CUSTOMER CONCENTRATION
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_8,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

high_value_summary AS (
    SELECT
        COUNT(*) AS high_value_customers,

        SUM(GREATEST(arpu_8, 0)) AS high_value_proxy

    FROM ranked_customers

    WHERE value_percentile >= 0.75
)

SELECT
    high_value_customers,

    ROUND(
        (
            100.0 * high_value_customers /
            (SELECT COUNT(*) FROM staging.train_raw)
        )::numeric,
        2
    ) AS customer_share_pct,

    ROUND(
        high_value_proxy::numeric,
        2
    ) AS high_value_proxy,

    ROUND(
        (
            100.0 * high_value_proxy /
            NULLIF(
                (
                    SELECT SUM(GREATEST(arpu_8, 0))
                    FROM staging.train_raw
                ),
                0
            )
        )::numeric,
        2
    ) AS value_share_pct

FROM high_value_summary;


/*==============================================================================
    09. HIGH-VALUE CHURN CONCENTRATION
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

high_value_churn AS (
    SELECT
        COUNT(*) AS high_value_churned_customers,

        SUM(GREATEST(arpu_8, 0)) AS high_value_churned_proxy

    FROM ranked_customers

    WHERE value_percentile >= 0.75
      AND churn_probability = 1
)

SELECT
    high_value_churned_customers,

    ROUND(
        (
            100.0 * high_value_churned_customers /
            NULLIF(
                (
                    SELECT COUNT(*)
                    FROM staging.train_raw
                    WHERE churn_probability = 1
                ),
                0
            )
        )::numeric,
        2
    ) AS share_of_churned_customers_pct,

    ROUND(
        high_value_churned_proxy::numeric,
        2
    ) AS high_value_churned_proxy,

    ROUND(
        (
            100.0 * high_value_churned_proxy /
            NULLIF(
                (
                    SELECT SUM(GREATEST(arpu_8, 0))
                    FROM staging.train_raw
                    WHERE churn_probability = 1
                ),
                0
            )
        )::numeric,
        2
    ) AS share_of_churned_value_proxy_pct

FROM high_value_churn;


/*==============================================================================
    10. HIGH-VALUE CUSTOMERS WITH DECLINING ARPU
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        arpu_7,
        arpu_8,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

high_value_decline AS (
    SELECT
        COUNT(*) AS declining_customers,

        AVG(
            CASE
                WHEN arpu_7 <> 0
                THEN ((arpu_8 - arpu_7) / ABS(arpu_7)) * 100
            END
        ) AS avg_arpu_change_pct

    FROM ranked_customers

    WHERE value_percentile >= 0.75
      AND arpu_7 IS NOT NULL
      AND arpu_8 IS NOT NULL
      AND arpu_8 < arpu_7
)

SELECT
    declining_customers,

    ROUND(
        (
            100.0 * declining_customers /
            NULLIF(
                (
                    SELECT COUNT(*)
                    FROM ranked_customers
                    WHERE value_percentile >= 0.75
                ),
                0
            )
        )::numeric,
        2
    ) AS share_of_high_value_customers_pct,

    ROUND(
        avg_arpu_change_pct::numeric,
        2
    ) AS avg_arpu_change_pct

FROM high_value_decline;


/*==============================================================================
    11. HIGH-VALUE CUSTOMERS WITH DECLINING RECHARGE
==============================================================================*/

WITH ranked_customers AS (
    SELECT
        id,
        total_rech_amt_7,
        total_rech_amt_8,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

high_value_decline AS (
    SELECT
        COUNT(*) AS declining_customers,

        AVG(
            CASE
                WHEN total_rech_amt_7 <> 0
                THEN (
                    (total_rech_amt_8 - total_rech_amt_7)
                    / ABS(total_rech_amt_7)
                ) * 100
            END
        ) AS avg_recharge_change_pct

    FROM ranked_customers

    WHERE value_percentile >= 0.75
      AND total_rech_amt_7 IS NOT NULL
      AND total_rech_amt_8 IS NOT NULL
      AND total_rech_amt_8 < total_rech_amt_7
)

SELECT
    declining_customers,

    ROUND(
        (
            100.0 * declining_customers /
            NULLIF(
                (
                    SELECT COUNT(*)
                    FROM ranked_customers
                    WHERE value_percentile >= 0.75
                ),
                0
            )
        )::numeric,
        2
    ) AS share_of_high_value_customers_pct,

    ROUND(
        avg_recharge_change_pct::numeric,
        2
    ) AS avg_recharge_change_pct

FROM high_value_decline;


/*==============================================================================
    12. VALUE × RECHARGE DETERIORATION
==============================================================================*/

WITH customer_profile AS (
    SELECT
        id,
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
            WHEN total_rech_amt_7 IS NULL
              OR total_rech_amt_8 IS NULL
                THEN 'Recharge Trend Unavailable'

            WHEN total_rech_amt_8 < total_rech_amt_7
                THEN 'Declining Recharge'

            ELSE 'Stable / Increasing'

        END AS recharge_status

    FROM customer_profile
)

SELECT
    value_group,
    recharge_status,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            ) / COUNT(*)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM classified

GROUP BY value_group, recharge_status

ORDER BY
    CASE
        WHEN value_group = 'High Value' THEN 1
        ELSE 2
    END,
    recharge_status;


/*==============================================================================
    13. VALUE × ENGAGEMENT EXPOSURE
==============================================================================*/

WITH customer_profile AS (
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
                THEN 'Usage Unavailable'

            WHEN total_og_mou_8 < 10
             AND total_ic_mou_8 < 10
                THEN 'Very Low Voice Engagement'

            WHEN total_og_mou_8 < 50
             AND total_ic_mou_8 < 50
                THEN 'Low Voice Engagement'

            ELSE 'Active Voice Engagement'

        END AS engagement_group

    FROM customer_profile
)

SELECT
    value_group,
    engagement_group,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            ) / COUNT(*)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM classified

GROUP BY value_group, engagement_group

ORDER BY
    CASE
        WHEN value_group = 'High Value' THEN 1
        ELSE 2
    END,
    engagement_group;


/*==============================================================================
    14. TENURE × VALUE EXPOSURE
==============================================================================*/

WITH customer_profile AS (
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
            WHEN aon < 365 THEN 'Under 1 Year'
            WHEN aon < 730 THEN '1–2 Years'
            WHEN aon < 1095 THEN '2–3 Years'
            WHEN aon < 1825 THEN '3–5 Years'
            ELSE '5+ Years'
        END AS tenure_band,

        CASE
            WHEN value_percentile >= 0.75
                THEN 'High Value'
            ELSE 'Other Value'
        END AS value_group

    FROM customer_profile
)

SELECT
    tenure_band,
    value_group,

    COUNT(*) AS customers,

    ROUND(
        AVG(arpu_8)::numeric,
        2
    ) AS avg_arpu,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            ) / COUNT(*)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM classified

GROUP BY tenure_band, value_group

ORDER BY
    CASE
        WHEN tenure_band = 'Under 1 Year' THEN 1
        WHEN tenure_band = '1–2 Years' THEN 2
        WHEN tenure_band = '2–3 Years' THEN 3
        WHEN tenure_band = '3–5 Years' THEN 4
        ELSE 5
    END,
    value_group;


/*==============================================================================
    15. VALUE × TEMPORAL DETERIORATION
==============================================================================*/

WITH customer_profile AS (
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

    FROM customer_profile
)

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

    ROUND(
        (
            100.0 *
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            ) / COUNT(*)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM classified

GROUP BY value_group, temporal_status

ORDER BY
    CASE
        WHEN value_group = 'High Value' THEN 1
        ELSE 2
    END,
    temporal_status;


/*==============================================================================
    16. SCENARIO-BASED POTENTIAL VALUE EXPOSURE
==============================================================================*/

WITH scenario_base AS (
    SELECT
        id,
        churn_probability,
        GREATEST(arpu_8, 0) AS value_proxy

    FROM staging.train_raw
),

scenario_summary AS (
    SELECT
        SUM(
            churn_probability * value_proxy
        ) AS probability_weighted_exposure,

        SUM(
            CASE
                WHEN churn_probability = 1
                THEN value_proxy
                ELSE 0
            END
        ) AS observed_churned_value_proxy

    FROM scenario_base
)

SELECT
    ROUND(
        probability_weighted_exposure::numeric,
        2
    ) AS probability_weighted_value_exposure,

    ROUND(
        observed_churned_value_proxy::numeric,
        2
    ) AS observed_churned_value_proxy,

    ROUND(
        (observed_churned_value_proxy * 0.75)::numeric,
        2
    ) AS conservative_recovery_scenario,

    ROUND(
        (observed_churned_value_proxy * 0.50)::numeric,
        2
    ) AS expected_recovery_scenario,

    ROUND(
        (observed_churned_value_proxy * 0.25)::numeric,
        2
    ) AS optimistic_recovery_scenario

FROM scenario_summary;


/*==============================================================================
    17. EXECUTIVE VALUE EXPOSURE SIGNALS
==============================================================================*/

WITH customer_profile AS (
    SELECT
        id,
        arpu_8,
        churn_probability,

        GREATEST(arpu_8, 0) AS value_proxy,

        PERCENT_RANK() OVER (
            ORDER BY GREATEST(arpu_8, 0), id
        ) AS value_percentile

    FROM staging.train_raw
),

executive_summary AS (
    SELECT
        COUNT(*) AS total_customers,

        SUM(
            CASE
                WHEN value_percentile >= 0.75
                THEN 1
                ELSE 0
            END
        ) AS high_value_customers,

        SUM(
            CASE
                WHEN value_percentile >= 0.75
                 AND churn_probability = 1
                THEN 1
                ELSE 0
            END
        ) AS high_value_churned_customers,

        SUM(value_proxy) AS total_value_proxy,

        SUM(
            CASE
                WHEN value_percentile >= 0.75
                THEN value_proxy
                ELSE 0
            END
        ) AS high_value_proxy,

        SUM(
            CASE
                WHEN value_percentile >= 0.75
                 AND churn_probability = 1
                THEN value_proxy
                ELSE 0
            END
        ) AS high_value_observed_churn_exposure

    FROM customer_profile
)

SELECT
    total_customers,
    high_value_customers,
    high_value_churned_customers,

    ROUND(
        total_value_proxy::numeric,
        2
    ) AS total_value_proxy,

    ROUND(
        high_value_proxy::numeric,
        2
    ) AS high_value_proxy,

    ROUND(
        high_value_observed_churn_exposure::numeric,
        2
    ) AS high_value_observed_churn_exposure,

    ROUND(
        (
            100.0 * high_value_proxy /
            NULLIF(total_value_proxy, 0)
        )::numeric,
        2
    ) AS high_value_share_of_total_value_pct,

    ROUND(
        (
            100.0 * high_value_observed_churn_exposure /
            NULLIF(high_value_proxy, 0)
        )::numeric,
        2
    ) AS high_value_observed_churn_exposure_pct

FROM executive_summary;


/*==============================================================================
    18. ANALYTICAL QUALITY GATE
==============================================================================*/

WITH quality_checks AS (
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
         AND missing_arpu_8_count = 0
         AND missing_recharge_8_count = 0
        THEN 'PASS'

        ELSE 'REVIEW'

    END AS analytical_quality_gate

FROM quality_checks;


/*==============================================================================
    END OF REVENUE & CUSTOMER-VALUE EXPOSURE ANALYSIS
==============================================================================*/