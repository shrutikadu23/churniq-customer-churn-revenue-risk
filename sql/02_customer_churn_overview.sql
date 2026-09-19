/* =============================================================================
   ChurnIQ — Customer Churn Overview
   File: 02_customer_churn_overview.sql

   Purpose:
   Stakeholder-oriented descriptive analysis of customer churn behavior.

   Business questions:
   1. What is the overall churn level?
   2. Which customer groups have the highest churn rates?
   3. Which groups contribute the most customers to total churn?
   4. How does churn vary by tenure, ARPU, usage and recharge behavior?
   5. Does customer value interact with engagement?
   6. How does data-service information availability relate to churn?

   Source:
   staging.train_raw

   Important:
   - This is descriptive / diagnostic analysis.
   - Results show associations, not causal relationships.
   - churn_probability is the observed target for the case-study dataset.
   - No model predictions or future-risk scores are used here.
=============================================================================*/


/* =============================================================================
   01. OVERALL CHURN SNAPSHOT
   =============================================================================
   Executive-level view of the customer base and observed churn.
============================================================================= */

SELECT
    COUNT(*) AS total_customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    SUM(
        CASE
            WHEN churn_probability = 0 THEN 1
            ELSE 0
        END
    ) AS retained_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct

FROM staging.train_raw;


/* =============================================================================
   02. CHURN DISTRIBUTION
   =============================================================================
   Shows both customer count and share of the full customer base.
============================================================================= */

SELECT
    churn_probability AS churn_flag,

    CASE
        WHEN churn_probability = 1 THEN 'Churned'
        WHEN churn_probability = 0 THEN 'Retained'
        ELSE 'Unknown'
    END AS customer_status,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM staging.train_raw

GROUP BY churn_probability

ORDER BY churn_probability;


/* =============================================================================
   03. CHURN BY TENURE
   =============================================================================
   Tenure bands help identify whether churn is concentrated among newer
   or longer-tenured customers.
============================================================================= */

WITH tenure_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN aon < 365 THEN '<1 year'
            WHEN aon < 730 THEN '1–2 years'
            WHEN aon < 1095 THEN '2–3 years'
            WHEN aon < 1825 THEN '3–5 years'
            ELSE '5+ years'
        END AS tenure_band

    FROM staging.train_raw

    WHERE aon IS NOT NULL
)

SELECT
    tenure_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 *
        COUNT(*) /
        SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM tenure_bands

GROUP BY tenure_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   04. CHURN BY AUGUST ARPU
   =============================================================================
   ARPU is treated as a customer-value indicator.

   NULL ARPU is kept separate rather than silently converted to zero.
============================================================================= */

WITH arpu_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 < 100 THEN '< ₹100'
            WHEN arpu_8 < 200 THEN '₹100–₹199'
            WHEN arpu_8 < 300 THEN '₹200–₹299'
            WHEN arpu_8 < 500 THEN '₹300–₹499'
            ELSE '₹500+'
        END AS arpu_band

    FROM staging.train_raw
)

SELECT
    arpu_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM arpu_bands

GROUP BY arpu_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   05. CHURN BY AUGUST OUTGOING USAGE
   =============================================================================
   Outgoing voice usage is used as an engagement indicator.

   Missing values remain identifiable.
============================================================================= */

WITH outgoing_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN total_og_mou_8 IS NULL THEN 'Usage missing'
            WHEN total_og_mou_8 = 0 THEN 'Zero usage'
            WHEN total_og_mou_8 < 50 THEN '<50 minutes'
            WHEN total_og_mou_8 < 100 THEN '50–99 minutes'
            WHEN total_og_mou_8 < 250 THEN '100–249 minutes'
            WHEN total_og_mou_8 < 500 THEN '250–499 minutes'
            ELSE '500+ minutes'
        END AS outgoing_usage_band

    FROM staging.train_raw
)

SELECT
    outgoing_usage_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM outgoing_bands

GROUP BY outgoing_usage_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   06. CHURN BY AUGUST INCOMING USAGE
   =============================================================================
   Incoming voice usage provides another engagement signal.
============================================================================= */

WITH incoming_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN total_ic_mou_8 IS NULL THEN 'Usage missing'
            WHEN total_ic_mou_8 = 0 THEN 'Zero usage'
            WHEN total_ic_mou_8 < 50 THEN '<50 minutes'
            WHEN total_ic_mou_8 < 100 THEN '50–99 minutes'
            WHEN total_ic_mou_8 < 250 THEN '100–249 minutes'
            WHEN total_ic_mou_8 < 500 THEN '250–499 minutes'
            ELSE '500+ minutes'
        END AS incoming_usage_band

    FROM staging.train_raw
)

SELECT
    incoming_usage_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM incoming_bands

GROUP BY incoming_usage_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   07. CHURN BY AUGUST RECHARGE VALUE
   ============================================================================= */

WITH recharge_value_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN total_rech_amt_8 IS NULL THEN 'Recharge amount missing'
            WHEN total_rech_amt_8 = 0 THEN '₹0'
            WHEN total_rech_amt_8 < 100 THEN '< ₹100'
            WHEN total_rech_amt_8 < 250 THEN '₹100–₹249'
            WHEN total_rech_amt_8 < 500 THEN '₹250–₹499'
            WHEN total_rech_amt_8 < 1000 THEN '₹500–₹999'
            ELSE '₹1000+'
        END AS recharge_value_band

    FROM staging.train_raw
)

SELECT
    recharge_value_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM recharge_value_bands

GROUP BY recharge_value_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   08. CHURN BY AUGUST RECHARGE FREQUENCY
   ============================================================================= */

WITH recharge_frequency_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN total_rech_num_8 IS NULL THEN 'Recharge frequency missing'
            WHEN total_rech_num_8 = 0 THEN '0 recharges'
            WHEN total_rech_num_8 <= 2 THEN '1–2 recharges'
            WHEN total_rech_num_8 <= 5 THEN '3–5 recharges'
            WHEN total_rech_num_8 <= 10 THEN '6–10 recharges'
            ELSE '11+ recharges'
        END AS recharge_frequency_band

    FROM staging.train_raw
)

SELECT
    recharge_frequency_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM recharge_frequency_bands

GROUP BY recharge_frequency_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   09. CHURN BY DATA-ENGAGEMENT STATUS
   =============================================================================
   Data-service fields have substantial missingness.

   Therefore:
   - Missing information is kept separate.
   - Missing is NOT automatically treated as zero usage.
============================================================================= */

SELECT
    CASE
        WHEN total_rech_data_8 IS NULL THEN 'Data recharge information missing'
        WHEN total_rech_data_8 = 0 THEN 'No data recharge'
        ELSE 'Data recharge present'
    END AS data_engagement_status,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM staging.train_raw

GROUP BY 1

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   10. CHURN BY AUGUST MOBILE-DATA USAGE
   ============================================================================= */

WITH data_usage_bands AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN 'Data usage missing'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) = 0
                THEN 'Zero data usage'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 100
                THEN '<100 MB'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 500
                THEN '100–499 MB'

            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 1000
                THEN '500–999 MB'

            ELSE '1000+ MB'
        END AS data_usage_band

    FROM staging.train_raw
)

SELECT
    data_usage_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM data_usage_bands

GROUP BY data_usage_band

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   11. OVERALL VOICE ENGAGEMENT
   =============================================================================
   Combines incoming and outgoing August voice usage.

   This is a descriptive engagement view, not a churn prediction rule.
============================================================================= */

SELECT
    CASE
        WHEN total_og_mou_8 IS NULL
             AND total_ic_mou_8 IS NULL
            THEN 'Voice usage missing'

        WHEN COALESCE(total_og_mou_8, 0)
           + COALESCE(total_ic_mou_8, 0) = 0
            THEN 'No voice usage'

        WHEN COALESCE(total_og_mou_8, 0)
           + COALESCE(total_ic_mou_8, 0) < 100
            THEN '<100 minutes'

        WHEN COALESCE(total_og_mou_8, 0)
           + COALESCE(total_ic_mou_8, 0) < 500
            THEN '100–499 minutes'

        WHEN COALESCE(total_og_mou_8, 0)
           + COALESCE(total_ic_mou_8, 0) < 1000
            THEN '500–999 minutes'

        ELSE '1000+ minutes'
    END AS voice_engagement_band,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM staging.train_raw

GROUP BY 1

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   12. CUSTOMER VALUE × ENGAGEMENT
   =============================================================================
   Combines August ARPU with August voice engagement.

   Purpose:
   Identify whether lower-engagement / higher-value groups show different
   observed churn patterns.

   ARPU threshold is an analytical segmentation threshold, not a business
   definition of "high value".
============================================================================= */

WITH customer_segments AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN arpu_8 IS NULL THEN 'ARPU missing'
            WHEN arpu_8 >= 300 THEN 'Higher ARPU'
            ELSE 'Lower ARPU'
        END AS value_group,

        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN 'Voice usage missing'

            WHEN COALESCE(total_og_mou_8, 0)
               + COALESCE(total_ic_mou_8, 0) = 0
                THEN 'No voice usage'

            ELSE 'Voice usage present'
        END AS engagement_group

    FROM staging.train_raw
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
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM customer_segments

GROUP BY
    value_group,
    engagement_group

ORDER BY
    churn_rate_pct DESC;


/* =============================================================================
   13. DATA-INFORMATION AVAILABILITY VS CHURN
   =============================================================================
   Distinguishes actual data-service activity from missing information.

   This is particularly important because data-related variables have high
   structural missingness in the source dataset.
============================================================================= */

SELECT
    CASE
        WHEN date_of_last_rech_data_8 IS NULL
             AND total_rech_data_8 IS NULL
            THEN 'Data-service information unavailable'

        WHEN date_of_last_rech_data_8 IS NOT NULL
             OR total_rech_data_8 IS NOT NULL
            THEN 'Data-service information available'

        ELSE 'Other'
    END AS data_information_status,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN churn_probability = 1 THEN 1
            ELSE 0
        END
    ) AS churned_customers,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS churn_rate_pct,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN churn_probability = 1 THEN 1
                ELSE 0
            END
        )
        /
        SUM(
            SUM(
                CASE
                    WHEN churn_probability = 1 THEN 1
                    ELSE 0
                END
            )
        ) OVER (),
        2
    ) AS churn_contribution_pct

FROM staging.train_raw

GROUP BY 1

ORDER BY churn_rate_pct DESC;


/* =============================================================================
   14. RETAINED VS CHURNED CUSTOMER PROFILE
   =============================================================================
   Compact executive comparison across major behavioral/value indicators.
============================================================================= */

SELECT
    CASE
        WHEN churn_probability = 1 THEN 'Churned'
        ELSE 'Retained'
    END AS customer_status,

    COUNT(*) AS customers,

    ROUND(
        AVG(aon)::numeric,
        1
    ) AS avg_tenure_days,

    ROUND(
        AVG(arpu_8)::numeric,
        2
    ) AS avg_august_arpu,

    ROUND(
        AVG(total_rech_amt_8)::numeric,
        2
    ) AS avg_august_recharge_amount,

    ROUND(
        AVG(total_rech_num_8)::numeric,
        2
    ) AS avg_august_recharge_count,

    ROUND(
        AVG(total_og_mou_8)::numeric,
        2
    ) AS avg_august_outgoing_minutes,

    ROUND(
        AVG(total_ic_mou_8)::numeric,
        2
    ) AS avg_august_incoming_minutes,

    ROUND(
        AVG(vol_2g_mb_8)::numeric,
        2
    ) AS avg_august_2g_mb,

    ROUND(
        AVG(vol_3g_mb_8)::numeric,
        2
    ) AS avg_august_3g_mb

FROM staging.train_raw

GROUP BY churn_probability

ORDER BY churn_probability;


/* =============================================================================
   15. EXECUTIVE CHURN SIGNALS
   =============================================================================
   Ranks selected customer groups by observed churn rate.

   Minimum-volume guardrail:
   Only groups containing at least 1% of the customer base are included.
   This prevents very small groups from appearing as misleading "highest risk"
   segments simply because of small sample sizes.
============================================================================= */

WITH signal_groups AS (

    /* Low ARPU */
    SELECT
        'Low August ARPU' AS signal,
        COUNT(*) AS customers,
        AVG(churn_probability::numeric) AS churn_rate
    FROM staging.train_raw
    WHERE arpu_8 < 100

    UNION ALL

    /* Low outgoing engagement */
    SELECT
        'Very low outgoing voice usage',
        COUNT(*),
        AVG(churn_probability::numeric)
    FROM staging.train_raw
    WHERE total_og_mou_8 IS NOT NULL
      AND total_og_mou_8 < 50

    UNION ALL

    /* Low incoming engagement */
    SELECT
        'Very low incoming voice usage',
        COUNT(*),
        AVG(churn_probability::numeric)
    FROM staging.train_raw
    WHERE total_ic_mou_8 IS NOT NULL
      AND total_ic_mou_8 < 50

    UNION ALL

    /* Low recharge value */
    SELECT
        'Low August recharge value',
        COUNT(*),
        AVG(churn_probability::numeric)
    FROM staging.train_raw
    WHERE total_rech_amt_8 < 100

    UNION ALL

    /* Short tenure */
    SELECT
        'Less than one year tenure',
        COUNT(*),
        AVG(churn_probability::numeric)
    FROM staging.train_raw
    WHERE aon < 365

    UNION ALL

    /* No voice usage */
    SELECT
        'No August voice usage',
        COUNT(*),
        AVG(churn_probability::numeric)
    FROM staging.train_raw
    WHERE COALESCE(total_og_mou_8, 0)
        + COALESCE(total_ic_mou_8, 0) = 0
)

SELECT
    signal,

    customers,

    ROUND(
        100.0 * customers /
        (SELECT COUNT(*) FROM staging.train_raw),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * churn_rate,
        2
    ) AS churn_rate_pct,

    RANK() OVER (
        ORDER BY churn_rate DESC
    ) AS churn_rate_rank

FROM signal_groups

WHERE customers >= (
    SELECT COUNT(*) * 0.01
    FROM staging.train_raw
)

ORDER BY churn_rate_rank;


/* =============================================================================
   16. ANALYTICAL QUALITY GATE
   =============================================================================
   Lightweight check that the source is suitable for this descriptive
   overview. Detailed data-quality validation belongs in #01.
============================================================================= */

SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT id) AS distinct_customer_ids,

    COUNT(*) FILTER (
        WHERE churn_probability IN (0, 1)
    ) AS valid_target_rows,

    CASE
        WHEN COUNT(*) > 0
         AND COUNT(DISTINCT id) = COUNT(*)
         AND COUNT(*) FILTER (
                WHERE churn_probability IN (0, 1)
             ) = COUNT(*)
        THEN 'PASS'
        ELSE 'REVIEW'
    END AS analytical_readiness

FROM staging.train_raw;


/* =============================================================================
   END OF FILE
   =============================================================================
*/