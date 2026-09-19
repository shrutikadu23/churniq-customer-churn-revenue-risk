/*==============================================================================
    CHURNIQ — TEMPORAL BEHAVIOR ANALYSIS

    Purpose:
    Analyze how customer behavior changes across June, July, and August 2014
    and identify temporal patterns associated with the observed churn outcome.

    Business Focus:
    - Measure month-to-month behavioral movement
    - Identify persistent deterioration and improvement
    - Compare August behavior with an earlier customer baseline
    - Identify multi-domain deterioration patterns
    - Understand temporal profiles associated with observed churn
    - Identify early-warning behavioral patterns
    - Assess deterioration among higher-value customers

    Prediction Framework:
    - Good Phase: June + July 2014
    - Action Phase: August 2014
    - Outcome Horizon: September 2014 / subsequent churn outcome
    - September attributes are not available and are never used as predictors.

    Analytical Guardrails:
    - Descriptive analysis only
    - No model predictions or risk scores
    - No thresholding or customer risk tiers
    - No retention recommendations
    - Missing values are not automatically interpreted as zero
    - Percentage changes require a meaningful non-zero baseline
    - Associations with churn are not interpreted as causal relationships
==============================================================================*/


/*==============================================================================
    01. OVERALL TEMPORAL BEHAVIOR SNAPSHOT
==============================================================================*/

SELECT
    COUNT(*) AS customer_count,

    ROUND(AVG(arpu_6)::numeric, 2) AS avg_arpu_june,
    ROUND(AVG(arpu_7)::numeric, 2) AS avg_arpu_july,
    ROUND(AVG(arpu_8)::numeric, 2) AS avg_arpu_august,

    ROUND(AVG(total_rech_amt_6)::numeric, 2) AS avg_recharge_june,
    ROUND(AVG(total_rech_amt_7)::numeric, 2) AS avg_recharge_july,
    ROUND(AVG(total_rech_amt_8)::numeric, 2) AS avg_recharge_august,

    ROUND(AVG(total_og_mou_6)::numeric, 2) AS avg_outgoing_june,
    ROUND(AVG(total_og_mou_7)::numeric, 2) AS avg_outgoing_july,
    ROUND(AVG(total_og_mou_8)::numeric, 2) AS avg_outgoing_august,

    ROUND(AVG(total_ic_mou_6)::numeric, 2) AS avg_incoming_june,
    ROUND(AVG(total_ic_mou_7)::numeric, 2) AS avg_incoming_july,
    ROUND(AVG(total_ic_mou_8)::numeric, 2) AS avg_incoming_august,

    ROUND(AVG(vol_2g_mb_6)::numeric, 2) AS avg_2g_june,
    ROUND(AVG(vol_2g_mb_7)::numeric, 2) AS avg_2g_july,
    ROUND(AVG(vol_2g_mb_8)::numeric, 2) AS avg_2g_august,

    ROUND(AVG(vol_3g_mb_6)::numeric, 2) AS avg_3g_june,
    ROUND(AVG(vol_3g_mb_7)::numeric, 2) AS avg_3g_july,
    ROUND(AVG(vol_3g_mb_8)::numeric, 2) AS avg_3g_august

FROM staging.train_raw;


/*==============================================================================
    02. ARPU TRAJECTORY
==============================================================================*/

SELECT
    CASE
        WHEN arpu_6 IS NULL
          OR arpu_7 IS NULL
          OR arpu_8 IS NULL
        THEN 'Incomplete ARPU history'

        WHEN arpu_7 > arpu_6
         AND arpu_8 > arpu_7
        THEN 'Consistently improving'

        WHEN arpu_7 < arpu_6
         AND arpu_8 < arpu_7
        THEN 'Consistently declining'

        WHEN arpu_7 > arpu_6
         AND arpu_8 < arpu_7
        THEN 'July improvement, August decline'

        WHEN arpu_7 < arpu_6
         AND arpu_8 > arpu_7
        THEN 'July decline, August recovery'

        ELSE 'Mixed / stable'
    END AS arpu_trajectory,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(arpu_8)::numeric,
        2
    ) AS avg_august_arpu

FROM staging.train_raw

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    03. ARPU DIRECTIONAL CHANGE
==============================================================================*/

WITH arpu_change AS (
    SELECT
        id,
        arpu_6,
        arpu_7,
        arpu_8,

        arpu_7 - arpu_6 AS june_to_july_change,
        arpu_8 - arpu_7 AS july_to_august_change,
        arpu_8 - arpu_6 AS june_to_august_change,

        CASE
            WHEN arpu_6 IS NOT NULL
             AND arpu_6 <> 0
            THEN
                100.0
                * (arpu_8 - arpu_6)
                / arpu_6
        END AS june_to_august_pct_change

    FROM staging.train_raw
)

SELECT
    CASE
        WHEN june_to_august_change > 0
            THEN 'Increasing'

        WHEN june_to_august_change < 0
            THEN 'Decreasing'

        WHEN june_to_august_change = 0
            THEN 'No change'

        ELSE 'Unavailable'
    END AS arpu_direction,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(june_to_august_change)::numeric,
        2
    ) AS avg_absolute_change,

    ROUND(
        AVG(june_to_august_pct_change)::numeric,
        2
    ) AS avg_pct_change

FROM arpu_change

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    04. RECHARGE TRAJECTORY
==============================================================================*/

SELECT
    CASE
        WHEN total_rech_amt_6 IS NULL
          OR total_rech_amt_7 IS NULL
          OR total_rech_amt_8 IS NULL
        THEN 'Incomplete recharge history'

        WHEN total_rech_amt_7 > total_rech_amt_6
         AND total_rech_amt_8 > total_rech_amt_7
        THEN 'Consistently increasing'

        WHEN total_rech_amt_7 < total_rech_amt_6
         AND total_rech_amt_8 < total_rech_amt_7
        THEN 'Consistently declining'

        WHEN total_rech_amt_7 > total_rech_amt_6
         AND total_rech_amt_8 < total_rech_amt_7
        THEN 'July increase, August decline'

        WHEN total_rech_amt_7 < total_rech_amt_6
         AND total_rech_amt_8 > total_rech_amt_7
        THEN 'July decline, August recovery'

        ELSE 'Mixed / stable'
    END AS recharge_trajectory,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_rech_amt_8)::numeric,
        2
    ) AS avg_august_recharge

FROM staging.train_raw

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    05. RECHARGE DIRECTIONAL CHANGE
==============================================================================*/

WITH recharge_change AS (
    SELECT
        id,

        total_rech_amt_6,
        total_rech_amt_8,

        total_rech_amt_8 - total_rech_amt_6
            AS june_to_august_change,

        CASE
            WHEN total_rech_amt_6 IS NOT NULL
             AND total_rech_amt_6 <> 0
            THEN
                100.0
                * (total_rech_amt_8 - total_rech_amt_6)
                / total_rech_amt_6
        END AS june_to_august_pct_change

    FROM staging.train_raw
)

SELECT
    CASE
        WHEN june_to_august_change > 0
            THEN 'Increasing'

        WHEN june_to_august_change < 0
            THEN 'Decreasing'

        WHEN june_to_august_change = 0
            THEN 'No change'

        ELSE 'Unavailable'
    END AS recharge_direction,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(june_to_august_change)::numeric,
        2
    ) AS avg_absolute_change,

    ROUND(
        AVG(june_to_august_pct_change)::numeric,
        2
    ) AS avg_pct_change

FROM recharge_change

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    06. VOICE USAGE TRAJECTORY
==============================================================================*/

WITH voice_trajectory AS (
    SELECT
        id,

        total_og_mou_6,
        total_og_mou_7,
        total_og_mou_8,

        total_ic_mou_6,
        total_ic_mou_7,
        total_ic_mou_8,

        CASE
            WHEN total_og_mou_6 IS NULL
              OR total_og_mou_7 IS NULL
              OR total_og_mou_8 IS NULL
            THEN 'Incomplete outgoing history'

            WHEN total_og_mou_7 > total_og_mou_6
             AND total_og_mou_8 > total_og_mou_7
            THEN 'Increasing'

            WHEN total_og_mou_7 < total_og_mou_6
             AND total_og_mou_8 < total_og_mou_7
            THEN 'Declining'

            ELSE 'Mixed / stable'
        END AS outgoing_trajectory,

        CASE
            WHEN total_ic_mou_6 IS NULL
              OR total_ic_mou_7 IS NULL
              OR total_ic_mou_8 IS NULL
            THEN 'Incomplete incoming history'

            WHEN total_ic_mou_7 > total_ic_mou_6
             AND total_ic_mou_8 > total_ic_mou_7
            THEN 'Increasing'

            WHEN total_ic_mou_7 < total_ic_mou_6
             AND total_ic_mou_8 < total_ic_mou_7
            THEN 'Declining'

            ELSE 'Mixed / stable'
        END AS incoming_trajectory

    FROM staging.train_raw
)

SELECT
    outgoing_trajectory,
    incoming_trajectory,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM voice_trajectory

GROUP BY
    outgoing_trajectory,
    incoming_trajectory

ORDER BY customers DESC;


/*==============================================================================
    07. VOICE CHANGE PATTERN
==============================================================================*/

WITH trajectory AS (
    SELECT
        id,

        total_og_mou_6,
        total_og_mou_8,
        total_ic_mou_6,
        total_ic_mou_8

    FROM staging.train_raw
)

SELECT
    CASE
        WHEN total_og_mou_6 IS NULL
          OR total_og_mou_8 IS NULL
          OR total_ic_mou_6 IS NULL
          OR total_ic_mou_8 IS NULL
        THEN 'Incomplete history'

        WHEN total_og_mou_8 < total_og_mou_6
         AND total_ic_mou_8 < total_ic_mou_6
        THEN 'Both declining'

        WHEN total_og_mou_8 < total_og_mou_6
         AND total_ic_mou_8 >= total_ic_mou_6
        THEN 'Outgoing declining only'

        WHEN total_og_mou_8 >= total_og_mou_6
         AND total_ic_mou_8 < total_ic_mou_6
        THEN 'Incoming declining only'

        ELSE 'Neither declining'
    END AS voice_change_pattern,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM trajectory

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    08. DATA-USAGE TRAJECTORY
==============================================================================*/

WITH data_trajectory AS (
    SELECT
        id,

        vol_2g_mb_6,
        vol_2g_mb_7,
        vol_2g_mb_8,

        vol_3g_mb_6,
        vol_3g_mb_7,
        vol_3g_mb_8

    FROM staging.train_raw
)

SELECT
    CASE
        WHEN vol_2g_mb_6 IS NULL
          OR vol_2g_mb_7 IS NULL
          OR vol_2g_mb_8 IS NULL
          OR vol_3g_mb_6 IS NULL
          OR vol_3g_mb_7 IS NULL
          OR vol_3g_mb_8 IS NULL
        THEN 'Incomplete data history'

        WHEN vol_2g_mb_7 > vol_2g_mb_6
         AND vol_2g_mb_8 > vol_2g_mb_7
         AND vol_3g_mb_7 > vol_3g_mb_6
         AND vol_3g_mb_8 > vol_3g_mb_7
        THEN 'Consistently increasing'

        WHEN vol_2g_mb_7 < vol_2g_mb_6
         AND vol_2g_mb_8 < vol_2g_mb_7
         AND vol_3g_mb_7 < vol_3g_mb_6
         AND vol_3g_mb_8 < vol_3g_mb_7
        THEN 'Consistently declining'

        WHEN vol_2g_mb_8 < vol_2g_mb_7
         AND vol_3g_mb_8 < vol_3g_mb_7
        THEN 'Recent decline'

        ELSE 'Mixed / stable'
    END AS data_trajectory,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM data_trajectory

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    09. MULTI-DOMAIN RECENT DETERIORATION
==============================================================================*/

WITH behavior_change AS (
    SELECT
        id,

        CASE
            WHEN arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND arpu_8 < arpu_7
            THEN 1 ELSE 0
        END AS arpu_decline,

        CASE
            WHEN total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_rech_amt_8 < total_rech_amt_7
            THEN 1 ELSE 0
        END AS recharge_decline,

        CASE
            WHEN total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND total_og_mou_8 < total_og_mou_7
            THEN 1 ELSE 0
        END AS outgoing_decline,

        CASE
            WHEN total_ic_mou_7 IS NOT NULL
             AND total_ic_mou_8 IS NOT NULL
             AND total_ic_mou_8 < total_ic_mou_7
            THEN 1 ELSE 0
        END AS incoming_decline,

        CASE
            WHEN vol_2g_mb_7 IS NOT NULL
             AND vol_2g_mb_8 IS NOT NULL
             AND vol_3g_mb_7 IS NOT NULL
             AND vol_3g_mb_8 IS NOT NULL
             AND vol_2g_mb_8 < vol_2g_mb_7
             AND vol_3g_mb_8 < vol_3g_mb_7
            THEN 1 ELSE 0
        END AS data_decline

    FROM staging.train_raw
)

SELECT
    (
        arpu_decline
        + recharge_decline
        + outgoing_decline
        + incoming_decline
        + data_decline
    ) AS deterioration_domain_count,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM behavior_change

GROUP BY 1

ORDER BY deterioration_domain_count;


/*==============================================================================
    10. TEMPORAL PROFILE CLASSIFICATION
==============================================================================*/

WITH directional_change AS (
    SELECT
        id,

        CASE
            WHEN arpu_8 IS NULL OR arpu_7 IS NULL THEN NULL
            WHEN arpu_8 > arpu_7 THEN 1
            WHEN arpu_8 < arpu_7 THEN -1
            ELSE 0
        END AS arpu_direction,

        CASE
            WHEN total_rech_amt_8 IS NULL
              OR total_rech_amt_7 IS NULL
            THEN NULL

            WHEN total_rech_amt_8 > total_rech_amt_7 THEN 1
            WHEN total_rech_amt_8 < total_rech_amt_7 THEN -1
            ELSE 0
        END AS recharge_direction,

        CASE
            WHEN total_og_mou_8 IS NULL
              OR total_og_mou_7 IS NULL
            THEN NULL

            WHEN total_og_mou_8 > total_og_mou_7 THEN 1
            WHEN total_og_mou_8 < total_og_mou_7 THEN -1
            ELSE 0
        END AS outgoing_direction,

        CASE
            WHEN total_ic_mou_8 IS NULL
              OR total_ic_mou_7 IS NULL
            THEN NULL

            WHEN total_ic_mou_8 > total_ic_mou_7 THEN 1
            WHEN total_ic_mou_8 < total_ic_mou_7 THEN -1
            ELSE 0
        END AS incoming_direction

    FROM staging.train_raw
),

profile AS (
    SELECT
        id,

        (
            CASE WHEN arpu_direction = -1 THEN 1 ELSE 0 END
            + CASE WHEN recharge_direction = -1 THEN 1 ELSE 0 END
            + CASE WHEN outgoing_direction = -1 THEN 1 ELSE 0 END
            + CASE WHEN incoming_direction = -1 THEN 1 ELSE 0 END
        ) AS declining_domains,

        (
            CASE WHEN arpu_direction = 1 THEN 1 ELSE 0 END
            + CASE WHEN recharge_direction = 1 THEN 1 ELSE 0 END
            + CASE WHEN outgoing_direction = 1 THEN 1 ELSE 0 END
            + CASE WHEN incoming_direction = 1 THEN 1 ELSE 0 END
        ) AS improving_domains,

        (
            CASE WHEN arpu_direction IS NOT NULL THEN 1 ELSE 0 END
            + CASE WHEN recharge_direction IS NOT NULL THEN 1 ELSE 0 END
            + CASE WHEN outgoing_direction IS NOT NULL THEN 1 ELSE 0 END
            + CASE WHEN incoming_direction IS NOT NULL THEN 1 ELSE 0 END
        ) AS available_domains

    FROM directional_change
)

SELECT
    CASE
        WHEN available_domains < 3
            THEN 'Insufficient behavioral history'

        WHEN declining_domains >= 3
         AND declining_domains > improving_domains
            THEN 'Declining'

        WHEN improving_domains >= 3
         AND improving_domains > declining_domains
            THEN 'Improving'

        WHEN declining_domains = 0
         AND improving_domains = 0
            THEN 'Stable'

        ELSE 'Mixed'
    END AS temporal_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM profile

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    11. AUGUST VS EARLY-PHASE BASELINE
==============================================================================*/

WITH baseline AS (
    SELECT
        id,

        CASE
            WHEN arpu_6 IS NOT NULL
             AND arpu_7 IS NOT NULL
            THEN (arpu_6 + arpu_7) / 2.0
        END AS baseline_arpu,

        CASE
            WHEN total_rech_amt_6 IS NOT NULL
             AND total_rech_amt_7 IS NOT NULL
            THEN
                (total_rech_amt_6 + total_rech_amt_7) / 2.0
        END AS baseline_recharge,

        CASE
            WHEN total_og_mou_6 IS NOT NULL
             AND total_og_mou_7 IS NOT NULL
            THEN
                (total_og_mou_6 + total_og_mou_7) / 2.0
        END AS baseline_outgoing,

        CASE
            WHEN total_ic_mou_6 IS NOT NULL
             AND total_ic_mou_7 IS NOT NULL
            THEN
                (total_ic_mou_6 + total_ic_mou_7) / 2.0
        END AS baseline_incoming,

        arpu_8,
        total_rech_amt_8,
        total_og_mou_8,
        total_ic_mou_8

    FROM staging.train_raw
)

SELECT
    CASE
        WHEN baseline_arpu IS NULL
          OR baseline_recharge IS NULL
          OR baseline_outgoing IS NULL
          OR baseline_incoming IS NULL
        THEN 'Incomplete baseline'

        WHEN arpu_8 < baseline_arpu
         AND total_rech_amt_8 < baseline_recharge
         AND total_og_mou_8 < baseline_outgoing
         AND total_ic_mou_8 < baseline_incoming
        THEN 'Broad deterioration vs baseline'

        WHEN arpu_8 > baseline_arpu
         AND total_rech_amt_8 > baseline_recharge
         AND total_og_mou_8 > baseline_outgoing
         AND total_ic_mou_8 > baseline_incoming
        THEN 'Broad improvement vs baseline'

        ELSE 'Mixed / partial movement'
    END AS recent_vs_baseline_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct

FROM baseline

GROUP BY 1

ORDER BY customers DESC;


/*==============================================================================
    12. TEMPORAL BEHAVIOR × CUSTOMER VALUE
==============================================================================*/

WITH temporal_profile AS (
    SELECT
        id,

        CASE
            WHEN arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND arpu_8 < arpu_7
             AND total_rech_amt_8 < total_rech_amt_7
             AND total_og_mou_8 < total_og_mou_7
            THEN 'Broad deterioration'

            WHEN arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND arpu_8 > arpu_7
             AND total_rech_amt_8 > total_rech_amt_7
             AND total_og_mou_8 > total_og_mou_7
            THEN 'Broad improvement'

            ELSE 'Mixed / partial movement'
        END AS temporal_profile

    FROM staging.train_raw
),

value_band AS (
    SELECT
        id,
        arpu_8,

        CASE
            WHEN arpu_8 IS NULL
                THEN 'Unavailable'

            WHEN arpu_8 < 100
                THEN 'Low'

            WHEN arpu_8 < 250
                THEN 'Medium'

            WHEN arpu_8 < 500
                THEN 'High'

            ELSE 'Very High'
        END AS value_band

    FROM staging.train_raw
)

SELECT
    v.value_band,
    t.temporal_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*)) OVER (
            PARTITION BY v.value_band
        ),
        2
    ) AS profile_share_within_value_band_pct,

    ROUND(
        AVG(v.arpu_8)::numeric,
        2
    ) AS avg_august_arpu

FROM temporal_profile t

JOIN value_band v
    ON t.id = v.id

GROUP BY
    v.value_band,
    t.temporal_profile

ORDER BY
    v.value_band,
    customers DESC;


/*==============================================================================
    13. TEMPORAL BEHAVIOR × OBSERVED CHURN
==============================================================================*/

WITH temporal_profile AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND total_ic_mou_7 IS NOT NULL
             AND total_ic_mou_8 IS NOT NULL
             AND arpu_8 < arpu_7
             AND total_rech_amt_8 < total_rech_amt_7
             AND total_og_mou_8 < total_og_mou_7
             AND total_ic_mou_8 < total_ic_mou_7
            THEN 'Broad deterioration'

            WHEN arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND total_ic_mou_7 IS NOT NULL
             AND total_ic_mou_8 IS NOT NULL
             AND arpu_8 > arpu_7
             AND total_rech_amt_8 > total_rech_amt_7
             AND total_og_mou_8 > total_og_mou_7
             AND total_ic_mou_8 > total_ic_mou_7
            THEN 'Broad improvement'

            ELSE 'Mixed / partial movement'
        END AS temporal_profile

    FROM staging.train_raw
)

SELECT
    temporal_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    SUM(churn_probability) AS observed_churned_customers,

    ROUND(
        100.0
        * SUM(churn_probability)
        / NULLIF(COUNT(*), 0),
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        100.0
        * SUM(churn_probability)
        / NULLIF(
            SUM(SUM(churn_probability)) OVER (),
            0
        ),
        2
    ) AS churn_contribution_pct

FROM temporal_profile

GROUP BY temporal_profile

ORDER BY observed_churn_rate_pct DESC;


/*==============================================================================
    14. PERSISTENT DETERIORATION / EARLY-WARNING PATTERNS
==============================================================================*/

WITH monthly_direction AS (
    SELECT
        id,
        churn_probability,

        CASE
            WHEN arpu_6 IS NOT NULL
             AND arpu_7 IS NOT NULL
             AND arpu_8 IS NOT NULL
             AND arpu_7 < arpu_6
             AND arpu_8 < arpu_7
            THEN 1 ELSE 0
        END AS arpu_persistent_decline,

        CASE
            WHEN total_rech_amt_6 IS NOT NULL
             AND total_rech_amt_7 IS NOT NULL
             AND total_rech_amt_8 IS NOT NULL
             AND total_rech_amt_7 < total_rech_amt_6
             AND total_rech_amt_8 < total_rech_amt_7
            THEN 1 ELSE 0
        END AS recharge_persistent_decline,

        CASE
            WHEN total_og_mou_6 IS NOT NULL
             AND total_og_mou_7 IS NOT NULL
             AND total_og_mou_8 IS NOT NULL
             AND total_og_mou_7 < total_og_mou_6
             AND total_og_mou_8 < total_og_mou_7
            THEN 1 ELSE 0
        END AS outgoing_persistent_decline,

        CASE
            WHEN total_ic_mou_6 IS NOT NULL
             AND total_ic_mou_7 IS NOT NULL
             AND total_ic_mou_8 IS NOT NULL
             AND total_ic_mou_7 < total_ic_mou_6
             AND total_ic_mou_8 < total_ic_mou_7
            THEN 1 ELSE 0
        END AS incoming_persistent_decline

    FROM staging.train_raw
),

early_warning AS (
    SELECT
        id,
        churn_probability,

        (
            arpu_persistent_decline
            + recharge_persistent_decline
            + outgoing_persistent_decline
            + incoming_persistent_decline
        ) AS persistent_decline_domains

    FROM monthly_direction
)

SELECT
    persistent_decline_domains,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0
        * SUM(churn_probability)
        / NULLIF(COUNT(*), 0),
        2
    ) AS observed_churn_rate_pct

FROM early_warning

GROUP BY persistent_decline_domains

ORDER BY persistent_decline_domains;


/*==============================================================================
    15. EXECUTIVE TEMPORAL SIGNALS
==============================================================================*/

WITH signals AS (
    SELECT
        COUNT(*) AS total_customers,

        COUNT(*) FILTER (
            WHERE arpu_7 IS NOT NULL
              AND arpu_8 IS NOT NULL
              AND arpu_8 < arpu_7
        ) AS arpu_declining,

        COUNT(*) FILTER (
            WHERE total_rech_amt_7 IS NOT NULL
              AND total_rech_amt_8 IS NOT NULL
              AND total_rech_amt_8 < total_rech_amt_7
        ) AS recharge_declining,

        COUNT(*) FILTER (
            WHERE total_og_mou_7 IS NOT NULL
              AND total_og_mou_8 IS NOT NULL
              AND total_og_mou_8 < total_og_mou_7
        ) AS outgoing_declining,

        COUNT(*) FILTER (
            WHERE total_ic_mou_7 IS NOT NULL
              AND total_ic_mou_8 IS NOT NULL
              AND total_ic_mou_8 < total_ic_mou_7
        ) AS incoming_declining,

        COUNT(*) FILTER (
            WHERE arpu_7 IS NOT NULL
              AND arpu_8 IS NOT NULL
              AND total_rech_amt_7 IS NOT NULL
              AND total_rech_amt_8 IS NOT NULL
              AND total_og_mou_7 IS NOT NULL
              AND total_og_mou_8 IS NOT NULL
              AND arpu_8 < arpu_7
              AND total_rech_amt_8 < total_rech_amt_7
              AND total_og_mou_8 < total_og_mou_7
        ) AS broad_deterioration,

        COUNT(*) FILTER (
            WHERE arpu_7 IS NOT NULL
              AND arpu_8 IS NOT NULL
              AND total_rech_amt_7 IS NOT NULL
              AND total_rech_amt_8 IS NOT NULL
              AND total_og_mou_7 IS NOT NULL
              AND total_og_mou_8 IS NOT NULL
              AND arpu_8 > arpu_7
              AND total_rech_amt_8 > total_rech_amt_7
              AND total_og_mou_8 > total_og_mou_7
        ) AS broad_improvement

    FROM staging.train_raw
)

SELECT
    total_customers,

    arpu_declining,

    ROUND(
        100.0 * arpu_declining
        / NULLIF(total_customers, 0),
        2
    ) AS arpu_declining_pct,

    recharge_declining,

    ROUND(
        100.0 * recharge_declining
        / NULLIF(total_customers, 0),
        2
    ) AS recharge_declining_pct,

    outgoing_declining,

    ROUND(
        100.0 * outgoing_declining
        / NULLIF(total_customers, 0),
        2
    ) AS outgoing_declining_pct,

    incoming_declining,

    ROUND(
        100.0 * incoming_declining
        / NULLIF(total_customers, 0),
        2
    ) AS incoming_declining_pct,

    broad_deterioration,

    ROUND(
        100.0 * broad_deterioration
        / NULLIF(total_customers, 0),
        2
    ) AS broad_deterioration_pct,

    broad_improvement,

    ROUND(
        100.0 * broad_improvement
        / NULLIF(total_customers, 0),
        2
    ) AS broad_improvement_pct

FROM signals;


/*==============================================================================
    16. ANALYTICAL QUALITY GATE
==============================================================================*/

WITH quality_checks AS (
    SELECT
        COUNT(*) AS row_count,

        COUNT(DISTINCT id) AS distinct_customer_ids,

        COUNT(*) - COUNT(DISTINCT id)
            AS duplicate_customer_ids,

        COUNT(*) FILTER (
            WHERE churn_probability IS NULL
        ) AS missing_targets,

        COUNT(*) FILTER (
            WHERE arpu_6 IS NULL
               OR arpu_7 IS NULL
               OR arpu_8 IS NULL
        ) AS incomplete_arpu_history,

        COUNT(*) FILTER (
            WHERE total_rech_amt_6 IS NULL
               OR total_rech_amt_7 IS NULL
               OR total_rech_amt_8 IS NULL
        ) AS incomplete_recharge_history,

        COUNT(*) FILTER (
            WHERE total_og_mou_6 IS NULL
               OR total_og_mou_7 IS NULL
               OR total_og_mou_8 IS NULL
        ) AS incomplete_outgoing_history,

        COUNT(*) FILTER (
            WHERE total_ic_mou_6 IS NULL
               OR total_ic_mou_7 IS NULL
               OR total_ic_mou_8 IS NULL
        ) AS incomplete_incoming_history

    FROM staging.train_raw
)

SELECT
    row_count,
    distinct_customer_ids,
    duplicate_customer_ids,
    missing_targets,
    incomplete_arpu_history,
    incomplete_recharge_history,
    incomplete_outgoing_history,
    incomplete_incoming_history,

    CASE
        WHEN row_count = 69999
         AND distinct_customer_ids = 69999
         AND duplicate_customer_ids = 0
         AND missing_targets = 0
        THEN 'PASS'

        ELSE 'REVIEW'
    END AS analytical_quality_gate

FROM quality_checks;


/*==============================================================================
    END OF TEMPORAL BEHAVIOR ANALYSIS
==============================================================================*/