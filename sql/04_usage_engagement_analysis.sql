-- ============================================================
-- CHURNIQ — USAGE & ENGAGEMENT ANALYSIS
-- ============================================================
-- Purpose:
-- Analyze customer voice, data, and engagement behavior across
-- June, July, and August 2014.
--
-- Prediction point:
-- End of August 2014
--
-- Data source:
-- staging.train_raw
--
-- Scope:
-- Descriptive business analysis only.
--
-- This script does NOT use:
--   - Model predictions
--   - Risk scores
--   - Thresholds
--   - Customer risk tiers
--   - Retention recommendations
--
-- Analytical principles:
--   1. Missing information is not automatically treated as zero.
--   2. August represents the latest observed action-period behavior.
--   3. July -> August deterioration is explicitly analyzed.
--   4. June -> July -> August trajectories are analyzed separately.
--   5. Minimum-volume guardrails reduce misleading small segments.
--   6. Findings represent associations, not causal effects.
-- ============================================================


-- ============================================================
-- 01. OVERALL USAGE & ENGAGEMENT SNAPSHOT
-- ============================================================

SELECT
    COUNT(*) AS total_customers,

    ROUND(AVG(total_og_mou_8)::numeric, 2)
        AS avg_august_outgoing_mou,

    ROUND(AVG(total_ic_mou_8)::numeric, 2)
        AS avg_august_incoming_mou,

    ROUND(AVG(onnet_mou_8)::numeric, 2)
        AS avg_august_onnet_mou,

    ROUND(AVG(offnet_mou_8)::numeric, 2)
        AS avg_august_offnet_mou,

    ROUND(AVG(vol_2g_mb_8)::numeric, 2)
        AS avg_august_2g_mb,

    ROUND(AVG(vol_3g_mb_8)::numeric, 2)
        AS avg_august_3g_mb,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS overall_observed_churn_rate_pct

FROM staging.train_raw;


-- ============================================================
-- 02. OUTGOING VOICE USAGE PROFILE
-- ============================================================

WITH outgoing_profile AS (
    SELECT
        CASE
            WHEN total_og_mou_8 IS NULL
                THEN 'Outgoing usage missing'
            WHEN total_og_mou_8 = 0
                THEN 'Zero outgoing usage'
            WHEN total_og_mou_8 < 50
                THEN 'Very low'
            WHEN total_og_mou_8 < 200
                THEN 'Low'
            WHEN total_og_mou_8 < 500
                THEN 'Moderate'
            ELSE 'High'
        END AS outgoing_usage_band,

        total_og_mou_8,
        churn_probability

    FROM staging.train_raw
)

SELECT
    outgoing_usage_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_og_mou_8)::numeric,
        2
    ) AS avg_outgoing_mou,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM outgoing_profile

GROUP BY outgoing_usage_band

ORDER BY
    CASE outgoing_usage_band
        WHEN 'Outgoing usage missing' THEN 1
        WHEN 'Zero outgoing usage' THEN 2
        WHEN 'Very low' THEN 3
        WHEN 'Low' THEN 4
        WHEN 'Moderate' THEN 5
        WHEN 'High' THEN 6
        ELSE 7
    END;


-- ============================================================
-- 03. INCOMING VOICE USAGE PROFILE
-- ============================================================

WITH incoming_profile AS (
    SELECT
        CASE
            WHEN total_ic_mou_8 IS NULL
                THEN 'Incoming usage missing'
            WHEN total_ic_mou_8 = 0
                THEN 'Zero incoming usage'
            WHEN total_ic_mou_8 < 25
                THEN 'Very low'
            WHEN total_ic_mou_8 < 100
                THEN 'Low'
            WHEN total_ic_mou_8 < 250
                THEN 'Moderate'
            ELSE 'High'
        END AS incoming_usage_band,

        total_ic_mou_8,
        churn_probability

    FROM staging.train_raw
)

SELECT
    incoming_usage_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_ic_mou_8)::numeric,
        2
    ) AS avg_incoming_mou,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM incoming_profile

GROUP BY incoming_usage_band

ORDER BY
    CASE incoming_usage_band
        WHEN 'Incoming usage missing' THEN 1
        WHEN 'Zero incoming usage' THEN 2
        WHEN 'Very low' THEN 3
        WHEN 'Low' THEN 4
        WHEN 'Moderate' THEN 5
        WHEN 'High' THEN 6
        ELSE 7
    END;


-- ============================================================
-- 04. TOTAL VOICE ENGAGEMENT
-- ============================================================

WITH voice_engagement AS (
    SELECT
        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN 'Voice usage missing'

            WHEN total_og_mou_8 = 0
                 AND total_ic_mou_8 = 0
                THEN 'Zero voice engagement'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) < 100
                THEN 'Very low'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) < 400
                THEN 'Low'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) < 1000
                THEN 'Moderate'

            ELSE 'High'
        END AS voice_engagement_band,

        CASE
            WHEN total_og_mou_8 IS NOT NULL
                 OR total_ic_mou_8 IS NOT NULL
                THEN
                    COALESCE(total_og_mou_8, 0)
                    + COALESCE(total_ic_mou_8, 0)
            ELSE NULL
        END AS total_voice_mou,

        churn_probability

    FROM staging.train_raw
)

SELECT
    voice_engagement_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_voice_mou)::numeric,
        2
    ) AS avg_total_voice_mou,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM voice_engagement

GROUP BY voice_engagement_band

ORDER BY
    CASE voice_engagement_band
        WHEN 'Voice usage missing' THEN 1
        WHEN 'Zero voice engagement' THEN 2
        WHEN 'Very low' THEN 3
        WHEN 'Low' THEN 4
        WHEN 'Moderate' THEN 5
        WHEN 'High' THEN 6
        ELSE 7
    END;


-- ============================================================
-- 05. ON-NET VS OFF-NET USAGE
-- ============================================================

WITH network_usage AS (
    SELECT
        CASE
            WHEN onnet_mou_8 IS NULL
                 AND offnet_mou_8 IS NULL
                THEN 'Usage information missing'

            WHEN onnet_mou_8 = 0
                 AND offnet_mou_8 = 0
                THEN 'No network voice usage'

            WHEN COALESCE(onnet_mou_8, 0)
                 > COALESCE(offnet_mou_8, 0)
                THEN 'On-net dominant'

            WHEN COALESCE(offnet_mou_8, 0)
                 > COALESCE(onnet_mou_8, 0)
                THEN 'Off-net dominant'

            ELSE 'Balanced'
        END AS network_usage_profile,

        churn_probability

    FROM staging.train_raw
)

SELECT
    network_usage_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM network_usage

GROUP BY network_usage_profile

ORDER BY customers DESC;


-- ============================================================
-- 06. LOCAL VS STD USAGE
-- ============================================================

WITH local_std_usage AS (
    SELECT
        CASE
            WHEN
                loc_og_t2t_mou_8 IS NULL
                AND loc_og_t2m_mou_8 IS NULL
                AND std_og_t2t_mou_8 IS NULL
                AND std_og_t2m_mou_8 IS NULL
                THEN 'Usage information missing'

            WHEN
                COALESCE(loc_og_t2t_mou_8, 0)
                + COALESCE(loc_og_t2m_mou_8, 0)
                + COALESCE(std_og_t2t_mou_8, 0)
                + COALESCE(std_og_t2m_mou_8, 0) = 0
                THEN 'No local/STD outgoing usage'

            WHEN
                COALESCE(loc_og_t2t_mou_8, 0)
                + COALESCE(loc_og_t2m_mou_8, 0)
                >
                COALESCE(std_og_t2t_mou_8, 0)
                + COALESCE(std_og_t2m_mou_8, 0)
                THEN 'Local dominant'

            WHEN
                COALESCE(std_og_t2t_mou_8, 0)
                + COALESCE(std_og_t2m_mou_8, 0)
                >
                COALESCE(loc_og_t2t_mou_8, 0)
                + COALESCE(loc_og_t2m_mou_8, 0)
                THEN 'STD dominant'

            ELSE 'Balanced'
        END AS local_std_profile,

        churn_probability

    FROM staging.train_raw
)

SELECT
    local_std_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM local_std_usage

GROUP BY local_std_profile

ORDER BY customers DESC;


-- ============================================================
-- 07. ROAMING BEHAVIOR
-- ============================================================

WITH roaming_profile AS (
    SELECT
        CASE
            WHEN roam_og_mou_8 IS NULL
                 AND roam_ic_mou_8 IS NULL
                THEN 'Roaming information missing'

            WHEN roam_og_mou_8 = 0
                 AND roam_ic_mou_8 = 0
                THEN 'No roaming usage'

            WHEN (
                COALESCE(roam_og_mou_8, 0)
                + COALESCE(roam_ic_mou_8, 0)
            ) < 50
                THEN 'Low roaming usage'

            WHEN (
                COALESCE(roam_og_mou_8, 0)
                + COALESCE(roam_ic_mou_8, 0)
            ) < 200
                THEN 'Moderate roaming usage'

            ELSE 'High roaming usage'
        END AS roaming_usage_band,

        churn_probability

    FROM staging.train_raw
)

SELECT
    roaming_usage_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM roaming_profile

GROUP BY roaming_usage_band

ORDER BY customers DESC;


-- ============================================================
-- 08. OUTGOING VS INCOMING BALANCE
-- ============================================================
-- Describes whether customer voice behavior is more outgoing
-- or incoming oriented.
-- ============================================================

WITH voice_balance AS (
    SELECT
        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN 'Voice information missing'

            WHEN total_og_mou_8 = 0
                 AND total_ic_mou_8 = 0
                THEN 'No voice activity'

            WHEN COALESCE(total_og_mou_8, 0)
                 >= 2 * COALESCE(total_ic_mou_8, 0)
                THEN 'Strongly outgoing-oriented'

            WHEN COALESCE(total_og_mou_8, 0)
                 > COALESCE(total_ic_mou_8, 0)
                THEN 'Outgoing-oriented'

            WHEN COALESCE(total_ic_mou_8, 0)
                 >= 2 * COALESCE(total_og_mou_8, 0)
                THEN 'Strongly incoming-oriented'

            WHEN COALESCE(total_ic_mou_8, 0)
                 > COALESCE(total_og_mou_8, 0)
                THEN 'Incoming-oriented'

            ELSE 'Balanced'
        END AS voice_balance_profile,

        churn_probability

    FROM staging.train_raw
)

SELECT
    voice_balance_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM voice_balance

GROUP BY voice_balance_profile

ORDER BY customers DESC;


-- ============================================================
-- 09. VOICE ACTIVITY CONSISTENCY
-- ============================================================

WITH monthly_voice AS (
    SELECT
        id,

        CASE
            WHEN total_og_mou_6 IS NULL
                 AND total_ic_mou_6 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_6, 0)
                + COALESCE(total_ic_mou_6, 0)
        END AS voice_6,

        CASE
            WHEN total_og_mou_7 IS NULL
                 AND total_ic_mou_7 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_7, 0)
                + COALESCE(total_ic_mou_7, 0)
        END AS voice_7,

        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
        END AS voice_8,

        churn_probability

    FROM staging.train_raw
),

voice_consistency AS (
    SELECT
        CASE
            WHEN voice_6 IS NULL
                 AND voice_7 IS NULL
                 AND voice_8 IS NULL
                THEN 'Voice information missing'

            WHEN voice_6 = 0
                 AND voice_7 = 0
                 AND voice_8 = 0
                THEN 'Consistently zero'

            WHEN voice_6 < 100
                 AND voice_7 < 100
                 AND voice_8 < 100
                THEN 'Consistently very low'

            WHEN voice_6 >= 100
                 AND voice_7 >= 100
                 AND voice_8 >= 100
                THEN 'Consistently active'

            WHEN voice_6 < voice_7
                 AND voice_7 < voice_8
                THEN 'Consistently increasing'

            WHEN voice_6 > voice_7
                 AND voice_7 > voice_8
                THEN 'Consistently declining'

            ELSE 'Mixed / variable'
        END AS voice_consistency_profile,

        churn_probability

    FROM monthly_voice
)

SELECT
    voice_consistency_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM voice_consistency

GROUP BY voice_consistency_profile

ORDER BY customers DESC;


-- ============================================================
-- 10. JULY → AUGUST VOICE DETERIORATION
-- ============================================================

WITH voice_change AS (
    SELECT
        CASE
            WHEN total_og_mou_7 IS NULL
                 AND total_ic_mou_7 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_7, 0)
                + COALESCE(total_ic_mou_7, 0)
        END AS july_voice,

        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
        END AS august_voice,

        churn_probability

    FROM staging.train_raw
),

voice_change_profile AS (
    SELECT
        CASE
            WHEN july_voice IS NULL
                 OR august_voice IS NULL
                THEN 'Insufficient voice information'

            WHEN july_voice = 0
                 AND august_voice = 0
                THEN 'Zero in both months'

            WHEN july_voice = 0
                 AND august_voice > 0
                THEN 'Started/increased from zero'

            WHEN august_voice = 0
                 AND july_voice > 0
                THEN 'Dropped to zero'

            WHEN august_voice < july_voice * 0.50
                THEN 'Severe decline (>50%)'

            WHEN august_voice < july_voice * 0.80
                THEN 'Moderate decline (20-50%)'

            WHEN august_voice < july_voice
                THEN 'Mild decline (<20%)'

            WHEN august_voice = july_voice
                THEN 'No change'

            ELSE 'Increased'
        END AS voice_change_profile,

        CASE
            WHEN july_voice > 0
                THEN
                    100.0
                    * (august_voice - july_voice)
                    / july_voice
            ELSE NULL
        END AS voice_change_pct,

        churn_probability

    FROM voice_change
)

SELECT
    voice_change_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(voice_change_pct)::numeric,
        2
    ) AS avg_voice_change_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM voice_change_profile

GROUP BY voice_change_profile

ORDER BY
    CASE voice_change_profile
        WHEN 'Insufficient voice information' THEN 1
        WHEN 'Zero in both months' THEN 2
        WHEN 'Dropped to zero' THEN 3
        WHEN 'Severe decline (>50%)' THEN 4
        WHEN 'Moderate decline (20-50%)' THEN 5
        WHEN 'Mild decline (<20%)' THEN 6
        WHEN 'No change' THEN 7
        WHEN 'Increased' THEN 8
        WHEN 'Started/increased from zero' THEN 9
        ELSE 10
    END;


-- ============================================================
-- 11. MULTI-MONTH VOICE ENGAGEMENT TRAJECTORY
-- ============================================================

WITH monthly_voice AS (
    SELECT
        CASE
            WHEN total_og_mou_6 IS NULL
                 AND total_ic_mou_6 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_6, 0)
                + COALESCE(total_ic_mou_6, 0)
        END AS voice_6,

        CASE
            WHEN total_og_mou_7 IS NULL
                 AND total_ic_mou_7 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_7, 0)
                + COALESCE(total_ic_mou_7, 0)
        END AS voice_7,

        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN NULL
            ELSE
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
        END AS voice_8,

        churn_probability

    FROM staging.train_raw
),

voice_trajectory AS (
    SELECT
        CASE
            WHEN voice_6 IS NULL
                 AND voice_7 IS NULL
                 AND voice_8 IS NULL
                THEN 'Insufficient information'

            WHEN voice_6 = 0
                 AND voice_7 = 0
                 AND voice_8 = 0
                THEN 'Consistently inactive'

            WHEN voice_6 < 100
                 AND voice_7 < 100
                 AND voice_8 < 100
                THEN 'Consistently low'

            WHEN voice_6 >= 100
                 AND voice_7 >= 100
                 AND voice_8 >= 100
                THEN 'Persistently active'

            WHEN voice_6 < voice_7
                 AND voice_7 < voice_8
                THEN 'Increasing trajectory'

            WHEN voice_6 > voice_7
                 AND voice_7 > voice_8
                THEN 'Declining trajectory'

            ELSE 'Variable trajectory'
        END AS voice_trajectory,

        churn_probability

    FROM monthly_voice
)

SELECT
    voice_trajectory,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM voice_trajectory

GROUP BY voice_trajectory

ORDER BY customers DESC;


-- ============================================================
-- 12. DATA USAGE PROFILE
-- ============================================================

WITH data_usage AS (
    SELECT
        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN 'Data usage missing'

            WHEN vol_2g_mb_8 = 0
                 AND vol_3g_mb_8 = 0
                THEN 'Zero data usage'

            WHEN (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) < 10
                THEN 'Very low'

            WHEN (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) < 100
                THEN 'Low'

            WHEN (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) < 500
                THEN 'Moderate'

            ELSE 'High'
        END AS data_usage_band,

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN NULL
            ELSE
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
        END AS total_data_mb,

        churn_probability

    FROM staging.train_raw
)

SELECT
    data_usage_band,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(total_data_mb)::numeric,
        2
    ) AS avg_total_data_mb,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM data_usage

GROUP BY data_usage_band

ORDER BY
    CASE data_usage_band
        WHEN 'Data usage missing' THEN 1
        WHEN 'Zero data usage' THEN 2
        WHEN 'Very low' THEN 3
        WHEN 'Low' THEN 4
        WHEN 'Moderate' THEN 5
        WHEN 'High' THEN 6
        ELSE 7
    END;


-- ============================================================
-- 13. 2G VS 3G DATA ENGAGEMENT
-- ============================================================

WITH data_mix AS (
    SELECT
        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN 'Data information missing'

            WHEN vol_2g_mb_8 = 0
                 AND vol_3g_mb_8 = 0
                THEN 'No data usage'

            WHEN COALESCE(vol_2g_mb_8, 0)
                 > COALESCE(vol_3g_mb_8, 0)
                THEN '2G dominant'

            WHEN COALESCE(vol_3g_mb_8, 0)
                 > COALESCE(vol_2g_mb_8, 0)
                THEN '3G dominant'

            ELSE 'Balanced 2G/3G'
        END AS data_mix_profile,

        churn_probability

    FROM staging.train_raw
)

SELECT
    data_mix_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM data_mix

GROUP BY data_mix_profile

ORDER BY customers DESC;


-- ============================================================
-- 14. DATA ENGAGEMENT CONSISTENCY
-- ============================================================

WITH monthly_data AS (
    SELECT
        CASE
            WHEN vol_2g_mb_6 IS NULL
                 AND vol_3g_mb_6 IS NULL
                THEN NULL
            ELSE
                COALESCE(vol_2g_mb_6, 0)
                + COALESCE(vol_3g_mb_6, 0)
        END AS data_6,

        CASE
            WHEN vol_2g_mb_7 IS NULL
                 AND vol_3g_mb_7 IS NULL
                THEN NULL
            ELSE
                COALESCE(vol_2g_mb_7, 0)
                + COALESCE(vol_3g_mb_7, 0)
        END AS data_7,

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN NULL
            ELSE
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
        END AS data_8,

        churn_probability

    FROM staging.train_raw
),

data_consistency AS (
    SELECT
        CASE
            WHEN data_6 IS NULL
                 AND data_7 IS NULL
                 AND data_8 IS NULL
                THEN 'Data information missing'

            WHEN data_6 = 0
                 AND data_7 = 0
                 AND data_8 = 0
                THEN 'Consistently zero'

            WHEN data_6 < 10
                 AND data_7 < 10
                 AND data_8 < 10
                THEN 'Consistently very low'

            WHEN data_6 >= 10
                 AND data_7 >= 10
                 AND data_8 >= 10
                THEN 'Consistently active'

            WHEN data_6 < data_7
                 AND data_7 < data_8
                THEN 'Consistently increasing'

            WHEN data_6 > data_7
                 AND data_7 > data_8
                THEN 'Consistently declining'

            ELSE 'Mixed / variable'
        END AS data_consistency_profile,

        churn_probability

    FROM monthly_data
)

SELECT
    data_consistency_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM data_consistency

GROUP BY data_consistency_profile

ORDER BY customers DESC;


-- ============================================================
-- 15. JULY → AUGUST DATA DETERIORATION
-- ============================================================

WITH data_change AS (
    SELECT
        CASE
            WHEN vol_2g_mb_7 IS NULL
                 AND vol_3g_mb_7 IS NULL
                THEN NULL
            ELSE
                COALESCE(vol_2g_mb_7, 0)
                + COALESCE(vol_3g_mb_7, 0)
        END AS july_data,

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN NULL
            ELSE
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
        END AS august_data,

        churn_probability

    FROM staging.train_raw
),

data_change_profile AS (
    SELECT
        CASE
            WHEN july_data IS NULL
                 OR august_data IS NULL
                THEN 'Insufficient data information'

            WHEN july_data = 0
                 AND august_data = 0
                THEN 'Zero in both months'

            WHEN july_data = 0
                 AND august_data > 0
                THEN 'Started/increased from zero'

            WHEN august_data = 0
                 AND july_data > 0
                THEN 'Dropped to zero'

            WHEN august_data < july_data * 0.50
                THEN 'Severe decline (>50%)'

            WHEN august_data < july_data * 0.80
                THEN 'Moderate decline (20-50%)'

            WHEN august_data < july_data
                THEN 'Mild decline (<20%)'

            WHEN august_data = july_data
                THEN 'No change'

            ELSE 'Increased'
        END AS data_change_profile,

        CASE
            WHEN july_data > 0
                THEN
                    100.0
                    * (august_data - july_data)
                    / july_data
            ELSE NULL
        END AS data_change_pct,

        churn_probability

    FROM data_change
)

SELECT
    data_change_profile,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        AVG(data_change_pct)::numeric,
        2
    ) AS avg_data_change_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM data_change_profile

GROUP BY data_change_profile

ORDER BY
    CASE data_change_profile
        WHEN 'Insufficient data information' THEN 1
        WHEN 'Zero in both months' THEN 2
        WHEN 'Dropped to zero' THEN 3
        WHEN 'Severe decline (>50%)' THEN 4
        WHEN 'Moderate decline (20-50%)' THEN 5
        WHEN 'Mild decline (<20%)' THEN 6
        WHEN 'No change' THEN 7
        WHEN 'Increased' THEN 8
        WHEN 'Started/increased from zero' THEN 9
        ELSE 10
    END;


-- ============================================================
-- 16. VOICE × DATA ENGAGEMENT MATRIX
-- ============================================================
-- Uses separate voice and data dimensions rather than adding
-- MOU and MB together.
-- ============================================================

WITH engagement AS (
    SELECT
        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN 'Voice information missing'

            WHEN COALESCE(total_og_mou_8, 0)
                 + COALESCE(total_ic_mou_8, 0) = 0
                THEN 'Low voice'

            WHEN COALESCE(total_og_mou_8, 0)
                 + COALESCE(total_ic_mou_8, 0) < 400
                THEN 'Moderate voice'

            ELSE 'High voice'
        END AS voice_band,

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN 'Data information missing'

            WHEN COALESCE(vol_2g_mb_8, 0)
                 + COALESCE(vol_3g_mb_8, 0) = 0
                THEN 'Low data'

            WHEN COALESCE(vol_2g_mb_8, 0)
                 + COALESCE(vol_3g_mb_8, 0) < 100
                THEN 'Moderate data'

            ELSE 'High data'
        END AS data_band,

        churn_probability

    FROM staging.train_raw
),

matrix_summary AS (
    SELECT
        voice_band,
        data_band,

        COUNT(*) AS customers,

        ROUND(
            100.0 * COUNT(*)
            / SUM(COUNT(*)) OVER (),
            2
        ) AS customer_share_pct,

        ROUND(
            100.0 * AVG(churn_probability::numeric),
            2
        ) AS observed_churn_rate_pct

    FROM engagement

    GROUP BY
        voice_band,
        data_band
)

SELECT
    voice_band,
    data_band,
    customers,
    customer_share_pct,
    observed_churn_rate_pct

FROM matrix_summary

WHERE customers >= 100

ORDER BY
    observed_churn_rate_pct DESC;


-- ============================================================
-- 17. ENGAGEMENT TRAJECTORY
-- ============================================================
-- Uses normalized month-level components so voice MOU and data MB
-- are not directly added together.
--
-- Voice component:
--   active = 1
--   inactive = 0
--
-- Data component:
--   active = 1
--   inactive = 0
--
-- The resulting score represents breadth of engagement, not
-- absolute usage volume.
-- ============================================================

WITH monthly_components AS (
    SELECT
        CASE
            WHEN total_og_mou_6 IS NULL
                 AND total_ic_mou_6 IS NULL
                THEN NULL
            WHEN COALESCE(total_og_mou_6, 0)
                 + COALESCE(total_ic_mou_6, 0) > 0
                THEN 1
            ELSE 0
        END AS voice_active_6,

        CASE
            WHEN total_og_mou_7 IS NULL
                 AND total_ic_mou_7 IS NULL
                THEN NULL
            WHEN COALESCE(total_og_mou_7, 0)
                 + COALESCE(total_ic_mou_7, 0) > 0
                THEN 1
            ELSE 0
        END AS voice_active_7,

        CASE
            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                THEN NULL
            WHEN COALESCE(total_og_mou_8, 0)
                 + COALESCE(total_ic_mou_8, 0) > 0
                THEN 1
            ELSE 0
        END AS voice_active_8,

        CASE
            WHEN vol_2g_mb_6 IS NULL
                 AND vol_3g_mb_6 IS NULL
                THEN NULL
            WHEN COALESCE(vol_2g_mb_6, 0)
                 + COALESCE(vol_3g_mb_6, 0) > 0
                THEN 1
            ELSE 0
        END AS data_active_6,

        CASE
            WHEN vol_2g_mb_7 IS NULL
                 AND vol_3g_mb_7 IS NULL
                THEN NULL
            WHEN COALESCE(vol_2g_mb_7, 0)
                 + COALESCE(vol_3g_mb_7, 0) > 0
                THEN 1
            ELSE 0
        END AS data_active_7,

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN NULL
            WHEN COALESCE(vol_2g_mb_8, 0)
                 + COALESCE(vol_3g_mb_8, 0) > 0
                THEN 1
            ELSE 0
        END AS data_active_8,

        churn_probability

    FROM staging.train_raw
),

engagement_score AS (
    SELECT
        (
            COALESCE(voice_active_6, 0)
            + COALESCE(data_active_6, 0)
        ) AS engagement_6,

        (
            COALESCE(voice_active_7, 0)
            + COALESCE(data_active_7, 0)
        ) AS engagement_7,

        (
            COALESCE(voice_active_8, 0)
            + COALESCE(data_active_8, 0)
        ) AS engagement_8,

        voice_active_6,
        voice_active_7,
        voice_active_8,
        data_active_6,
        data_active_7,
        data_active_8,

        churn_probability

    FROM monthly_components
),

engagement_trajectory AS (
    SELECT
        CASE
            WHEN voice_active_6 IS NULL
                 AND voice_active_7 IS NULL
                 AND voice_active_8 IS NULL
                 AND data_active_6 IS NULL
                 AND data_active_7 IS NULL
                 AND data_active_8 IS NULL
                THEN 'Insufficient information'

            WHEN engagement_6 = 0
                 AND engagement_7 = 0
                 AND engagement_8 = 0
                THEN 'Consistently inactive'

            WHEN engagement_6 < engagement_7
                 AND engagement_7 < engagement_8
                THEN 'Increasing engagement'

            WHEN engagement_6 > engagement_7
                 AND engagement_7 > engagement_8
                THEN 'Declining engagement'

            WHEN engagement_8 = 0
                THEN 'Inactive in August'

            WHEN engagement_6 = engagement_7
                 AND engagement_7 = engagement_8
                THEN 'Stable engagement'

            ELSE 'Variable engagement'
        END AS engagement_trajectory,

        churn_probability

    FROM engagement_score
)

SELECT
    engagement_trajectory,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM engagement_trajectory

GROUP BY engagement_trajectory

ORDER BY customers DESC;


-- ============================================================
-- 18. LOW-ENGAGEMENT CUSTOMER CONCENTRATION
-- ============================================================

WITH engagement AS (
    SELECT
        CASE
            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) = 0
            AND (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) = 0
                THEN 'Zero voice and data engagement'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) < 100
            AND (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) < 100
                THEN 'Very low voice and data engagement'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) < 400
            AND (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) < 500
                THEN 'Moderate engagement'

            ELSE 'Higher engagement'
        END AS engagement_group,

        churn_probability

    FROM staging.train_raw
)

SELECT
    engagement_group,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM engagement

GROUP BY engagement_group

ORDER BY
    CASE engagement_group
        WHEN 'Zero voice and data engagement' THEN 1
        WHEN 'Very low voice and data engagement' THEN 2
        WHEN 'Moderate engagement' THEN 3
        WHEN 'Higher engagement' THEN 4
        ELSE 5
    END;


-- ============================================================
-- 19. EXECUTIVE ENGAGEMENT SIGNALS
-- ============================================================

WITH engagement_signals AS (

    SELECT
        'Outgoing voice' AS profile_dimension,

        CASE
            WHEN total_og_mou_8 IS NULL
                THEN 'Missing'
            WHEN total_og_mou_8 = 0
                THEN 'Zero'
            WHEN total_og_mou_8 < 100
                THEN 'Low'
            WHEN total_og_mou_8 < 400
                THEN 'Moderate'
            ELSE 'High'
        END AS profile_group,

        churn_probability

    FROM staging.train_raw


    UNION ALL


    SELECT
        'Incoming voice',

        CASE
            WHEN total_ic_mou_8 IS NULL
                THEN 'Missing'
            WHEN total_ic_mou_8 = 0
                THEN 'Zero'
            WHEN total_ic_mou_8 < 50
                THEN 'Low'
            WHEN total_ic_mou_8 < 200
                THEN 'Moderate'
            ELSE 'High'
        END,

        churn_probability

    FROM staging.train_raw


    UNION ALL


    SELECT
        'Data engagement',

        CASE
            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN 'Missing'
            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) = 0
                THEN 'Zero'
            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 100
                THEN 'Low'
            WHEN COALESCE(vol_2g_mb_8, 0)
               + COALESCE(vol_3g_mb_8, 0) < 500
                THEN 'Moderate'
            ELSE 'High'
        END,

        churn_probability

    FROM staging.train_raw


    UNION ALL


    SELECT
        'Voice deterioration',

        CASE
            WHEN total_og_mou_7 IS NULL
                 AND total_ic_mou_7 IS NULL
                 THEN 'Missing July information'

            WHEN total_og_mou_8 IS NULL
                 AND total_ic_mou_8 IS NULL
                 THEN 'Missing August information'

            WHEN (
                COALESCE(total_og_mou_7, 0)
                + COALESCE(total_ic_mou_7, 0)
            ) > 0
            AND (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) = 0
                THEN 'Dropped to zero'

            WHEN (
                COALESCE(total_og_mou_8, 0)
                + COALESCE(total_ic_mou_8, 0)
            ) <
            (
                COALESCE(total_og_mou_7, 0)
                + COALESCE(total_ic_mou_7, 0)
            )
                THEN 'Declined'

            ELSE 'Stable / increased'
        END,

        churn_probability

    FROM staging.train_raw


    UNION ALL


    SELECT
        'Data deterioration',

        CASE
            WHEN vol_2g_mb_7 IS NULL
                 AND vol_3g_mb_7 IS NULL
                THEN 'Missing July information'

            WHEN vol_2g_mb_8 IS NULL
                 AND vol_3g_mb_8 IS NULL
                THEN 'Missing August information'

            WHEN (
                COALESCE(vol_2g_mb_7, 0)
                + COALESCE(vol_3g_mb_7, 0)
            ) > 0
            AND (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) = 0
                THEN 'Dropped to zero'

            WHEN (
                COALESCE(vol_2g_mb_8, 0)
                + COALESCE(vol_3g_mb_8, 0)
            ) <
            (
                COALESCE(vol_2g_mb_7, 0)
                + COALESCE(vol_3g_mb_7, 0)
            )
                THEN 'Declined'

            ELSE 'Stable / increased'
        END,

        churn_probability

    FROM staging.train_raw
)

SELECT
    profile_dimension,

    profile_group,

    COUNT(*) AS customers,

    ROUND(
        100.0 * COUNT(*)
        / SUM(COUNT(*))
          OVER (PARTITION BY profile_dimension),
        2
    ) AS customer_share_pct,

    ROUND(
        100.0 * AVG(churn_probability::numeric),
        2
    ) AS observed_churn_rate_pct

FROM engagement_signals

GROUP BY
    profile_dimension,
    profile_group

HAVING COUNT(*) >= 100

ORDER BY
    profile_dimension,
    observed_churn_rate_pct DESC;


-- ============================================================
-- 20. ANALYTICAL QUALITY GATE
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
-- END OF USAGE & ENGAGEMENT ANALYSIS
-- ============================================================