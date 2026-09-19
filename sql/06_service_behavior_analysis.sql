/*==============================================================================
    CHURNIQ — SQL BUSINESS ANALYSIS
    06_SERVICE_BEHAVIOR_ANALYSIS.SQL

    Purpose:
    Analyze customer service adoption, service mix, service breadth,
    service persistence, customer value, and observed churn patterns.

    Primary business question:
    How do customers adopt different telecom services, how broad is their
    recorded service adoption, and how do service-adoption patterns differ
    across customer-value and observed-churn groups?

    Analytical boundaries:
    - Focuses on service adoption and service mix.
    - Detailed voice/data usage is covered in analysis #04.
    - Recharge value and ARPU are covered in analysis #05.
    - Missing service information is NOT automatically treated as zero.
    - Observed churn is descriptive only.
    - No model predictions, risk scores, thresholds, or retention tiers.
==============================================================================*/


/*==============================================================================
    01. OVERALL SERVICE ADOPTION SNAPSHOT

    High-level August service participation.
==============================================================================*/

SELECT
    COUNT(*) AS customers,

    COUNT(*) FILTER (
        WHERE fb_user_8 = 1
    ) AS social_service_users,

    COUNT(*) FILTER (
        WHERE night_pck_user_8 = 1
    ) AS night_service_users,

    COUNT(*) FILTER (
        WHERE monthly_2g_8 > 0
    ) AS monthly_2g_users,

    COUNT(*) FILTER (
        WHERE sachet_2g_8 > 0
    ) AS sachet_2g_users,

    COUNT(*) FILTER (
        WHERE monthly_3g_8 > 0
    ) AS monthly_3g_users,

    COUNT(*) FILTER (
        WHERE sachet_3g_8 > 0
    ) AS sachet_3g_users,

    COUNT(*) FILTER (
        WHERE total_rech_data_8 > 0
    ) AS data_recharge_users,

    COUNT(*) FILTER (
        WHERE total_rech_data_8 IS NULL
    ) AS missing_data_recharge_information

FROM staging.train_raw;


/*==============================================================================
    02. SOCIAL-SERVICE ADOPTION

    Missing values remain separate from explicit non-adoption.
==============================================================================*/

WITH social_service AS (
    SELECT
        CASE
            WHEN fb_user_8 = 1
                THEN 'Adopted'
            WHEN fb_user_8 = 0
                THEN 'Not adopted'
            ELSE 'Information unavailable'
        END AS adoption_status,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        adoption_status,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM social_service
    GROUP BY adoption_status
)
SELECT
    adoption_status,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    03. NIGHT-SERVICE ADOPTION

    Missing values remain separate from explicit non-adoption.
==============================================================================*/

WITH night_service AS (
    SELECT
        CASE
            WHEN night_pck_user_8 = 1
                THEN 'Adopted'
            WHEN night_pck_user_8 = 0
                THEN 'Not adopted'
            ELSE 'Information unavailable'
        END AS adoption_status,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        adoption_status,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM night_service
    GROUP BY adoption_status
)
SELECT
    adoption_status,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    04. 2G SERVICE ADOPTION

    Combines monthly and sachet 2G service participation.

    NULL values are kept separate from explicit zero values.
==============================================================================*/

WITH service_profile AS (
    SELECT
        CASE
            WHEN monthly_2g_8 > 0
             AND sachet_2g_8 > 0
                THEN 'Monthly + Sachet 2G'

            WHEN monthly_2g_8 > 0
                THEN 'Monthly 2G'

            WHEN sachet_2g_8 > 0
                THEN 'Sachet 2G'

            WHEN monthly_2g_8 = 0
             AND sachet_2g_8 = 0
                THEN 'No recorded 2G service'

            ELSE 'Information unavailable'
        END AS service_group,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        service_group,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM service_profile
    GROUP BY service_group
)
SELECT
    service_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    05. 3G SERVICE ADOPTION

    Combines monthly and sachet 3G service participation.
==============================================================================*/

WITH service_profile AS (
    SELECT
        CASE
            WHEN monthly_3g_8 > 0
             AND sachet_3g_8 > 0
                THEN 'Monthly + Sachet 3G'

            WHEN monthly_3g_8 > 0
                THEN 'Monthly 3G'

            WHEN sachet_3g_8 > 0
                THEN 'Sachet 3G'

            WHEN monthly_3g_8 = 0
             AND sachet_3g_8 = 0
                THEN 'No recorded 3G service'

            ELSE 'Information unavailable'
        END AS service_group,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        service_group,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM service_profile
    GROUP BY service_group
)
SELECT
    service_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    06. MONTHLY VS SACHET SERVICE ADOPTION

    Compares monthly-oriented and sachet-oriented service adoption.

    Minimum-volume guardrail:
    Groups with fewer than 100 customers are excluded.
==============================================================================*/

WITH service_mix AS (
    SELECT
        CASE
            WHEN
                (monthly_2g_8 > 0 OR monthly_3g_8 > 0)
                AND
                (sachet_2g_8 > 0 OR sachet_3g_8 > 0)
                THEN 'Monthly + Sachet services'

            WHEN
                monthly_2g_8 > 0
                OR monthly_3g_8 > 0
                THEN 'Monthly-oriented services'

            WHEN
                sachet_2g_8 > 0
                OR sachet_3g_8 > 0
                THEN 'Sachet-oriented services'

            WHEN
                monthly_2g_8 = 0
                AND sachet_2g_8 = 0
                AND monthly_3g_8 = 0
                AND sachet_3g_8 = 0
                THEN 'No recorded pack adoption'

            ELSE 'Information unavailable'
        END AS service_mix_group,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        service_mix_group,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM service_mix
    GROUP BY service_mix_group
    HAVING COUNT(*) >= 100
)
SELECT
    service_mix_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    07. DATA-RECHARGE SERVICE PARTICIPATION

    NULL = information unavailable.
    Zero = explicitly recorded as no data recharge.
==============================================================================*/

WITH data_recharge AS (
    SELECT
        CASE
            WHEN total_rech_data_8 IS NULL
                THEN 'Information unavailable'

            WHEN total_rech_data_8 = 0
                THEN 'No recorded data recharge'

            WHEN total_rech_data_8 > 0
                THEN 'Data recharge recorded'

            ELSE 'Other'
        END AS data_recharge_status,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        data_recharge_status,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM data_recharge
    GROUP BY data_recharge_status
)
SELECT
    data_recharge_status,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    08. MULTI-MONTH DATA-SERVICE ADOPTION

    Examines persistence and change in recorded data-recharge participation
    across June, July and August.
==============================================================================*/

WITH service_trajectory AS (
    SELECT
        CASE
            WHEN total_rech_data_6 > 0
             AND total_rech_data_7 > 0
             AND total_rech_data_8 > 0
                THEN 'Consistently recorded data recharge'

            WHEN total_rech_data_6 IS NULL
             AND total_rech_data_7 IS NULL
             AND total_rech_data_8 IS NULL
                THEN 'No recorded information'

            WHEN total_rech_data_8 > 0
             AND (
                    total_rech_data_6 = 0
                 OR total_rech_data_7 = 0
                 OR total_rech_data_6 IS NULL
                 OR total_rech_data_7 IS NULL
             )
                THEN 'Recent data-service adoption'

            WHEN total_rech_data_8 = 0
             AND (
                    total_rech_data_6 > 0
                 OR total_rech_data_7 > 0
             )
                THEN 'Recent data-service reduction'

            ELSE 'Variable / incomplete trajectory'
        END AS trajectory_group,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        trajectory_group,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM service_trajectory
    GROUP BY trajectory_group
)
SELECT
    trajectory_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    09. RECORDED SERVICE BREADTH

    Counts positive recorded adoption across seven service categories.

    Categories:
    1. Social service
    2. Night service
    3. Monthly 2G
    4. Sachet 2G
    5. Monthly 3G
    6. Sachet 3G
    7. Data recharge

    IMPORTANT:
    A separate information-availability flag is retained so that a low
    breadth score is not automatically interpreted as low actual adoption.
==============================================================================*/

WITH breadth AS (
    SELECT

        (
            CASE WHEN fb_user_8 = 1 THEN 1 ELSE 0 END
          + CASE WHEN night_pck_user_8 = 1 THEN 1 ELSE 0 END
          + CASE WHEN monthly_2g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN sachet_2g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN monthly_3g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN sachet_3g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN total_rech_data_8 > 0 THEN 1 ELSE 0 END
        ) AS service_breadth,

        (
            CASE WHEN fb_user_8 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN night_pck_user_8 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN monthly_2g_8 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN sachet_2g_8 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN monthly_3g_8 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN sachet_3g_8 IS NOT NULL THEN 1 ELSE 0 END
          + CASE WHEN total_rech_data_8 IS NOT NULL THEN 1 ELSE 0 END
        ) AS available_service_fields,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT

        CASE
            WHEN available_service_fields = 0
                THEN 'Information unavailable'

            WHEN service_breadth = 0
                THEN '0 recorded services'

            WHEN service_breadth = 1
                THEN '1 recorded service'

            WHEN service_breadth = 2
                THEN '2 recorded services'

            WHEN service_breadth <= 4
                THEN '3-4 recorded services'

            ELSE '5+ recorded services'
        END AS breadth_group,

        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM breadth
    GROUP BY breadth_group
)
SELECT
    breadth_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    10. SERVICE-MIX ARCHETYPES

    Uses ONLY service-adoption variables.

    This intentionally avoids voice-usage fields already analyzed in #04.

    Archetypes:
    - Data-service oriented
    - Pack/add-on oriented
    - Social/night oriented
    - Multi-service
    - Limited recorded adoption
    - Information unavailable
==============================================================================*/

WITH service_archetypes AS (
    SELECT

        CASE
            WHEN
                total_rech_data_8 > 0
                AND (
                    fb_user_8 = 1
                    OR night_pck_user_8 = 1
                    OR monthly_2g_8 > 0
                    OR sachet_2g_8 > 0
                    OR monthly_3g_8 > 0
                    OR sachet_3g_8 > 0
                )
                THEN 'Multi-service adopter'

            WHEN
                total_rech_data_8 > 0
                THEN 'Data-service oriented'

            WHEN
                fb_user_8 = 1
                OR night_pck_user_8 = 1
                OR monthly_2g_8 > 0
                OR sachet_2g_8 > 0
                OR monthly_3g_8 > 0
                OR sachet_3g_8 > 0
                THEN 'Pack / add-on oriented'

            WHEN
                total_rech_data_8 = 0
                AND (
                    fb_user_8 = 0
                    OR fb_user_8 IS NULL
                )
                AND (
                    night_pck_user_8 = 0
                    OR night_pck_user_8 IS NULL
                )
                AND (
                    monthly_2g_8 = 0
                    OR monthly_2g_8 IS NULL
                )
                AND (
                    sachet_2g_8 = 0
                    OR sachet_2g_8 IS NULL
                )
                AND (
                    monthly_3g_8 = 0
                    OR monthly_3g_8 IS NULL
                )
                AND (
                    sachet_3g_8 = 0
                    OR sachet_3g_8 IS NULL
                )
                THEN 'Limited recorded adoption'

            ELSE 'Information unavailable'
        END AS service_archetype,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        service_archetype,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM service_archetypes
    GROUP BY service_archetype
    HAVING COUNT(*) >= 100
)
SELECT
    service_archetype,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    11. SERVICE ADOPTION × CUSTOMER VALUE

    Connects recorded service breadth with August ARPU.

    ARPU is treated as a customer-value proxy, not booked revenue.

    Minimum-volume guardrail:
    Groups with fewer than 100 customers are excluded.
==============================================================================*/

WITH customer_profile AS (
    SELECT

        CASE
            WHEN arpu_8 < 0
                THEN 'Negative ARPU'

            WHEN arpu_8 = 0
                THEN 'Zero ARPU'

            WHEN arpu_8 < 200
                THEN 'Lower value'

            WHEN arpu_8 < 500
                THEN 'Mid value'

            ELSE 'Higher value'
        END AS value_band,

        (
            CASE WHEN fb_user_8 = 1 THEN 1 ELSE 0 END
          + CASE WHEN night_pck_user_8 = 1 THEN 1 ELSE 0 END
          + CASE WHEN monthly_2g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN sachet_2g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN monthly_3g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN sachet_3g_8 > 0 THEN 1 ELSE 0 END
          + CASE WHEN total_rech_data_8 > 0 THEN 1 ELSE 0 END
        ) AS service_breadth,

        churn_probability

    FROM staging.train_raw
),
profile AS (
    SELECT

        value_band,

        CASE
            WHEN service_breadth = 0
                THEN '0 recorded services'

            WHEN service_breadth = 1
                THEN '1 recorded service'

            WHEN service_breadth = 2
                THEN '2 recorded services'

            WHEN service_breadth <= 4
                THEN '3-4 recorded services'

            ELSE '5+ recorded services'
        END AS breadth_group,

        churn_probability

    FROM customer_profile
),
summary AS (
    SELECT
        value_band,
        breadth_group,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM profile
    GROUP BY value_band, breadth_group
    HAVING COUNT(*) >= 100
)
SELECT
    value_band,
    breadth_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    12. SERVICE ADOPTION × OBSERVED CHURN

    Compares major service groups against observed churn.

    Churn contribution is calculated against ALL observed churners so the
    result can be interpreted consistently across service dimensions.

    Minimum-volume guardrail:
    Groups with fewer than 100 customers are excluded.
==============================================================================*/

WITH service_groups AS (

    SELECT
        'Social service' AS service_dimension,

        CASE
            WHEN fb_user_8 = 1
                THEN 'Adopted'
            WHEN fb_user_8 = 0
                THEN 'Not adopted'
            ELSE 'Information unavailable'
        END AS service_group,

        churn_probability

    FROM staging.train_raw

    UNION ALL

    SELECT
        'Night service',

        CASE
            WHEN night_pck_user_8 = 1
                THEN 'Adopted'
            WHEN night_pck_user_8 = 0
                THEN 'Not adopted'
            ELSE 'Information unavailable'
        END,

        churn_probability

    FROM staging.train_raw

    UNION ALL

    SELECT
        'Data recharge service',

        CASE
            WHEN total_rech_data_8 IS NULL
                THEN 'Information unavailable'
            WHEN total_rech_data_8 = 0
                THEN 'No recorded data recharge'
            ELSE 'Data recharge recorded'
        END,

        churn_probability

    FROM staging.train_raw

    UNION ALL

    SELECT
        '2G service',

        CASE
            WHEN monthly_2g_8 > 0
             OR sachet_2g_8 > 0
                THEN 'Any recorded 2G service'

            WHEN monthly_2g_8 = 0
             AND sachet_2g_8 = 0
                THEN 'No recorded 2G service'

            ELSE 'Information unavailable'
        END,

        churn_probability

    FROM staging.train_raw

    UNION ALL

    SELECT
        '3G service',

        CASE
            WHEN monthly_3g_8 > 0
             OR sachet_3g_8 > 0
                THEN 'Any recorded 3G service'

            WHEN monthly_3g_8 = 0
             AND sachet_3g_8 = 0
                THEN 'No recorded 3G service'

            ELSE 'Information unavailable'
        END,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        service_dimension,
        service_group,

        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM service_groups
    GROUP BY service_dimension, service_group
    HAVING COUNT(*) >= 100
)
SELECT
    service_dimension,
    service_group,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        (
            100.0 * churners
            / NULLIF(
                SUM(churners) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_contribution_pct

FROM summary
ORDER BY
    observed_churn_rate_pct DESC,
    customers DESC;


/*==============================================================================
    13. SERVICE ADOPTION CONSISTENCY

    Evaluates whether recorded service adoption persists across
    June, July and August.

    Covers:
    - Social service
    - Night service
    - Data recharge service

    Missing information remains explicitly identified.
==============================================================================*/

WITH consistency AS (
    SELECT

        CASE
            WHEN fb_user_6 = 1
             AND fb_user_7 = 1
             AND fb_user_8 = 1
                THEN 'Social — consistently adopted'

            WHEN fb_user_6 = 0
             AND fb_user_7 = 0
             AND fb_user_8 = 0
                THEN 'Social — consistently not adopted'

            WHEN
                fb_user_6 IS NULL
                OR fb_user_7 IS NULL
                OR fb_user_8 IS NULL
                THEN 'Social — incomplete information'

            ELSE 'Social — variable adoption'
        END AS social_consistency,

        CASE
            WHEN night_pck_user_6 = 1
             AND night_pck_user_7 = 1
             AND night_pck_user_8 = 1
                THEN 'Night — consistently adopted'

            WHEN night_pck_user_6 = 0
             AND night_pck_user_7 = 0
             AND night_pck_user_8 = 0
                THEN 'Night — consistently not adopted'

            WHEN
                night_pck_user_6 IS NULL
                OR night_pck_user_7 IS NULL
                OR night_pck_user_8 IS NULL
                THEN 'Night — incomplete information'

            ELSE 'Night — variable adoption'
        END AS night_consistency,

        CASE
            WHEN total_rech_data_6 > 0
             AND total_rech_data_7 > 0
             AND total_rech_data_8 > 0
                THEN 'Data — consistently recorded'

            WHEN total_rech_data_6 = 0
             AND total_rech_data_7 = 0
             AND total_rech_data_8 = 0
                THEN 'Data — consistently no recorded recharge'

            WHEN
                total_rech_data_6 IS NULL
                OR total_rech_data_7 IS NULL
                OR total_rech_data_8 IS NULL
                THEN 'Data — incomplete information'

            ELSE 'Data — variable adoption'
        END AS data_consistency,

        churn_probability

    FROM staging.train_raw
),
summary AS (
    SELECT
        social_consistency,
        night_consistency,
        data_consistency,

        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM consistency
    GROUP BY
        social_consistency,
        night_consistency,
        data_consistency

    HAVING COUNT(*) >= 100
)
SELECT
    social_consistency,
    night_consistency,
    data_consistency,
    customers,

    ROUND(
        (
            100.0 * customers
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct

FROM summary
ORDER BY customers DESC;


/*==============================================================================
    14. EXECUTIVE SERVICE SIGNALS

    Compact management-facing summary of major service adoption signals.

    These are descriptive associations only.

    Minimum-volume guardrail:
    Each signal must contain at least 100 customers.
==============================================================================*/

WITH signal_base AS (

    SELECT
        'Data service' AS signal_dimension,
        'Recorded data recharge users' AS signal,
        COUNT(*) AS customers,

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        ) AS churners

    FROM staging.train_raw

    WHERE total_rech_data_8 > 0


    UNION ALL


    SELECT
        'Data service',
        'Information unavailable',
        COUNT(*),

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )

    FROM staging.train_raw

    WHERE total_rech_data_8 IS NULL


    UNION ALL


    SELECT
        'Social service',
        'Social service adopted',
        COUNT(*),

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )

    FROM staging.train_raw

    WHERE fb_user_8 = 1


    UNION ALL


    SELECT
        'Night service',
        'Night service adopted',
        COUNT(*),

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )

    FROM staging.train_raw

    WHERE night_pck_user_8 = 1


    UNION ALL


    SELECT
        '2G service',
        'Any recorded 2G service',
        COUNT(*),

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )

    FROM staging.train_raw

    WHERE monthly_2g_8 > 0
       OR sachet_2g_8 > 0


    UNION ALL


    SELECT
        '3G service',
        'Any recorded 3G service',
        COUNT(*),

        COUNT(*) FILTER (
            WHERE churn_probability = 1
        )

    FROM staging.train_raw

    WHERE monthly_3g_8 > 0
       OR sachet_3g_8 > 0
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
            / SUM(customers) OVER ()
        )::numeric,
        2
    ) AS customer_share_pct,

    churners,

    ROUND(
        (
            100.0 * churners
            / NULLIF(customers, 0)
        )::numeric,
        2
    ) AS observed_churn_rate_pct,

    ROUND(
        (
            100.0 * churners
            / NULLIF(
                SUM(churners) OVER (),
                0
            )
        )::numeric,
        2
    ) AS churn_contribution_pct

FROM filtered
ORDER BY
    observed_churn_rate_pct DESC,
    customers DESC;


/*==============================================================================
    15. ANALYTICAL QUALITY GATE

    Confirms that the service-behavior analysis uses the complete training
    population and has no duplicate customer IDs or missing target values.

    Service-level missingness is expected and is reported separately.
==============================================================================*/

SELECT

    COUNT(*) AS row_count,

    COUNT(DISTINCT id) AS distinct_customer_ids,

    COUNT(*) - COUNT(DISTINCT id)
        AS duplicate_id_count,

    COUNT(*) FILTER (
        WHERE churn_probability IS NULL
    ) AS missing_target,

    COUNT(*) FILTER (
        WHERE total_rech_data_8 IS NULL
    ) AS missing_data_recharge_information,

    COUNT(*) FILTER (
        WHERE fb_user_8 IS NULL
    ) AS missing_social_service_information,

    COUNT(*) FILTER (
        WHERE night_pck_user_8 IS NULL
    ) AS missing_night_service_information,

    CASE
        WHEN COUNT(*) = 69999
         AND COUNT(DISTINCT id) = 69999

         AND COUNT(*) - COUNT(DISTINCT id) = 0

         AND COUNT(*) FILTER (
                WHERE churn_probability IS NULL
             ) = 0

        THEN 'PASS'

        ELSE 'REVIEW'
    END AS analytical_quality_gate

FROM staging.train_raw;


/*==============================================================================
    END OF SERVICE BEHAVIOR ANALYSIS
==============================================================================*/