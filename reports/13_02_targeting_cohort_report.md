# Phase 13.2 — Targeting Cohort Analysis

## Purpose

Identify economically concentrated customer cohorts for potential retention targeting.

The analysis uses the Phase 12 potential monthly revenue exposure and the locked Phase 11 risk classification.

## Customer Universe

**69,999 customers**

## Total Potential Revenue Exposure

- Monthly: **₹776,208.24**
- Annualized: **₹9,314,498.90**

## Historical Churn Context

Historical churn is used only as a descriptive validation measure.

It is **not** used to calculate retention ROI or future retention success.

Historical churned customers:

**7,132**

## Targeting Cohorts

1. Top 1% Exposure
2. Top 5% Exposure
3. Top 10% Exposure
4. Top 20% Exposure
5. High + Very High Risk

## Cohort Summary

| targeting_cohort      |   targeted_customers |   targeting_population_pct |   positive_exposure_customers |   positive_exposure_share_pct |   monthly_revenue_exposure_inr |   monthly_exposure_capture_pct |   annualized_revenue_exposure_inr |   average_monthly_exposure_per_customer_inr |   exposure_capture_efficiency |   exposure_captured_per_1pct_population_targeted |   historical_churned_customers |   historical_churn_capture_pct |   average_calibrated_churn_probability |   high_very_high_risk_customers |   high_very_high_risk_share_pct |
|:----------------------|---------------------:|---------------------------:|------------------------------:|------------------------------:|-------------------------------:|-------------------------------:|----------------------------------:|--------------------------------------------:|------------------------------:|-------------------------------------------------:|-------------------------------:|-------------------------------:|---------------------------------------:|--------------------------------:|--------------------------------:|
| Top 1% Exposure       |                  700 |                    1.00001 |                           700 |                      100      |                         248555 |                        32.0217 |                       2.98266e+06 |                                    355.078  |                      32.0212  |                                         32.0212  |                            395 |                        5.53842 |                               0.555859 |                             681 |                         97.2857 |
| Top 5% Exposure       |                 3500 |                    5.00007 |                          3500 |                      100      |                         495515 |                        63.8379 |                       5.94618e+06 |                                    141.576  |                      12.7674  |                                         12.7674  |                           1464 |                       20.5272  |                               0.404447 |                            3108 |                         88.8    |
| Top 10% Exposure      |                 7000 |                   10.0001  |                          7000 |                      100      |                         602398 |                        77.6078 |                       7.22878e+06 |                                     86.0569 |                       7.76067 |                                          7.76067 |                           2213 |                       31.0292  |                               0.306106 |                            5080 |                         72.5714 |
| Top 20% Exposure      |                14000 |                   20.0003  |                         14000 |                      100      |                         684559 |                        88.1927 |                       8.21471e+06 |                                     48.8971 |                       4.40957 |                                          4.40957 |                           2957 |                       41.461   |                               0.203432 |                            6648 |                         47.4857 |
| High + Very High Risk |                12492 |                   17.846   |                          8750 |                       70.0448 |                         551433 |                        71.0418 |                       6.61719e+06 |                                     44.1429 |                       3.98083 |                                          3.98083 |                           6279 |                       88.0398  |                               0.499863 |                           12492 |                        100      |

## Exposure / Risk Overlap

| exposure_cohort   |   exposure_cohort_customers |   high_very_high_overlap_customers |   high_very_high_overlap_share_of_cohort_pct |   overlap_monthly_revenue_exposure_inr |   overlap_exposure_share_of_cohort_pct |   overlap_historical_churned_customers |
|:------------------|----------------------------:|-----------------------------------:|---------------------------------------------:|---------------------------------------:|---------------------------------------:|---------------------------------------:|
| Top 1% Exposure   |                         700 |                                681 |                                      97.2857 |                                 242587 |                                97.5989 |                                    394 |
| Top 5% Exposure   |                        3500 |                               3108 |                                      88.8    |                                 462532 |                                93.3437 |                                   1432 |
| Top 10% Exposure  |                        7000 |                               5080 |                                      72.5714 |                                 525677 |                                87.2641 |                                   2081 |
| Top 20% Exposure  |                       14000 |                               6648 |                                      47.4857 |                                 546111 |                                79.7756 |                                   2594 |

## Interpretation

Exposure capture measures how much of the Phase 12 revenue-exposure proxy is contained within each cohort.

Historical churn capture provides descriptive evidence about where previously observed churn was concentrated.

The High + Very High Risk overlap analysis shows whether economically valuable exposure is also concentrated among customers classified as higher model risk.

These measures are decision-support metrics and do not represent guaranteed future churn, guaranteed revenue loss, or guaranteed retention success.

## Ranking Basis

Customers are ranked primarily by:

**Potential Monthly Revenue Exposure**

Calibrated churn probability and customer ID are used only as deterministic tie-breakers.

## Governance

- Model retraining: **NO**
- Probability recalibration: **NO**
- Threshold changes: **NO**
- Test data loaded: **NO**
- Actual churn used for ROI: **NO**
- Historical churn used: **DESCRIPTIVE VALIDATION ONLY**

## Phase Boundary

Phase 13.2 identifies and compares targeting cohorts.

It does **not** select a final intervention strategy.

Retention economics and ROI sensitivity are evaluated in subsequent Phase 13 steps.

## Quality Gate

**PASS**
