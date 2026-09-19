# Phase 13.3 — Retention Economics by Targeting Cohort

## Purpose

Evaluate whether the targeting cohorts identified in Phase 13.2 produce economically meaningful retention scenarios under the locked Phase 13.1 assumptions.

## Customer Universe

**69,999 customers**

## Evaluation Horizon

**12 months**

## Intervention Cost Timing

Intervention cost is modeled as a **one-time cost per targeted customer** over the evaluation horizon.

## Locked Retention Scenarios

The analysis uses the three Phase 13.1 scenario assumptions:

- Low-Touch Retention: ₹100/customer, 20% assumed success
- Targeted Retention: ₹250/customer, 35% assumed success
- High-Touch Retention: ₹500/customer, 50% assumed success

These are analytical assumptions and are not observed company outcomes or costs.

## Economic Formula

Expected Revenue Retained:

**Monthly Revenue Exposure × Assumed Retention Success Rate × Evaluation Horizon**

Expected Net Value:

**Expected Revenue Retained − Intervention Cost**

ROI:

**Expected Net Value ÷ Intervention Cost**

Break-even Retention Success Rate:

**Intervention Cost ÷ Annualized Revenue Exposure**

## Cohort × Scenario Results

| targeting_cohort      | scenario_id   | scenario_name        |   targeted_customers |   positive_exposure_customers |   targeting_population_pct |   monthly_revenue_exposure_inr |   monthly_exposure_capture_pct |   annualized_revenue_exposure_inr |   average_monthly_exposure_per_customer_inr |   average_calibrated_churn_probability |   historical_churned_customers |   intervention_cost_per_customer_inr |   retention_success_rate |   expected_monthly_revenue_retained_inr |   expected_revenue_retained_over_horizon_inr |   total_intervention_cost_inr |   expected_net_value_inr |   net_value_per_targeted_customer_inr |        roi |   roi_pct |   break_even_retention_success_rate |   break_even_retention_success_rate_pct | break_even_feasible   | retention_success_above_break_even   |   intervention_cost_as_pct_of_annualized_exposure | economically_viable   |
|:----------------------|:--------------|:---------------------|---------------------:|------------------------------:|---------------------------:|-------------------------------:|-------------------------------:|----------------------------------:|--------------------------------------------:|---------------------------------------:|-------------------------------:|-------------------------------------:|-------------------------:|----------------------------------------:|---------------------------------------------:|------------------------------:|-------------------------:|--------------------------------------:|-----------:|----------:|------------------------------------:|----------------------------------------:|:----------------------|:-------------------------------------|--------------------------------------------------:|:----------------------|
| Top 1% Exposure       | low_touch     | Low-Touch Retention  |                  700 |                           700 |                    1.00001 |                         248555 |                        32.0217 |                       2.98266e+06 |                                    355.078  |                               0.555859 |                            395 |                                  100 |                     0.2  |                                 49711   |                             596532           |                70000          |         526532           |                             752.188   |  7.52188   | 752.188   |                           0.023469  |                                 2.3469  | True                  | True                                 |                                           2.3469  | True                  |
| Top 1% Exposure       | targeted      | Targeted Retention   |                  700 |                           700 |                    1.00001 |                         248555 |                        32.0217 |                       2.98266e+06 |                                    355.078  |                               0.555859 |                            395 |                                  250 |                     0.35 |                                 86994.2 |                                  1.04393e+06 |               175000          |         868930           |                            1241.33    |  4.96532   | 496.532   |                           0.0586725 |                                 5.86725 | True                  | True                                 |                                           5.86725 | True                  |
| Top 1% Exposure       | high_touch    | High-Touch Retention |                  700 |                           700 |                    1.00001 |                         248555 |                        32.0217 |                       2.98266e+06 |                                    355.078  |                               0.555859 |                            395 |                                  500 |                     0.5  |                                124277   |                                  1.49133e+06 |               350000          |              1.14133e+06 |                            1630.47    |  3.26094   | 326.094   |                           0.117345  |                                11.7345  | True                  | True                                 |                                          11.7345  | True                  |
| Top 5% Exposure       | low_touch     | Low-Touch Retention  |                 3500 |                          3500 |                    5.00007 |                         495515 |                        63.8379 |                       5.94618e+06 |                                    141.576  |                               0.404447 |                           1464 |                                  100 |                     0.2  |                                 99103   |                                  1.18924e+06 |               350000          |         839237           |                             239.782   |  2.39782   | 239.782   |                           0.0588613 |                                 5.88613 | True                  | True                                 |                                           5.88613 | True                  |
| Top 5% Exposure       | targeted      | Targeted Retention   |                 3500 |                          3500 |                    5.00007 |                         495515 |                        63.8379 |                       5.94618e+06 |                                    141.576  |                               0.404447 |                           1464 |                                  250 |                     0.35 |                                173430   |                                  2.08116e+06 |               875000          |              1.20616e+06 |                             344.618   |  1.37847   | 137.847   |                           0.147153  |                                14.7153  | True                  | True                                 |                                          14.7153  | True                  |
| Top 5% Exposure       | high_touch    | High-Touch Retention |                 3500 |                          3500 |                    5.00007 |                         495515 |                        63.8379 |                       5.94618e+06 |                                    141.576  |                               0.404447 |                           1464 |                                  500 |                     0.5  |                                247758   |                                  2.97309e+06 |                    1.75e+06   |              1.22309e+06 |                             349.455   |  0.698909  |  69.8909  |                           0.294306  |                                29.4306  | True                  | True                                 |                                          29.4306  | True                  |
| Top 10% Exposure      | low_touch     | Low-Touch Retention  |                 7000 |                          7000 |                   10.0001  |                         602398 |                        77.6078 |                       7.22878e+06 |                                     86.0569 |                               0.306106 |                           2213 |                                  100 |                     0.2  |                                120480   |                                  1.44576e+06 |               700000          |         745755           |                             106.536   |  1.06536   | 106.536   |                           0.0968352 |                                 9.68352 | True                  | True                                 |                                           9.68352 | True                  |
| Top 10% Exposure      | targeted      | Targeted Retention   |                 7000 |                          7000 |                   10.0001  |                         602398 |                        77.6078 |                       7.22878e+06 |                                     86.0569 |                               0.306106 |                           2213 |                                  250 |                     0.35 |                                210839   |                                  2.53007e+06 |                    1.75e+06   |         780072           |                             111.439   |  0.445755  |  44.5755  |                           0.242088  |                                24.2088  | True                  | True                                 |                                          24.2088  | True                  |
| Top 10% Exposure      | high_touch    | High-Touch Retention |                 7000 |                          7000 |                   10.0001  |                         602398 |                        77.6078 |                       7.22878e+06 |                                     86.0569 |                               0.306106 |                           2213 |                                  500 |                     0.5  |                                301199   |                                  3.61439e+06 |                    3.5e+06    |         114389           |                              16.3412  |  0.0326825 |   3.26825 |                           0.484176  |                                48.4176  | True                  | True                                 |                                          48.4176  | True                  |
| Top 20% Exposure      | low_touch     | Low-Touch Retention  |                14000 |                         14000 |                   20.0003  |                         684559 |                        88.1927 |                       8.21471e+06 |                                     48.8971 |                               0.203432 |                           2957 |                                  100 |                     0.2  |                                136912   |                                  1.64294e+06 |                    1.4e+06    |         242942           |                              17.353   |  0.17353   |  17.353   |                           0.170426  |                                17.0426  | True                  | True                                 |                                          17.0426  | True                  |
| Top 20% Exposure      | targeted      | Targeted Retention   |                14000 |                         14000 |                   20.0003  |                         684559 |                        88.1927 |                       8.21471e+06 |                                     48.8971 |                               0.203432 |                           2957 |                                  250 |                     0.35 |                                239596   |                                  2.87515e+06 |                    3.5e+06    |        -624852           |                             -44.6323  | -0.178529  | -17.8529  |                           0.426065  |                                42.6065  | True                  | False                                |                                          42.6065  | False                 |
| Top 20% Exposure      | high_touch    | High-Touch Retention |                14000 |                         14000 |                   20.0003  |                         684559 |                        88.1927 |                       8.21471e+06 |                                     48.8971 |                               0.203432 |                           2957 |                                  500 |                     0.5  |                                342280   |                                  4.10735e+06 |                    7e+06      |             -2.89265e+06 |                            -206.618   | -0.413235  | -41.3235  |                           0.85213   |                                85.213   | True                  | False                                |                                          85.213   | False                 |
| High + Very High Risk | low_touch     | Low-Touch Retention  |                12492 |                          8750 |                   17.846   |                         551433 |                        71.0418 |                       6.61719e+06 |                                     44.1429 |                               0.499863 |                           6279 |                                  100 |                     0.2  |                                110287   |                                  1.32344e+06 |                    1.2492e+06 |          74238.2         |                               5.94286 |  0.0594286 |   5.94286 |                           0.188781  |                                18.8781  | True                  | True                                 |                                          18.8781  | True                  |
| High + Very High Risk | targeted      | Targeted Retention   |                12492 |                          8750 |                   17.846   |                         551433 |                        71.0418 |                       6.61719e+06 |                                     44.1429 |                               0.499863 |                           6279 |                                  250 |                     0.35 |                                193001   |                                  2.31602e+06 |                    3.123e+06  |        -806983           |                             -64.6     | -0.2584    | -25.84    |                           0.471953  |                                47.1953  | True                  | False                                |                                          47.1953  | False                 |
| High + Very High Risk | high_touch    | High-Touch Retention |                12492 |                          8750 |                   17.846   |                         551433 |                        71.0418 |                       6.61719e+06 |                                     44.1429 |                               0.499863 |                           6279 |                                  500 |                     0.5  |                                275716   |                                  3.3086e+06  |                    6.246e+06  |             -2.9374e+06  |                            -235.143   | -0.470286  | -47.0286  |                           0.943905  |                                94.3905  | True                  | False                                |                                          94.3905  | False                 |

## Economic Interpretation

**11 of 15 scenarios** have positive expected net value under the stated assumptions.

**15 of 15 scenarios** have a mathematically feasible break-even retention rate at or below 100%.

A positive scenario means that the stated assumptions produce positive expected economics. It does not mean that an intervention will generate realized ROI.

## Governance

- No model retraining
- No probability recalibration
- No threshold changes
- No test data
- Historical churn is descriptive only
- Retention success rates are assumptions
- Intervention costs are assumptions
- Intervention cost is modeled as one-time per targeted customer
- Revenue retained is scenario-based
- ROI is scenario-based, not realized ROI
- No guaranteed revenue savings are claimed

## Phase Boundary

Phase 13.3 evaluates the economics of the locked targeting cohorts under the locked Phase 13.1 scenarios.

It does not determine final intervention policy.

Further independent cost/success sensitivity analysis is reserved for Phase 13.5.

## Quality Gate

**PASS**
