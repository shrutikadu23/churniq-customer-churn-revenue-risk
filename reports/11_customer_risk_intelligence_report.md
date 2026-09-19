# Phase 11 — Customer Risk Intelligence

## Purpose

Build an auditable customer-level risk dataset for the complete 69,999-customer universe.

The workflow is:

**Calibrated Churn Probability → Risk Score → Risk Level → Primary Threshold Flag → Risk Ranking**

## Customer Universe

- Total customers: 69,999
- Development OOF customers: 55,999
- Validation customers: 14,000
- Actual churners: 7,132

## Risk Probability

Customer risk is based on the selected tuned XGBoost model outputs with the exact Step 7.11 sigmoid probability calibration.

No additional model training, tuning, recalibration, or feature selection was performed.

### Important interpretation

`calibrated_churn_probability` is the model-estimated probability of churn.

`actual_churn` is the observed historical outcome.

These fields are intentionally kept separate.

A high probability does not guarantee that an individual customer will churn.

## Primary Threshold

Locked Step 7.12 primary operating threshold:

**0.100000**

Customers at or above the threshold:

- Count: 12,492
- Share: 17.85%
- Observed historical churners within threshold population: 6,279
- Observed historical churn rate within threshold population: 50.26%
- Observed historical churn captured: 88.04%

These retrospective figures describe historical model performance.

They are not guarantees of future churn or future retention outcomes.

## Risk Level Framework

| Risk Level | Calibrated Probability |
|---|---:|
| Below Primary Threshold | < 10% |
| High | 10% to < 50% |
| Very High | >= 50% |

## Risk Score

Risk score is the calibrated churn probability expressed on a 0–100 scale.

It is a communication-friendly representation of the calibrated probability and does not represent a separate model.

## Risk Percentile

Risk percentile represents a customer's relative position in the full 69,999-customer calibrated-risk distribution.

It is descriptive only and does not replace the calibrated probability or the locked primary threshold.

## Risk Distribution

| risk_level              |   customer_count |   customer_share |   mean_churn_probability |   median_churn_probability |   actual_churn_count |   actual_churn_rate |   actual_churn_capture_share |
|:------------------------|-----------------:|-----------------:|-------------------------:|---------------------------:|---------------------:|--------------------:|-----------------------------:|
| Below Primary Threshold |            57507 |        0.82154   |                0.0147234 |                 0.00677967 |                  853 |            0.014833 |                     0.119602 |
| High                    |             6340 |        0.0905727 |                0.237609  |                 0.207956   |                 1590 |            0.250789 |                     0.222939 |
| Very High               |             6152 |        0.087887  |                0.770131  |                 0.791907   |                 4689 |            0.762191 |                     0.657459 |

## Probability Summary

- Minimum calibrated probability: 0.0003
- Mean calibrated probability: 0.1013
- Median calibrated probability: 0.0097
- Maximum calibrated probability: 0.9901

## Full-Universe Ranking

The customer risk dataset ranks all 69,999 customers using descending calibrated churn probability.

Ties are resolved deterministically using customer ID.

The top 100 output therefore represents the highest-risk customers across the complete customer universe, not a sample.

## Governance

- Model retraining: **NO**
- Additional tuning: **NO**
- Additional calibration: **NO**
- Feature selection: **NO**
- Test data loaded: **NO**
- Feature freeze: **ACTIVE**

## Phase Boundary

Customer value and revenue exposure are intentionally excluded from this phase.

They will be addressed in:

- **Phase 12 — Customer Value & Revenue Risk**
- **Phase 13 — Retention Prioritization & ROI**

## Quality Gate

**PASS**
