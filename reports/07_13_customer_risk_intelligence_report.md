# Step 7.13 — Customer Risk Intelligence

## Purpose

Convert calibrated customer-level churn probabilities into an auditable business prioritization framework.

The workflow is:

**Calibrated Risk → Risk Band → Customer Value → Risk × Value → Potential Revenue Exposure → Priority → Intervention Tier**

## Customer Universe

- Total customers: 69,999
- Development customers: 55,999
- Validation customers: 14,000
- Actual churners: 7,132

## Risk Source

Customer churn risk comes from the selected **tuned XGBoost model** from Step 7.10 and the **sigmoid probability calibration** selected in Step 7.11.

No additional model training, tuning, feature selection, or recalibration was performed in Step 7.13.

## Primary Threshold

Step 7.12 primary analytical threshold:

**0.100000**

Customers at or above this threshold:

- Count: 12,492
- Share: 17.85%

The threshold is treated as an analytical operating point. It is not interpreted as a guarantee that an individual customer will churn.

## Risk Bands

| Risk Band | Probability |
|---|---:|
| Very Low | 0% to <20% |
| Low | 20% to <40% |
| Medium | 40% to <60% |
| High | 60% to <80% |
| Very High | 80%+ |

## Customer Value

Customer value is represented using **August ARPU (`arpu_8`)** as a practical value proxy.

Raw ARPU values are preserved.

Negative ARPU values are not deleted. They are clipped to zero only when calculating potential revenue exposure because negative exposure is not meaningful for this prioritization framework.

## Potential Revenue Exposure

Potential revenue exposure is defined as:

**Calibrated churn probability × non-negative August ARPU**

This is a **scenario-based exposure estimate**, not a forecast of guaranteed revenue loss.

Total potential revenue exposure across the customer universe:

**776,208.24**

## Risk × Value Prioritization

The risk-value priority score combines:

- calibrated churn probability
- normalized customer value percentile

The score is a prioritization aid only. It does not replace the calibrated churn probability.

## Intervention Framework

### Tier 1 — Priority Retention

High-risk + high-value customers.

Customer count:

**182**

Recommended action:

**Immediate retention review**

### Tier 2 — Targeted Retention

High/Very High risk customers who do not fall into Tier 1 because they are standard-value customers.

Customer count:

**5,085**

Recommended action:

**Targeted retention outreach**

### Tier 3 — Proactive Engagement

Medium-risk customers with high customer value who are not in the immediate or targeted retention tiers.

Recommended action:

**Proactive engagement**

### Tier 4 — Monitor

Remaining customers.

Recommended action:

**Monitor and maintain engagement**

## Validation Governance

The 14,000-customer validation set was scored and included in the customer-level universe.

It was **not used to select**:

- the model
- the calibration method
- the threshold
- the value framework
- the intervention framework

## Test Data

Test data was not loaded.

## Governance Checks

- Model retraining: **NO**
- Additional tuning: **NO**
- Additional calibration: **NO**
- Feature selection: **NO**
- Test data loaded: **NO**
- Validation used for framework selection: **NO**
- Feature freeze: **ACTIVE**

## Quality Gate

**PASS**

## Important Interpretation

Risk probability, customer value, and potential revenue exposure represent different business concepts.

A high-risk customer is not automatically a high-value customer.

A high-value customer is not automatically high risk.

The most actionable segment is therefore the intersection of **risk and customer value**, supported by the separate revenue-exposure estimate.

Potential revenue exposure should be treated as a scenario-based prioritization measure rather than a guaranteed financial loss.
