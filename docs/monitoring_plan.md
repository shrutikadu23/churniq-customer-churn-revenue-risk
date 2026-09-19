# ChurnIQ — Monitoring Plan

## 1. Purpose

This document defines the monitoring framework for ChurnIQ after deployment.

The objective is to detect changes in data quality, customer behavior, model predictions, probability calibration, model performance, and business conditions that could reduce the reliability of ChurnIQ outputs.

Monitoring should support timely investigation and model governance rather than assuming that historical model performance will remain stable indefinitely.

---

## 2. Monitoring Scope

ChurnIQ monitoring covers the following areas:

1. Data quality
2. Feature distributions and drift
3. Missingness patterns
4. Prediction distributions
5. Probability calibration
6. Model performance
7. Customer-risk population stability
8. Revenue-exposure stability
9. Retention-targeting assumptions
10. Model and artifact governance

---

## 3. Data Quality Monitoring

Data quality checks should be performed before customer records are passed to the inference pipeline.

### 3.1 Schema Checks

Monitor for:

- Unexpected columns
- Missing required columns
- Changed column names
- Changed data types
- Missing customer identifiers
- Duplicate customer identifiers
- Unexpected feature-count changes

The production input should remain compatible with the frozen ChurnIQ feature schema.

### 3.2 Numeric Validation

Monitor numeric inputs for:

- Missing values
- Infinite values
- Unexpected negative values where not logically valid
- Extreme values
- Unexpected zero concentrations
- Invalid numeric conversions

### 3.3 Customer Identifier Checks

Monitor:

- Duplicate IDs
- Missing IDs
- Unexpected ID formats
- Sudden changes in customer population size

Records failing critical validation should not silently enter the scoring pipeline.

---

## 4. Missingness Monitoring

The feature-engineering pipeline uses documented missingness treatment and development-only median imputation.

After deployment, monitor:

- Overall missing-value rate
- Missingness by feature
- Missingness by customer population
- Newly emerging missingness patterns
- Features with rapidly increasing missingness

Particular attention should be given to important model features because substantial changes in their availability may affect prediction reliability.

A persistent or material change in missingness should trigger investigation of the underlying data pipeline before assuming that model behavior remains valid.

---

## 5. Feature Drift Monitoring

Feature distributions should be compared between the development population and incoming scoring populations.

### 5.1 Features to Monitor

Monitoring should prioritize:

- High-importance model features
- Customer-value features
- Usage and engagement features
- Recharge-related features
- Recency features
- Temporal-change features
- Features with historically stable distributions

### 5.2 Drift Methods

Possible statistical monitoring methods include:

- Population Stability Index (PSI)
- Distribution comparisons
- Quantile comparisons
- Mean and median changes
- Missingness-rate comparisons
- Category-frequency changes where applicable

Drift should be evaluated over time rather than from a single observation.

No universal PSI cutoff is imposed by this portfolio project. Production alert thresholds should be established using historical variation, data-quality expectations, and business tolerance before deployment.

### 5.3 Investigation

Observed drift should be investigated for possible causes such as:

- Customer population changes
- Product or service changes
- Pricing changes
- Data-pipeline changes
- Measurement-definition changes
- Seasonal behavior
- Business-process changes

Drift does not automatically mean that the model has failed. It indicates that further investigation is required.

---

## 6. Prediction Monitoring

Monitor the distribution of ChurnIQ predictions over time.

Key measures include:

- Mean calibrated churn probability
- Median calibrated churn probability
- Probability distribution
- Percentage of customers above the 0.10 primary threshold
- Percentage classified as Very High Risk
- Number of customers scored
- Changes in risk-level distribution

The historical development population provides a reference point for comparison.

A substantial change in the predicted-risk distribution should be investigated before changing thresholds or retraining the model.

---

## 7. Threshold Monitoring

The primary analytical threshold is currently:

`0.10`

Monitor:

- Percentage of customers above threshold
- Number of customers above threshold
- Risk population by risk level
- Changes in threshold population over time

The threshold should not be changed solely because the size of the flagged population changes.

Any threshold change should be supported by updated business capacity, intervention economics, model validation, and documented governance review.

---

## 8. Probability Calibration Monitoring

Because ChurnIQ produces calibrated churn probabilities, calibration should be monitored separately from ranking performance.

### 8.1 Calibration Measures

Where actual outcomes become available, monitor:

- Brier score
- Log loss
- Expected Calibration Error (ECE)
- Predicted probability versus observed churn rate
- Reliability curves
- Calibration by risk band

### 8.2 Current Calibration Reference

The historical calibration evaluation provides the following reference values:

| Metric | OOF Reference |
|---|---:|
| Brier Score | 0.041740 |
| Log Loss | 0.144759 |
| ECE | 0.005859 |

The calibrated validation ECE was **0.006148**.

These values are historical reference points, not permanent production targets.

### 8.3 Calibration Investigation

Calibration deterioration may indicate:

- Population shift
- Feature drift
- Changing churn behavior
- Model degradation
- Changes in the relationship between customer behavior and churn

If calibration deteriorates materially, recalibration should be evaluated before assuming that the existing probability estimates remain reliable.

---

## 9. Model Performance Monitoring

Model performance can only be measured after actual outcomes become available for scored customers.

Monitor:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1 score
- Top-k churn capture
- Precision at relevant targeting capacities

PR-AUC should remain an important metric because churn is a minority outcome in the historical development population.

Performance should be evaluated on appropriate time-based outcome windows rather than relying only on aggregate historical measurements.

### 9.1 Outcome Availability

Performance metrics cannot necessarily be calculated immediately after scoring.

There may be a delay between:

1. Customer scoring
2. Outcome period completion
3. Actual churn-label availability
4. Model-performance evaluation

Monitoring systems should therefore distinguish between:

- Predictions awaiting outcomes
- Predictions with observed outcomes
- Completed evaluation windows

This prevents incomplete outcome data from being incorrectly interpreted as model degradation.

## 10. Risk-Level Monitoring

Monitor customer counts and observed outcomes by:

- Below Primary Threshold
- High
- Very High

Once actual outcomes become available, compare:

- Observed churn rate by risk level
- Churn capture by risk level
- Risk-level population size
- Changes in the relationship between risk level and observed outcomes

The purpose is to determine whether the model's risk stratification continues to separate customers with different observed churn rates.

---

## 11. Customer-Value Monitoring

August ARPU is used as a customer-value proxy in the current analytical framework.

For ongoing monitoring, track:

- Distribution of customer-value proxy
- Median and mean value
- Negative-value frequency
- Value tiers
- Changes in exposure-eligible value
- Changes in exposure concentration

Customer-value monitoring should distinguish between changes in the customer population and changes caused by data-quality or calculation issues.

---

## 12. Revenue-Exposure Monitoring

Potential monthly revenue exposure is derived from calibrated churn probability and exposure-eligible customer value.

Monitor:

- Total potential monthly exposure
- Exposure by risk level
- Exposure by customer-value tier
- Exposure concentration among top-risk customers
- Exposure distribution over time

Changes in revenue exposure should be interpreted alongside changes in customer population, customer-value distributions, and predicted-risk distributions.

Revenue exposure remains a scenario estimate and should not be treated as realized revenue loss.

---

## 13. Retention Economics Monitoring

Retention economics are based on analyst-defined assumptions.

Monitor the assumptions used for scenario analysis, including:

- Retention success rate
- Intervention cost
- Targeting cohort
- Revenue-exposure methodology

Where real intervention data becomes available, compare actual outcomes against scenario assumptions.

Relevant observed business measures may include:

- Customers contacted
- Customers retained
- Intervention cost
- Incremental revenue
- Customer response rate
- Cost per retained customer

Actual campaign outcomes should be kept separate from the original scenario-based ROI estimates.

---

## 14. Model Drift and Performance Investigation

Monitoring signals should be evaluated together.

For example:

    Data Quality Issue
            ↓
    Feature Distribution Change
            ↓
    Prediction Distribution Change
            ↓
    Calibration / Performance Change
            ↓
    Model Investigation

A single metric change should not automatically trigger retraining.

Investigation should first determine whether the observed change is caused by:

- Data-quality problems
- Pipeline changes
- Population changes
- Seasonal effects
- Business-process changes
- Genuine model degradation

A deterioration in model metrics should be investigated alongside data quality and population drift before attributing the change to the model itself.

---

## 15. Retraining Triggers

Model retraining should be considered when one or more of the following conditions persist after investigation:

- Material feature drift
- Persistent missingness changes
- Significant deterioration in PR-AUC
- Significant deterioration in recall or precision
- Reduced top-k churn capture
- Material calibration deterioration
- Major changes in customer behavior
- Major product or pricing changes
- Changes in the business definition of churn
- Significant changes in available input data

No single universal numeric trigger is prescribed by this portfolio project.

Production thresholds should be established from historical model behavior, business tolerance, operational requirements, and validation evidence.

Retraining should not be performed automatically from a single anomalous observation.

---

## 16. Retraining and Revalidation Process

A model update should follow a controlled process:

1. Investigate the monitoring signal.
2. Validate the underlying data.
3. Rebuild the feature pipeline if required.
4. Recreate development and validation datasets.
5. Re-evaluate candidate models.
6. Re-evaluate probability calibration.
7. Re-evaluate threshold selection.
8. Re-evaluate SHAP explanations.
9. Re-evaluate customer-value and revenue-exposure calculations.
10. Re-evaluate retention-prioritization scenarios where relevant.
11. Compare the new model against the existing production model.
12. Document the changes.
13. Approve the new model version before deployment.

The complete pipeline should be revalidated rather than replacing only one component in isolation.

---

## 17. Model Artifact Governance

The following artifacts should be version-controlled and kept synchronized:

- Final model artifact
- Probability calibrator
- Frozen feature schema
- Feature-engineering logic
- Development-only imputation values
- Primary threshold
- Model metadata
- Model-card documentation
- Limitations documentation
- Monitoring plan

A model, calibrator, feature schema, or threshold should not be replaced independently without compatibility and validation checks.

---

## 18. Monitoring Frequency

A production implementation can use different monitoring frequencies depending on the metric.

| Monitoring Area | Suggested Frequency |
|---|---|
| Input schema validation | Every scoring run |
| Data-quality checks | Every scoring run |
| Missingness monitoring | Every scoring run / daily |
| Prediction distribution | Daily or weekly |
| Feature drift | Weekly or monthly |
| Customer-value distribution | Monthly |
| Revenue-exposure distribution | Weekly or monthly |
| Calibration | After outcomes become available |
| Model performance | After outcome windows become available |
| Retention economics | After campaign outcomes become available |
| Full model review | Periodically and after major business changes |

These frequencies are operational starting points rather than fixed business requirements.

---

## 19. Monitoring Alerts

Monitoring alerts should distinguish between:

### Informational

Small changes that remain within expected historical variation.

### Investigation Required

Persistent or material changes in:

- Data quality
- Missingness
- Feature distributions
- Prediction distributions
- Customer-value distributions

### Model Review Required

Evidence of:

- Material calibration deterioration
- Material performance deterioration
- Persistent population shift
- Major business or data changes

### Deployment Block

Critical schema or data-quality failures that could make scoring unreliable.

Alert thresholds should be configured using historical baselines and business/operational tolerance before production deployment.

---

## 20. Governance and Audit Trail

Each production model version should maintain a record containing:

- Model version
- Feature-schema version
- Calibration version
- Threshold version
- Training period
- Validation period
- Deployment date
- Monitoring review dates
- Known limitations
- Detected drift events
- Performance results
- Retraining decisions
- Approval status

This creates traceability between the deployed model and the evidence used to validate it.

## 21. Current ChurnIQ Reference Baseline

The following historical results provide reference points for monitoring:

### 21.1 Population Reference

- Development population: 55,999 customers
- Held-out validation population: 14,000 customers
- Full historical customer universe: 69,999 customers

The development population is used for model fitting, the validation population is held out for model evaluation, and the full historical customer universe represents the complete customer population used for customer-level risk intelligence.

### 21.2 Model Reference

- Historical churn rate: 10.19%
- Final model features: 169
- Validation PR-AUC: 0.7614
- Validation ROC-AUC: 0.9502
- Validation precision: 0.7686
- Validation recall: 0.6220
- Validation F1: 0.6876
- Top-10% churn capture: 71.04%
- Primary threshold: 0.10

### 21.3 Calibration Reference

- OOF Brier Score: 0.041740
- OOF Log Loss: 0.144759
- OOF ECE: 0.005859
- Validation ECE after calibration: 0.006148

### 21.4 Customer-Value Reference

- Customers above threshold in the scored universe: 12,492
- Potential monthly revenue exposure: ₹776,208.24

These values are historical project reference points, not production targets or guaranteed future results.

---

## 22. Monitoring Principles

The ChurnIQ monitoring framework follows these principles:

- Monitor data before monitoring predictions.
- Monitor both ranking performance and probability calibration.
- Separate model degradation from data-pipeline problems.
- Investigate persistent changes rather than reacting to isolated anomalies.
- Do not change thresholds without documented analysis.
- Do not treat scenario-based revenue exposure as realized revenue loss.
- Do not treat scenario-based ROI as realized business performance.
- Maintain version control across the model, calibrator, feature schema, and threshold.
- Revalidate the complete pipeline when material changes occur.
- Account for the delay between prediction generation and actual outcome availability.
- Establish production alert thresholds from historical behavior and business tolerance rather than arbitrary universal cutoffs.
- Keep monitoring evidence as part of model governance.

---

## 23. Monitoring Decision Framework

A practical monitoring workflow is:

    Monitor
       ↓
    Detect Change
       ↓
    Validate Data
       ↓
    Investigate Population / Feature Drift
       ↓
    Evaluate Prediction & Calibration Behavior
       ↓
    Evaluate Model Performance When Outcomes Are Available
       ↓
    Determine Root Cause
       ↓
    Continue Monitoring OR Recalibrate OR Retrain
       ↓
    Validate and Approve Before Deployment

This framework helps distinguish operational data problems from genuine model degradation.

---

## 24. Production Readiness Boundary

The monitoring plan defines what should be monitored if ChurnIQ is deployed in a production environment.

The current portfolio implementation demonstrates the analytical and inference workflow but does not represent a continuously operating production monitoring platform.

A production implementation would require additional infrastructure for:

- Automated data-quality checks
- Scheduled monitoring jobs
- Drift calculation
- Alert delivery
- Outcome collection
- Performance evaluation
- Model-version tracking
- Audit logging
- Monitoring dashboards
- Operational ownership

These requirements are part of the transition from a portfolio prototype to a production ML system.

---

## 25. Summary

ChurnIQ should be treated as a monitored analytical system rather than a static model artifact.

Ongoing monitoring should evaluate data quality, feature stability, prediction behavior, probability calibration, eventual model performance, customer-value exposure, and retention-economics assumptions.

When material changes are detected, the appropriate response is investigation followed by revalidation, recalibration, or retraining when supported by evidence.

The monitoring framework deliberately avoids arbitrary universal thresholds. Production alert and retraining criteria should be established from historical model behavior, operational requirements, business tolerance, and documented validation evidence.

This monitoring plan is intended to preserve the reliability, traceability, and responsible interpretation of ChurnIQ as the underlying customer population and business environment change.