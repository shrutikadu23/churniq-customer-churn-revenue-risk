# ChurnIQ — Limitations

## 1. Purpose

This document records the known limitations and interpretation boundaries of ChurnIQ.

ChurnIQ is designed to estimate customer churn risk using historical customer behavior and to support customer-retention analysis. Its outputs should be interpreted as analytical decision-support signals rather than guaranteed future outcomes.

---

## 2. Data Limitations

### 2.1 Dataset Scope

ChurnIQ is developed using the Telecom Churn Case Study Hackathon dataset.

The development dataset contains 69,999 customers and historical behavioral, usage, recharge, and related customer-level variables.

The dataset represents a specific historical customer population and may not fully represent a different company, market, geography, product portfolio, or customer population.

### 2.2 Available Business Variables

The dataset does not contain several business attributes that could influence churn, including:

- Contract type
- Payment method
- Service-plan information
- Customer tenure segments defined by the business
- Explicit customer satisfaction measures
- Customer-support interaction history
- Marketing campaign exposure
- Competitor information
- Detailed customer lifecycle events

Therefore, the model cannot directly account for these factors.

### 2.3 Historical Data

The model learns patterns from historical observations.

Customer behavior, pricing, products, competition, economic conditions, and business processes can change over time. Historical relationships should therefore not automatically be assumed to remain stable indefinitely.

---

## 3. Class Imbalance Limitation

The historical churn rate in the development population is 10.19%.

This means churn is a minority outcome in the dataset.

Because of this imbalance, overall accuracy is not treated as the primary measure of model usefulness. ChurnIQ emphasizes metrics such as:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1 score
- Top-k churn capture
- Probability calibration

The selected model achieved the following held-out validation results:

| Metric | Validation Result |
|---|---:|
| PR-AUC | 0.7614 |
| ROC-AUC | 0.9502 |
| Precision | 0.7686 |
| Recall | 0.6220 |
| F1 | 0.6876 |
| Top-10% churn capture | 71.04% |

These results describe performance on the historical validation population and should not be interpreted as guaranteed future production performance.

---

## 4. Temporal Limitations

### 4.1 Prediction Point

The project uses the end of August 2014 as the prediction point.

Features used for prediction are restricted to information available by that point.

### 4.2 Outcome Horizon

September 2014 is treated as the subsequent outcome period.

Information from the outcome period is not used as a predictor.

This prevents future information from being incorporated into the prediction features.

### 4.3 Future Generalization

The model's historical validation performance does not guarantee the same performance on future customer populations.

A production implementation would require continuous monitoring for changes in customer behavior and data distributions.

---

## 5. Model Limitations

### 5.1 Probability Is Not Certainty

The calibrated churn probability represents the model's estimated likelihood of churn under the development data and modeling framework.

A customer with a high predicted probability should therefore not automatically be treated as certain to churn.

The probability is a decision-support signal.

### 5.2 Classification Threshold

The primary threshold of 0.10 is an analytical operating point selected during threshold optimization.

It should not be interpreted as a universal definition of a "churner."

Changing business capacity, intervention costs, retention objectives, or risk tolerance may justify a different operating threshold.

### 5.3 Model Performance

The selected XGBoost model demonstrated strong historical validation performance within the available development and validation framework.

However, validation metrics describe historical holdout performance and may change when the model encounters a different customer population or future data distribution.

### 5.4 Calibration

ChurnIQ applies sigmoid probability calibration using out-of-fold model predictions.

The calibration process improved probability calibration within the evaluated population. However, calibration quality can deteriorate when the future customer population or relationship between features and outcomes changes.

Calibration should therefore be monitored after deployment.

### 5.5 Explainability Does Not Establish Causality

SHAP explanations describe how model features contributed to an individual prediction or to overall model behavior.

A feature being strongly associated with model predictions does not establish that changing that feature would cause a customer to stop churning.

ChurnIQ identifies statistical patterns associated with churn risk; it does not estimate the causal effect of a retention intervention.

---

## 6. Missing-Data Limitations

The feature-engineering pipeline applies documented missingness treatment and development-only median imputation where required.

This allows the model to score customers when certain feature values are unavailable.

However, imputation introduces assumptions about missing values and may reduce the accuracy of individual predictions when a customer's missingness pattern differs substantially from the development population.

New production data should therefore be monitored for changes in missingness rates and patterns.

---

## 7. Customer-Value Limitations

### 7.1 August ARPU as a Value Proxy

August ARPU is used as a customer-value proxy for the revenue-risk analysis.

It should not be interpreted as booked accounting revenue, customer lifetime value, or guaranteed future revenue.

### 7.2 Revenue Exposure

Potential monthly revenue exposure is calculated by combining calibrated churn probability with exposure-eligible August ARPU.

This represents a modeled scenario estimate.

It does not represent revenue that is guaranteed to be lost.

### 7.3 Negative ARPU

Negative August ARPU values are preserved for customer-value analysis.

For revenue-exposure calculations, exposure eligibility is restricted to non-negative values.

This is an analytical treatment rather than an accounting rule.

---

## 8. Retention Economics Limitations

### 8.1 Scenario-Based ROI

Retention ROI calculations are based on analyst-defined assumptions for:

- Retention success rate
- Intervention cost
- Targeting cohort

These assumptions are scenario inputs and are not observed business outcomes.

### 8.2 No Realized ROI Claim

The project does not claim that a particular retention campaign will achieve the modeled ROI.

Actual retention effectiveness, intervention costs, customer response, and realized revenue would need to be measured through real business experiments or historical intervention data.

### 8.3 Cohort Overlap

The targeting cohorts used in the retention analysis are analytical targeting definitions and may overlap.

Therefore, aggregate values across overlapping cohorts should not automatically be interpreted as mutually exclusive customer populations.

### 8.4 Break-Even Analysis

Break-even success rates indicate the retention success rate required for a scenario to cover its assumed intervention cost under the specified revenue-exposure framework.

They do not guarantee that the required success rate can be achieved operationally.

### 8.5 Sensitivity Analysis

The sensitivity analysis evaluates a defined range of analyst-selected retention success rates, intervention costs, and targeting cohorts.

The resulting robustness classifications describe behavior within those tested scenarios. They should not be interpreted as forecasts of actual campaign performance.

---

## 9. Operational and Deployment Limitations

### 9.1 Existing-Customer Scoring

The current inference engine is designed around the frozen ChurnIQ feature schema and available customer-level data.

A production system would require a reliable process for collecting, validating, transforming, and supplying new customer data in the expected structure.

### 9.2 Prototype Application

The Streamlit application demonstrates the ChurnIQ analytical workflow as a portfolio/prototype implementation.

Production deployment would additionally require appropriate:

- Authentication and authorization
- Access controls
- Data-pipeline reliability
- Model and artifact version management
- Monitoring infrastructure
- Logging and auditability
- Error handling
- Operational ownership

### 9.3 Model Artifact Management

The final tuned XGBoost model, probability calibrator, and frozen feature schema must remain version-controlled and synchronized.

Changing one component without validating compatibility with the others can produce inconsistent predictions.

### 9.4 Monitoring Requirement

Model performance should not be assumed to remain stable after deployment.

Data quality, feature distributions, predicted-risk distributions, calibration, and eventual prediction performance should be monitored over time.

---

## 10. Responsible Interpretation

ChurnIQ should be used to support informed retention analysis, not to make automatic decisions about individual customers without appropriate business review.

Recommended interpretation principles include:

- Treat churn probability as an estimate, not a certainty.
- Treat SHAP explanations as model explanations, not causal conclusions.
- Treat August ARPU as a value proxy.
- Treat revenue exposure as a scenario estimate.
- Treat ROI and break-even results as assumption-based scenarios.
- Validate important business actions using appropriate real-world evidence.
- Reassess the model when customer populations, products, processes, or data sources materially change.

---

## 11. Summary

ChurnIQ provides an integrated framework for predicting churn risk, explaining model signals, quantifying customer-value exposure, and analyzing retention-targeting scenarios.

Its outputs are subject to limitations arising from the available historical dataset, feature coverage, temporal scope, class imbalance, modeling assumptions, probability calibration, value-proxy methodology, and scenario-based retention economics.

These limitations do not invalidate the analytical framework, but they define the boundaries within which its outputs should be interpreted.

Future production use should therefore combine the model with data-quality controls, monitoring, periodic validation, appropriate governance, and informed business judgment.