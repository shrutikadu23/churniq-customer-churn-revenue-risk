# ChurnIQ — Model Card

## 1. Model Overview

### Model Name

ChurnIQ Customer Churn Risk Model

### Model Version

ChurnIQ v1.0

### Purpose

ChurnIQ is an end-to-end customer retention intelligence system designed to identify customers with elevated churn risk, explain the behavioral signals associated with their predictions, quantify customer-value exposure, and support retention prioritization.

The system follows the framework:

> Predict → Explain → Quantify Risk → Prioritize → Recommend Action

### Intended Users

- Customer Retention Managers
- Business Managers
- Business Analysts
- Customer Analytics Teams

### Intended Use

The model is intended for decision support and customer-risk analysis. It can help identify customers who may warrant further retention attention based on their historical behavioral patterns and modeled churn probability.

The model should not be interpreted as determining with certainty that a customer will churn.

### Not Intended For

The model is not intended to:

- Guarantee that an individual customer will churn
- Automatically determine customer treatment without human review
- Establish causal reasons for churn
- Guarantee revenue recovery from retention actions
- Replace business judgment or operational constraints
- Serve as evidence of realized retention ROI

### Model Artifacts

Final model:

`models/xgboost_final_tuned.ubj`

Probability calibration artifact:

`models/xgb_probability_calibrator.joblib`

Feature set:

169 frozen model features

---

## 2. Prediction Definition

### Target

The prediction target is customer churn.

- `0` = Retained
- `1` = Churned

### Prediction Point

The prediction framework uses information available up to the end of August 2014.

### Outcome Horizon

Customer churn is evaluated using the subsequent observed churn outcome following the August 2014 prediction point.

### Temporal Control

Information from September 2014 is excluded from the predictive feature set to prevent future-information leakage.

---

## 3. Dataset

### Dataset Structure

The project uses a telecom customer churn dataset containing:

- 69,999 training customers
- 30,000 test customers
- 172 columns in the training dataset
- 171 columns in the test dataset

The training dataset contains:

- 7,132 churned customers
- 62,867 retained customers
- 10.19% historical churn rate

Duplicate customer IDs were checked and no duplicate IDs were found in either train or test data.

Train/test customer-ID overlap was also checked and none was found.

### Available Business Signals

The dataset contains customer-level information related to:

- Voice usage
- Incoming and outgoing usage
- Recharge behavior
- Data usage
- Recency
- Activity
- Tenure
- Value-related behavioral measures

The source dataset does not provide several potentially important business attributes such as:

- Contract type
- Service plan
- Payment method
- Subscription plan
- Explicit customer segment

The absence of these attributes limits the range of business explanations and operational targeting strategies that can be derived from the dataset.

---

## 4. Feature Engineering

The feature-engineering pipeline was designed around the prediction point and temporal eligibility.

### Feature Development

- Raw predictors: 170
- Engineered candidate features: 427
- Final frozen model features: 169

Feature engineering included:

- Temporal change measures
- Percentage-change measures
- Multi-month behavioral measures
- Recent-vs-baseline behavior
- Recency and activity measures
- Customer-value measures
- Missingness treatment
- Redundancy review
- Predictive screening
- Stability review

### Feature Freeze

The final model uses a frozen feature set containing exactly 169 features.

The feature matrix was validated to ensure:

- Customer ID is excluded
- Target is excluded
- Future information is excluded
- Target-derived information is excluded
- Duplicate features are not present
- Infinite values are not present
- Missing values are handled through the defined preprocessing process

---

## 5. Data Splitting & Preprocessing

The development dataset contains:

- 55,999 development customers
- 14,000 validation customers

The split uses stratification and a fixed random seed of 42.

Numeric missing values for final inference are handled using median values calculated from the development dataset only.

This prevents validation or future customer information from influencing preprocessing parameters.

---

## 6. Model Selection

Multiple models were evaluated during development.

### Baseline

A dummy classifier was used as a reference point.

Baseline PR-AUC:

`0.1019`

### Logistic Regression

Validation:

- PR-AUC: 0.6800
- ROC-AUC: 0.9186
- Precision: 0.7661
- Recall: 0.5603
- F1: 0.6472

### HistGradientBoosting

Validation:

- PR-AUC: 0.7551
- ROC-AUC: 0.9477
- Recall: 0.6192
- F1: 0.6826

### XGBoost Baseline

Validation:

- PR-AUC: 0.7576
- ROC-AUC: 0.9500
- Recall: 0.6192
- F1: 0.6866

The baseline XGBoost model captured approximately 70.20% of observed churn within the highest-risk 10% of customers.

---

## 7. Final Model

The selected model is a tuned XGBoost classifier.

### Selected Configuration

- Candidate: `XGB_02_SLOWER_MORE_TREES`
- Estimators: 200
- Learning rate: 0.05
- Maximum depth: 6
- Minimum child weight: 1
- Subsample: 0.90
- Column sampling: 0.90
- Gamma: 0
- L1 regularization: 0
- L2 regularization: 1
- Random state: 42

The final model was retrained on the complete development dataset using the frozen 169-feature set.

---

## 8. Model Performance

### Cross-Validation

The tuned model achieved:

- 5-fold CV PR-AUC: 0.7769

This represented an improvement over the baseline XGBoost configuration.

### Held-Out Validation

The selected tuned model achieved:

- PR-AUC: 0.7614
- ROC-AUC: 0.9502
- Precision: 0.7686
- Recall: 0.6220
- F1: 0.6876

At the highest-risk 10% of customers:

- Churn capture: 71.04%
- Lift: 7.10×

PR-AUC is emphasized because the dataset is imbalanced, with churn representing approximately 10.19% of the training population.

## 9. Probability Calibration

The raw XGBoost probabilities were calibrated using sigmoid calibration.

The calibration process was performed using out-of-fold predictions.

### Calibration Results

OOF results:

- Brier score: 0.041740
- Log loss: 0.144759
- Expected Calibration Error: 0.005859
- PR-AUC: 0.776913
- ROC-AUC: 0.955720

Validation ECE changed from:

`0.009384 → 0.006148`

Sigmoid calibration reduced the observed calibration error on the evaluated validation population while preserving the model's ranking behavior.

The sigmoid calibrator is stored separately as a model artifact and reproduced during inference.

---

## 10. Risk Threshold

The primary decision-support threshold is:

`0.10`

At this threshold on the OOF population:

- Target rate: 17.93%
- Observed historical churn capture: 88.21%
- Precision: 50.14%

### Threshold vs Capacity

The `0.10` threshold identifies customers whose modeled churn probability is at least 10%. It does not imply that 10% of customers should or can be targeted.

Operational targeting capacity must be evaluated separately.

For example, a separate analysis found that approximately 10% targeting capacity corresponds to a threshold near `0.402`.

---

## 11. Risk Intelligence

The final customer-risk scoring layer evaluates the complete 69,999-customer universe.

### Risk Levels

Customers are classified as:

- Below Primary Threshold: probability < 0.10
- High: probability ≥ 0.10 and < 0.50
- Very High: probability ≥ 0.50

Final scoring results:

- Customers above primary threshold: 12,492
- Very High Risk: 6,152
- High Risk: 6,340
- Below Primary Threshold: 57,507
- Observed historical churn captured above threshold: 88.04%

The risk score is represented on a 0–100 scale derived from calibrated churn probability.

---

## 12. Explainability

ChurnIQ uses SHAP to explain model predictions.

A reproducible sample of 10,000 customers was used for global SHAP analysis.

The top global model drivers included:

1. `total_activity_8`
2. `voice_rech_recency_8`
3. `last_day_rch_amt_8`
4. `total_ic_mou_8`
5. `roam_og_mou_8`
6. `total_activity_pct_change_7_to_8`
7. `total_ic_mou_min_6_8`
8. `spl_ic_mou_8`
9. `aon`
10. `total_voice_mou_8`

SHAP explanations describe how model features contribute to a prediction. They should not be interpreted as causal explanations of why a customer will churn.

---

## 13. Customer Value & Revenue Risk

ChurnIQ uses August ARPU as a customer-value proxy.

It is not treated as booked revenue.

For revenue-exposure analysis:

`Potential Monthly Revenue Exposure = Calibrated Churn Probability × Exposure-Eligible August ARPU`

Negative August ARPU values are preserved in the raw value field but excluded from exposure calculations by clipping the exposure-eligible value to zero.

The final customer universe produced:

- Average August ARPU: ₹278.86
- Median August ARPU: ₹192.23
- Potential monthly exposure: ₹776,208.24
- Annualized scenario exposure: ₹9,314,498.90

These values represent modeled scenario exposure rather than guaranteed revenue loss.

---

## 14. Retention Prioritization & ROI

ChurnIQ evaluates targeting cohorts including:

- Top 1% Exposure
- Top 5% Exposure
- Top 10% Exposure
- Top 20% Exposure
- High + Very High Risk

The targeting analysis evaluates:

- Customers targeted
- Historical churn descriptively
- Revenue exposure captured
- Assumed retention success
- Assumed intervention cost
- Scenario ROI
- Break-even success rate
- Sensitivity and robustness

The ROI analysis contains 135 sensitivity scenarios.

Of these:

- 94 scenarios were economically viable
- Tested break-even success rates ranged from 2.35% to 94.39%

These are analyst-defined scenarios and should not be interpreted as realized business outcomes.

Historical churn is used descriptively in targeting analysis and is not used as an input to the ROI calculation.

---

## 15. Application Architecture

ChurnIQ is implemented as an end-to-end analytical application.

The architecture is:

    Streamlit Interface
            ↓
    ChurnIQ Inference Engine
            ↓
    Frozen Feature Schema
            ↓
    Final Tuned XGBoost Model
            ↓
    Sigmoid Probability Calibration
            ↓
    Risk Classification
            ↓
    SHAP Explanation
            ↓
    Customer Value
            ↓
    Revenue Exposure
            ↓
    Business Interpretation

The Streamlit application allows a user to enter a customer ID and review the customer's:

- Calibrated churn probability
- Risk score and risk level
- SHAP-based prediction explanation
- Customer-value metrics
- Revenue-exposure scenario

Power BI provides a supporting executive-level view of overall risk, customer value, revenue exposure, and retention economics.

---

## 16. Limitations

The model has several important limitations.

### Data Limitations

The dataset does not contain all customer attributes that may influence churn, including contract, plan, and payment information.

This limits the range of business explanations and operational targeting strategies that can be derived from the available data.

### Temporal Limitations

The model is based on historical customer behavior and should not automatically be assumed to perform identically on future populations.

Customer behavior, product offerings, market conditions, and business processes may change over time.

### Causality

The model identifies predictive associations. It does not establish that changing a particular behavior will cause a customer to remain.

### Probability Interpretation

A calibrated probability represents modeled likelihood, not certainty.

A customer with a high predicted probability is not guaranteed to churn.

### Threshold Interpretation

The primary threshold of `0.10` is a decision-support threshold and should be reassessed when operational capacity, intervention costs, or business objectives change.

The threshold should not be interpreted as an automatic targeting rule.

### Explainability

SHAP explains model behavior but does not establish causal relationships.

A feature with a strong SHAP contribution should not automatically be interpreted as a root cause of churn.

### Revenue Exposure

August ARPU is used as a customer-value proxy.

Revenue exposure is a modeled scenario estimate rather than booked revenue or guaranteed future revenue loss.

### ROI

Retention economics depend on analyst-defined assumptions regarding retention success and intervention costs.

The resulting ROI values represent scenario economics rather than realized business ROI.

### Production Use

Before production use, the model should be monitored for data drift, behavior drift, calibration changes, and performance degradation.

---

## 17. Responsible Use

ChurnIQ should be used as decision support rather than as an automatic decision-maker.

Customer risk scores should be combined with appropriate business context and operational information.

The system should not be used to:

- Guarantee customer outcomes
- Guarantee revenue recovery
- Automatically determine consequential customer treatment without appropriate human review
- Treat model explanations as causal conclusions
- Treat scenario ROI as realized financial performance

---

## 18. Reproducibility & Model Governance

Key reproducibility controls include:

- Random seed: `42`
- Frozen 169-feature schema
- Development-only preprocessing statistics
- Saved final XGBoost model artifact
- Saved sigmoid calibration artifact
- Deterministic customer-risk scoring logic
- Reproducible SHAP sampling configuration

### Model Artifacts

Final model:

`models/xgboost_final_tuned.ubj`

Probability calibration artifact:

`models/xgb_probability_calibrator.joblib`

Frozen feature schema:

`data/processed/final_model_features.csv`

### Inference Controls

The inference engine validates:

- Expected feature count
- Feature ordering
- Numeric feature values
- Missing-value treatment
- Infinite-value absence
- Model loading
- Calibration method
- Customer lookup

The deployed application uses the same frozen model, feature schema, preprocessing logic, and calibration artifact rather than retraining the model at inference time.

---

## 19. Monitoring Plan

If ChurnIQ is used with continuously generated customer data, monitoring should cover three areas.

### Data Monitoring

Monitor:

- Feature missingness
- Feature distributions
- Schema changes
- Unexpected values
- New or unavailable fields

### Model Monitoring

Monitor:

- Prediction probability distribution
- Risk-level distribution
- Calibration error
- PR-AUC when future labels become available
- Recall and precision when future labels become available
- Ranking behavior
- Model performance by relevant customer populations

### Business Monitoring

Monitor:

- Customers identified as high risk
- Observed churn among scored customers
- Churn capture within targeted populations
- Revenue exposure distribution
- Retention intervention outcomes
- Actual financial outcomes where reliable intervention and outcome data become available

---

## 20. Drift & Retraining Considerations

Model retraining should be considered when meaningful changes occur in:

- Customer behavior
- Data distributions
- Product or service offerings
- Customer acquisition patterns
- Business processes
- Prediction performance
- Probability calibration

### Potential Retraining Triggers

1. Sustained feature-distribution drift
2. Material deterioration in model discrimination
3. Material deterioration in probability calibration
4. Significant changes in the observed churn rate
5. Changes to the available feature schema
6. Changes in the business definition of churn

Retraining should be performed using an updated, temporally appropriate dataset and should repeat the relevant validation, calibration, threshold, and explainability checks before deployment.

---

## 21. Model Governance Summary

ChurnIQ separates:

- Prediction from intervention
- Probability from certainty
- Association from causation
- Customer-value proxy from booked revenue
- Scenario economics from realized ROI
- Historical evaluation from future performance

Any future production version should preserve these distinctions in both technical documentation and business communication.

---

## 22. Summary

ChurnIQ combines predictive modeling, probability calibration, explainable AI, customer-value analysis, revenue-risk exposure, retention prioritization, scenario economics, and business dashboards into a single customer-retention intelligence workflow.

The system is designed to help move customer retention analysis from broad historical reporting toward evidence-based identification, explanation, quantification, and prioritization of potential churn risk.

The current model is a decision-support system and should be evaluated and monitored continuously before being relied upon in a production environment.