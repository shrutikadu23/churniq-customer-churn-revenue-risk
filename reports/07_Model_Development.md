# ChurnIQ — Model Development

## 1. Purpose

This phase develops and evaluates machine learning models for customer churn prediction.

The objective is not simply to maximize predictive accuracy. The modeling process is designed to identify a reliable model that can distinguish customers with higher churn risk while supporting downstream customer prioritization, revenue-risk analysis, and retention decision-making.

The modeling framework follows the ChurnIQ principle:

**Predict → Explain → Quantify Risk → Prioritize → Recommend Action**

---

## 2. Modeling Objective

### ML Objective

Estimate the probability that an individual customer will churn based exclusively on information available at the prediction point.

### Business Objective

Use customer-level churn predictions to:

- identify customers with elevated churn risk,
- understand predictive model performance,
- prioritize customers for potential retention intervention,
- quantify the concentration of churn and revenue risk,
- support retention strategy and scenario analysis.

The model output is a risk estimate and must not be interpreted as a guaranteed future outcome.

---

## 3. Prediction Framework

| Component | Definition |
|---|---|
| Prediction unit | Individual customer |
| Target | `churn_probability` |
| Target type | Binary classification |
| Target = 0 | Retained customer |
| Target = 1 | Churned customer |
| Prediction point | End of August 2014 |
| Prediction horizon | Subsequent churn outcome |
| Available behavioral period | June–August 2014 |
| Model feature set | 169 frozen features |
| Future information | Excluded |
| September information | Excluded |
| Test data | Not loaded during baseline development |

The source column `churn_probability` is treated as the binary target variable in this dataset. It is not used as an existing probability score.

---

## 4. Prediction Data Contract

The model may use only information that would have been available by the prediction point.

### Allowed

- June customer behavior
- July customer behavior
- August customer behavior
- Customer tenure information available by the prediction point
- Revenue and recharge behavior available by the prediction point
- Usage and engagement behavior available by the prediction point
- Engineered features derived exclusively from June–August information
- The frozen 169-feature model input set

### Not Allowed

- `churn_probability` as a model input
- Customer identifier as a predictive feature
- September or future-period information
- Outcome-derived information
- Features created using future observations
- Transformations fitted using validation or test data
- Any information unavailable at the prediction point

This contract also establishes the expected input structure for the future prediction application.

---

## 5. Model-Ready Dataset

The finalized feature-selection process produced:

- 427 candidate features
- 258 excluded features
- 169 retained model features

The 169 retained features are frozen for baseline model development.

No additional feature selection will be performed using validation performance during this phase.

---

## 6. Development and Validation Strategy

The model-ready dataset is divided into:

| Dataset | Rows | Purpose |
|---|---:|---|
| Development | 55,999 | Model training and development |
| Validation | 14,000 | Unseen model evaluation and comparison |
| Test | Not loaded | Final unseen evaluation |

The development/validation split uses:

- Stratified sampling
- Random seed: `42`
- Target stratification

The validation dataset is not used for model fitting, feature selection, preprocessing fitting, threshold optimization, or hyperparameter tuning during baseline development.

The test dataset remains completely isolated until the final evaluation stage.

---

## 7. Class Imbalance

The complete modeling population contains:

| Outcome | Customers | Share |
|---|---:|---:|
| Retained | 62,867 | 89.81% |
| Churned | 7,132 | 10.19% |
| Total | 69,999 | 100.00% |

The churn class therefore represents approximately 10.19% of customers.

The approximate majority-to-minority class ratio is:

**8.81 : 1**

This represents a meaningful class imbalance.

A model could achieve high overall accuracy by predominantly predicting the majority class while performing poorly at identifying churners.

Therefore, accuracy will not be treated as the primary model-selection criterion.

---

## 8. Evaluation Hierarchy

Model evaluation will be performed at four levels.

### 8.1 Discrimination

Measures how effectively the model ranks churners above retained customers.

Primary metric:

- PR-AUC

Secondary metric:

- ROC-AUC

PR-AUC receives particular attention because churn is a relatively rare class.

---

### 8.2 Classification Performance

Measures performance at a defined classification threshold.

Metrics:

- Recall
- Precision
- F1-score
- Confusion matrix

For initial baseline comparison, a common threshold of `0.50` will be used.

This threshold is a diagnostic comparison threshold and is **not** treated as the final business decision threshold.

---

### 8.3 Customer-Targeting Performance

Measures whether high-risk customers are concentrated near the top of the model's ranking.

Planned measures:

- Top 5% churn capture
- Top 10% churn capture
- Top 20% churn capture
- Lift
- Gains

These measures will be introduced after baseline predictive performance has been established.

---

### 8.4 Business Impact

Later stages will connect model predictions with:

- customer value,
- revenue exposure,
- revenue risk,
- retention prioritization,
- hypothetical retention scenarios.

Business-impact calculations will clearly distinguish observed evaluation results from hypothetical scenarios.

No projected financial outcome will be presented as guaranteed.

---

## 9. Model Development Sequence

The initial modeling sequence is:

1. Dummy Classifier
2. Logistic Regression
3. Tree-based baseline
4. XGBoost

The purpose is to establish progressively stronger predictive baselines rather than testing a large number of algorithms without business justification.

---

## 10. Baseline Comparison Protocol

Every baseline model will use:

- the same frozen 169-feature dataset,
- the same development/validation split,
- the same preprocessing principles,
- the same initial classification threshold,
- the same core evaluation metrics,
- the same random-state policy where applicable.

This creates a controlled comparison between candidate models.

---

## 11. Model Selection Philosophy

No single metric will automatically determine the final model.

Model selection will consider:

1. PR-AUC
2. Recall
3. Precision
4. F1-score
5. ROC-AUC
6. Stability
7. Probability-ranking quality
8. Customer-targeting usefulness
9. Business relevance

A model with strong aggregate discrimination but poor practical targeting performance will not automatically be considered the best business model.

---

## 12. Threshold Optimization Boundary

Threshold optimization is intentionally excluded from the initial baseline comparison.

The initial `0.50` threshold exists only to provide a common classification reference.

After the strongest model candidate has been identified, threshold analysis will consider:

- precision-recall trade-offs,
- churn capture,
- intervention capacity,
- potential cost of false positives,
- potential cost of false negatives,
- business targeting objectives.

Any financial assumptions used for threshold or ROI analysis will be explicitly labeled as hypothetical assumptions.

---

## 13. Reproducibility

Each modeling experiment will record:

- random seed,
- feature-set version,
- preprocessing configuration,
- model type,
- model parameters,
- training dataset,
- validation dataset,
- evaluation metrics,
- threshold,
- model-selection decision.

The modeling process should be reproducible without relying on undocumented manual steps.

---

## 14. Modeling Controls

The following controls remain active:

- 169-feature freeze
- No target leakage
- No future-period leakage
- No validation leakage during fitting
- No test-data usage
- Development-only preprocessing fitting
- Common evaluation framework
- Common initial classification threshold
- No threshold optimization during baseline comparison
- No hyperparameter tuning during baseline comparison
- No SHAP explainability during baseline comparison
- Fixed random-state policy

---

## 15. Quality Gate

Before baseline model training begins, the following conditions must pass:

- Exactly 169 model features are present.
- Development and validation feature schemas match.
- Development and validation target schemas match.
- Target contains only binary values.
- Feature/target row counts align.
- No duplicate feature names exist.
- No model-ready missing values exist.
- No infinite feature values exist.
- All model features are numeric.
- Development and validation target distributions are compatible.
- Validation remains separate from training.
- Test data remains unloaded.
- Initial threshold is defined consistently.
- No threshold optimization has occurred.

---

## Final Status

Model Development & Validation: COMPLETE

Modeling foundation and target validation: COMPLETE

Dummy baseline: COMPLETE

Logistic Regression baseline: COMPLETE

HistGradientBoosting baseline: COMPLETE

XGBoost baseline: COMPLETE

5-fold cross-validation stability analysis: COMPLETE

Controlled hyperparameter tuning: COMPLETE

Held-out validation: COMPLETE

Final model selection: COMPLETE

Selected model: XGBoost — XGB_02_SLOWER_MORE_TREES

Probability calibration: COMPLETE

Selected calibration method: Sigmoid

Threshold optimization: COMPLETE

Primary analytical threshold: 0.10

Customer risk intelligence: COMPLETE

Feature freeze: ACTIVE

Validation/test separation: PASS

Model leakage controls: PASS

Model Development & Validation is complete. The selected tuned XGBoost model and calibrated risk framework are approved for downstream explainability, revenue-risk analysis, prioritization, dashboarding, and application development.