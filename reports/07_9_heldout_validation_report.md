# ChurnIQ — Step 7.9
## Held-Out Validation & Model Comparison

### Objective

Step 7.9 evaluates the baseline and tuned candidate models on the untouched validation dataset.

### Validation design

- Development rows: 55,999
- Validation rows: 14,000
- Frozen features: 169
- Validation was not used for feature selection
- Validation was not used for hyperparameter tuning
- Validation was not used for threshold optimization
- Validation was not used for calibration
- Test data was not loaded
- Threshold fixed at 0.50
- Primary metric: PR-AUC

### Models evaluated

1. Logistic Regression baseline
2. HistGradientBoosting baseline
3. HistGradientBoosting tuned
4. XGBoost baseline
5. XGBoost tuned

### Validation results

#### 1. XGBoost Tuned

- Development CV PR-AUC: 0.7769
- Validation PR-AUC: 0.7614
- Validation ROC-AUC: 0.9502
- Precision: 0.7686
- Recall: 0.6220
- F1: 0.6876
- Log Loss: 0.1509
- Brier Score: 0.0432
- Top-10 churn capture: 71.04%
- Top-10 lift: 7.10x
- Predicted churn rate: 8.24%
- CV → validation PR-AUC change: -0.0156

#### 2. HistGradientBoosting Tuned

- Development CV PR-AUC: 0.7748
- Validation PR-AUC: 0.7581
- Validation ROC-AUC: 0.9496
- Precision: 0.7525
- Recall: 0.6227
- F1: 0.6815
- Log Loss: 0.1522
- Brier Score: 0.0438
- Top-10 churn capture: 70.20%
- Top-10 lift: 7.02x
- Predicted churn rate: 8.43%
- CV → validation PR-AUC change: -0.0167

#### 3. XGBoost Baseline

- Development CV PR-AUC: 0.7720
- Validation PR-AUC: 0.7576
- Validation ROC-AUC: 0.9500
- Precision: 0.7705
- Recall: 0.6192
- F1: 0.6866
- Log Loss: 0.1519
- Brier Score: 0.0436
- Top-10 churn capture: 70.20%
- Top-10 lift: 7.02x
- Predicted churn rate: 8.19%
- CV → validation PR-AUC change: -0.0144

#### 4. HistGradientBoosting Baseline

- Development CV PR-AUC: 0.7705
- Validation PR-AUC: 0.7551
- Validation ROC-AUC: 0.9477
- Precision: 0.7606
- Recall: 0.6192
- F1: 0.6826
- Log Loss: 0.1537
- Brier Score: 0.0439
- Top-10 churn capture: 69.78%
- Top-10 lift: 6.98x
- Predicted churn rate: 8.29%
- CV → validation PR-AUC change: -0.0154

#### 5. Logistic Regression

- Development CV PR-AUC: 0.6959
- Validation PR-AUC: 0.6800
- Validation ROC-AUC: 0.9186
- Precision: 0.7661
- Recall: 0.5603
- F1: 0.6472
- Log Loss: 0.1825
- Brier Score: 0.0500
- Top-10 churn capture: 65.85%
- Top-10 lift: 6.58x
- Predicted churn rate: 7.45%
- CV → validation PR-AUC change: -0.0159

### Tuning transfer

- HGB baseline → tuned validation PR-AUC: 0.7551 → 0.7581
- HGB validation improvement: +0.0030
- XGBoost baseline → tuned validation PR-AUC: 0.7576 → 0.7614
- XGBoost validation improvement: +0.0038

### Held-out validation winner

**XGBoost Tuned**

- Validation PR-AUC: **0.7614**
- Validation ROC-AUC: **0.9502**
- Top-10 churn capture: **71.04%**
- Top-10 lift: **7.10x**

### Interpretation

The held-out validation set was evaluated only after feature selection, baseline modeling, cross-validation, and controlled hyperparameter tuning were completed.

PR-AUC is treated as the primary selection metric because the churn target is imbalanced and the business objective requires ranking customers by churn risk.

Top-K capture and lift provide a business-oriented view of how effectively the model concentrates actual churners within the highest-risk customer groups.

A small PR-AUC difference should not automatically be interpreted as a meaningful business advantage. Validation performance, ranking metrics, and stability are considered together.

### Controls maintained

- 169 features remained frozen
- Validation was not used for feature selection
- Validation was not used for tuning
- Validation was not used for threshold optimization
- Validation was not used for calibration
- Test data remained untouched
- No resampling
- No class weighting
- No SHAP

### Quality gate

**PASS — Step 7.9 completed successfully.**

### Next step

Step 7.10 will finalize the selected model based on held-out validation evidence before probability calibration and threshold optimization.