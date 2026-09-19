# ChurnIQ — Step 7.7
## Cross-Validation & Model Stability

### Purpose

Step 7.7 evaluates the stability of the three baseline models using development data only.

The held-out validation dataset remains untouched and the test dataset is not loaded.

### Evaluation design

- Development data only
- 55,999 development observations
- 169 frozen model features
- Stratified 5-fold cross-validation
- Shuffle enabled
- Random state: 42
- Same folds used for model comparison
- Exact baseline model configurations reused
- No hyperparameter tuning
- No threshold optimization
- No calibration
- No SHAP
- No feature selection
- No resampling
- No class weighting

### Primary metric

PR-AUC is the primary model discrimination metric because churn is an imbalanced binary outcome.

### Current CV ranking

1. **XGBoost** — PR-AUC 0.7720 ± 0.0078

2. **HistGradientBoosting** — PR-AUC 0.7705 ± 0.0086

3. **Logistic Regression** — PR-AUC 0.6959 ± 0.0063

### Fold ranking consistency

Current CV leader: **XGBoost**

It achieved the highest PR-AUC in 4/5 folds.

### Pairwise model comparison

- **XGBoost vs HistGradientBoosting**: mean PR-AUC difference = 0.0016; 4 fold(s) favored the first model, 1 fold(s) favored the second model.
- **XGBoost vs Logistic Regression**: mean PR-AUC difference = 0.0761; 5 fold(s) favored the first model, 0 fold(s) favored the second model.
- **HistGradientBoosting vs Logistic Regression**: mean PR-AUC difference = 0.0745; 5 fold(s) favored the first model, 0 fold(s) favored the second model.

### Out-of-fold predictions

Out-of-fold probability predictions were generated for every development observation. Each observation received a prediction from a model that was not trained on that observation.

These predictions are saved as an analytical artifact for later probability calibration and decision-threshold analysis. No calibration or threshold optimization was performed in Step 7.7.

### Interpretation rule

The CV leader is not automatically declared the final model. Model selection will consider:

- Mean PR-AUC
- Fold-to-fold variability
- ROC-AUC stability
- Classification metric stability
- Top-K churn capture stability
- Pairwise fold consistency
- Previously observed held-out validation performance

### Validation boundary

The held-out validation dataset was not loaded or evaluated during this step. It remains reserved for later model comparison and final evaluation.

### Quality gate

**PASS — Step 7.7 Cross-Validation & Model Stability completed.**

### Next step

Proceed to Step 7.8 Controlled Hyperparameter Tuning only after reviewing the CV evidence and confirming the model candidate(s) to tune.