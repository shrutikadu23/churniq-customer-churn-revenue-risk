# ChurnIQ — Step 7.8
## Controlled Hyperparameter Tuning

### Objective

Step 7.8 performs controlled hyperparameter tuning for HistGradientBoosting and XGBoost, the two strongest candidates identified during Step 7.7.

### Experimental design

- Development dataset only
- 55,999 development observations
- 169 frozen model features
- Stratified 5-fold cross-validation
- Same CV folds reused across every candidate
- Random state: 42
- 10 candidates per model
- 20 candidates total
- 100 fold-level evaluations
- Primary metric: PR-AUC
- Secondary metrics: ROC-AUC, Precision, Recall, F1
- Business metric: Top-10 churn capture and lift

### Validation boundary

The held-out validation dataset was not loaded during hyperparameter tuning. The test dataset was also not loaded.

### HGB result

- Baseline CV PR-AUC: 0.7705
- Best candidate: HGB_09_STRONG_REGULARIZATION
- Best CV PR-AUC: 0.7748
- Improvement: +0.0044
- PR-AUC standard deviation: 0.0104
- Top-10 churn capture: 71.22%

### XGBoost result

- Baseline CV PR-AUC: 0.7720
- Best candidate: XGB_02_SLOWER_MORE_TREES
- Best CV PR-AUC: 0.7769
- Improvement: +0.0049
- PR-AUC standard deviation: 0.0091
- Top-10 churn capture: 71.21%

### Current development-CV leader

**XGBoost (XGB_02_SLOWER_MORE_TREES)**

Development CV PR-AUC: **0.7769**

Top-10 churn capture: **71.21%**

This is a development cross-validation result, not the final held-out validation result.

### Selection rule

Candidates are ranked using mean PR-AUC as the primary criterion. Lower PR-AUC variability, higher Top-10 capture, and lower generalization gap are used as secondary criteria.

### Controls maintained

- Feature freeze remained active
- No feature selection
- No resampling
- No class weighting
- No threshold optimization
- No probability calibration
- No SHAP analysis
- Validation remained untouched
- Test remained untouched

### Interpretation

Tuning is considered useful only when it produces a repeatable improvement over the corresponding baseline without an obvious increase in instability or generalization gap.

### Quality gate

**PASS — Step 7.8 completed successfully.**

### Next step

Step 7.9 will evaluate the tuned candidates and untuned baselines on the untouched validation dataset for final model comparison.