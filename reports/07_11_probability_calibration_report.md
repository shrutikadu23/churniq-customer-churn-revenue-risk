# Step 7.11 — Probability Calibration & Reliability Analysis

## Objective
Assess whether the selected tuned XGBoost model produces probabilities that are sufficiently reliable for downstream customer-risk interpretation.

## Selected Model
- Model: XGBoost
- Step 7.8 candidate: `XGB_02_SLOWER_MORE_TREES`
- Step 7.8 development CV PR-AUC: 0.776913
- Step 7.9 validation model label: `XGBoost Tuned`

## Methodology
- Generated fresh 5-fold OOF predictions for the exact Step 7.8 tuned XGBoost configuration.
- Calibration-method selection used development OOF data only.
- Compared uncalibrated, sigmoid and isotonic calibration.
- Primary calibration metrics: Brier score, log loss and ECE.
- PR-AUC and ROC-AUC were monitored to ensure predictive ranking was not materially degraded.
- The Step 7.9 validation set remained untouched until final confirmation.

## Calibration Method Selection
### Sigmoid
- Mean Brier score: 0.041740
- Mean log loss: 0.144759
- Mean ECE: 0.005859
- Mean PR-AUC: 0.776913
- Mean ROC-AUC: 0.955720
- Selection rank: 1

### Isotonic
- Mean Brier score: 0.041771
- Mean log loss: 0.145131
- Mean ECE: 0.006341
- Mean PR-AUC: 0.767137
- Mean ROC-AUC: 0.955568
- Selection rank: 2

### Uncalibrated
- Mean Brier score: 0.041797
- Mean log loss: 0.144972
- Mean ECE: 0.007504
- Mean PR-AUC: 0.776913
- Mean ROC-AUC: 0.955720
- Selection rank: 3

**Selected calibration method: `sigmoid`**

## Validation Reliability Results
| Metric | Before | After |
|---|---:|---:|
| Brier score | 0.043227 | 0.043096 |
| Log loss | 0.150891 | 0.150287 |
| ECE | 0.009384 | 0.006148 |
| PR-AUC | 0.761352 | 0.761352 |
| ROC-AUC | 0.950225 | 0.950225 |
| Calibration slope | 0.956697 | 0.989230 |
| Calibration intercept | 0.113079 | 0.057775 |

## Ranking Preservation
- Spearman correlation between uncalibrated and calibrated validation probabilities: `1.000000`
Calibration should preserve customer-risk ordering as much as possible because thresholding and Top-K prioritization depend on ranking as well as probability quality.

## Governance Controls
- Frozen 169-feature schema: PASS
- Fresh OOF predictions for exact tuned model: PASS
- Validation used for calibration-method selection: NO
- Validation used for calibrator fitting: NO
- Test data loaded: NO
- Additional hyperparameter tuning: NO
- Threshold optimization: NO
- Feature selection: NO
- Resampling: NO
- Class weighting: NO
- SHAP analysis: NO

## Interpretation
The selected calibration method is `sigmoid` based on development-only calibration evidence. Validation results are treated as final confirmation rather than as a method selection mechanism.

Calibration does not change the underlying churn label. It attempts to make predicted probabilities better aligned with observed outcome frequencies.

Probability calibration is intentionally separated from threshold optimization. The operating threshold will be evaluated independently in Step 7.12.

## Artifacts
- `reports\07_11_tuned_xgb_oof_predictions.csv`
- `reports\07_11_tuned_xgb_oof_fold_results.csv`
- `reports\07_11_calibration_method_fold_results.csv`
- `reports\07_11_calibration_method_comparison.csv`
- `reports\07_11_validation_calibration_comparison.csv`
- `reports\07_11_reliability_curve.csv`
- `reports\07_11_calibrated_validation_predictions.csv`
- `models\xgb_probability_calibrator.joblib`
- `reports\07_11_calibration_metadata.json`