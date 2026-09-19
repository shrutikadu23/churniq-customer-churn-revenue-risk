# ChurnIQ — Step 7.10 Final Model Selection

## 1. Purpose

This step formally selects the final candidate model using previously generated cross-validation, tuning, and held-out validation evidence.

No model training, hyperparameter tuning, feature selection, threshold optimization, probability calibration, SHAP analysis, or test-data evaluation is performed in this step.

## 2. Selection Framework

- **Primary metric:** Held-out Validation PR-AUC
- Secondary metrics: ROC-AUC, Precision, Recall, F1
- Business metrics: Top-10% churn capture and lift
- Stability evidence: 5-fold cross-validation
- Generalization evidence: CV-to-held-out transfer
- Governance: leakage and validation isolation

## 3. Final Selection

**Selected model: `XGB_02_SLOWER_MORE_TREES`**

Selected held-out validation PR-AUC: **0.7614**

The selected model is the held-out validation PR-AUC leader. This prevents subjective model selection after observing the validation results.

## 4. Held-Out Validation Ranking

| Rank | Model | Validation PR-AUC | ROC-AUC | Precision | Recall | F1 | Top-10 Capture | Top-10 Lift |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | XGB_02_SLOWER_MORE_TREES | 0.7614 | 0.9502 | 0.7686 | 0.6220 | 0.6876 | 0.7104 | 7.1038 |
| 2 | HGB_09_STRONG_REGULARIZATION | 0.7581 | 0.9496 | 0.7525 | 0.6227 | 0.6815 | 0.7020 | 7.0196 |
| 3 | XGB_00_BASELINE | 0.7576 | 0.9500 | 0.7705 | 0.6192 | 0.6866 | 0.7020 | 7.0196 |
| 4 | HGB_00_BASELINE | 0.7551 | 0.9477 | 0.7606 | 0.6192 | 0.6826 | 0.6978 | 6.9776 |
| 5 | Logistic Regression | 0.6800 | 0.9186 | 0.7661 | 0.5603 | 0.6472 | 0.6585 | 6.5849 |

## 5. Close Competitor

The tuned HGB candidate remains a close alternative.

- XGBoost tuned validation PR-AUC: **0.7614**
- HGB tuned validation PR-AUC: **0.7581**
- Difference: **0.0033**

The difference is modest, so the selection should not be presented as evidence that XGBoost is categorically superior to HGB. It is the current winner under the locked validation selection framework.

## 6. Tuning Transfer

- Selected model CV-to-holdout absolute PR-AUC gap: **0.0156**

The held-out validation result is used as the final decision evidence rather than selecting a model solely from cross-validation.

## 7. Business Interpretation

The selected model captures approximately **71.04%** of actual churners within the top 10% highest-risk validation customers.

This corresponds to approximately **7.10× lift** over untargeted selection.

These metrics support prioritization of a limited retention capacity. They do not imply that every selected customer will churn.

## 8. Governance Controls

- **169-feature freeze active: PASS**
- **No training performed: PASS**
- **No additional tuning performed: PASS**
- **No threshold optimization performed: PASS**
- **No probability calibration performed: PASS**
- **No SHAP analysis performed: PASS**
- **No test data loaded: PASS**
- **Validation not reused for tuning: PASS**
- **Validation not reused for feature selection: PASS**

## 9. What This Step Does Not Decide

Final model selection does not determine the operational probability threshold.

The threshold will be addressed separately using business-aware cost considerations, precision/recall trade-offs, churn capture, and revenue-risk prioritization.

Probability calibration is also a separate step and has not been performed here.

## 10. Next Step

**Step 7.11 — Probability Calibration & Reliability Analysis**

The selected XGBoost candidate will next be assessed for probability reliability before operational risk thresholds and customer prioritization are finalized.

## 11. Status

**Step 7.10 Final Model Selection: COMPLETE — PASS**
