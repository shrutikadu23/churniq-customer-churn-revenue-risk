"""
ChurnIQ — Step 7.6
XGBoost Baseline

Purpose
-------
Establish a controlled XGBoost benchmark before:
- hyperparameter tuning
- threshold optimization
- probability calibration
- SHAP explainability
- cost-sensitive modeling

Experimental controls
---------------------
- 169 frozen model features
- Development data used for fitting only
- Validation data used for evaluation only
- Test data not loaded
- No feature selection
- No hyperparameter tuning
- No threshold optimization
- No class weighting
- No resampling
- No early stopping
- No calibration
- No SHAP
- Initial threshold = 0.50
- Same evaluation framework as previous baselines
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

X_DEV_PATH = DATA_DIR / "X_dev.csv"
X_VAL_PATH = DATA_DIR / "X_validation.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"
Y_VAL_PATH = DATA_DIR / "y_validation.csv"

RESULTS_PATH = REPORT_DIR / "07_6_xgboost_baseline_results.csv"
CONFUSION_PATH = REPORT_DIR / "07_6_xgboost_baseline_confusion_matrix.csv"
METADATA_PATH = REPORT_DIR / "07_6_xgboost_baseline_metadata.json"
MODEL_PATH = MODEL_DIR / "xgboost_baseline.joblib"

TARGET_NAME = "churn_probability"

EXPECTED_FEATURE_COUNT = 169

RANDOM_STATE = 42

INITIAL_THRESHOLD = 0.50


# ============================================================================
# CONTROLLED BASELINE CONFIGURATION
# ============================================================================

# IMPORTANT:
# These parameters are deliberately fixed.
# They are NOT tuned in Step 7.6.

XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.10,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 75)
    print("ChurnIQ — STEP 7.6")
    print(title)
    print("=" * 75)


def validate_binary_target(
    y: pd.Series,
    name: str,
) -> None:

    assert y.notna().all(), (
        f"{name} contains missing target values."
    )

    unique_values = set(
        y.unique()
    )

    assert unique_values.issubset({0, 1}), (
        f"{name} contains non-binary values: "
        f"{unique_values}"
    )


def validate_model_features(
    X: pd.DataFrame,
    name: str,
) -> None:

    assert X.shape[1] == EXPECTED_FEATURE_COUNT, (
        f"{name} feature count mismatch: "
        f"{X.shape[1]} != {EXPECTED_FEATURE_COUNT}"
    )

    assert X.columns.is_unique, (
        f"{name} contains duplicate feature names."
    )

    assert all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ), (
        f"{name} contains non-numeric features."
    )

    assert not X.isna().any().any(), (
        f"{name} contains missing feature values."
    )

    values = X.to_numpy(dtype=float)

    assert np.isfinite(values).all(), (
        f"{name} contains infinite feature values."
    )


def calculate_top_k_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict:

    y_true_array = np.asarray(y_true)
    probability_array = np.asarray(probabilities)

    total_actual_churners = int(
        y_true_array.sum()
    )

    assert total_actual_churners > 0

    # Stable descending sort makes tied probabilities
    # deterministic and reproducible.
    ranking = np.argsort(
        -probability_array,
        kind="mergesort",
    )

    results = {}

    for percentage in [5, 10, 20]:

        k = max(
            1,
            int(
                np.ceil(
                    len(y_true_array)
                    * percentage
                    / 100
                )
            ),
        )

        selected_indices = ranking[:k]

        captured_churners = int(
            y_true_array[
                selected_indices
            ].sum()
        )

        capture = (
            captured_churners
            / total_actual_churners
        )

        baseline_capture = (
            percentage / 100
        )

        lift = (
            capture / baseline_capture
        )

        results[
            f"top_{percentage}_capture"
        ] = capture

        results[
            f"top_{percentage}_lift"
        ] = lift

        results[
            f"top_{percentage}_customers"
        ] = k

        results[
            f"top_{percentage}_captured_churners"
        ] = captured_churners

    return results


# ============================================================================
# 3. START
# ============================================================================

print_header("XGBoost Baseline")


# ============================================================================
# 4. LOAD MODEL-READY DATA
# ============================================================================

print("\n[1] Loading model-ready datasets...")

X_dev = pd.read_csv(
    X_DEV_PATH
)

X_validation = pd.read_csv(
    X_VAL_PATH
)

y_dev = pd.read_csv(
    Y_DEV_PATH
).squeeze("columns")

y_validation = pd.read_csv(
    Y_VAL_PATH
).squeeze("columns")


print(
    f"Development features: {X_dev.shape}"
)

print(
    f"Validation features:  {X_validation.shape}"
)

print(
    f"Development target:   {y_dev.shape}"
)

print(
    f"Validation target:    {y_validation.shape}"
)


# ============================================================================
# 5. STRUCTURAL VALIDATION
# ============================================================================

print("\n[2] Validating dataset structure...")

assert X_dev.shape[1] == EXPECTED_FEATURE_COUNT
assert X_validation.shape[1] == EXPECTED_FEATURE_COUNT

assert len(X_dev) == len(y_dev)
assert len(X_validation) == len(y_validation)

assert list(X_dev.columns) == list(
    X_validation.columns
)

assert X_dev.columns.is_unique
assert X_validation.columns.is_unique

print(
    f"Feature count: "
    f"{EXPECTED_FEATURE_COUNT} PASS"
)

print(
    "Development row alignment PASS"
)

print(
    "Validation row alignment PASS"
)

print(
    "Feature schema consistency PASS"
)

print(
    "Duplicate feature-name check PASS"
)


# ============================================================================
# 6. TARGET VALIDATION
# ============================================================================

print("\n[3] Validating target...")

validate_binary_target(
    y_dev,
    "Development target",
)

validate_binary_target(
    y_validation,
    "Validation target",
)

assert y_dev.name == TARGET_NAME
assert y_validation.name == TARGET_NAME

print(
    f"Target name: "
    f"{TARGET_NAME} PASS"
)

print(
    "Binary target validation PASS"
)

print(
    "Target missing-value check PASS"
)


# ============================================================================
# 7. FEATURE VALUE VALIDATION
# ============================================================================

print("\n[4] Validating model feature values...")

validate_model_features(
    X_dev,
    "Development features",
)

validate_model_features(
    X_validation,
    "Validation features",
)

print(
    "Numeric feature validation PASS"
)

print(
    "Development missing-value check PASS"
)

print(
    "Validation missing-value check PASS"
)

print(
    "Development infinite-value check PASS"
)

print(
    "Validation infinite-value check PASS"
)


# ============================================================================
# 8. TARGET DISTRIBUTION
# ============================================================================

print("\n[5] Target distribution")

dev_counts = (
    y_dev
    .value_counts()
    .sort_index()
)

val_counts = (
    y_validation
    .value_counts()
    .sort_index()
)

dev_retained = int(
    dev_counts.get(0, 0)
)

dev_churned = int(
    dev_counts.get(1, 0)
)

val_retained = int(
    val_counts.get(0, 0)
)

val_churned = int(
    val_counts.get(1, 0)
)

dev_churn_rate = float(
    y_dev.mean()
)

val_churn_rate = float(
    y_validation.mean()
)

print("\nDevelopment:")

print(
    f"  Retained: {dev_retained:,} "
    f"({(1 - dev_churn_rate) * 100:.2f}%)"
)

print(
    f"  Churned:  {dev_churned:,} "
    f"({dev_churn_rate * 100:.2f}%)"
)

print("\nValidation:")

print(
    f"  Retained: {val_retained:,} "
    f"({(1 - val_churn_rate) * 100:.2f}%)"
)

print(
    f"  Churned:  {val_churned:,} "
    f"({val_churn_rate * 100:.2f}%)"
)


# ============================================================================
# 9. CLASS IMBALANCE
# ============================================================================

print("\n[6] Class imbalance assessment")

rate_difference = abs(
    dev_churn_rate
    - val_churn_rate
)

minority_ratio = (
    dev_retained
    / dev_churned
    if dev_churned > 0
    else np.inf
)

print(
    f"Development churn rate: "
    f"{dev_churn_rate * 100:.4f}%"
)

print(
    f"Validation churn rate:  "
    f"{val_churn_rate * 100:.4f}%"
)

print(
    f"Absolute rate difference: "
    f"{rate_difference * 100:.4f}%"
)

print(
    f"Development majority/minority ratio: "
    f"{minority_ratio:.2f}:1"
)

assert rate_difference < 0.01

print(
    "Stratified target-rate preservation PASS"
)


# ============================================================================
# 10. NO-SKILL BENCHMARK
# ============================================================================

print("\n[7] No-skill benchmark")

no_skill_pr_auc = val_churn_rate

print(
    f"Validation churn prevalence / "
    f"no-skill PR-AUC: "
    f"{no_skill_pr_auc:.6f}"
)

print(
    "This represents the approximate PR-AUC "
    "expected from a non-informative classifier."
)


# ============================================================================
# 11. BUILD XGBOOST MODEL
# ============================================================================

print("\n[8] Building XGBoost baseline...")

print(
    "Model: XGBClassifier"
)

print(
    "Model type: gradient-boosted decision trees"
)

print(
    "Baseline configuration:"
)

for parameter, value in XGB_PARAMS.items():

    print(
        f"  {parameter}: {value}"
    )

print(
    "Class weighting: None"
)

print(
    "Resampling: None"
)

print(
    "Early stopping: None"
)

print(
    "Hyperparameter tuning: False"
)


model = XGBClassifier(
    **XGB_PARAMS
)


# ============================================================================
# 12. TRAIN MODEL
# ============================================================================

print("\n[9] Training XGBoost baseline...")

start_time = time.perf_counter()

model.fit(
    X_dev,
    y_dev,
)

training_time = (
    time.perf_counter()
    - start_time
)

print(
    "Development-only fitting: PASS"
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# ============================================================================
# 13. MODEL FIT DIAGNOSTICS
# ============================================================================

print("\n[10] Checking model fit diagnostics...")

assert hasattr(
    model,
    "feature_names_in_"
)

assert list(
    model.feature_names_in_
) == list(
    X_dev.columns
)

print(
    "Fitted feature schema matches frozen schema: PASS"
)

assert hasattr(
    model,
    "n_estimators"
)

print(
    f"Configured estimators: "
    f"{model.n_estimators}"
)

assert model.n_estimators == (
    XGB_PARAMS["n_estimators"]
)

print(
    "Configured estimator count validated: PASS"
)


# ============================================================================
# 14. GENERATE VALIDATION PROBABILITIES
# ============================================================================

print(
    "\n[11] Generating validation probabilities..."
)

validation_probabilities = (
    model
    .predict_proba(
        X_validation
    )[:, 1]
)

assert (
    len(validation_probabilities)
    == len(y_validation)
)

assert np.isfinite(
    validation_probabilities
).all()

assert (
    (validation_probabilities >= 0).all()
    and
    (validation_probabilities <= 1).all()
)

print(
    "Validation probability generation: PASS"
)

print(
    "Probability count validation: PASS"
)

print(
    "Probability range validation: PASS"
)


# ============================================================================
# 15. REPRODUCIBILITY CHECK
# ============================================================================

print(
    "\n[12] Running deterministic prediction check..."
)

repeat_probabilities = (
    model
    .predict_proba(
        X_validation
    )[:, 1]
)

reproducible = np.array_equal(
    validation_probabilities,
    repeat_probabilities,
)

assert reproducible

print(
    "Repeated prediction consistency: PASS"
)


# ============================================================================
# 16. INITIAL THRESHOLD CLASSIFICATION
# ============================================================================

validation_predictions = (
    validation_probabilities
    >= INITIAL_THRESHOLD
).astype(int)


# ============================================================================
# 17. VALIDATION EVALUATION
# ============================================================================

print(
    "\n[13] Evaluating validation performance..."
)

accuracy = accuracy_score(
    y_validation,
    validation_predictions,
)

precision = precision_score(
    y_validation,
    validation_predictions,
    zero_division=0,
)

recall = recall_score(
    y_validation,
    validation_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities,
)

pr_auc = average_precision_score(
    y_validation,
    validation_probabilities,
)

logloss = log_loss(
    y_validation,
    validation_probabilities,
)

brier = brier_score_loss(
    y_validation,
    validation_probabilities,
)

predicted_churn_rate = (
    validation_predictions.mean()
)

top_k = calculate_top_k_metrics(
    y_validation,
    validation_probabilities,
)

cm = confusion_matrix(
    y_validation,
    validation_predictions,
)

print("\nXGBoost")
print("-" * 50)

print(
    f"Accuracy:                "
    f"{accuracy:.4f}"
)

print(
    f"Precision:               "
    f"{precision:.4f}"
)

print(
    f"Recall:                  "
    f"{recall:.4f}"
)

print(
    f"F1:                      "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC:                 "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC:                  "
    f"{pr_auc:.4f}"
)

print(
    f"Log Loss:                "
    f"{logloss:.4f}"
)

print(
    f"Brier Score:             "
    f"{brier:.4f}"
)

print(
    f"Predicted churn rate:    "
    f"{predicted_churn_rate * 100:.2f}%"
)

print(
    f"Top-5% churn capture:    "
    f"{top_k['top_5_capture'] * 100:.2f}%"
)

print(
    f"Top-5% lift:             "
    f"{top_k['top_5_lift']:.2f}x"
)

print(
    f"Top-10% churn capture:   "
    f"{top_k['top_10_capture'] * 100:.2f}%"
)

print(
    f"Top-10% lift:            "
    f"{top_k['top_10_lift']:.2f}x"
)

print(
    f"Top-20% churn capture:   "
    f"{top_k['top_20_capture'] * 100:.2f}%"
)

print(
    f"Top-20% lift:            "
    f"{top_k['top_20_lift']:.2f}x"
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================================
# 18. BASELINE COMPARISON
# ============================================================================

print(
    "\n[14] Baseline comparison"
)

print(
    f"No-skill PR-AUC:       "
    f"{no_skill_pr_auc:.4f}"
)

print(
    f"XGBoost PR-AUC:        "
    f"{pr_auc:.4f}"
)

pr_auc_beats_baseline = (
    pr_auc > no_skill_pr_auc
)

print(
    "PR-AUC beats no-skill benchmark: "
    f"{'PASS' if pr_auc_beats_baseline else 'FAIL'}"
)

assert pr_auc_beats_baseline


# ============================================================================
# 19. SAVE RESULTS
# ============================================================================

print(
    "\n[15] Saving model artifacts..."
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results = pd.DataFrame(
    [
        {
            "model": "XGBoost",
            "threshold": INITIAL_THRESHOLD,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "log_loss": logloss,
            "brier_score": brier,
            "predicted_churn_rate": predicted_churn_rate,
            "top_5_capture": top_k[
                "top_5_capture"
            ],
            "top_5_lift": top_k[
                "top_5_lift"
            ],
            "top_10_capture": top_k[
                "top_10_capture"
            ],
            "top_10_lift": top_k[
                "top_10_lift"
            ],
            "top_20_capture": top_k[
                "top_20_capture"
            ],
            "top_20_lift": top_k[
                "top_20_lift"
            ],
            "top_5_customers": top_k[
                "top_5_customers"
            ],
            "top_10_customers": top_k[
                "top_10_customers"
            ],
            "top_20_customers": top_k[
                "top_20_customers"
            ],
            "top_5_captured_churners": top_k[
                "top_5_captured_churners"
            ],
            "top_10_captured_churners": top_k[
                "top_10_captured_churners"
            ],
            "top_20_captured_churners": top_k[
                "top_20_captured_churners"
            ],
            "training_time_seconds": training_time,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "random_state": RANDOM_STATE,
        }
    ]
)

results.to_csv(
    RESULTS_PATH,
    index=False,
)


confusion_df = pd.DataFrame(
    cm,
    index=[
        "actual_retained",
        "actual_churned",
    ],
    columns=[
        "predicted_retained",
        "predicted_churned",
    ],
)

confusion_df.to_csv(
    CONFUSION_PATH
)


metadata = {
    "model": "XGBoost",
    "model_class": "XGBClassifier",
    "target": TARGET_NAME,
    "feature_count": EXPECTED_FEATURE_COUNT,
    "random_state": RANDOM_STATE,
    "initial_threshold": INITIAL_THRESHOLD,

    "class_weighting": False,
    "resampling": False,
    "early_stopping": False,
    "hyperparameter_tuning": False,
    "threshold_optimization": False,
    "calibration": False,
    "shap_explainability": False,

    "test_data_loaded": False,

    "validation_used_for_feature_selection": False,
    "validation_used_for_hyperparameter_tuning": False,
    "validation_used_for_threshold_optimization": False,

    "development_rows": len(X_dev),
    "validation_rows": len(X_validation),

    "training_time_seconds": training_time,

    "xgboost_parameters": XGB_PARAMS,

    "feature_names": list(
        X_dev.columns
    ),
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=2,
    )


joblib.dump(
    model,
    MODEL_PATH,
)


print(
    f"Saved: {RESULTS_PATH}"
)

print(
    f"Saved: {CONFUSION_PATH}"
)

print(
    f"Saved: {METADATA_PATH}"
)

print(
    f"Saved: {MODEL_PATH}"
)


# ============================================================================
# 20. QUALITY CHECKS
# ============================================================================

print(
    "\n[16] Running quality checks..."
)

assert (
    X_dev.shape[1]
    == EXPECTED_FEATURE_COUNT
)

print(
    "169 frozen features: PASS"
)

assert (
    len(X_dev)
    == len(y_dev)
)

assert (
    len(X_validation)
    == len(y_validation)
)

print(
    "Development/validation alignment: PASS"
)

assert (
    list(X_dev.columns)
    == list(X_validation.columns)
)

print(
    "Feature schemas aligned: PASS"
)

assert X_dev.columns.is_unique

print(
    "No duplicate feature names: PASS"
)

assert set(
    y_dev.unique()
).issubset({0, 1})

assert set(
    y_validation.unique()
).issubset({0, 1})

print(
    "Binary target: PASS"
)

assert not X_dev.isna().any().any()
assert not X_validation.isna().any().any()

print(
    "No missing model features: PASS"
)

assert np.isfinite(
    X_dev.to_numpy(dtype=float)
).all()

assert np.isfinite(
    X_validation.to_numpy(dtype=float)
).all()

print(
    "No infinite model features: PASS"
)

assert (
    len(validation_probabilities)
    == len(y_validation)
)

print(
    "Probability count matches validation: PASS"
)

assert (
    (validation_probabilities >= 0).all()
    and
    (validation_probabilities <= 1).all()
)

print(
    "Probability range valid: PASS"
)

assert list(
    model.feature_names_in_
) == list(
    X_dev.columns
)

print(
    "Fitted feature order matches frozen schema: PASS"
)

assert np.array_equal(
    validation_probabilities,
    repeat_probabilities,
)

print(
    "Prediction reproducibility check: PASS"
)

print(
    "Development-only fitting: PASS"
)

print(
    "Validation-only evaluation: PASS"
)

assert (
    INITIAL_THRESHOLD == 0.50
)

print(
    "Initial threshold fixed at 0.50: PASS"
)

print(
    "No threshold optimization: PASS"
)

print(
    "No hyperparameter tuning: PASS"
)

print(
    "No class weighting: PASS"
)

print(
    "No resampling: PASS"
)

print(
    "No early stopping: PASS"
)

print(
    "Test data not loaded: PASS"
)

assert np.isfinite(pr_auc)

assert np.isfinite(roc_auc)

print(
    "PR-AUC calculated: PASS"
)

print(
    "ROC-AUC calculated: PASS"
)

assert np.isfinite(logloss)

assert np.isfinite(brier)

print(
    "Log Loss calculated: PASS"
)

print(
    "Brier Score calculated: PASS"
)

assert all(
    key in top_k
    for key in [
        "top_5_capture",
        "top_5_lift",
        "top_10_capture",
        "top_10_lift",
        "top_20_capture",
        "top_20_lift",
    ]
)

print(
    "Top-K targeting metrics calculated: PASS"
)

assert RESULTS_PATH.exists()

print(
    "Results artifact saved: PASS"
)

assert CONFUSION_PATH.exists()

print(
    "Confusion matrix saved: PASS"
)

assert MODEL_PATH.exists()

print(
    "Model artifact saved: PASS"
)

assert METADATA_PATH.exists()

print(
    "Metadata artifact saved: PASS"
)


# ============================================================================
# 21. FINAL QUALITY GATE
# ============================================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 7.6 QUALITY GATE"
)

print(
    "=" * 75
)

print(
    "169 frozen features: PASS"
)

print(
    "Development data used for fitting only: PASS"
)

print(
    "Validation data used for evaluation only: PASS"
)

print(
    "Test data not loaded: PASS"
)

print(
    "XGBoost baseline completed: PASS"
)

print(
    "No-skill PR-AUC benchmark established: PASS"
)

print(
    "PR-AUC beats no-skill benchmark: PASS"
)

print(
    "PR-AUC calculated: PASS"
)

print(
    "ROC-AUC calculated: PASS"
)

print(
    "Classification metrics calculated: PASS"
)

print(
    "Probability metrics calculated: PASS"
)

print(
    "Top-K targeting metrics calculated: PASS"
)

print(
    "Confusion matrix saved: PASS"
)

print(
    "Model artifact saved: PASS"
)

print(
    "Metadata artifact saved: PASS"
)

print(
    "Initial threshold fixed at 0.50: PASS"
)

print(
    "Threshold optimization not performed: PASS"
)

print(
    "Hyperparameter tuning not performed: PASS"
)

print(
    "Class weighting not performed: PASS"
)

print(
    "Resampling not performed: PASS"
)

print(
    "Early stopping not performed: PASS"
)

print(
    "Prediction reproducibility verified: PASS"
)

print(
    "\nFINAL STATUS: PASS"
)

print(
    "Ready for Step 7.7 — Cross-Validation & Model Stability."
)

print(
    "=" * 75
)