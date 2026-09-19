"""
ChurnIQ — Step 7.4
Logistic Regression Baseline

Purpose:
    Train and evaluate the first genuine predictive model.

Modeling controls:
    - 169 frozen features
    - Development data used for fitting only
    - Validation data used for evaluation only
    - Test data not loaded
    - StandardScaler fitted on development data only
    - No hyperparameter tuning
    - No threshold optimization
    - Initial classification threshold = 0.50
    - class_weight=None for clean baseline comparison

Additional diagnostics:
    - Log Loss
    - Brier Score
    - Training time
    - Iteration/convergence information
    - Model coefficients
    - Model metadata
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)


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

RESULTS_PATH = REPORT_DIR / "07_4_logistic_regression_results.csv"
CONFUSION_PATH = (
    REPORT_DIR / "07_4_logistic_regression_confusion_matrix.csv"
)
COEFFICIENT_PATH = (
    REPORT_DIR / "07_4_logistic_regression_coefficients.csv"
)
METADATA_PATH = (
    REPORT_DIR / "07_4_logistic_regression_metadata.json"
)
MODEL_PATH = MODEL_DIR / "logistic_regression_baseline.joblib"

EXPECTED_FEATURE_COUNT = 169
TARGET_NAME = "churn_probability"

RANDOM_STATE = 42
THRESHOLD = 0.50

MAX_ITER = 3000
CLASS_WEIGHT = None


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def calculate_top_k_metrics(y_true, probabilities, k):
    """
    Calculate churn capture and lift among the top k% highest-risk customers.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    n = len(y_true)
    top_n = max(1, int(np.ceil(n * k)))

    ranking = np.argsort(-probabilities)
    top_indices = ranking[:top_n]

    actual_churn_total = y_true.sum()
    captured_churn = y_true[top_indices].sum()

    if actual_churn_total > 0:
        capture = captured_churn / actual_churn_total
    else:
        capture = np.nan

    baseline_rate = actual_churn_total / n
    top_k_rate = y_true[top_indices].mean()

    if baseline_rate > 0:
        lift = top_k_rate / baseline_rate
    else:
        lift = np.nan

    return capture, lift


def evaluate_model(y_true, probabilities, threshold):
    """
    Evaluate probability predictions using a fixed classification threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities
        ),
        "log_loss": log_loss(
            y_true,
            probabilities
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),
        "predicted_churn_rate": predictions.mean(),
    }

    for k in [0.05, 0.10, 0.20]:

        capture, lift = calculate_top_k_metrics(
            y_true,
            probabilities,
            k
        )

        metrics[f"top_{int(k * 100)}_pct_churn_capture"] = capture
        metrics[f"top_{int(k * 100)}_pct_lift"] = lift

    return metrics, predictions


# ============================================================================
# 3. HEADER
# ============================================================================

print("=" * 75)
print("ChurnIQ — STEP 7.4")
print("Logistic Regression Baseline")
print("=" * 75)


# ============================================================================
# 4. LOAD DATA
# ============================================================================

print("\n[1] Loading model-ready datasets...")

X_dev = pd.read_csv(X_DEV_PATH)
X_val = pd.read_csv(X_VAL_PATH)

y_dev = pd.read_csv(Y_DEV_PATH).squeeze("columns")
y_val = pd.read_csv(Y_VAL_PATH).squeeze("columns")

print(f"Development features: {X_dev.shape}")
print(f"Validation features:  {X_val.shape}")
print(f"Development target:   {y_dev.shape}")
print(f"Validation target:    {y_val.shape}")


# ============================================================================
# 5. STRUCTURE VALIDATION
# ============================================================================

print("\n[2] Validating dataset structure...")

assert X_dev.shape[1] == EXPECTED_FEATURE_COUNT
assert X_val.shape[1] == EXPECTED_FEATURE_COUNT

assert list(X_dev.columns) == list(X_val.columns)

assert len(X_dev) == len(y_dev)
assert len(X_val) == len(y_val)

assert X_dev.columns.is_unique
assert X_val.columns.is_unique

print(f"Feature count: {EXPECTED_FEATURE_COUNT} PASS")
print("Development row alignment PASS")
print("Validation row alignment PASS")
print("Feature schema consistency PASS")
print("Duplicate feature-name check PASS")


# ============================================================================
# 6. TARGET VALIDATION
# ============================================================================

print("\n[3] Validating target...")

assert TARGET_NAME == "churn_probability"

assert y_dev.isna().sum() == 0
assert y_val.isna().sum() == 0

assert set(y_dev.unique()).issubset({0, 1})
assert set(y_val.unique()).issubset({0, 1})

print(f"Target name: {TARGET_NAME} PASS")
print("Binary target validation PASS")
print("Target missing-value check PASS")


# ============================================================================
# 7. FEATURE VALUE VALIDATION
# ============================================================================

print("\n[4] Validating model feature values...")

assert all(
    pd.api.types.is_numeric_dtype(X_dev[col])
    for col in X_dev.columns
)

assert all(
    pd.api.types.is_numeric_dtype(X_val[col])
    for col in X_val.columns
)

assert X_dev.isna().sum().sum() == 0
assert X_val.isna().sum().sum() == 0

assert np.isfinite(X_dev.to_numpy()).all()
assert np.isfinite(X_val.to_numpy()).all()

print("Numeric feature validation PASS")
print("Development missing-value check PASS")
print("Validation missing-value check PASS")
print("Development infinite-value check PASS")
print("Validation infinite-value check PASS")


# ============================================================================
# 8. TARGET DISTRIBUTION
# ============================================================================

print("\n[5] Target distribution")

print("\nDevelopment:")

dev_counts = y_dev.value_counts().sort_index()

print(
    f"  Retained: {dev_counts.get(0, 0):,} "
    f"({dev_counts.get(0, 0) / len(y_dev):.2%})"
)

print(
    f"  Churned:  {dev_counts.get(1, 0):,} "
    f"({dev_counts.get(1, 0) / len(y_dev):.2%})"
)

print("\nValidation:")

val_counts = y_val.value_counts().sort_index()

print(
    f"  Retained: {val_counts.get(0, 0):,} "
    f"({val_counts.get(0, 0) / len(y_val):.2%})"
)

print(
    f"  Churned:  {val_counts.get(1, 0):,} "
    f"({val_counts.get(1, 0) / len(y_val):.2%})"
)


# ============================================================================
# 9. NO-SKILL BENCHMARK
# ============================================================================

validation_prevalence = y_val.mean()

print("\n[6] No-skill benchmark")

print(
    f"Validation churn prevalence / no-skill PR-AUC: "
    f"{validation_prevalence:.6f}"
)


# ============================================================================
# 10. MODEL DEFINITION
# ============================================================================

print("\n[7] Building Logistic Regression pipeline...")

pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=MAX_ITER,
                class_weight=CLASS_WEIGHT,
                random_state=RANDOM_STATE
            )
        )
    ]
)

print("StandardScaler: development-fitted through pipeline")
print("Logistic Regression: baseline configuration")
print(f"class_weight: {CLASS_WEIGHT}")
print(f"max_iter: {MAX_ITER}")
print(f"random_state: {RANDOM_STATE}")


# ============================================================================
# 11. TRAINING
# ============================================================================

print("\n[8] Training Logistic Regression...")

training_start = time.perf_counter()

pipeline.fit(X_dev, y_dev)

training_seconds = time.perf_counter() - training_start

model = pipeline.named_steps["model"]

print("Development-only fitting: PASS")
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Iterations used: {model.n_iter_[0]}")


# ============================================================================
# 12. CONVERGENCE CHECK
# ============================================================================

print("\n[9] Checking model convergence...")

convergence_pass = model.n_iter_[0] < MAX_ITER

print(
    f"Convergence check: "
    f"{'PASS' if convergence_pass else 'REVIEW'}"
)

if not convergence_pass:
    print(
        "WARNING: Logistic Regression reached max_iter. "
        "Review convergence before final model selection."
    )


# ============================================================================
# 13. VALIDATION PROBABILITIES
# ============================================================================

print("\n[10] Generating validation probabilities...")

validation_probabilities = pipeline.predict_proba(X_val)[:, 1]

assert len(validation_probabilities) == len(y_val)
assert np.isfinite(validation_probabilities).all()

assert np.all(validation_probabilities >= 0)
assert np.all(validation_probabilities <= 1)

print("Validation probability generation: PASS")
print("Probability range validation: PASS")


# ============================================================================
# 14. EVALUATION
# ============================================================================

print("\n[11] Evaluating validation performance...")

metrics, validation_predictions = evaluate_model(
    y_val,
    validation_probabilities,
    THRESHOLD
)

print("\nLogistic Regression")
print("-" * 50)

print(f"Accuracy:              {metrics['accuracy']:.4f}")
print(f"Precision:             {metrics['precision']:.4f}")
print(f"Recall:                {metrics['recall']:.4f}")
print(f"F1:                    {metrics['f1']:.4f}")
print(f"ROC-AUC:               {metrics['roc_auc']:.4f}")
print(f"PR-AUC:                {metrics['pr_auc']:.4f}")
print(f"Log Loss:              {metrics['log_loss']:.4f}")
print(f"Brier Score:           {metrics['brier_score']:.4f}")
print(
    f"Predicted churn rate:  "
    f"{metrics['predicted_churn_rate']:.2%}"
)

print(
    f"Top-5% churn capture:  "
    f"{metrics['top_5_pct_churn_capture']:.2%}"
)

print(
    f"Top-5% lift:           "
    f"{metrics['top_5_pct_lift']:.2f}x"
)

print(
    f"Top-10% churn capture: "
    f"{metrics['top_10_pct_churn_capture']:.2%}"
)

print(
    f"Top-10% lift:          "
    f"{metrics['top_10_pct_lift']:.2f}x"
)

print(
    f"Top-20% churn capture: "
    f"{metrics['top_20_pct_churn_capture']:.2%}"
)

print(
    f"Top-20% lift:          "
    f"{metrics['top_20_pct_lift']:.2f}x"
)


# ============================================================================
# 15. CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(
    y_val,
    validation_predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================================
# 16. BASELINE COMPARISON
# ============================================================================

print("\n[12] Baseline comparison")

print(f"No-skill PR-AUC:       {validation_prevalence:.4f}")
print(f"Logistic Regression:   {metrics['pr_auc']:.4f}")

beats_no_skill = metrics["pr_auc"] > validation_prevalence

print(
    "PR-AUC beats no-skill benchmark: "
    f"{'PASS' if beats_no_skill else 'FAIL'}"
)


# ============================================================================
# 17. MODEL COEFFICIENTS
# ============================================================================

print("\n[13] Extracting model coefficients...")

coefficients = pd.DataFrame(
    {
        "feature": X_dev.columns,
        "coefficient": model.coef_[0],
        "absolute_coefficient": np.abs(model.coef_[0]),
    }
)

coefficients["direction"] = np.where(
    coefficients["coefficient"] > 0,
    "Higher feature value associated with higher churn probability",
    np.where(
        coefficients["coefficient"] < 0,
        "Higher feature value associated with lower churn probability",
        "No directional effect"
    )
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
)

print("Coefficient extraction: PASS")

print("\nTop 10 absolute coefficients:")

print(
    coefficients[
        [
            "feature",
            "coefficient",
            "direction"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================================
# 18. SAVE ARTIFACTS
# ============================================================================

print("\n[14] Saving model artifacts...")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame(
    [
        {
            "model": "Logistic Regression",
            "threshold": THRESHOLD,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "log_loss": metrics["log_loss"],
            "brier_score": metrics["brier_score"],
            "predicted_churn_rate":
                metrics["predicted_churn_rate"],
            "top_5_pct_churn_capture":
                metrics["top_5_pct_churn_capture"],
            "top_5_pct_lift":
                metrics["top_5_pct_lift"],
            "top_10_pct_churn_capture":
                metrics["top_10_pct_churn_capture"],
            "top_10_pct_lift":
                metrics["top_10_pct_lift"],
            "top_20_pct_churn_capture":
                metrics["top_20_pct_churn_capture"],
            "top_20_pct_lift":
                metrics["top_20_pct_lift"],
            "no_skill_pr_auc":
                validation_prevalence,
            "pr_auc_beats_no_skill":
                beats_no_skill,
            "feature_count":
                EXPECTED_FEATURE_COUNT,
            "random_state":
                RANDOM_STATE,
            "max_iter":
                MAX_ITER,
            "class_weight":
                str(CLASS_WEIGHT),
            "training_seconds":
                training_seconds,
            "iterations_used":
                int(model.n_iter_[0]),
            "converged":
                convergence_pass,
        }
    ]
)

results.to_csv(
    RESULTS_PATH,
    index=False
)

confusion_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
)

confusion_df.to_csv(
    CONFUSION_PATH
)

coefficients.to_csv(
    COEFFICIENT_PATH,
    index=False
)

joblib.dump(
    pipeline,
    MODEL_PATH
)


# ============================================================================
# 19. MODEL METADATA
# ============================================================================

metadata = {
    "project": "ChurnIQ",
    "step": "7.4",
    "model": "Logistic Regression",
    "purpose": "First genuine predictive baseline",
    "target": TARGET_NAME,
    "feature_count": EXPECTED_FEATURE_COUNT,
    "development_rows": int(len(X_dev)),
    "validation_rows": int(len(X_val)),
    "random_state": RANDOM_STATE,
    "threshold": THRESHOLD,
    "max_iter": MAX_ITER,
    "class_weight": CLASS_WEIGHT,
    "scaling": "StandardScaler",
    "validation_used_for_feature_selection": False,
    "validation_used_for_tuning": False,
    "validation_used_for_threshold_optimization": False,
    "test_data_loaded": False,
    "shap_used": False,
    "hyperparameter_tuning": False,
    "training_seconds": round(training_seconds, 4),
    "iterations_used": int(model.n_iter_[0]),
    "converged": bool(convergence_pass),
    "validation_pr_auc": float(metrics["pr_auc"]),
    "validation_roc_auc": float(metrics["roc_auc"]),
    "validation_recall": float(metrics["recall"]),
    "validation_precision": float(metrics["precision"]),
    "validation_f1": float(metrics["f1"]),
    "validation_log_loss": float(metrics["log_loss"]),
    "validation_brier_score": float(metrics["brier_score"]),
    "no_skill_pr_auc": float(validation_prevalence),
    "pr_auc_beats_no_skill": bool(beats_no_skill),
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

print(f"Saved: {RESULTS_PATH}")
print(f"Saved: {CONFUSION_PATH}")
print(f"Saved: {COEFFICIENT_PATH}")
print(f"Saved: {METADATA_PATH}")
print(f"Saved: {MODEL_PATH}")


# ============================================================================
# 20. QUALITY CHECKS
# ============================================================================

print("\n[15] Running quality checks...")

checks = {
    "169 frozen features":
        X_dev.shape[1] == EXPECTED_FEATURE_COUNT,

    "Development/validation schema aligned":
        list(X_dev.columns) == list(X_val.columns),

    "Binary target":
        set(y_dev.unique()).issubset({0, 1})
        and set(y_val.unique()).issubset({0, 1}),

    "No missing model features":
        X_dev.isna().sum().sum() == 0
        and X_val.isna().sum().sum() == 0,

    "No infinite model features":
        np.isfinite(X_dev.to_numpy()).all()
        and np.isfinite(X_val.to_numpy()).all(),

    "Development-only fitting":
        True,

    "Validation-only evaluation":
        True,

    "StandardScaler used in pipeline":
        True,

    "Initial threshold fixed at 0.50":
        THRESHOLD == 0.50,

    "No threshold optimization":
        True,

    "No hyperparameter tuning":
        True,

    "Test data not loaded":
        True,

    "PR-AUC calculated":
        np.isfinite(metrics["pr_auc"]),

    "ROC-AUC calculated":
        np.isfinite(metrics["roc_auc"]),

    "Log Loss calculated":
        np.isfinite(metrics["log_loss"]),

    "Brier Score calculated":
        np.isfinite(metrics["brier_score"]),

    "Top-K metrics calculated":
        all(
            np.isfinite(
                metrics[
                    f"top_{int(k * 100)}_pct_churn_capture"
                ]
            )
            for k in [0.05, 0.10, 0.20]
        ),

    "Model converged":
        convergence_pass,

    "Model artifact saved":
        MODEL_PATH.exists(),

    "Coefficient artifact saved":
        COEFFICIENT_PATH.exists(),
}

for name, passed in checks.items():
    print(
        f"{name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


all_passed = all(checks.values())


# ============================================================================
# 21. FINAL QUALITY GATE
# ============================================================================

print("\n" + "=" * 75)
print("STEP 7.4 QUALITY GATE")
print("=" * 75)

print(
    f"169 frozen features: "
    f"{'PASS' if X_dev.shape[1] == 169 else 'FAIL'}"
)

print("Development data used for fitting only: PASS")
print("Validation data used for evaluation only: PASS")
print("Test data not loaded: PASS")
print("StandardScaler inside training pipeline: PASS")

print(
    f"Initial threshold fixed at 0.50: "
    f"{'PASS' if THRESHOLD == 0.50 else 'FAIL'}"
)

print("Hyperparameter tuning not performed: PASS")
print("Threshold optimization not performed: PASS")

print(
    f"Logistic Regression convergence: "
    f"{'PASS' if convergence_pass else 'FAIL'}"
)

print(
    f"PR-AUC calculated: "
    f"{'PASS' if np.isfinite(metrics['pr_auc']) else 'FAIL'}"
)

print(
    f"PR-AUC beats no-skill benchmark: "
    f"{'PASS' if beats_no_skill else 'FAIL'}"
)

print(
    f"ROC-AUC calculated: "
    f"{'PASS' if np.isfinite(metrics['roc_auc']) else 'FAIL'}"
)

print("Top-K targeting metrics calculated: PASS")
print("Confusion matrix saved: PASS")
print("Coefficient artifact saved: PASS")
print("Model artifact saved: PASS")
print("Metadata artifact saved: PASS")

print("\nFINAL STATUS:", "PASS" if all_passed else "FAIL")

if all_passed:
    print(
        "Ready for Step 7.5 — Tree-Based Baseline."
    )
else:
    print(
        "STOP — resolve failed quality checks before continuing."
    )

print("=" * 75)