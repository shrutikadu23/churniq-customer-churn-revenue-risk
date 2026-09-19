"""
ChurnIQ — Step 7.5
Tree-Based Baseline

Purpose
-------
Establish a controlled nonlinear tree-based benchmark against
the Logistic Regression baseline.

Model
-----
HistGradientBoostingClassifier

Experimental controls
----------------------
- 169 frozen model features
- Development data used for fitting only
- Validation data used for evaluation only
- Test data not loaded
- No hyperparameter tuning
- No threshold optimization
- Initial classification threshold fixed at 0.50
- No class weighting
- No resampling / SMOTE
- No SHAP
- No calibration
- No validation-driven feature selection

Evaluation
----------
Primary:
    PR-AUC

Secondary:
    ROC-AUC

Classification:
    Accuracy, Precision, Recall, F1, Confusion Matrix

Probability quality:
    Log Loss, Brier Score

Business targeting:
    Top-5%, Top-10%, Top-20% churn capture
    Top-5%, Top-10%, Top-20% lift
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
THRESHOLD = 0.50
EXPECTED_FEATURE_COUNT = 169

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Input files
X_DEV_PATH = DATA_DIR / "X_dev.csv"
X_VALIDATION_PATH = DATA_DIR / "X_validation.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"
Y_VALIDATION_PATH = DATA_DIR / "y_validation.csv"


# Output files
RESULTS_PATH = REPORT_DIR / "07_5_tree_based_baseline_results.csv"
CM_PATH = REPORT_DIR / "07_5_tree_based_baseline_confusion_matrix.csv"
METADATA_PATH = REPORT_DIR / "07_5_tree_based_baseline_metadata.json"

MODEL_PATH = MODEL_DIR / "tree_based_baseline.joblib"


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def load_target(path: Path) -> pd.Series:
    """
    Load a one-column target CSV while preserving the target column name.
    """
    df = pd.read_csv(path)

    if df.shape[1] != 1:
        raise ValueError(
            f"Expected one target column in {path.name}; "
            f"found {df.shape[1]} columns."
        )

    return df.iloc[:, 0]


def calculate_top_k_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    k: float,
) -> tuple[float, float]:
    """
    Calculate:
        - churn capture
        - lift

    among the top k% customers ranked by predicted churn probability.

    Ties are resolved deterministically using original row order.
    """

    n = len(y_true)

    if n == 0:
        raise ValueError("Cannot calculate Top-K metrics on empty data.")

    if len(probabilities) != n:
        raise ValueError(
            "Probability length does not match target length."
        )

    k_count = max(1, int(np.ceil(n * k)))

    ranking = pd.DataFrame(
        {
            "probability": probabilities,
            "actual": np.asarray(y_true),
            "row_order": np.arange(n),
        }
    )

    ranking = ranking.sort_values(
        by=["probability", "row_order"],
        ascending=[False, True],
        kind="mergesort",
    )

    top_k = ranking.head(k_count)

    total_churners = int(ranking["actual"].sum())
    captured_churners = int(top_k["actual"].sum())

    if total_churners == 0:
        return np.nan, np.nan

    churn_capture = captured_churners / total_churners

    selected_churn_rate = captured_churners / k_count
    overall_churn_rate = total_churners / n

    lift = (
        selected_churn_rate / overall_churn_rate
        if overall_churn_rate > 0
        else np.nan
    )

    return churn_capture, lift


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict, np.ndarray]:

    """
    Evaluate validation performance using predicted probabilities
    and an explicitly controlled classification threshold.
    """

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Target and probability lengths do not match."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite probability values detected."
        )

    if probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError(
            "Predicted probabilities fall outside [0, 1]."
        )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1],
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    predicted_churn_rate = predictions.mean()

    top5_capture, top5_lift = calculate_top_k_metrics(
        y_true,
        probabilities,
        0.05,
    )

    top10_capture, top10_lift = calculate_top_k_metrics(
        y_true,
        probabilities,
        0.10,
    )

    top20_capture, top20_lift = calculate_top_k_metrics(
        y_true,
        probabilities,
        0.20,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    metrics = {
        "model": model_name,
        "threshold": threshold,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "log_loss": logloss,
        "brier_score": brier,

        "predicted_churn_rate": predicted_churn_rate,

        "top_5_churn_capture": top5_capture,
        "top_5_lift": top5_lift,

        "top_10_churn_capture": top10_capture,
        "top_10_lift": top10_lift,

        "top_20_churn_capture": top20_capture,
        "top_20_lift": top20_lift,
    }

    return metrics, cm


# ============================================================================
# 3. START
# ============================================================================

print_header("ChurnIQ — STEP 7.5")
print("Tree-Based Baseline")
print("=" * 75)


# ============================================================================
# 4. LOAD MODEL-READY DATA
# ============================================================================

print("\n[1] Loading model-ready datasets...")

X_dev = pd.read_csv(X_DEV_PATH)
X_validation = pd.read_csv(X_VALIDATION_PATH)

y_dev = load_target(Y_DEV_PATH)
y_validation = load_target(Y_VALIDATION_PATH)

print(f"Development features: {X_dev.shape}")
print(f"Validation features:  {X_validation.shape}")
print(f"Development target:   {y_dev.shape}")
print(f"Validation target:    {y_validation.shape}")


# ============================================================================
# 5. DATASET STRUCTURE VALIDATION
# ============================================================================

print("\n[2] Validating dataset structure...")

if X_dev.shape[1] != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} development features; "
        f"found {X_dev.shape[1]}."
    )

if X_validation.shape[1] != EXPECTED_FEATURE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_FEATURE_COUNT} validation features; "
        f"found {X_validation.shape[1]}."
    )

print(f"Feature count: {EXPECTED_FEATURE_COUNT} PASS")


if len(X_dev) != len(y_dev):
    raise ValueError(
        "Development feature/target row count mismatch."
    )

if len(X_validation) != len(y_validation):
    raise ValueError(
        "Validation feature/target row count mismatch."
    )

print("Development row alignment PASS")
print("Validation row alignment PASS")


if list(X_dev.columns) != list(X_validation.columns):
    raise ValueError(
        "Development and validation feature schemas differ."
    )

print("Feature schema consistency PASS")


if X_dev.columns.duplicated().any():
    raise ValueError(
        "Duplicate development feature names detected."
    )

if X_validation.columns.duplicated().any():
    raise ValueError(
        "Duplicate validation feature names detected."
    )

print("Duplicate feature-name check PASS")


# ============================================================================
# 6. TARGET VALIDATION
# ============================================================================

print("\n[3] Validating target...")

if y_dev.name != "churn_probability":
    raise ValueError(
        f"Unexpected development target name: {y_dev.name}"
    )

if y_validation.name != "churn_probability":
    raise ValueError(
        f"Unexpected validation target name: {y_validation.name}"
    )

print("Target name: churn_probability PASS")


if not set(y_dev.unique()).issubset({0, 1}):
    raise ValueError(
        "Development target is not binary."
    )

if not set(y_validation.unique()).issubset({0, 1}):
    raise ValueError(
        "Validation target is not binary."
    )

print("Binary target validation PASS")


if y_dev.isna().any() or y_validation.isna().any():
    raise ValueError(
        "Missing target values detected."
    )

print("Target missing-value check PASS")


# ============================================================================
# 7. MODEL FEATURE VALIDATION
# ============================================================================

print("\n[4] Validating model feature values...")

if not all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in X_dev.dtypes
):
    raise ValueError(
        "Non-numeric development feature detected."
    )

if not all(
    pd.api.types.is_numeric_dtype(dtype)
    for dtype in X_validation.dtypes
):
    raise ValueError(
        "Non-numeric validation feature detected."
    )

print("Numeric feature validation PASS")


if X_dev.isna().any().any():
    raise ValueError(
        "Missing values found in development features."
    )

if X_validation.isna().any().any():
    raise ValueError(
        "Missing values found in validation features."
    )

print("Development missing-value check PASS")
print("Validation missing-value check PASS")


dev_array = X_dev.to_numpy(dtype=float)
val_array = X_validation.to_numpy(dtype=float)

if not np.isfinite(dev_array).all():
    raise ValueError(
        "Infinite values found in development features."
    )

if not np.isfinite(val_array).all():
    raise ValueError(
        "Infinite values found in validation features."
    )

print("Development infinite-value check PASS")
print("Validation infinite-value check PASS")


# ============================================================================
# 8. TARGET DISTRIBUTION
# ============================================================================

print("\n[5] Target distribution")

dev_counts = y_dev.value_counts().sort_index()
val_counts = y_validation.value_counts().sort_index()

dev_retained = int(dev_counts.get(0, 0))
dev_churned = int(dev_counts.get(1, 0))

val_retained = int(val_counts.get(0, 0))
val_churned = int(val_counts.get(1, 0))

dev_rate = float(y_dev.mean())
val_rate = float(y_validation.mean())


print("\nDevelopment:")
print(
    f"  Retained: {dev_retained:,} "
    f"({(1 - dev_rate) * 100:.2f}%)"
)
print(
    f"  Churned:  {dev_churned:,} "
    f"({dev_rate * 100:.2f}%)"
)


print("\nValidation:")
print(
    f"  Retained: {val_retained:,} "
    f"({(1 - val_rate) * 100:.2f}%)"
)
print(
    f"  Churned:  {val_churned:,} "
    f"({val_rate * 100:.2f}%)"
)


# ============================================================================
# 9. CLASS IMBALANCE
# ============================================================================

print("\n[6] Class imbalance assessment")

if dev_churned == 0:
    raise ValueError(
        "Development set contains no churned customers."
    )

rate_difference = abs(dev_rate - val_rate)

majority_minority_ratio = (
    dev_retained / dev_churned
)

print(
    f"Development churn rate: "
    f"{dev_rate * 100:.4f}%"
)

print(
    f"Validation churn rate:  "
    f"{val_rate * 100:.4f}%"
)

print(
    f"Absolute rate difference: "
    f"{rate_difference * 100:.4f}%"
)

print(
    f"Development majority/minority ratio: "
    f"{majority_minority_ratio:.2f}:1"
)


if rate_difference > 0.02:
    raise ValueError(
        "Development/validation churn-rate difference "
        "exceeds 2 percentage points."
    )

print("Stratified target-rate preservation PASS")


# ============================================================================
# 10. NO-SKILL BENCHMARK
# ============================================================================

print("\n[7] No-skill benchmark")

no_skill_pr_auc = val_rate

print(
    f"Validation churn prevalence / no-skill PR-AUC: "
    f"{no_skill_pr_auc:.6f}"
)

print(
    "This represents the approximate PR-AUC expected "
    "from a non-informative classifier."
)


# ============================================================================
# 11. BUILD TREE-BASED BASELINE
# ============================================================================

print("\n[8] Building tree-based baseline...")

model = HistGradientBoostingClassifier(
    learning_rate=0.10,
    max_iter=100,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=RANDOM_STATE,
)


print("Model: HistGradientBoostingClassifier")
print("Model type: nonlinear tree-based ensemble")
print("Baseline configuration:")
print("  learning_rate:     0.10")
print("  max_iter:          100")
print("  max_leaf_nodes:    31")
print("  min_samples_leaf:  20")
print("  l2_regularization: 0.0")
print(f"  random_state:      {RANDOM_STATE}")
print("Class weighting:     None")
print("Hyperparameter tuning: False")


# ============================================================================
# 12. TRAINING
# ============================================================================

print("\n[9] Training tree-based baseline...")

start_time = time.perf_counter()

model.fit(
    X_dev,
    y_dev,
)

training_time = time.perf_counter() - start_time

print("Development-only fitting: PASS")
print(f"Training time: {training_time:.2f} seconds")


# ============================================================================
# 13. VALIDATION PROBABILITIES
# ============================================================================

print("\n[10] Generating validation probabilities...")

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


if len(validation_probabilities) != len(X_validation):
    raise ValueError(
        "Validation probability count does not match "
        "validation row count."
    )

if not np.isfinite(validation_probabilities).all():
    raise ValueError(
        "Invalid probability values detected."
    )

if (
    validation_probabilities.min() < 0
    or validation_probabilities.max() > 1
):
    raise ValueError(
        "Predicted probabilities outside [0, 1]."
    )

print("Validation probability generation: PASS")
print("Probability count validation: PASS")
print("Probability range validation: PASS")


# ============================================================================
# 14. VALIDATION EVALUATION
# ============================================================================

print("\n[11] Evaluating validation performance...")

metrics, cm = evaluate_model(
    model_name="HistGradientBoosting",
    y_true=y_validation,
    probabilities=validation_probabilities,
    threshold=THRESHOLD,
)


print("\nHistGradientBoosting")
print("-" * 50)

print(
    f"{'Accuracy:':<24} "
    f"{metrics['accuracy']:.4f}"
)

print(
    f"{'Precision:':<24} "
    f"{metrics['precision']:.4f}"
)

print(
    f"{'Recall:':<24} "
    f"{metrics['recall']:.4f}"
)

print(
    f"{'F1:':<24} "
    f"{metrics['f1']:.4f}"
)

print(
    f"{'ROC-AUC:':<24} "
    f"{metrics['roc_auc']:.4f}"
)

print(
    f"{'PR-AUC:':<24} "
    f"{metrics['pr_auc']:.4f}"
)

print(
    f"{'Log Loss:':<24} "
    f"{metrics['log_loss']:.4f}"
)

print(
    f"{'Brier Score:':<24} "
    f"{metrics['brier_score']:.4f}"
)

print(
    f"{'Predicted churn rate:':<24} "
    f"{metrics['predicted_churn_rate'] * 100:.2f}%"
)

print(
    f"{'Top-5% churn capture:':<24} "
    f"{metrics['top_5_churn_capture'] * 100:.2f}%"
)

print(
    f"{'Top-5% lift:':<24} "
    f"{metrics['top_5_lift']:.2f}x"
)

print(
    f"{'Top-10% churn capture:':<24} "
    f"{metrics['top_10_churn_capture'] * 100:.2f}%"
)

print(
    f"{'Top-10% lift:':<24} "
    f"{metrics['top_10_lift']:.2f}x"
)

print(
    f"{'Top-20% churn capture:':<24} "
    f"{metrics['top_20_churn_capture'] * 100:.2f}%"
)

print(
    f"{'Top-20% lift:':<24} "
    f"{metrics['top_20_lift']:.2f}x"
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================================
# 15. NO-SKILL COMPARISON
# ============================================================================

print("\n[12] Baseline comparison")

print(
    f"No-skill PR-AUC:       "
    f"{no_skill_pr_auc:.4f}"
)

print(
    f"Tree-based PR-AUC:     "
    f"{metrics['pr_auc']:.4f}"
)

beats_no_skill = (
    metrics["pr_auc"] > no_skill_pr_auc
)

print(
    "PR-AUC beats no-skill benchmark: "
    + ("PASS" if beats_no_skill else "FAIL")
)

if not beats_no_skill:
    raise ValueError(
        "Tree-based baseline did not beat the "
        "no-skill PR-AUC benchmark."
    )


# ============================================================================
# 16. SAVE RESULTS
# ============================================================================

print("\n[13] Saving model artifacts...")

results_df = pd.DataFrame([metrics])

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


cm_df = pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
)

cm_df.to_csv(
    CM_PATH
)


metadata = {
    "project": "ChurnIQ",
    "step": "7.5",

    "model_name": "HistGradientBoostingClassifier",
    "model_type": "nonlinear_tree_based_ensemble",
    "purpose": "Controlled tree-based baseline",

    "random_state": RANDOM_STATE,
    "threshold": THRESHOLD,

    "feature_count": EXPECTED_FEATURE_COUNT,
    "feature_names": X_dev.columns.tolist(),

    "development_rows": int(len(X_dev)),
    "validation_rows": int(len(X_validation)),

    "development_churn_rate": dev_rate,
    "validation_churn_rate": val_rate,

    "no_skill_pr_auc": float(
        no_skill_pr_auc
    ),

    "training_time_seconds": float(
        training_time
    ),

    "hyperparameter_tuning": False,
    "threshold_optimization": False,
    "class_weighting": False,
    "resampling": False,
    "validation_used_for_training": False,
    "validation_used_for_feature_selection": False,
    "test_data_loaded": False,
    "shap_used": False,
    "calibration_performed": False,

    "model_parameters": {
        "learning_rate": 0.10,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "random_state": RANDOM_STATE,
    },

    "evaluation": metrics,
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )


joblib.dump(
    model,
    MODEL_PATH,
)


print(f"Saved: {RESULTS_PATH}")
print(f"Saved: {CM_PATH}")
print(f"Saved: {METADATA_PATH}")
print(f"Saved: {MODEL_PATH}")


# ============================================================================
# 17. QUALITY CHECKS
# ============================================================================

print("\n[14] Running quality checks...")

checks = {
    "169 frozen features":
        X_dev.shape[1] == EXPECTED_FEATURE_COUNT,

    "Development/validation alignment":
        (
            len(X_dev) == len(y_dev)
            and
            len(X_validation) == len(y_validation)
        ),

    "Feature schemas aligned":
        list(X_dev.columns)
        == list(X_validation.columns),

    "No duplicate feature names":
        (
            not X_dev.columns.duplicated().any()
            and
            not X_validation.columns.duplicated().any()
        ),

    "Binary target":
        (
            set(y_dev.unique()).issubset({0, 1})
            and
            set(y_validation.unique()).issubset({0, 1})
        ),

    "No target missing values":
        (
            not y_dev.isna().any()
            and
            not y_validation.isna().any()
        ),

    "No missing model features":
        (
            not X_dev.isna().any().any()
            and
            not X_validation.isna().any().any()
        ),

    "No infinite model features":
        (
            np.isfinite(dev_array).all()
            and
            np.isfinite(val_array).all()
        ),

    "Probability count matches validation":
        len(validation_probabilities)
        == len(y_validation),

    "Probability range valid":
        (
            validation_probabilities.min() >= 0
            and
            validation_probabilities.max() <= 1
        ),

    "Development-only fitting": True,

    "Validation-only evaluation": True,

    "Initial threshold fixed at 0.50":
        THRESHOLD == 0.50,

    "No threshold optimization": True,

    "No hyperparameter tuning": True,

    "No class weighting": True,

    "No resampling": True,

    "Test data not loaded": True,

    "PR-AUC calculated":
        np.isfinite(metrics["pr_auc"]),

    "PR-AUC beats no-skill benchmark":
        beats_no_skill,

    "ROC-AUC calculated":
        np.isfinite(metrics["roc_auc"]),

    "Log Loss calculated":
        np.isfinite(metrics["log_loss"]),

    "Brier Score calculated":
        np.isfinite(metrics["brier_score"]),

    "Top-K metrics calculated":
        all(
            np.isfinite(metrics[key])
            for key in [
                "top_5_churn_capture",
                "top_5_lift",
                "top_10_churn_capture",
                "top_10_lift",
                "top_20_churn_capture",
                "top_20_lift",
            ]
        ),

    "Results artifact saved":
        RESULTS_PATH.exists(),

    "Confusion matrix saved":
        CM_PATH.exists(),

    "Model artifact saved":
        MODEL_PATH.exists(),

    "Metadata artifact saved":
        METADATA_PATH.exists(),
}


for check_name, passed in checks.items():
    print(
        f"{check_name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


if not all(checks.values()):
    raise RuntimeError(
        "One or more Step 7.5 quality checks failed."
    )


# ============================================================================
# 18. FINAL QUALITY GATE
# ============================================================================

print_header("STEP 7.5 QUALITY GATE")


final_checks = [
    (
        "169 frozen features",
        True,
    ),
    (
        "Development data used for fitting only",
        True,
    ),
    (
        "Validation data used for evaluation only",
        True,
    ),
    (
        "Test data not loaded",
        True,
    ),
    (
        "Tree-based baseline completed",
        True,
    ),
    (
        "No-skill PR-AUC benchmark established",
        True,
    ),
    (
        "PR-AUC beats no-skill benchmark",
        beats_no_skill,
    ),
    (
        "ROC-AUC calculated",
        True,
    ),
    (
        "Classification metrics calculated",
        True,
    ),
    (
        "Probability metrics calculated",
        True,
    ),
    (
        "Top-K targeting metrics calculated",
        True,
    ),
    (
        "Confusion matrix saved",
        CM_PATH.exists(),
    ),
    (
        "Model artifact saved",
        MODEL_PATH.exists(),
    ),
    (
        "Metadata artifact saved",
        METADATA_PATH.exists(),
    ),
    (
        "Initial threshold fixed at 0.50",
        THRESHOLD == 0.50,
    ),
    (
        "Threshold optimization not performed",
        True,
    ),
    (
        "Hyperparameter tuning not performed",
        True,
    ),
    (
        "Class weighting not performed",
        True,
    ),
    (
        "Resampling not performed",
        True,
    ),
]


for name, passed in final_checks:
    print(
        f"{name}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


if all(
    passed
    for _, passed in final_checks
):
    print("\nFINAL STATUS: PASS")
    print(
        "Ready for Step 7.6 — XGBoost Baseline."
    )
else:
    print("\nFINAL STATUS: FAIL")
    raise RuntimeError(
        "Step 7.5 quality gate failed."
    )


print("=" * 75)