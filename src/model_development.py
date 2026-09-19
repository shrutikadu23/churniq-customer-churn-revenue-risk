"""
ChurnIQ — Model Development

Step 7.1 + 7.2
Modeling Foundation & Evaluation Protocol

Purpose:
- Validate the model-ready datasets
- Confirm the 169-feature freeze
- Validate the prediction data contract
- Confirm class imbalance
- Establish the baseline evaluation protocol
- Produce a reproducible modeling-foundation audit artifact

No predictive model is trained in this step.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
EXPECTED_FEATURE_COUNT = 169
INITIAL_THRESHOLD = 0.50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

X_DEV_PATH = DATA_DIR / "X_dev.csv"
X_VALIDATION_PATH = DATA_DIR / "X_validation.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"
Y_VALIDATION_PATH = DATA_DIR / "y_validation.csv"

AUDIT_OUTPUT_PATH = REPORTS_DIR / "07_modeling_foundation_audit.csv"

TARGET = "churn_probability"


# ============================================================
# 2. START
# ============================================================

print("=" * 75)
print("ChurnIQ — STEP 7.1 + 7.2")
print("Modeling Foundation & Evaluation Protocol")
print("=" * 75)


# ============================================================
# 3. LOAD MODEL-READY DATA
# ============================================================

print("\n[1] Loading model-ready datasets...")

X_dev = pd.read_csv(X_DEV_PATH)
X_validation = pd.read_csv(X_VALIDATION_PATH)

y_dev = pd.read_csv(Y_DEV_PATH)
y_validation = pd.read_csv(Y_VALIDATION_PATH)

print(f"Development features: {X_dev.shape}")
print(f"Validation features:  {X_validation.shape}")
print(f"Development target:   {y_dev.shape}")
print(f"Validation target:    {y_validation.shape}")


# ============================================================
# 4. STRUCTURAL VALIDATION
# ============================================================

print("\n[2] Validating dataset structure...")

assert X_dev.shape[1] == EXPECTED_FEATURE_COUNT, (
    f"Expected {EXPECTED_FEATURE_COUNT} development features, "
    f"found {X_dev.shape[1]}"
)

assert X_validation.shape[1] == EXPECTED_FEATURE_COUNT, (
    f"Expected {EXPECTED_FEATURE_COUNT} validation features, "
    f"found {X_validation.shape[1]}"
)

assert len(X_dev) == len(y_dev), (
    "Development feature/target row counts do not match."
)

assert len(X_validation) == len(y_validation), (
    "Validation feature/target row counts do not match."
)

assert list(X_dev.columns) == list(X_validation.columns), (
    "Development and validation feature schemas do not match."
)

duplicate_dev_columns = X_dev.columns[X_dev.columns.duplicated()].tolist()
duplicate_val_columns = X_validation.columns[
    X_validation.columns.duplicated()
].tolist()

assert not duplicate_dev_columns, (
    f"Duplicate development feature columns: {duplicate_dev_columns}"
)

assert not duplicate_val_columns, (
    f"Duplicate validation feature columns: {duplicate_val_columns}"
)

print(f"Feature count: {EXPECTED_FEATURE_COUNT} PASS")
print("Development row alignment PASS")
print("Validation row alignment PASS")
print("Feature schema consistency PASS")
print("Duplicate feature-name check PASS")


# ============================================================
# 5. TARGET VALIDATION
# ============================================================

print("\n[3] Validating target...")

assert list(y_dev.columns) == [TARGET], (
    f"Unexpected development target columns: {list(y_dev.columns)}"
)

assert list(y_validation.columns) == [TARGET], (
    f"Unexpected validation target columns: {list(y_validation.columns)}"
)

dev_targets = set(y_dev[TARGET].dropna().unique())
val_targets = set(y_validation[TARGET].dropna().unique())

assert dev_targets.issubset({0, 1}), (
    f"Unexpected development target values: {dev_targets}"
)

assert val_targets.issubset({0, 1}), (
    f"Unexpected validation target values: {val_targets}"
)

assert y_dev[TARGET].isna().sum() == 0
assert y_validation[TARGET].isna().sum() == 0

print("Target name: churn_probability PASS")
print("Binary target validation PASS")
print("Target missing-value check PASS")


# ============================================================
# 6. FEATURE VALUE VALIDATION
# ============================================================

print("\n[4] Validating model feature values...")

non_numeric_dev = X_dev.select_dtypes(exclude=np.number).columns.tolist()
non_numeric_val = X_validation.select_dtypes(exclude=np.number).columns.tolist()

assert not non_numeric_dev, (
    f"Non-numeric development features found: {non_numeric_dev}"
)

assert not non_numeric_val, (
    f"Non-numeric validation features found: {non_numeric_val}"
)

dev_missing = int(X_dev.isna().sum().sum())
val_missing = int(X_validation.isna().sum().sum())

assert dev_missing == 0, (
    f"Development dataset contains {dev_missing:,} missing values."
)

assert val_missing == 0, (
    f"Validation dataset contains {val_missing:,} missing values."
)

dev_infinite = int(np.isinf(X_dev.to_numpy(dtype=float)).sum())
val_infinite = int(np.isinf(X_validation.to_numpy(dtype=float)).sum())

assert dev_infinite == 0, (
    f"Development dataset contains {dev_infinite:,} infinite values."
)

assert val_infinite == 0, (
    f"Validation dataset contains {val_infinite:,} infinite values."
)

print("Numeric feature validation PASS")
print("Development missing-value check PASS")
print("Validation missing-value check PASS")
print("Development infinite-value check PASS")
print("Validation infinite-value check PASS")


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n[5] Target distribution")

dev_counts = y_dev[TARGET].value_counts().sort_index()
val_counts = y_validation[TARGET].value_counts().sort_index()

dev_total = len(y_dev)
val_total = len(y_validation)

dev_retained = int(dev_counts.get(0, 0))
dev_churned = int(dev_counts.get(1, 0))

val_retained = int(val_counts.get(0, 0))
val_churned = int(val_counts.get(1, 0))

dev_churn_rate = dev_churned / dev_total
val_churn_rate = val_churned / val_total

print("\nDevelopment:")
print(
    f"  Retained: {dev_retained:,} "
    f"({dev_retained / dev_total:.2%})"
)
print(
    f"  Churned:  {dev_churned:,} "
    f"({dev_churn_rate:.2%})"
)

print("\nValidation:")
print(
    f"  Retained: {val_retained:,} "
    f"({val_retained / val_total:.2%})"
)
print(
    f"  Churned:  {val_churned:,} "
    f"({val_churn_rate:.2%})"
)


# ============================================================
# 8. CLASS IMBALANCE
# ============================================================

print("\n[6] Class imbalance assessment")

majority_count = max(dev_retained, dev_churned)
minority_count = min(dev_retained, dev_churned)

class_ratio = majority_count / minority_count

rate_difference = abs(dev_churn_rate - val_churn_rate)

print(f"Development churn rate: {dev_churn_rate:.4%}")
print(f"Validation churn rate:  {val_churn_rate:.4%}")
print(f"Absolute rate difference: {rate_difference:.4%}")
print(f"Development majority/minority ratio: {class_ratio:.2f}:1")

assert rate_difference < 0.02, (
    "Development and validation churn rates differ by more than "
    "2 percentage points."
)

print("Stratified target-rate preservation PASS")


# ============================================================
# 9. EVALUATION PROTOCOL
# ============================================================

print("\n[7] Evaluation protocol")

primary_metrics = [
    "PR-AUC",
]

secondary_metrics = [
    "ROC-AUC",
]

classification_metrics = [
    "Recall",
    "Precision",
    "F1",
    "Confusion Matrix",
]

targeting_metrics = [
    "Top-5% Churn Capture",
    "Top-10% Churn Capture",
    "Top-20% Churn Capture",
    "Lift",
    "Gains",
]

print("\nPrimary discrimination metric:")
for metric in primary_metrics:
    print(f"  - {metric}")

print("\nSecondary discrimination metric:")
for metric in secondary_metrics:
    print(f"  - {metric}")

print("\nClassification metrics:")
for metric in classification_metrics:
    print(f"  - {metric}")

print("\nCustomer-targeting metrics:")
for metric in targeting_metrics:
    print(f"  - {metric}")

print(
    f"\nInitial classification threshold: "
    f"{INITIAL_THRESHOLD:.2f}"
)


# ============================================================
# 10. MODELING CONTROLS
# ============================================================

print("\n[8] Modeling controls")

controls = {
    "Feature freeze": f"{EXPECTED_FEATURE_COUNT} features",
    "Random seed": RANDOM_STATE,
    "Validation used for feature selection": False,
    "Validation used for threshold optimization": False,
    "Validation used for hyperparameter tuning": False,
    "Test data loaded": False,
    "Threshold optimization": False,
    "Hyperparameter tuning": False,
    "SHAP explainability": False,
    "Future-period information": False,
}

for control, value in controls.items():
    print(f"  {control}: {value}")


# ============================================================
# 11. FOUNDATION AUDIT ARTIFACT
# ============================================================

print("\n[9] Saving modeling-foundation audit artifact...")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

audit_rows = [
    ["feature_count", EXPECTED_FEATURE_COUNT],
    ["development_rows", len(X_dev)],
    ["validation_rows", len(X_validation)],
    ["development_churn_rate", dev_churn_rate],
    ["validation_churn_rate", val_churn_rate],
    ["class_ratio_majority_to_minority", class_ratio],
    ["initial_threshold", INITIAL_THRESHOLD],
    ["random_state", RANDOM_STATE],
    ["development_missing_values", dev_missing],
    ["validation_missing_values", val_missing],
    ["development_infinite_values", dev_infinite],
    ["validation_infinite_values", val_infinite],
    ["validation_used_for_feature_selection", False],
    ["validation_used_for_threshold_optimization", False],
    ["test_data_loaded", False],
]

audit_df = pd.DataFrame(
    audit_rows,
    columns=["check", "value"]
)

audit_df.to_csv(
    AUDIT_OUTPUT_PATH,
    index=False
)

print(f"Saved: {AUDIT_OUTPUT_PATH}")


# ============================================================
# 12. FINAL QUALITY GATE
# ============================================================

print("\n" + "=" * 75)
print("STEP 7.1 + 7.2 QUALITY GATE")
print("=" * 75)

checks = {
    "169 frozen features":
        X_dev.shape[1] == EXPECTED_FEATURE_COUNT
        and X_validation.shape[1] == EXPECTED_FEATURE_COUNT,

    "Development rows aligned":
        len(X_dev) == len(y_dev),

    "Validation rows aligned":
        len(X_validation) == len(y_validation),

    "Feature schemas aligned":
        list(X_dev.columns) == list(X_validation.columns),

    "No duplicate feature names":
        not duplicate_dev_columns
        and not duplicate_val_columns,

    "Binary target":
        dev_targets.issubset({0, 1})
        and val_targets.issubset({0, 1}),

    "No target missing values":
        y_dev[TARGET].isna().sum() == 0
        and y_validation[TARGET].isna().sum() == 0,

    "No model-ready missing values":
        dev_missing == 0
        and val_missing == 0,

    "Numeric features only":
        not non_numeric_dev
        and not non_numeric_val,

    "No infinite feature values":
        dev_infinite == 0
        and val_infinite == 0,

    "Target-rate preservation":
        rate_difference < 0.02,

    "Test data not loaded":
        True,

    "Threshold optimization not performed":
        True,

    "Hyperparameter tuning not performed":
        True,
}

all_pass = True

for check, result in checks.items():
    status = "PASS" if result else "FAIL"
    print(f"{check}: {status}")

    if not result:
        all_pass = False


# ============================================================
# 13. FINAL STATUS
# ============================================================

if not all_pass:
    raise RuntimeError(
        "\nStep 7.1 + 7.2 quality gate FAILED.\n"
        "Resolve the failed checks before proceeding."
    )

print("\nFINAL STATUS: PASS")
print("Ready for Step 7.3 — Dummy Classifier Baseline.")
print("=" * 75)