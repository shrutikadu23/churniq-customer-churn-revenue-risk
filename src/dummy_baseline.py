"""
ChurnIQ — Step 7.3
Dummy Classifier Baseline

Purpose:
Establish a naive performance benchmark before training
real predictive models.

Baselines:
1. Most Frequent Class
2. Prior Probability

Evaluation:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Predicted Churn Rate
- Top-5%, Top-10%, Top-20% Churn Capture
- Lift
- Gains

Validation discipline:
- Development data used for fitting only
- Validation data used for evaluation only
- Test data is NOT loaded
- Threshold optimization is NOT performed
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

X_DEV_PATH = DATA_DIR / "X_dev.csv"
X_VAL_PATH = DATA_DIR / "X_validation.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"
Y_VAL_PATH = DATA_DIR / "y_validation.csv"

RESULTS_PATH = REPORTS_DIR / "07_3_dummy_baseline_results.csv"
CONFUSION_PATH = REPORTS_DIR / "07_3_dummy_baseline_confusion_matrix.csv"

TARGET = "churn_probability"
THRESHOLD = 0.50
RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 75)
print("ChurnIQ — STEP 7.3")
print("Dummy Classifier Baseline")
print("=" * 75)

print("\n[1] Loading model-ready datasets...")

X_dev = pd.read_csv(X_DEV_PATH)
X_val = pd.read_csv(X_VAL_PATH)

y_dev = pd.read_csv(Y_DEV_PATH).squeeze("columns")
y_val = pd.read_csv(Y_VAL_PATH).squeeze("columns")

print(f"Development features: {X_dev.shape}")
print(f"Validation features:  {X_val.shape}")
print(f"Development target:   {y_dev.shape}")
print(f"Validation target:    {y_val.shape}")


# ============================================================
# 3. STRUCTURAL VALIDATION
# ============================================================

print("\n[2] Validating dataset structure...")

assert X_dev.shape[1] == 169
assert X_val.shape[1] == 169

assert list(X_dev.columns) == list(X_val.columns)

assert X_dev.index.equals(y_dev.index)
assert X_val.index.equals(y_val.index)

assert X_dev.columns.is_unique
assert X_val.columns.is_unique

print("Feature count: 169 PASS")
print("Development row alignment PASS")
print("Validation row alignment PASS")
print("Feature schema consistency PASS")
print("Duplicate feature-name check PASS")


# ============================================================
# 4. TARGET VALIDATION
# ============================================================

print("\n[3] Validating target...")

assert TARGET == "churn_probability"
assert y_dev.isna().sum() == 0
assert y_val.isna().sum() == 0

assert set(y_dev.unique()).issubset({0, 1})
assert set(y_val.unique()).issubset({0, 1})

print(f"Target name: {TARGET} PASS")
print("Binary target validation PASS")
print("Target missing-value check PASS")


# ============================================================
# 5. FEATURE VALUE VALIDATION
# ============================================================

print("\n[4] Validating model feature values...")

assert all(pd.api.types.is_numeric_dtype(X_dev[col]) for col in X_dev.columns)
assert all(pd.api.types.is_numeric_dtype(X_val[col]) for col in X_val.columns)

assert not X_dev.isna().any().any()
assert not X_val.isna().any().any()

assert np.isfinite(X_dev.to_numpy(dtype=float)).all()
assert np.isfinite(X_val.to_numpy(dtype=float)).all()

print("Numeric feature validation PASS")
print("Development missing-value check PASS")
print("Validation missing-value check PASS")
print("Development infinite-value check PASS")
print("Validation infinite-value check PASS")


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n[5] Target distribution")

dev_retained = int((y_dev == 0).sum())
dev_churned = int((y_dev == 1).sum())

val_retained = int((y_val == 0).sum())
val_churned = int((y_val == 1).sum())

dev_churn_rate = y_dev.mean()
val_churn_rate = y_val.mean()

print("\nDevelopment:")
print(f"  Retained: {dev_retained:,} ({(1 - dev_churn_rate) * 100:.2f}%)")
print(f"  Churned:  {dev_churned:,} ({dev_churn_rate * 100:.2f}%)")

print("\nValidation:")
print(f"  Retained: {val_retained:,} ({(1 - val_churn_rate) * 100:.2f}%)")
print(f"  Churned:  {val_churned:,} ({val_churn_rate * 100:.2f}%)")


# ============================================================
# 7. CLASS IMBALANCE CHECK
# ============================================================

print("\n[6] Class imbalance assessment")

rate_difference = abs(dev_churn_rate - val_churn_rate)

print(f"Development churn rate: {dev_churn_rate * 100:.4f}%")
print(f"Validation churn rate:  {val_churn_rate * 100:.4f}%")
print(f"Absolute rate difference: {rate_difference * 100:.4f}%")

print(
    f"Development majority/minority ratio: "
    f"{dev_retained / dev_churned:.2f}:1"
)

assert rate_difference < 0.01

print("Stratified target-rate preservation PASS")


# ============================================================
# 8. NO-SKILL PR-AUC BENCHMARK
# ============================================================

print("\n[7] No-skill benchmark")

no_skill_pr_auc = val_churn_rate

print(
    f"Validation churn prevalence / no-skill PR-AUC: "
    f"{no_skill_pr_auc:.6f}"
)

print(
    "This represents the approximate PR-AUC expected from a "
    "non-informative classifier."
)


# ============================================================
# 9. BASELINE MODELS
# ============================================================

print("\n[8] Training dummy baselines...")

models = {
    "Most Frequent": DummyClassifier(
        strategy="most_frequent"
    ),
    "Prior Probability": DummyClassifier(
        strategy="prior"
    ),
}

results = []
confusion_results = []


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model):

    model.fit(X_dev, y_dev)

    # Explicit probability-based threshold.
    # This keeps the evaluation framework consistent
    # with later predictive models.
    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (probabilities >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    predicted_churn_rate = predictions.mean()

    # --------------------------------------------------------
    # Top-K targeting metrics
    # --------------------------------------------------------

    ranking = pd.DataFrame(
        {
            "actual": y_val.to_numpy(),
            "probability": probabilities,
        }
    ).sort_values(
        "probability",
        ascending=False
    ).reset_index(drop=True)

    total_churners = ranking["actual"].sum()

    top_k_metrics = {}

    for pct in [0.05, 0.10, 0.20]:

        n_customers = max(
            1,
            int(np.ceil(len(ranking) * pct))
        )

        top_k = ranking.head(n_customers)

        captured_churners = top_k["actual"].sum()

        capture_rate = (
            captured_churners / total_churners
            if total_churners > 0
            else 0
        )

        lift = (
            capture_rate / pct
            if pct > 0
            else np.nan
        )

        top_k_metrics[f"top_{int(pct * 100)}_capture"] = capture_rate
        top_k_metrics[f"top_{int(pct * 100)}_lift"] = lift

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Store model results
    # --------------------------------------------------------

    results.append(
        {
            "model": model_name,
            "threshold": THRESHOLD,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "no_skill_pr_auc": no_skill_pr_auc,
            "predicted_churn_rate": predicted_churn_rate,
            "top_5_churn_capture": top_k_metrics["top_5_capture"],
            "top_5_lift": top_k_metrics["top_5_lift"],
            "top_10_churn_capture": top_k_metrics["top_10_capture"],
            "top_10_lift": top_k_metrics["top_10_lift"],
            "top_20_churn_capture": top_k_metrics["top_20_capture"],
            "top_20_lift": top_k_metrics["top_20_lift"],
        }
    )

    confusion_results.append(
        {
            "model": model_name,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        }
    )

    print(f"\n{model_name}")
    print("-" * 50)
    print(f"Accuracy:              {accuracy:.4f}")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1:                    {f1:.4f}")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"PR-AUC:                {pr_auc:.4f}")
    print(f"Predicted churn rate:  {predicted_churn_rate * 100:.2f}%")
    print(f"Top-5% churn capture:  {top_k_metrics['top_5_capture'] * 100:.2f}%")
    print(f"Top-5% lift:           {top_k_metrics['top_5_lift']:.2f}x")
    print(f"Top-10% churn capture: {top_k_metrics['top_10_capture'] * 100:.2f}%")
    print(f"Top-10% lift:          {top_k_metrics['top_10_lift']:.2f}x")
    print(f"Top-20% churn capture: {top_k_metrics['top_20_capture'] * 100:.2f}%")
    print(f"Top-20% lift:          {top_k_metrics['top_20_lift']:.2f}x")

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# 11. RUN BASELINES
# ============================================================

for model_name, model in models.items():
    evaluate_model(model_name, model)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\n[9] Saving baseline artifacts...")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
confusion_df = pd.DataFrame(confusion_results)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

confusion_df.to_csv(
    CONFUSION_PATH,
    index=False
)

print(f"Saved: {RESULTS_PATH}")
print(f"Saved: {CONFUSION_PATH}")


# ============================================================
# 13. BASELINE QUALITY CHECKS
# ============================================================

print("\n[10] Running quality checks...")

assert len(results_df) == 2
assert len(confusion_df) == 2

assert results_df["pr_auc"].notna().all()
assert results_df["roc_auc"].notna().all()

assert (results_df["pr_auc"] >= 0).all()
assert (results_df["pr_auc"] <= 1).all()

assert (results_df["roc_auc"] >= 0).all()
assert (results_df["roc_auc"] <= 1).all()

assert results_df["threshold"].eq(THRESHOLD).all()

assert results_df["no_skill_pr_auc"].eq(no_skill_pr_auc).all()

assert X_dev.shape[1] == 169
assert X_val.shape[1] == 169

print("169 frozen features: PASS")
print("Development/validation alignment: PASS")
print("Binary target: PASS")
print("Two dummy baselines evaluated: PASS")
print("PR-AUC calculated: PASS")
print("ROC-AUC calculated: PASS")
print("No-skill PR-AUC benchmark calculated: PASS")
print("Predicted churn rate calculated: PASS")
print("Top-K churn capture calculated: PASS")
print("Lift calculated: PASS")
print("Initial threshold fixed at 0.50: PASS")
print("Validation-only evaluation: PASS")
print("Test data not loaded: PASS")
print("Threshold optimization not performed: PASS")
print("Hyperparameter tuning not performed: PASS")


# ============================================================
# 14. FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("STEP 7.3 QUALITY GATE")
print("=" * 75)

print("169 frozen features: PASS")
print("Development data used for fitting only: PASS")
print("Validation data used for evaluation only: PASS")
print("Test data not loaded: PASS")
print("Two naive baselines completed: PASS")
print("PR-AUC benchmark established: PASS")
print("ROC-AUC calculated: PASS")
print("Classification metrics calculated: PASS")
print("Top-K targeting metrics calculated: PASS")
print("Confusion matrices saved: PASS")
print("Threshold optimization not performed: PASS")
print("Hyperparameter tuning not performed: PASS")

print("\nFINAL STATUS: PASS")
print("Ready for Step 7.4 — Logistic Regression Baseline.")
print("=" * 75)