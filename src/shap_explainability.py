from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

X_DEV_PATH = ROOT / "data" / "processed" / "X_dev.csv"
Y_DEV_PATH = ROOT / "data" / "processed" / "y_dev.csv"
ID_DEV_PATH = ROOT / "data" / "processed" / "id_dev.csv"

FEATURES_PATH = ROOT / "data" / "processed" / "final_model_features.csv"
PARAMS_PATH = ROOT / "reports" / "07_8_selected_model_parameters.json"
CALIBRATOR_PATH = ROOT / "models" / "xgb_probability_calibrator.joblib"

OUTPUT_DIR = ROOT / "reports" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1 — LOAD FROZEN MODEL DATA
# ============================================================

print("=" * 80)
print("STEP 1 — LOAD FROZEN MODEL DATA")
print("=" * 80)

X_dev = pd.read_csv(X_DEV_PATH)
y_dev = pd.read_csv(Y_DEV_PATH).squeeze()
id_dev = pd.read_csv(ID_DEV_PATH).squeeze()
feature_register = pd.read_csv(FEATURES_PATH)

print(f"X_dev shape: {X_dev.shape}")
print(f"y_dev shape: {y_dev.shape}")
print(f"id_dev shape: {id_dev.shape}")
print(f"Feature register shape: {feature_register.shape}")


# ============================================================
# STEP 2 — VERIFY DATA ALIGNMENT
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 — VERIFY DATA ALIGNMENT")
print("=" * 80)

if len(X_dev) != len(y_dev):
    raise ValueError("X_dev and y_dev row counts do not match.")

if len(X_dev) != len(id_dev):
    raise ValueError("X_dev and id_dev row counts do not match.")

print("X_dev / y_dev / id_dev alignment: PASS")


# ============================================================
# STEP 3 — IDENTIFY FROZEN FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 — VERIFY FROZEN 169 FEATURES")
print("=" * 80)

# final_model_features.csv is a customer-level register.
# "id" and "churn_probability" are metadata columns.
# All remaining columns represent the frozen model features.

frozen_features = [
    col
    for col in feature_register.columns
    if col not in ["id", "churn_probability"]
]

print(f"Frozen feature count: {len(frozen_features)}")

if len(frozen_features) != 169:
    raise ValueError(
        f"Expected 169 frozen features, found {len(frozen_features)}."
    )


# ============================================================
# STEP 4 — VERIFY FEATURE PRESENCE AND ORDER
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 — VERIFY FEATURE PRESENCE AND ORDER")
print("=" * 80)

missing_features = [
    col
    for col in frozen_features
    if col not in X_dev.columns
]

extra_features = [
    col
    for col in X_dev.columns
    if col not in frozen_features
]

if missing_features:
    raise ValueError(
        f"Frozen features missing from X_dev: {missing_features}"
    )

if extra_features:
    raise ValueError(
        f"Unexpected features in X_dev: {extra_features}"
    )

# Explicitly reconstruct X in frozen feature order.
X = X_dev[frozen_features].copy()

# Defensive feature-order check.
if list(X.columns) != frozen_features:
    raise ValueError(
        "Feature order mismatch detected."
    )

print("Feature presence: PASS")
print("Feature order: PASS")
print(f"Final SHAP matrix shape: {X.shape}")


# ============================================================
# STEP 5 — LOAD SELECTED MODEL PARAMETERS
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 — LOAD SELECTED XGBOOST PARAMETERS")
print("=" * 80)

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    model_selection = json.load(f)

selected_model = model_selection["selected_model"]
selected_candidate = model_selection["selected_candidate"]
selected_cv_pr_auc = model_selection["selected_cv_pr_auc"]
selected_parameters = model_selection["selected_parameters"]

print(f"Selected model: {selected_model}")
print(f"Selected candidate: {selected_candidate}")
print(f"Selected CV PR-AUC: {selected_cv_pr_auc:.6f}")
print(f"Selected parameters: {selected_parameters}")

if selected_model != "XGBoost":
    raise ValueError(
        f"Expected XGBoost as selected model, found {selected_model}"
    )


# ============================================================
# STEP 6 — REBUILD EXACT SELECTED MODEL
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 — REBUILD EXACT SELECTED XGBOOST MODEL")
print("=" * 80)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    **selected_parameters
)

model.fit(X, y_dev)

print("XGBoost model rebuild: PASS")


# ============================================================
# STEP 7 — VERIFY CALIBRATION ARTIFACT
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 — VERIFY PROBABILITY CALIBRATION")
print("=" * 80)

calibration_artifact = joblib.load(CALIBRATOR_PATH)

if not isinstance(calibration_artifact, dict):
    raise ValueError(
        "Unexpected calibration artifact format."
    )

calibration_method = calibration_artifact.get("method")
calibrator = calibration_artifact.get("calibrator")

print(f"Calibration method: {calibration_method}")
print(f"Calibrator type: {type(calibrator)}")

if calibration_method != "sigmoid":
    raise ValueError(
        f"Expected sigmoid calibration, found {calibration_method}"
    )

print("Calibration artifact verification: PASS")

print(
    "\nMethodology note:"
    "\nSHAP explains the underlying tuned XGBoost model."
    "\nProbability calibration remains a separate prediction layer."
)


# ============================================================
# STEP 8 — CREATE REPRODUCIBLE SHAP SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 — CREATE REPRODUCIBLE SHAP SAMPLE")
print("=" * 80)

SHAP_SAMPLE_SIZE = min(10000, len(X))

sample_indices = X.sample(
    n=SHAP_SAMPLE_SIZE,
    random_state=42
).index

X_shap = X.loc[sample_indices].copy()
id_shap = id_dev.loc[sample_indices].copy()
y_shap = y_dev.loc[sample_indices].copy()

print(f"SHAP sample size: {len(X_shap)}")


# ============================================================
# STEP 9 — SAVE SHAP SAMPLE IDENTIFIERS
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 — SAVE SHAP SAMPLE IDENTIFIERS")
print("=" * 80)

shap_sample_registry = pd.DataFrame({
    "row_index": sample_indices,
    "id": id_shap.values,
    "churn_probability": y_shap.values
})

sample_registry_path = (
    OUTPUT_DIR / "shap_sample_registry.csv"
)

shap_sample_registry.to_csv(
    sample_registry_path,
    index=False
)

print(f"Saved: {sample_registry_path}")


# ============================================================
# STEP 10 — CALCULATE SHAP VALUES
# ============================================================

print("\n" + "=" * 80)
print("STEP 10 — CALCULATE SHAP VALUES")
print("=" * 80)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_shap)

# Handle SHAP versions that return a list.
if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

print(f"SHAP value matrix shape: {shap_values.shape}")

if shap_values.shape != X_shap.shape:
    raise ValueError(
        "SHAP value matrix does not match SHAP feature matrix."
    )

print("SHAP calculation: PASS")


# ============================================================
# STEP 11 — GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 — GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 80)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

global_importance = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": mean_abs_shap
})

global_importance = (
    global_importance
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

global_importance["rank"] = (
    np.arange(1, len(global_importance) + 1)
)

global_importance = global_importance[
    ["rank", "feature", "mean_abs_shap"]
]

importance_path = (
    OUTPUT_DIR / "global_shap_feature_importance.csv"
)

global_importance.to_csv(
    importance_path,
    index=False
)

print("\nTop 20 SHAP features:")
print(
    global_importance
    .head(20)
    .to_string(index=False)
)

print(f"\nSaved: {importance_path}")


# ============================================================
# STEP 12 — SHAP SUMMARY BEESWARM
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 — CREATE SHAP SUMMARY PLOT")
print("=" * 80)

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    show=False,
    max_display=20
)

plt.tight_layout()

summary_path = (
    OUTPUT_DIR / "shap_summary_beeswarm.png"
)

plt.savefig(
    summary_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {summary_path}")


# ============================================================
# STEP 13 — SHAP GLOBAL BAR PLOT
# ============================================================

print("\n" + "=" * 80)
print("STEP 13 — CREATE SHAP GLOBAL BAR PLOT")
print("=" * 80)

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

bar_path = (
    OUTPUT_DIR / "shap_global_importance_bar.png"
)

plt.savefig(
    bar_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {bar_path}")


# ============================================================
# STEP 14 — SHAP QUALITY GATE
# ============================================================

print("\n" + "=" * 80)
print("STEP 14 — SHAP QUALITY GATE")
print("=" * 80)

checks = {
    "feature_count_169": X.shape[1] == 169,
    "feature_order_verified": list(X.columns) == frozen_features,
    "x_y_alignment": len(X) == len(y_dev),
    "x_id_alignment": len(X) == len(id_dev),
    "shap_rows_match": shap_values.shape[0] == X_shap.shape[0],
    "shap_columns_match": shap_values.shape[1] == X_shap.shape[1],
    "importance_rows_169": len(global_importance) == 169,
    "importance_sorted": global_importance[
        "mean_abs_shap"
    ].is_monotonic_decreasing,
    "sample_registry_exists": sample_registry_path.exists(),
    "summary_plot_exists": summary_path.exists(),
    "bar_plot_exists": bar_path.exists(),
}

for name, result in checks.items():
    print(
        f"{name}: {'PASS' if result else 'FAIL'}"
    )

overall = all(checks.values())

print(
    f"\nOVERALL SHAP QUALITY GATE: "
    f"{'PASS' if overall else 'FAIL'}"
)

if not overall:
    raise ValueError(
        "SHAP quality gate failed."
    )

print("\nPhase 10.1 — Global SHAP completed successfully.")