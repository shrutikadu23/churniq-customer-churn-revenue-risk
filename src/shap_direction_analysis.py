from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

X_DEV_PATH = ROOT / "data" / "processed" / "X_dev.csv"
Y_DEV_PATH = ROOT / "data" / "processed" / "y_dev.csv"

FEATURES_PATH = ROOT / "data" / "processed" / "final_model_features.csv"
PARAMS_PATH = ROOT / "reports" / "07_8_selected_model_parameters.json"

SHAP_SAMPLE_REGISTRY_PATH = (
    ROOT / "reports" / "shap" / "shap_sample_registry.csv"
)

OUTPUT_DIR = ROOT / "reports" / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("=" * 80)
print("STEP 1 — LOAD FROZEN MODEL DATA")
print("=" * 80)

X_dev = pd.read_csv(X_DEV_PATH)
y_dev = pd.read_csv(Y_DEV_PATH).squeeze()
feature_register = pd.read_csv(FEATURES_PATH)

print(f"X_dev shape: {X_dev.shape}")
print(f"y_dev shape: {y_dev.shape}")


# ============================================================
# STEP 2 — LOAD FROZEN FEATURES
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 — VERIFY FROZEN FEATURES")
print("=" * 80)

frozen_features = [
    col
    for col in feature_register.columns
    if col not in ["id", "churn_probability"]
]

if len(frozen_features) != 169:
    raise ValueError(
        f"Expected 169 frozen features, found {len(frozen_features)}."
    )

X = X_dev[frozen_features].copy()

if list(X.columns) != frozen_features:
    raise ValueError("Feature order mismatch detected.")

print(f"Frozen feature count: {len(frozen_features)}")
print("Feature order: PASS")


# ============================================================
# STEP 3 — LOAD REPRODUCIBLE SHAP SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("STEP 3 — LOAD SHAP SAMPLE")
print("=" * 80)

sample_registry = pd.read_csv(
    SHAP_SAMPLE_REGISTRY_PATH
)

sample_indices = (
    sample_registry["row_index"]
    .astype(int)
    .tolist()
)

X_shap = X.loc[sample_indices].copy()

print(f"SHAP sample size: {len(X_shap)}")

if len(X_shap) != len(sample_registry):
    raise ValueError(
        "SHAP sample registry and feature matrix do not match."
    )

print("SHAP sample reproducibility check: PASS")


# ============================================================
# STEP 4 — LOAD SELECTED MODEL PARAMETERS
# ============================================================

print("\n" + "=" * 80)
print("STEP 4 — LOAD SELECTED XGBOOST PARAMETERS")
print("=" * 80)

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    model_selection = json.load(f)

selected_parameters = model_selection[
    "selected_parameters"
]

print(
    f"Selected candidate: "
    f"{model_selection['selected_candidate']}"
)

print(
    f"Selected CV PR-AUC: "
    f"{model_selection['selected_cv_pr_auc']:.6f}"
)


# ============================================================
# STEP 5 — REBUILD EXACT SELECTED MODEL
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 — REBUILD EXACT SELECTED XGBOOST MODEL")
print("=" * 80)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    **selected_parameters
)

model.fit(X, y_dev)

print("Model rebuild: PASS")


# ============================================================
# STEP 6 — CALCULATE SHAP VALUES
# ============================================================

print("\n" + "=" * 80)
print("STEP 6 — CALCULATE SHAP VALUES")
print("=" * 80)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_shap)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

if shap_values.shape != X_shap.shape:
    raise ValueError(
        "SHAP values and feature matrix shapes do not match."
    )

print(f"SHAP matrix shape: {shap_values.shape}")
print("SHAP calculation: PASS")


# ============================================================
# STEP 7 — CALCULATE GLOBAL DIRECTION METRICS
# ============================================================

print("\n" + "=" * 80)
print("STEP 7 — CALCULATE SHAP DIRECTION METRICS")
print("=" * 80)

rows = []

for i, feature in enumerate(X_shap.columns):

    values = shap_values[:, i]
    feature_values = X_shap[feature]

    positive_mask = values > 0
    negative_mask = values < 0
    zero_mask = values == 0

    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    zero_count = int(zero_mask.sum())

    non_zero_count = (
        positive_count + negative_count
    )

    if non_zero_count > 0:

        positive_pct_nonzero = (
            positive_count / non_zero_count * 100
        )

        negative_pct_nonzero = (
            negative_count / non_zero_count * 100
        )

        direction_consistency = max(
            positive_pct_nonzero,
            negative_pct_nonzero
        )

    else:

        positive_pct_nonzero = 0.0
        negative_pct_nonzero = 0.0
        direction_consistency = 0.0

    # --------------------------------------------------------
    # Direction classification
    #
    # IMPORTANT:
    # This describes the average SHAP contribution.
    # It does NOT claim that higher feature values cause
    # higher/lower churn risk.
    # --------------------------------------------------------

    mean_shap = float(values.mean())
    mean_abs_shap = float(np.abs(values).mean())

    if mean_shap > 0:
        overall_direction = (
            "Positive average contribution"
        )

    elif mean_shap < 0:
        overall_direction = (
            "Negative average contribution"
        )

    else:
        overall_direction = (
            "Near-neutral average contribution"
        )

    # --------------------------------------------------------
    # Direction consistency classification
    # --------------------------------------------------------

    if non_zero_count == 0:

        direction_consistency_label = (
            "No SHAP contribution"
        )

    elif direction_consistency >= 75:

        direction_consistency_label = (
            "Strongly consistent"
        )

    elif direction_consistency >= 60:

        direction_consistency_label = (
            "Moderately consistent"
        )

    else:

        direction_consistency_label = (
            "Mixed"
        )

    rows.append({
        "feature": feature,
        "mean_shap": mean_shap,
        "mean_abs_shap": mean_abs_shap,
        "positive_shap_pct_nonzero":
            positive_pct_nonzero,
        "negative_shap_pct_nonzero":
            negative_pct_nonzero,
        "zero_shap_pct":
            zero_count / len(values) * 100,
        "direction_consistency_pct":
            direction_consistency,
        "direction_consistency_label":
            direction_consistency_label,
        "mean_feature_value":
            feature_values.mean(),
        "median_feature_value":
            feature_values.median(),
        "overall_direction":
            overall_direction
    })


direction_df = pd.DataFrame(rows)

direction_df = (
    direction_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

direction_df["rank"] = (
    np.arange(1, len(direction_df) + 1)
)

direction_df = direction_df[
    [
        "rank",
        "feature",
        "mean_abs_shap",
        "mean_shap",
        "positive_shap_pct_nonzero",
        "negative_shap_pct_nonzero",
        "zero_shap_pct",
        "direction_consistency_pct",
        "direction_consistency_label",
        "mean_feature_value",
        "median_feature_value",
        "overall_direction"
    ]
]


# ============================================================
# STEP 8 — SAVE FULL DIRECTION TABLE
# ============================================================

print("\n" + "=" * 80)
print("STEP 8 — SAVE SHAP DIRECTION ANALYSIS")
print("=" * 80)

direction_path = (
    OUTPUT_DIR /
    "shap_feature_direction_analysis.csv"
)

direction_df.to_csv(
    direction_path,
    index=False
)

print(f"Saved: {direction_path}")


# ============================================================
# STEP 9 — DISPLAY TOP 20
# ============================================================

print("\n" + "=" * 80)
print("STEP 9 — TOP 20 SHAP DRIVERS WITH DIRECTION")
print("=" * 80)

display_columns = [
    "rank",
    "feature",
    "mean_abs_shap",
    "mean_shap",
    "positive_shap_pct_nonzero",
    "negative_shap_pct_nonzero",
    "direction_consistency_pct",
    "direction_consistency_label",
    "overall_direction"
]

print(
    direction_df
    .head(20)[display_columns]
    .to_string(index=False)
)


# ============================================================
# STEP 10 — STRONGEST POSITIVE CONTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("STEP 10 — STRONGEST POSITIVE AVERAGE CONTRIBUTIONS")
print("=" * 80)

positive_drivers = (
    direction_df[
        direction_df["mean_shap"] > 0
    ]
    .sort_values(
        "mean_shap",
        ascending=False
    )
    .head(15)
)

print(
    positive_drivers[
        [
            "rank",
            "feature",
            "mean_shap",
            "mean_abs_shap",
            "positive_shap_pct_nonzero",
            "direction_consistency_pct",
            "direction_consistency_label"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 11 — STRONGEST NEGATIVE CONTRIBUTIONS
# ============================================================

print("\n" + "=" * 80)
print("STEP 11 — STRONGEST NEGATIVE AVERAGE CONTRIBUTIONS")
print("=" * 80)

negative_drivers = (
    direction_df[
        direction_df["mean_shap"] < 0
    ]
    .sort_values(
        "mean_shap",
        ascending=True
    )
    .head(15)
)

print(
    negative_drivers[
        [
            "rank",
            "feature",
            "mean_shap",
            "mean_abs_shap",
            "negative_shap_pct_nonzero",
            "direction_consistency_pct",
            "direction_consistency_label"
        ]
    ].to_string(index=False)
)


# ============================================================
# STEP 12 — DIRECTION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP 12 — DIRECTION SUMMARY")
print("=" * 80)

direction_counts = (
    direction_df["overall_direction"]
    .value_counts()
)

for label, count in direction_counts.items():
    print(f"{label}: {count}")


consistency_counts = (
    direction_df["direction_consistency_label"]
    .value_counts()
)

print("\nDirection consistency:")
for label, count in consistency_counts.items():
    print(f"{label}: {count}")


# ============================================================
# STEP 13 — QUALITY GATE
# ============================================================

print("\n" + "=" * 80)
print("STEP 13 — SHAP DIRECTION QUALITY GATE")
print("=" * 80)

direction_sum = (
    direction_df["positive_shap_pct_nonzero"]
    + direction_df["negative_shap_pct_nonzero"]
)

valid_direction_sum = (
    direction_sum.between(99.999, 100.001)
    | (direction_sum == 0)
)

checks = {
    "feature_count_169":
        len(direction_df) == 169,

    "importance_values_present":
        direction_df["mean_abs_shap"].notna().all(),

    "direction_values_present":
        direction_df["mean_shap"].notna().all(),

    "direction_percentages_valid":
        valid_direction_sum.all(),

    "consistency_values_valid":
        direction_df[
            "direction_consistency_pct"
        ].between(0, 100).all(),

    "importance_sorted":
        direction_df[
            "mean_abs_shap"
        ].is_monotonic_decreasing,

    "output_exists":
        direction_path.exists()
}

for name, result in checks.items():
    print(
        f"{name}: {'PASS' if result else 'FAIL'}"
    )

overall = all(checks.values())

print(
    f"\nOVERALL SHAP DIRECTION QUALITY GATE: "
    f"{'PASS' if overall else 'FAIL'}"
)

if not overall:
    raise ValueError(
        "SHAP direction quality gate failed."
    )

print(
    "\nPhase 10.2 — SHAP Direction & Driver "
    "Interpretation completed successfully."
)