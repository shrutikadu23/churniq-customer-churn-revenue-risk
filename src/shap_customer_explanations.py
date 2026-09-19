from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

X_DEV_PATH = PROJECT_ROOT / "data" / "processed" / "X_dev.csv"
Y_DEV_PATH = PROJECT_ROOT / "data" / "processed" / "y_dev.csv"
ID_DEV_PATH = PROJECT_ROOT / "data" / "processed" / "id_dev.csv"

FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "processed" / "final_model_features.csv"
PARAMETERS_PATH = PROJECT_ROOT / "reports" / "07_8_selected_model_parameters.json"

CALIBRATOR_PATH = PROJECT_ROOT / "models" / "xgb_probability_calibrator.joblib"

SHAP_REGISTRY_PATH = (
    PROJECT_ROOT / "reports" / "shap" / "shap_sample_registry.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "shap"
OUTPUT_PATH = OUTPUT_DIR / "customer_shap_explanations.csv"

TOP_N = 5

# Locked primary threshold from Phase 09
PRIMARY_THRESHOLD = 0.10


# =============================================================================
# STEP 1 — LOAD FROZEN MODEL DATA
# =============================================================================

print("=" * 80)
print("STEP 1 — LOAD FROZEN MODEL DATA")
print("=" * 80)

X_dev = pd.read_csv(X_DEV_PATH)
y_dev = pd.read_csv(Y_DEV_PATH).squeeze("columns")
id_dev = pd.read_csv(ID_DEV_PATH).squeeze("columns")

print(f"X_dev shape: {X_dev.shape}")
print(f"y_dev shape: {y_dev.shape}")
print(f"id_dev shape: {id_dev.shape}")


# =============================================================================
# STEP 2 — VERIFY FROZEN 169 FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2 — VERIFY FROZEN FEATURES")
print("=" * 80)

feature_register = pd.read_csv(FEATURE_LIST_PATH)

# X_dev is the authoritative frozen model matrix.
# final_model_features.csv is used as an additional structural validation.
frozen_features = X_dev.columns.astype(str).tolist()

if len(frozen_features) != 169:
    raise ValueError(
        f"Expected 169 frozen model features, found {len(frozen_features)}"
    )

# Validate that the feature register contains the expected customer-level
# structure without assuming a particular feature-column name.
register_columns = feature_register.columns.astype(str).tolist()

if "id" not in register_columns:
    raise ValueError(
        "final_model_features.csv does not contain the expected 'id' column."
    )

if "churn_probability" not in register_columns:
    raise ValueError(
        "final_model_features.csv does not contain the expected "
        "'churn_probability' column."
    )

# The feature register should therefore contain:
# id + churn_probability + 169 frozen features = 171 columns
expected_register_columns = 171

if len(register_columns) != expected_register_columns:
    raise ValueError(
        f"Expected {expected_register_columns} columns in "
        f"final_model_features.csv, found {len(register_columns)}"
    )

# Validate that all X_dev features are represented in the register.
register_feature_columns = [
    col
    for col in register_columns
    if col not in {"id", "churn_probability"}
]

missing_from_register = [
    feature
    for feature in frozen_features
    if feature not in register_feature_columns
]

if missing_from_register:
    raise ValueError(
        f"{len(missing_from_register)} frozen features are missing "
        "from final_model_features.csv"
    )

# Validate exact feature order.
if register_feature_columns != frozen_features:
    raise ValueError(
        "Feature order mismatch between X_dev and final_model_features.csv."
    )

print(f"Frozen feature count: {len(frozen_features)}")
print(f"Feature register columns: {len(register_columns)}")
print("Feature presence: PASS")
print("Feature order: PASS")

# =============================================================================
# STEP 3 — VERIFY DATA ALIGNMENT
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3 — VERIFY DATA ALIGNMENT")
print("=" * 80)

if len(X_dev) != len(y_dev):
    raise ValueError("X_dev and y_dev row counts do not match.")

if len(X_dev) != len(id_dev):
    raise ValueError("X_dev and id_dev row counts do not match.")

if id_dev.duplicated().any():
    raise ValueError("Duplicate customer IDs detected in id_dev.")

print("X / y alignment: PASS")
print("X / ID alignment: PASS")


# =============================================================================
# STEP 4 — LOAD SHAP SAMPLE REGISTRY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4 — LOAD SHAP SAMPLE REGISTRY")
print("=" * 80)

registry = pd.read_csv(SHAP_REGISTRY_PATH)

if "id" not in registry.columns:
    raise ValueError(
        "SHAP registry must contain an 'id' column."
    )

sample_ids = registry["id"].tolist()

if len(sample_ids) != 10000:
    raise ValueError(
        f"Expected 10000 SHAP sample IDs, found {len(sample_ids)}"
    )

if len(set(sample_ids)) != len(sample_ids):
    raise ValueError(
        "Duplicate IDs found in SHAP sample registry."
    )

id_to_position = {
    customer_id: idx
    for idx, customer_id in enumerate(id_dev)
}

missing_ids = [
    customer_id
    for customer_id in sample_ids
    if customer_id not in id_to_position
]

if missing_ids:
    raise ValueError(
        f"{len(missing_ids)} SHAP sample IDs were not found in id_dev."
    )

sample_positions = [
    id_to_position[customer_id]
    for customer_id in sample_ids
]

X_sample = X_dev.iloc[sample_positions].copy()
y_sample = y_dev.iloc[sample_positions].copy()
id_sample = id_dev.iloc[sample_positions].copy()

print(f"SHAP sample size: {len(X_sample)}")
print("SHAP sample registry alignment: PASS")


# =============================================================================
# STEP 5 — LOAD SELECTED MODEL PARAMETERS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5 — LOAD SELECTED XGBOOST PARAMETERS")
print("=" * 80)

with open(PARAMETERS_PATH, "r", encoding="utf-8") as f:
    model_config = json.load(f)

selected_parameters = model_config["selected_parameters"]

print(
    f"Selected candidate: "
    f"{model_config['selected_candidate']}"
)

print(
    f"Selected CV PR-AUC: "
    f"{model_config['selected_cv_pr_auc']:.6f}"
)


# =============================================================================
# STEP 6 — REBUILD EXACT SELECTED XGBOOST MODEL
# =============================================================================

print("\n" + "=" * 80)
print("STEP 6 — REBUILD EXACT SELECTED XGBOOST MODEL")
print("=" * 80)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    **selected_parameters,
)

model.fit(X_dev, y_dev)

print("Model rebuild: PASS")


# =============================================================================
# STEP 7 — LOAD PROBABILITY CALIBRATION
# =============================================================================

print("\n" + "=" * 80)
print("STEP 7 — LOAD PROBABILITY CALIBRATION")
print("=" * 80)

calibration_artifact = joblib.load(CALIBRATOR_PATH)

if not isinstance(calibration_artifact, dict):
    raise ValueError(
        "Calibration artifact must be a dictionary."
    )

calibration_method = calibration_artifact.get("method")
calibrator = calibration_artifact.get("calibrator")

if calibration_method != "sigmoid":
    raise ValueError(
        f"Expected sigmoid calibration, found {calibration_method}"
    )

if calibrator is None:
    raise ValueError(
        "Calibrator object is missing."
    )

print(f"Calibration method: {calibration_method}")
print(f"Calibrator type: {type(calibrator)}")
print("Calibration artifact verification: PASS")

print("\nMethodology note:")
print(
    "SHAP explains the underlying tuned XGBoost raw output. "
    "Probability calibration remains a separate prediction layer."
)


# =============================================================================
# STEP 8 — CALCULATE SHAP VALUES
# =============================================================================

print("\n" + "=" * 80)
print("STEP 8 — CALCULATE INDIVIDUAL SHAP VALUES")
print("=" * 80)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_sample,
    check_additivity=True
)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_values = np.asarray(shap_values)

expected_shape = (
    len(X_sample),
    len(frozen_features)
)

if shap_values.shape != expected_shape:
    raise ValueError(
        f"Unexpected SHAP shape: {shap_values.shape}; "
        f"expected {expected_shape}"
    )

print(f"SHAP matrix shape: {shap_values.shape}")
print("SHAP calculation: PASS")


# =============================================================================
# STEP 9 — VERIFY SHAP ADDITIVITY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 9 — VERIFY SHAP ADDITIVITY")
print("=" * 80)

expected_value = explainer.expected_value

if isinstance(expected_value, np.ndarray):
    expected_value = float(expected_value.reshape(-1)[0])
else:
    expected_value = float(expected_value)

raw_margin = model.predict(
    X_sample,
    output_margin=True
)

reconstructed_margin = (
    expected_value
    + shap_values.sum(axis=1)
)

max_abs_error = np.max(
    np.abs(
        raw_margin
        - reconstructed_margin
    )
)

mean_abs_error = np.mean(
    np.abs(
        raw_margin
        - reconstructed_margin
    )
)

ADDITIVITY_TOLERANCE = 1e-5

print(f"Expected value: {expected_value:.6f}")
print(f"Maximum absolute error: {max_abs_error:.10f}")
print(f"Mean absolute error: {mean_abs_error:.10f}")

additivity_pass = (
    max_abs_error <= ADDITIVITY_TOLERANCE
)

print(
    "SHAP additivity check: "
    f"{'PASS' if additivity_pass else 'FAIL'}"
)

if not additivity_pass:
    raise ValueError(
        "SHAP additivity validation failed."
    )


# =============================================================================
# STEP 10 — CALCULATE CALIBRATED CHURN PROBABILITY
# =============================================================================

print("\n" + "=" * 80)
print("STEP 10 — CALCULATE CALIBRATED CHURN PROBABILITY")
print("=" * 80)

raw_probability = model.predict_proba(
    X_sample
)[:, 1]

eps = 1e-15

clipped_probability = np.clip(
    raw_probability,
    eps,
    1 - eps
)

logit = np.log(
    clipped_probability
    / (1 - clipped_probability)
)

calibrated_probability = calibrator.predict_proba(
    logit.reshape(-1, 1)
)[:, 1]

if not np.all(
    (calibrated_probability >= 0)
    & (calibrated_probability <= 1)
):
    raise ValueError(
        "Invalid calibrated probability detected."
    )

print(
    f"Calibrated probability range: "
    f"{calibrated_probability.min():.6f} - "
    f"{calibrated_probability.max():.6f}"
)

print("Calibrated probability calculation: PASS")


# =============================================================================
# STEP 11 — ASSIGN RISK LEVEL
# =============================================================================

print("\n" + "=" * 80)
print("STEP 11 — ASSIGN RISK LEVEL")
print("=" * 80)

def assign_risk_level(probability):
    if probability >= 0.50:
        return "Very High"
    elif probability >= PRIMARY_THRESHOLD:
        return "High"
    else:
        return "Below Primary Threshold"


risk_levels = [
    assign_risk_level(probability)
    for probability in calibrated_probability
]

print(
    pd.Series(risk_levels)
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    f"\nPrimary threshold: {PRIMARY_THRESHOLD:.2f}"
)


# =============================================================================
# STEP 12 — BUILD CUSTOMER-LEVEL EXPLANATIONS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 12 — BUILD CUSTOMER-LEVEL EXPLANATIONS")
print("=" * 80)

records = []

for row_idx in range(len(X_sample)):

    row_shap = shap_values[row_idx]
    row_values = X_sample.iloc[row_idx]

    # Highest positive SHAP contributions
    positive_indices = np.argsort(
        row_shap
    )[::-1]

    positive_indices = [
        idx
        for idx in positive_indices
        if row_shap[idx] > 0
    ][:TOP_N]

    # Highest negative SHAP contributions
    negative_indices = np.argsort(
        row_shap
    )

    negative_indices = [
        idx
        for idx in negative_indices
        if row_shap[idx] < 0
    ][:TOP_N]

    record = {
        "id": id_sample.iloc[row_idx],
        "churn_probability": calibrated_probability[row_idx],
        "risk_level": risk_levels[row_idx],
        "primary_threshold": PRIMARY_THRESHOLD,
        "above_primary_threshold": (
            calibrated_probability[row_idx]
            >= PRIMARY_THRESHOLD
        ),
        "actual_churn": int(
            y_sample.iloc[row_idx]
        ),
    }

    # Positive contributors
    for rank in range(1, TOP_N + 1):

        if rank <= len(positive_indices):

            feature_idx = positive_indices[
                rank - 1
            ]

            feature = frozen_features[
                feature_idx
            ]

            shap_value = row_shap[
                feature_idx
            ]

            record[
                f"positive_driver_{rank}"
            ] = feature

            record[
                f"positive_driver_{rank}_shap"
            ] = shap_value

            record[
                f"positive_driver_{rank}_abs_shap"
            ] = abs(shap_value)

            record[
                f"positive_driver_{rank}_value"
            ] = row_values[feature]

            record[
                f"positive_driver_{rank}_type"
            ] = "Risk-increasing contribution"

        else:

            record[
                f"positive_driver_{rank}"
            ] = np.nan

            record[
                f"positive_driver_{rank}_shap"
            ] = np.nan

            record[
                f"positive_driver_{rank}_abs_shap"
            ] = np.nan

            record[
                f"positive_driver_{rank}_value"
            ] = np.nan

            record[
                f"positive_driver_{rank}_type"
            ] = np.nan

    # Negative contributors
    for rank in range(1, TOP_N + 1):

        if rank <= len(negative_indices):

            feature_idx = negative_indices[
                rank - 1
            ]

            feature = frozen_features[
                feature_idx
            ]

            shap_value = row_shap[
                feature_idx
            ]

            record[
                f"negative_driver_{rank}"
            ] = feature

            record[
                f"negative_driver_{rank}_shap"
            ] = shap_value

            record[
                f"negative_driver_{rank}_abs_shap"
            ] = abs(shap_value)

            record[
                f"negative_driver_{rank}_value"
            ] = row_values[feature]

            record[
                f"negative_driver_{rank}_type"
            ] = "Risk-reducing contribution"

        else:

            record[
                f"negative_driver_{rank}"
            ] = np.nan

            record[
                f"negative_driver_{rank}_shap"
            ] = np.nan

            record[
                f"negative_driver_{rank}_abs_shap"
            ] = np.nan

            record[
                f"negative_driver_{rank}_value"
            ] = np.nan

            record[
                f"negative_driver_{rank}_type"
            ] = np.nan

    records.append(record)


customer_explanations = pd.DataFrame(
    records
)


# =============================================================================
# STEP 13 — SAVE OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("STEP 13 — SAVE CUSTOMER SHAP EXPLANATIONS")
print("=" * 80)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

customer_explanations.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Saved: {OUTPUT_PATH}")


# =============================================================================
# STEP 14 — DISPLAY EXAMPLE EXPLANATIONS
# =============================================================================

print("\n" + "=" * 80)
print("STEP 14 — EXAMPLE CUSTOMER EXPLANATIONS")
print("=" * 80)

display_columns = [
    "id",
    "churn_probability",
    "risk_level",
    "above_primary_threshold",
    "positive_driver_1",
    "positive_driver_1_shap",
    "positive_driver_1_value",
    "negative_driver_1",
    "negative_driver_1_shap",
    "negative_driver_1_value",
]

print(
    customer_explanations[
        display_columns
    ]
    .head(10)
    .to_string(index=False)
)


# =============================================================================
# STEP 15 — QUALITY GATE
# =============================================================================

print("\n" + "=" * 80)
print("STEP 15 — CUSTOMER SHAP QUALITY GATE")
print("=" * 80)

checks = {
    "sample_rows_10000": (
        len(customer_explanations) == 10000
    ),

    "feature_count_169": (
        len(frozen_features) == 169
    ),

    "unique_customer_ids": (
        customer_explanations["id"].nunique()
        == 10000
    ),

    "probabilities_valid": (
        customer_explanations[
            "churn_probability"
        ]
        .between(0, 1)
        .all()
    ),

    "risk_levels_present": (
        customer_explanations[
            "risk_level"
        ]
        .notna()
        .all()
    ),

    "threshold_logic_valid": (
        customer_explanations[
            "above_primary_threshold"
        ]
        ==
        (
            customer_explanations[
                "churn_probability"
            ]
            >= PRIMARY_THRESHOLD
        )
    ).all(),

    "positive_driver_columns_present": all(
        f"positive_driver_{i}"
        in customer_explanations.columns
        for i in range(1, TOP_N + 1)
    ),

    "negative_driver_columns_present": all(
        f"negative_driver_{i}"
        in customer_explanations.columns
        for i in range(1, TOP_N + 1)
    ),

    "positive_shap_values_valid": all(
        customer_explanations[
            f"positive_driver_{i}_shap"
        ]
        .dropna()
        .ge(0)
        .all()
        for i in range(1, TOP_N + 1)
    ),

    "negative_shap_values_valid": all(
        customer_explanations[
            f"negative_driver_{i}_shap"
        ]
        .dropna()
        .le(0)
        .all()
        for i in range(1, TOP_N + 1)
    ),

    "shap_additivity_valid": (
        max_abs_error
        <= ADDITIVITY_TOLERANCE
    ),

    "output_exists": (
        OUTPUT_PATH.exists()
    ),
}


for check_name, result in checks.items():

    if isinstance(result, (bool, np.bool_)):
        status = bool(result)
    else:
        status = bool(result)

    print(
        f"{check_name}: "
        f"{'PASS' if status else 'FAIL'}"
    )


overall_pass = all(
    bool(value)
    for value in checks.values()
)


print(
    "\nOVERALL CUSTOMER SHAP QUALITY GATE: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)


if not overall_pass:

    raise RuntimeError(
        "Customer SHAP quality gate failed."
    )


print(
    "\nPhase 10.3 — Individual Customer SHAP "
    "Explanations completed successfully."
)