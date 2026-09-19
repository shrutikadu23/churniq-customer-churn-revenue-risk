"""
ChurnIQ — Final Tuned XGBoost Model Builder

Purpose
-------
Rebuild the locked final XGBoost model from the frozen development dataset
and the Step 7.8 selected-parameter artifact.

This script:
    1. Loads frozen development features.
    2. Loads frozen development labels.
    3. Loads the Step 7.8 selected parameters.
    4. Rebuilds the selected XGBoost configuration.
    5. Fits on development data only.
    6. Saves the model using XGBoost's native model format.
    7. Writes a small metadata artifact for traceability.

This script does NOT:
    - use test data
    - perform tuning
    - perform feature selection
    - perform calibration
    - change the probability threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

X_DEV_PATH = PROCESSED_DIR / "X_dev.csv"
Y_DEV_PATH = PROCESSED_DIR / "y_dev.csv"

SELECTED_PARAMS_PATH = (
    REPORTS_DIR / "07_8_selected_model_parameters.json"
)

FINAL_MODEL_PATH = (
    MODELS_DIR / "xgboost_final_tuned.ubj"
)

FINAL_METADATA_PATH = (
    MODELS_DIR / "xgboost_final_tuned_metadata.json"
)


# ============================================================================
# LOCKED EXPECTATIONS
# ============================================================================

EXPECTED_FEATURE_COUNT = 169
EXPECTED_DEV_ROWS = 55_999
EXPECTED_CANDIDATE = "XGB_02_SLOWER_MORE_TREES"
RANDOM_STATE = 42


# ============================================================================
# LOAD SELECTED PARAMETERS
# ============================================================================

def load_selected_parameters() -> dict:

    if not SELECTED_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing Step 7.8 parameter artifact:\n"
            f"{SELECTED_PARAMS_PATH}"
        )

    with open(
        SELECTED_PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    candidate = payload.get(
        "selected_candidate"
    )

    if candidate != EXPECTED_CANDIDATE:
        raise ValueError(
            "Selected model candidate mismatch.\n"
            f"Expected: {EXPECTED_CANDIDATE}\n"
            f"Found: {candidate}"
        )

    parameters = payload.get(
        "selected_parameters"
    )

    if not isinstance(parameters, dict):
        raise ValueError(
            "selected_parameters is missing or invalid."
        )

    required = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }

    missing = required - set(parameters)

    if missing:
        raise ValueError(
            "Selected parameter artifact is incomplete.\n"
            f"Missing: {sorted(missing)}"
        )

    return parameters


# ============================================================================
# BUILD MODEL
# ============================================================================

def build_model(parameters: dict) -> XGBClassifier:

    model_parameters = dict(parameters)

    model_parameters.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    )

    return XGBClassifier(
        **model_parameters
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("CHURNIQ — BUILD FINAL TUNED XGBOOST")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 1. Validate artifacts
    # ------------------------------------------------------------------------

    print("\n[1/6] Checking required artifacts...")

    for path in (
        X_DEV_PATH,
        Y_DEV_PATH,
        SELECTED_PARAMS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    print("Artifact check: PASS")

    # ------------------------------------------------------------------------
    # 2. Load frozen development data
    # ------------------------------------------------------------------------

    print("\n[2/6] Loading development data...")

    X_dev = pd.read_csv(X_DEV_PATH)
    y_dev = pd.read_csv(Y_DEV_PATH).squeeze("columns")

    print(f"Rows     : {len(X_dev):,}")
    print(f"Features : {X_dev.shape[1]}")

    # ------------------------------------------------------------------------
    # 3. Validate feature matrix
    # ------------------------------------------------------------------------

    print("\n[3/6] Validating frozen feature matrix...")

    if X_dev.shape != (
        EXPECTED_DEV_ROWS,
        EXPECTED_FEATURE_COUNT,
    ):
        raise ValueError(
            "Unexpected development matrix shape.\n"
            f"Expected: "
            f"({EXPECTED_DEV_ROWS}, {EXPECTED_FEATURE_COUNT})\n"
            f"Found: {X_dev.shape}"
        )

    if len(y_dev) != EXPECTED_DEV_ROWS:
        raise ValueError(
            "Unexpected y_dev row count."
        )

    if X_dev.columns.duplicated().any():
        raise ValueError(
            "Duplicate feature names detected."
        )

    if X_dev.isna().any().any():
        raise ValueError(
            "Missing values detected in X_dev."
        )

    print("Frozen feature matrix: PASS")

    # ------------------------------------------------------------------------
    # 4. Load exact selected parameters
    # ------------------------------------------------------------------------

    print("\n[4/6] Loading Step 7.8 parameters...")

    parameters = load_selected_parameters()

    print(
        f"Selected candidate: {EXPECTED_CANDIDATE}"
    )

    for key in sorted(parameters):
        print(
            f"  {key}: {parameters[key]}"
        )

    # ------------------------------------------------------------------------
    # 5. Fit final model
    # ------------------------------------------------------------------------

    print("\n[5/6] Fitting final model...")

    model = build_model(
        parameters
    )

    model.fit(
        X_dev,
        y_dev,
    )

    print("Model fitting: PASS")

    # ------------------------------------------------------------------------
    # 6. Save native XGBoost model
    # ------------------------------------------------------------------------

    print("\n[6/6] Saving final model...")

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(
        FINAL_MODEL_PATH
    )

    if not FINAL_MODEL_PATH.exists():
        raise RuntimeError(
            "Final XGBoost model was not created."
        )

    metadata = {
        "model_name": "ChurnIQ Final Tuned XGBoost",
        "candidate": EXPECTED_CANDIDATE,
        "feature_count": EXPECTED_FEATURE_COUNT,
        "development_rows": EXPECTED_DEV_ROWS,
        "random_state": RANDOM_STATE,
        "calibration": "separate sigmoid calibrator",
        "primary_threshold": 0.10,
        "parameter_source": (
            "reports/07_8_selected_model_parameters.json"
        ),
        "model_file": FINAL_MODEL_PATH.name,
        "test_data_used": False,
        "feature_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "calibration_fitted": False,
    }

    with open(
        FINAL_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Reload verification
    # ------------------------------------------------------------------------

    print("\nVerifying saved model...")

    verification_model = XGBClassifier()

    verification_model.load_model(
        FINAL_MODEL_PATH
    )

    if verification_model.get_booster().feature_names:
        saved_feature_count = len(
            verification_model.get_booster().feature_names
        )

        if saved_feature_count != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Saved model feature count mismatch."
            )

    print()
    print("=" * 80)
    print("FINAL MODEL ARTIFACT QUALITY GATE: PASS")
    print("=" * 80)

    print(
        f"Candidate : {EXPECTED_CANDIDATE}"
    )

    print(
        f"Features  : {EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"Rows      : {EXPECTED_DEV_ROWS:,}"
    )

    print(
        f"Model     : {FINAL_MODEL_PATH.name}"
    )

    print(
        f"Metadata  : {FINAL_METADATA_PATH.name}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()