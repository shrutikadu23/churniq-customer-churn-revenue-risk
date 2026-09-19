"""
ChurnIQ — Reusable Inference Engine

Inference only.

Pipeline
--------
Frozen 169 features
        ↓
Development-only median imputation
        ↓
Final Tuned XGBoost
        ↓
Raw probability
        ↓
Probability → logit
        ↓
Saved sigmoid calibrator
        ↓
Calibrated probability
        ↓
Risk score
        ↓
Risk level
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEPLOYMENT_DATA_DIR = PROJECT_ROOT / "deployment_data"
MODEL_DIR = PROJECT_ROOT / "models"

CUSTOMER_FEATURE_FILE = (
    DEPLOYMENT_DATA_DIR / "customer_features.parquet"
)

IMPUTATION_MEDIANS_FILE = (
    DEPLOYMENT_DATA_DIR / "imputation_medians.csv"
)

FINAL_MODEL_FILE = (
    MODEL_DIR / "xgboost_final_tuned.ubj"
)

CALIBRATOR_FILE = (
    MODEL_DIR / "xgb_probability_calibrator.joblib"
)


# ============================================================================
# LOCKED SETTINGS
# ============================================================================

EXPECTED_FEATURE_COUNT = 169

PRIMARY_THRESHOLD = 0.10

VERY_HIGH_THRESHOLD = 0.50


# ============================================================================
# CACHED ARTIFACTS
# ============================================================================

_MODEL = None
_CALIBRATOR = None
_FEATURE_NAMES = None
_IMPUTATION_MEDIANS = None


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_inference_engine():

    global _MODEL
    global _CALIBRATOR
    global _FEATURE_NAMES
    global _IMPUTATION_MEDIANS

    if (
        _MODEL is not None
        and _CALIBRATOR is not None
        and _FEATURE_NAMES is not None
        and _IMPUTATION_MEDIANS is not None
    ):
        return (
            _MODEL,
            _CALIBRATOR,
            _FEATURE_NAMES,
            _IMPUTATION_MEDIANS,
        )

    required_files = (
        FINAL_MODEL_FILE,
        CALIBRATOR_FILE,
        CUSTOMER_FEATURE_FILE,
        IMPUTATION_MEDIANS_FILE,
    )

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required ChurnIQ artifact not found:\n{path}"
            )

    # ------------------------------------------------------------------------
    # Load native XGBoost model
    # ------------------------------------------------------------------------

    model = XGBClassifier()

    model.load_model(
        FINAL_MODEL_FILE
    )

    # ------------------------------------------------------------------------
    # Validate model feature count
    # ------------------------------------------------------------------------

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        model_feature_count is not None
        and model_feature_count != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            "Loaded model feature count mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {model_feature_count}"
        )

    # ------------------------------------------------------------------------
    # Load sigmoid calibrator
    # ------------------------------------------------------------------------

    calibration_artifact = joblib.load(
        CALIBRATOR_FILE
    )

    if not isinstance(
        calibration_artifact,
        dict,
    ):
        raise TypeError(
            "Calibration artifact must be a dictionary."
        )

    method = calibration_artifact.get(
        "method"
    )

    if method != "sigmoid":
        raise ValueError(
            f"Expected sigmoid calibration, found: {method}"
        )

    calibrator = calibration_artifact.get(
        "calibrator"
    )

    if calibrator is None:
        raise ValueError(
            "Sigmoid calibrator is missing."
        )

    # ------------------------------------------------------------------------
    # Load frozen feature schema
    # ------------------------------------------------------------------------

    sample = pd.read_parquet(
         CUSTOMER_FEATURE_FILE
    )

    excluded_columns = {
        "id",
    }

    feature_names = [
        column
        for column in sample.columns
        if column not in excluded_columns
    ]

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Frozen feature count mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {len(feature_names)}"
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError(
            "Duplicate feature names detected."
        )

    # ------------------------------------------------------------------------
    # Load locked development-only imputation medians
    # ------------------------------------------------------------------------

    median_table = pd.read_csv(
        IMPUTATION_MEDIANS_FILE
    )

    if median_table.shape[0] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Imputation median count mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {median_table.shape[0]}"
        )

    if median_table.columns.tolist() != ["feature", "median"]:
        raise ValueError(
            "Imputation median file must contain exactly "
            "'feature' and 'median' columns."
        )

    if median_table["feature"].tolist() != feature_names:
        raise ValueError(
            "Imputation feature order does not match "
            "the frozen feature schema."
        )

    imputation_medians = (
        pd.Series(
            median_table["median"].astype(float).to_numpy(),
            index=median_table["feature"].tolist(),
        )
    )

    if imputation_medians.isna().any():
        raise ValueError(
            "Imputation median file contains missing values."
        )
    

    # ------------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------------

    _MODEL = model
    _CALIBRATOR = calibrator
    _FEATURE_NAMES = feature_names
    _IMPUTATION_MEDIANS = imputation_medians

    return (
        _MODEL,
        _CALIBRATOR,
        _FEATURE_NAMES,
        _IMPUTATION_MEDIANS,
    )


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(
    features: pd.DataFrame,
) -> pd.DataFrame:

    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError(
            "features must be a pandas DataFrame."
        )

    (
        _,
        _,
        feature_names,
        imputation_medians,
    ) = load_inference_engine()

    # ------------------------------------------------------------------------
    # Feature presence
    # ------------------------------------------------------------------------

    missing = [
        column
        for column in feature_names
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            "Missing frozen features:\n"
            + "\n".join(missing)
        )

    # ------------------------------------------------------------------------
    # Exact frozen feature order
    # ------------------------------------------------------------------------

    prepared = features[
        feature_names
    ].copy()

    if prepared.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Prepared feature count mismatch.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Found: {prepared.shape[1]}"
        )

    # ------------------------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------------------------

    non_numeric = prepared.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            "Non-numeric inference features:\n"
            + "\n".join(non_numeric)
        )

    # ------------------------------------------------------------------------
    # Development-only median imputation
    # ------------------------------------------------------------------------

    missing_before = int(
        prepared.isna()
        .sum()
        .sum()
    )

    if missing_before > 0:

        prepared = prepared.fillna(
            imputation_medians
        )

    missing_after = int(
        prepared.isna()
        .sum()
        .sum()
    )

    if missing_after > 0:
        raise ValueError(
            "Inference features still contain missing values "
            "after development-only median imputation."
        )

    # ------------------------------------------------------------------------
    # Infinite-value validation
    # ------------------------------------------------------------------------

    if np.isinf(
        prepared.to_numpy(
            dtype=float
        )
    ).any():
        raise ValueError(
            "Inference features contain infinite values."
        )

    return prepared


# ============================================================================
# PROBABILITY → LOGIT
# ============================================================================

def probability_to_logit(
    probability,
) -> np.ndarray:

    probability = np.asarray(
        probability,
        dtype=float,
    )

    probability = np.clip(
        probability,
        1e-7,
        1 - 1e-7,
    )

    return np.log(
        probability
        / (1 - probability)
    )


# ============================================================================
# CALIBRATION
# ============================================================================

def calibrate_probability(
    calibrator,
    raw_probability,
) -> np.ndarray:

    logits = probability_to_logit(
        raw_probability
    )

    calibrated = (
        calibrator.predict_proba(
            logits.reshape(-1, 1)
        )[:, 1]
    )

    return np.clip(
        calibrated,
        0.0,
        1.0,
    )


# ============================================================================
# PREDICTION
# ============================================================================

def predict(
    features: pd.DataFrame,
) -> np.ndarray:

    (
        model,
        calibrator,
        _,
        _,
    ) = load_inference_engine()

    prepared = prepare_features(
        features
    )

    raw_probability = (
        model.predict_proba(
            prepared
        )[:, 1]
    )

    return calibrate_probability(
        calibrator,
        raw_probability,
    )


# ============================================================================
# RISK LEVEL
# ============================================================================

def assign_risk_level(
    probability: float,
) -> str:

    if probability < PRIMARY_THRESHOLD:
        return "Below Primary Threshold"

    if probability < VERY_HIGH_THRESHOLD:
        return "High"

    return "Very High"


# ============================================================================
# CUSTOMER FEATURE LOOKUP
# ============================================================================

def load_customer_features(
    customer_id,
) -> pd.DataFrame:

    data = pd.read_parquet(
       CUSTOMER_FEATURE_FILE
    )
    if "id" not in data.columns:
        raise ValueError(
           "customer_features.parquet has no id column."
        )

    matches = data[
        data["id"] == customer_id
    ]

    if len(matches) == 0:
        raise ValueError(
            f"Customer ID {customer_id} not found."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Customer ID {customer_id} appears more than once."
        )

    (
        _,
        _,
        feature_names,
        _,
    ) = load_inference_engine()

    return matches[
        feature_names
    ].copy()


# ============================================================================
# SCORE CUSTOMER
# ============================================================================

def score_existing_customer(
    customer_id,
) -> dict:

    features = load_customer_features(
        customer_id
    )

    missing_count = int(
        features.isna()
        .sum()
        .sum()
    )

    probability = float(
        predict(features)[0]
    )

    return {
        "id": customer_id,
        "calibrated_churn_probability": probability,
        "risk_score": probability * 100.0,
        "risk_level": assign_risk_level(
            probability
        ),
        "primary_threshold": PRIMARY_THRESHOLD,
        "above_primary_threshold": (
            probability >= PRIMARY_THRESHOLD
        ),
        "missing_values_before_imputation": (
            missing_count
        ),
    }


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("CHURNIQ — INFERENCE ENGINE SMOKE TEST")
    print("=" * 80)

    (
        model,
        calibrator,
        feature_names,
        imputation_medians,
    ) = load_inference_engine()

    print(
        f"Model             : "
        f"{type(model).__name__}"
    )

    print(
        f"Calibrator        : "
        f"{type(calibrator).__name__}"
    )

    print(
        f"Frozen features   : "
        f"{len(feature_names)}"
    )

    print(
        f"Primary threshold : "
        f"{PRIMARY_THRESHOLD:.2f}"
    )

    print(
       f"Imputation source : "
       f"deployment_data/imputation_medians.csv"
    )

    print(
        f"Median values     : "
        f"{len(imputation_medians)}"
    )

    ids = pd.read_parquet(
        CUSTOMER_FEATURE_FILE,
        columns=["id"],
    )

    customer_id = ids.iloc[0]["id"]

    result = score_existing_customer(
        customer_id
    )

    print(
        f"\nCustomer ID       : "
        f"{result['id']}"
    )

    print(
        f"Missing before    : "
        f"{result['missing_values_before_imputation']}"
    )

    print(
        f"Calibrated prob.  : "
        f"{result['calibrated_churn_probability']:.6f}"
    )

    print(
        f"Risk score        : "
        f"{result['risk_score']:.2f}"
    )

    print(
        f"Risk level        : "
        f"{result['risk_level']}"
    )

    print(
        f"Above threshold   : "
        f"{result['above_primary_threshold']}"
    )

    print()
    print("=" * 80)
    print("INFERENCE ENGINE SMOKE TEST: PASS")
    print("=" * 80)