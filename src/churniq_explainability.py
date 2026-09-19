"""
ChurnIQ — Customer SHAP Explainability

Phase 14.3
----------
Customer-level explainability using SHAP.

Purpose:
- Explain why a customer's model prediction is higher or lower.
- Reuse the locked ChurnIQ model and frozen 169-feature schema.
- Reproduce the Phase 10 SHAP methodology at customer level.
- Provide structured explanation output for the Streamlit application.

Important:
- SHAP values describe model-associated signals, not causal effects.
- SHAP values are calculated in the model's raw-output / log-odds space.
- Calibrated churn probability remains the customer-facing probability.
- No model, calibration, threshold, or feature-selection logic is changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# IMPORT LOCKED INFERENCE ENGINE
# =============================================================================

from churniq_inference import (
    load_customer_features,
    load_inference_engine,
)


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

EXPECTED_FEATURE_COUNT = 169

SHAP_TOP_N = 5

ADDITIVITY_TOLERANCE = 1e-4


# =============================================================================
# EXPLAINABILITY ENGINE LOADER
# =============================================================================

def load_explainability_engine():
    """
    Load the locked ChurnIQ inference artifacts and validate
    the frozen feature schema.
    """

    (
        model,
        calibrator,
        feature_names,
        imputation_medians,
    ) = load_inference_engine()

    feature_names = list(feature_names)

    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Frozen feature count mismatch: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"found {len(feature_names)}."
        )

    if len(imputation_medians) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Imputation median count mismatch: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"found {len(imputation_medians)}."
        )

    if list(imputation_medians.index) != feature_names:
        raise ValueError(
            "Imputation median index does not match "
            "the frozen feature order."
        )

    return (
        model,
        calibrator,
        feature_names,
        imputation_medians,
    )


# =============================================================================
# SHAP EXPLAINER CACHE
# =============================================================================

_EXPLAINER = None


def get_shap_explainer(model):
    """
    Create the TreeExplainer once and reuse it.
    """

    global _EXPLAINER

    if _EXPLAINER is None:
        _EXPLAINER = shap.TreeExplainer(model)

    return _EXPLAINER


# =============================================================================
# CUSTOMER FEATURE PREPARATION
# =============================================================================

def prepare_customer_for_shap(customer_id: int):
    """
    Load and prepare one customer using the exact frozen
    inference feature schema and development-only medians.

    Returns
    -------
    X_customer : pandas.DataFrame
        One-row feature matrix in frozen feature order.

    missing_before_imputation : int
        Number of missing frozen feature values before imputation.
    """

    (
        model,
        calibrator,
        feature_names,
        imputation_medians,
    ) = load_explainability_engine()

    customer = load_customer_features(customer_id)

    if customer.empty:
        raise ValueError(
            f"Customer ID {customer_id} was not found."
        )

    # -------------------------------------------------------------------------
    # Frozen schema validation
    # -------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in customer.columns
    ]

    if missing_features:
        raise ValueError(
            "Customer feature record is missing frozen features: "
            f"{missing_features[:10]}"
        )

    # -------------------------------------------------------------------------
    # Exact feature order
    # -------------------------------------------------------------------------

    X_customer = customer[feature_names].copy()

    if X_customer.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Customer feature count mismatch: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"found {X_customer.shape[1]}."
        )

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    for feature in feature_names:
        X_customer[feature] = pd.to_numeric(
            X_customer[feature],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Missingness before imputation
    # -------------------------------------------------------------------------

    missing_before_imputation = int(
        X_customer.isna().sum().sum()
    )

    # -------------------------------------------------------------------------
    # Development-only median imputation
    # -------------------------------------------------------------------------

    X_customer = X_customer.fillna(imputation_medians)

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    if X_customer.isna().any().any():
        raise ValueError(
            "Customer SHAP features still contain missing values "
            "after development-only median imputation."
        )

    if not all(
        pd.api.types.is_numeric_dtype(X_customer[column])
        for column in X_customer.columns
    ):
        raise ValueError(
            "Customer SHAP feature matrix contains non-numeric columns."
        )

    values = X_customer.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Customer SHAP feature matrix contains "
            "infinite or non-finite values."
        )

    if list(X_customer.columns) != feature_names:
        raise ValueError(
            "Customer SHAP feature order does not match "
            "the frozen feature order."
        )

    return X_customer, missing_before_imputation


# =============================================================================
# SHAP EXPECTED VALUE
# =============================================================================

def get_base_value(explainer) -> float:
    """
    Return the scalar SHAP base value for binary classification.
    """

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)

        if expected_value.size == 1:
            return float(expected_value[0])

        # Binary XGBoost explanations can expose multiple outputs.
        # Use the positive-class output when present.
        return float(expected_value[-1])

    return float(expected_value)


# =============================================================================
# RAW MODEL OUTPUT
# =============================================================================

def get_raw_model_output(model, X_customer: pd.DataFrame) -> float:
    """
    Get the XGBoost raw model margin / log-odds.

    This is the correct output space for validating the default
    TreeExplainer SHAP decomposition.
    """

    try:
        raw_output = model.predict(
            X_customer,
            output_margin=True,
        )

        raw_output = np.asarray(raw_output).reshape(-1)

        if raw_output.size != 1:
            raise ValueError(
                "Unexpected raw model output shape: "
                f"{raw_output.shape}"
            )

        return float(raw_output[0])

    except TypeError:
        # Compatibility fallback for XGBoost versions that do not
        # expose output_margin through XGBClassifier.predict().
        booster = model.get_booster()

        import xgboost as xgb

        matrix = xgb.DMatrix(
            X_customer,
            feature_names=list(X_customer.columns),
        )

        raw_output = booster.predict(
            matrix,
            output_margin=True,
        )

        raw_output = np.asarray(raw_output).reshape(-1)

        if raw_output.size != 1:
            raise ValueError(
                "Unexpected booster raw output shape: "
                f"{raw_output.shape}"
            )

        return float(raw_output[0])


# =============================================================================
# CUSTOMER SHAP EXPLANATION
# =============================================================================

def explain_customer(customer_id: int):
    """
    Generate a complete customer-level SHAP explanation.

    Returns
    -------
    dict
        Structured explanation suitable for Streamlit.
    """

    (
        model,
        calibrator,
        feature_names,
        imputation_medians,
    ) = load_explainability_engine()

    X_customer, missing_before_imputation = (
        prepare_customer_for_shap(customer_id)
    )

    explainer = get_shap_explainer(model)

    # -------------------------------------------------------------------------
    # SHAP calculation
    # -------------------------------------------------------------------------

    shap_result = explainer(
        X_customer,
        check_additivity=True,
    )

    shap_values = np.asarray(shap_result.values)

    # -------------------------------------------------------------------------
    # Handle possible SHAP output dimensions
    # -------------------------------------------------------------------------

    if shap_values.ndim == 3:

        # Expected shape:
        # (rows, features, outputs)

        if shap_values.shape[0] != 1:
            raise ValueError(
                "Unexpected SHAP row count: "
                f"{shap_values.shape}"
            )

        if shap_values.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Unexpected SHAP feature count: "
                f"{shap_values.shape}"
            )

        # Positive churn class
        shap_values = shap_values[0, :, -1]

    elif shap_values.ndim == 2:

        if shap_values.shape != (
            1,
            EXPECTED_FEATURE_COUNT,
        ):
            raise ValueError(
                "Unexpected SHAP matrix shape: "
                f"{shap_values.shape}"
            )

        shap_values = shap_values[0]

    else:
        raise ValueError(
            "Unsupported SHAP output dimensions: "
            f"{shap_values.shape}"
        )

    # -------------------------------------------------------------------------
    # SHAP base value
    # -------------------------------------------------------------------------

    base_value = get_base_value(explainer)

    # -------------------------------------------------------------------------
    # SHAP reconstruction
    #
    # For the default TreeExplainer configuration this is in the
    # model's raw-output / log-odds space.
    # -------------------------------------------------------------------------

    reconstructed_raw_output = float(
        base_value + np.sum(shap_values)
    )

    # -------------------------------------------------------------------------
    # Independent model raw output
    # -------------------------------------------------------------------------

    raw_model_output = get_raw_model_output(
        model,
        X_customer,
    )

    # -------------------------------------------------------------------------
    # Correct additivity validation
    # -------------------------------------------------------------------------

    additivity_error = float(
        abs(
            reconstructed_raw_output
            - raw_model_output
        )
    )

    if additivity_error > ADDITIVITY_TOLERANCE:
        raise ValueError(
            "SHAP additivity validation failed. "
            f"Error={additivity_error:.10f}, "
            f"Tolerance={ADDITIVITY_TOLERANCE:.10f}"
        )

    # -------------------------------------------------------------------------
    # Raw model probability
    # -------------------------------------------------------------------------

    raw_model_probability = float(
        model.predict_proba(X_customer)[0, 1]
    )

    # -------------------------------------------------------------------------
    # Customer feature values
    # -------------------------------------------------------------------------

    feature_values = X_customer.iloc[0].to_dict()

    # -------------------------------------------------------------------------
    # Structured SHAP table
    # -------------------------------------------------------------------------

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": shap_values,
            "feature_value": [
                feature_values[feature]
                for feature in feature_names
            ],
        }
    )

    explanation_df["absolute_shap"] = (
        explanation_df["shap_value"].abs()
    )

    # -------------------------------------------------------------------------
    # Top positive drivers
    # -------------------------------------------------------------------------

    top_positive = (
        explanation_df[
            explanation_df["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False,
        )
        .head(SHAP_TOP_N)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Top negative drivers
    # -------------------------------------------------------------------------

    top_negative = (
        explanation_df[
            explanation_df["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .head(SHAP_TOP_N)
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Return structured result
    # -------------------------------------------------------------------------

    return {
        "customer_id": int(customer_id),

        "feature_count": EXPECTED_FEATURE_COUNT,

        "missing_values_before_imputation": (
            missing_before_imputation
        ),

        "raw_model_output": raw_model_output,

        "raw_model_probability": raw_model_probability,

        "base_value": base_value,

        "reconstructed_raw_output": (
            reconstructed_raw_output
        ),

        "additivity_error": additivity_error,

        "additivity_tolerance": ADDITIVITY_TOLERANCE,

        "explanation": explanation_df,

        "top_positive": top_positive,

        "top_negative": top_negative,

        "interpretation_note": (
            "SHAP values represent model-associated signals "
            "that push the model prediction higher or lower. "
            "They should not be interpreted as causal effects."
        ),

        "probability_space_note": (
            "SHAP values are calculated in the model's "
            "raw-output/log-odds space. The customer-facing "
            "churn probability is produced separately by the "
            "locked ChurnIQ inference and calibration pipeline."
        ),

        "calibration_note": (
            "The raw XGBoost probability is shown only for "
            "technical validation. Customer-facing probability "
            "should use the locked calibrated inference output."
        ),
    }


# =============================================================================
# EXPLANATION QUALITY VALIDATION
# =============================================================================

def validate_explanation(result: dict):
    """
    Validate the structure and numerical integrity of a
    customer-level SHAP explanation.
    """

    if int(result["feature_count"]) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Explanation feature count validation failed."
        )

    explanation_df = result["explanation"]

    if len(explanation_df) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Explanation row count validation failed: "
            f"expected {EXPECTED_FEATURE_COUNT}, "
            f"found {len(explanation_df)}."
        )

    required_columns = {
        "feature",
        "shap_value",
        "feature_value",
        "absolute_shap",
    }

    if not required_columns.issubset(
        explanation_df.columns
    ):
        raise ValueError(
            "Explanation DataFrame is missing required columns."
        )

    if explanation_df["shap_value"].isna().any():
        raise ValueError(
            "SHAP explanation contains NaN SHAP values."
        )

    if explanation_df["feature_value"].isna().any():
        raise ValueError(
            "SHAP explanation contains NaN feature values."
        )

    if explanation_df["absolute_shap"].isna().any():
        raise ValueError(
            "SHAP explanation contains NaN absolute SHAP values."
        )

    if len(result["top_positive"]) > SHAP_TOP_N:
        raise ValueError(
            "Too many positive SHAP drivers returned."
        )

    if len(result["top_negative"]) > SHAP_TOP_N:
        raise ValueError(
            "Too many negative SHAP drivers returned."
        )

    additivity_error = float(
        result["additivity_error"]
    )

    if additivity_error > ADDITIVITY_TOLERANCE:
        raise ValueError(
            "Stored SHAP additivity error exceeds tolerance."
        )

    raw_output = float(
        result["raw_model_output"]
    )

    reconstructed_output = float(
        result["reconstructed_raw_output"]
    )

    if not np.isfinite(raw_output):
        raise ValueError(
            "Raw model output is not finite."
        )

    if not np.isfinite(reconstructed_output):
        raise ValueError(
            "Reconstructed SHAP output is not finite."
        )

    return True


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("CHURNIQ — CUSTOMER SHAP EXPLANATION SMOKE TEST")
    print("=" * 80)
    print()

    test_customer_id = 0

    print(
        f"Analyzing customer: {test_customer_id}"
    )
    print()

    result = explain_customer(
        test_customer_id
    )

    validate_explanation(result)

    print(
        f"Customer ID                 : "
        f"{result['customer_id']}"
    )

    print(
        f"Frozen feature count        : "
        f"{result['feature_count']}"
    )

    print(
        f"Missing before imputation  : "
        f"{result['missing_values_before_imputation']}"
    )

    print(
        f"Raw model probability      : "
        f"{result['raw_model_probability']:.6f}"
    )

    print(
        f"SHAP base value            : "
        f"{result['base_value']:.6f}"
    )

    print(
        f"Raw model output           : "
        f"{result['raw_model_output']:.6f}"
    )

    print(
        f"Reconstructed SHAP output  : "
        f"{result['reconstructed_raw_output']:.6f}"
    )

    print(
        f"Additivity validation error: "
        f"{result['additivity_error']:.10f}"
    )

    print()
    print("Top factors increasing churn risk:")
    print("-" * 70)

    for _, row in result["top_positive"].iterrows():
        print(
            f"  {row['feature']:<45}"
            f"{row['shap_value']:>12.6f}"
        )

    print()
    print("Top factors decreasing churn risk:")
    print("-" * 70)

    for _, row in result["top_negative"].iterrows():
        print(
            f"  {row['feature']:<45}"
            f"{row['shap_value']:>12.6f}"
        )

    print()
    print("=" * 80)
    print("CUSTOMER SHAP EXPLANATION SMOKE TEST: PASS")
    print("=" * 80)
    print()