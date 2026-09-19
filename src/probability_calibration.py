"""
ChurnIQ — Step 7.11
Probability Calibration & Reliability Analysis

Purpose
-------
Evaluate whether the selected tuned XGBoost model produces probabilities
that are reliable enough to interpret as churn-risk probabilities.

Methodology
-----------
1. Load the frozen 169-feature development dataset.
2. Load the exact Step 7.8 selected XGBoost parameters.
3. Generate fresh 5-fold OOF predictions for the exact tuned XGBoost model.
4. Compare:
      - Uncalibrated
      - Sigmoid / Platt scaling
      - Isotonic regression
   using development OOF data only.
5. Select the calibration method using:
      1) Brier score
      2) Log loss
      3) Expected Calibration Error (ECE)
6. Fit the selected calibrator on all development OOF predictions.
7. Apply it to the untouched Step 7.9 validation predictions.
8. Compare validation calibration before vs after.
9. Save reliability curves, predictions, metrics, metadata and calibrator.
10. Keep threshold optimization separate for Step 7.12.

Governance
----------
- Frozen 169-feature schema.
- Development data only for calibration-method selection.
- Fresh OOF predictions are generated for the exact tuned XGB model.
- Step 7.9 validation data is used only for final confirmation.
- Test data is never loaded.
- No threshold optimization.
- No feature selection.
- No resampling.
- No class weighting.
- No SHAP.
- No additional hyperparameter tuning.
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from xgboost import XGBClassifier


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 5
EXPECTED_FEATURE_COUNT = 169
EXPECTED_VALIDATION_ROWS = 14000

FINAL_MODEL_NAME = "XGB_02_SLOWER_MORE_TREES"

# Step 7.9 uses this display/model label.
VALIDATION_MODEL_NAME = "XGBoost Tuned"

CALIBRATION_METHODS = [
    "uncalibrated",
    "sigmoid",
    "isotonic",
]

CALIBRATION_BINS = 10


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

X_DEV_PATH = PROCESSED_DIR / "X_dev.csv"
Y_DEV_PATH = PROCESSED_DIR / "y_dev.csv"

X_VALIDATION_PATH = PROCESSED_DIR / "X_validation.csv"
Y_VALIDATION_PATH = PROCESSED_DIR / "y_validation.csv"

SELECTED_PARAMS_PATH = (
    REPORTS_DIR / "07_8_selected_model_parameters.json"
)

VALIDATION_PREDICTIONS_PATH = (
    REPORTS_DIR / "07_9_validation_predictions.csv"
)


# =============================================================================
# OUTPUT ARTIFACTS
# =============================================================================

OOF_PREDICTIONS_PATH = (
    REPORTS_DIR / "07_11_tuned_xgb_oof_predictions.csv"
)

OOF_FOLD_RESULTS_PATH = (
    REPORTS_DIR / "07_11_tuned_xgb_oof_fold_results.csv"
)

CALIBRATION_METHOD_FOLD_RESULTS_PATH = (
    REPORTS_DIR / "07_11_calibration_method_fold_results.csv"
)

CALIBRATION_METHOD_COMPARISON_PATH = (
    REPORTS_DIR / "07_11_calibration_method_comparison.csv"
)

VALIDATION_COMPARISON_PATH = (
    REPORTS_DIR / "07_11_validation_calibration_comparison.csv"
)

RELIABILITY_CURVE_PATH = (
    REPORTS_DIR / "07_11_reliability_curve.csv"
)

CALIBRATED_VALIDATION_PREDICTIONS_PATH = (
    REPORTS_DIR / "07_11_calibrated_validation_predictions.csv"
)

METADATA_PATH = (
    REPORTS_DIR / "07_11_calibration_metadata.json"
)

REPORT_PATH = (
    REPORTS_DIR / "07_11_probability_calibration_report.md"
)

CALIBRATOR_PATH = (
    MODELS_DIR / "xgb_probability_calibrator.joblib"
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def print_header(text: str) -> None:
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def ensure_output_directories() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names for robust matching while preserving
    the original dataframe separately.
    """
    result = df.copy()
    result.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in result.columns
    ]
    return result


def assert_binary_target(y: pd.Series, name: str) -> None:
    values = set(pd.Series(y).dropna().unique().tolist())

    if values != {0, 1}:
        raise ValueError(
            f"{name} must contain exactly binary values {{0, 1}}. "
            f"Found: {sorted(values)}"
        )


def assert_no_missing_or_infinite(
    X: pd.DataFrame,
    name: str,
) -> None:
    missing = int(X.isna().sum().sum())

    numeric = X.select_dtypes(include=[np.number])

    if numeric.shape[1] != X.shape[1]:
        raise ValueError(
            f"{name} contains non-numeric columns. "
            "All frozen model features must be numeric."
        )

    infinite = int(
        np.isinf(numeric.to_numpy(dtype=float)).sum()
    )

    if missing > 0:
        raise ValueError(
            f"{name} contains {missing:,} missing values."
        )

    if infinite > 0:
        raise ValueError(
            f"{name} contains {infinite:,} infinite values."
        )


def safe_float(value):
    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def safe_metric_function(
    func,
    y_true,
    probabilities,
):
    try:
        value = float(func(y_true, probabilities))

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


# =============================================================================
# STEP 7.8 PARAMETER LOADING
# =============================================================================

def load_selected_xgb_parameters() -> dict:
    """
    Load the exact selected candidate and parameters from Step 7.8.

    Actual Step 7.8 JSON structure:

    {
        "selected_candidate": "XGB_02_SLOWER_MORE_TREES",
        "selected_parameters": {...}
    }
    """

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

    selected_candidate = payload.get("selected_candidate")

    if selected_candidate != FINAL_MODEL_NAME:
        raise ValueError(
            "Step 7.8 selected candidate mismatch.\n"
            f"Expected: {FINAL_MODEL_NAME}\n"
            f"Found:    {selected_candidate}"
        )

    selected_parameters = payload.get("selected_parameters")

    if not isinstance(selected_parameters, dict):
        raise ValueError(
            "Step 7.8 JSON does not contain a valid "
            "'selected_parameters' dictionary."
        )

    required_parameters = {
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

    missing_parameters = (
        required_parameters
        - set(selected_parameters.keys())
    )

    if missing_parameters:
        raise ValueError(
            "Selected XGB parameter artifact is incomplete. "
            f"Missing: {sorted(missing_parameters)}"
        )

    return {
        key: selected_parameters[key]
        for key in sorted(selected_parameters.keys())
    }


# =============================================================================
# XGBOOST MODEL
# =============================================================================

def build_tuned_xgb(parameters: dict) -> XGBClassifier:
    """
    Build the exact Step 7.8 tuned XGBoost configuration.

    Only the selected hyperparameters come from the artifact.
    Fixed training settings are kept explicit for reproducibility.
    """

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

    return XGBClassifier(**model_parameters)


# =============================================================================
# CALIBRATION HELPERS
# =============================================================================

def clip_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Prevent numerical issues in log-loss calculations.
    """
    return np.clip(
        np.asarray(probabilities, dtype=float),
        1e-7,
        1 - 1e-7,
    )


def probability_to_logit(
    probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = clip_probabilities(probabilities)

    return np.log(
        probabilities / (1 - probabilities)
    )


def fit_sigmoid_calibrator(
    probabilities: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """
    Platt scaling implemented as logistic regression on model logits.
    """

    logits = probability_to_logit(probabilities)

    calibrator = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        logits.reshape(-1, 1),
        y,
    )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:

    logits = probability_to_logit(probabilities)

    calibrated = calibrator.predict_proba(
        logits.reshape(-1, 1)
    )[:, 1]

    return clip_probabilities(calibrated)


def fit_isotonic_calibrator(
    probabilities: np.ndarray,
    y: np.ndarray,
) -> IsotonicRegression:

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probabilities,
        y,
    )

    return calibrator


def apply_isotonic_calibrator(
    calibrator: IsotonicRegression,
    probabilities: np.ndarray,
) -> np.ndarray:

    calibrated = calibrator.predict(
        probabilities
    )

    return clip_probabilities(calibrated)


def fit_calibrator(
    method: str,
    probabilities: np.ndarray,
    y: np.ndarray,
):
    if method == "uncalibrated":
        return None

    if method == "sigmoid":
        return fit_sigmoid_calibrator(
            probabilities,
            y,
        )

    if method == "isotonic":
        return fit_isotonic_calibrator(
            probabilities,
            y,
        )

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


def apply_calibrator(
    method: str,
    calibrator,
    probabilities: np.ndarray,
) -> np.ndarray:

    if method == "uncalibrated":
        return clip_probabilities(probabilities)

    if method == "sigmoid":
        return apply_sigmoid_calibrator(
            calibrator,
            probabilities,
        )

    if method == "isotonic":
        return apply_isotonic_calibrator(
            calibrator,
            probabilities,
        )

    raise ValueError(
        f"Unknown calibration method: {method}"
    )


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> float:

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        bin_y = y_true[mask]
        bin_p = probabilities[mask]

        confidence = float(np.mean(bin_p))
        accuracy = float(np.mean(bin_y))

        weight = len(bin_y) / total

        ece += weight * abs(
            confidence - accuracy
        )

    return float(ece)


def calibration_slope_intercept(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float | None, float | None]:

    """
    Estimate calibration intercept and slope using logistic regression
    on predicted logits.

    Ideal:
        intercept ≈ 0
        slope ≈ 1
    """

    logits = probability_to_logit(probabilities)

    model = LogisticRegression(
        fit_intercept=True,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    try:
        model.fit(
            logits.reshape(-1, 1),
            y_true,
        )

        intercept = float(model.intercept_[0])
        slope = float(model.coef_[0][0])

        return slope, intercept

    except Exception:
        return None, None


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict:

    probabilities = clip_probabilities(probabilities)

    pr_auc = safe_metric_function(
        average_precision_score,
        y_true,
        probabilities,
    )

    roc_auc = safe_metric_function(
        roc_auc_score,
        y_true,
        probabilities,
    )

    brier = safe_metric_function(
        brier_score_loss,
        y_true,
        probabilities,
    )

    try:
        loss = float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        )

        if not math.isfinite(loss):
            loss = None

    except Exception:
        loss = None

    ece = expected_calibration_error(
        y_true,
        probabilities,
    )

    slope, intercept = calibration_slope_intercept(
        y_true,
        probabilities,
    )

    return {
        "brier_score": safe_float(brier),
        "log_loss": safe_float(loss),
        "ece": safe_float(ece),
        "pr_auc": safe_float(pr_auc),
        "roc_auc": safe_float(roc_auc),
        "calibration_slope": safe_float(slope),
        "calibration_intercept": safe_float(intercept),
        "mean_predicted_probability": safe_float(
            np.mean(probabilities)
        ),
        "actual_churn_rate": safe_float(
            np.mean(y_true)
        ),
    }


# =============================================================================
# RELIABILITY CURVE
# =============================================================================

def reliability_curve_dataframe(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:

    y_true = np.asarray(y_true)
    probabilities = clip_probabilities(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "method": method,
                    "bin": i + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "sample_count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_churn_rate": np.nan,
                }
            )

            continue

        rows.append(
            {
                "method": method,
                "bin": i + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_count": count,
                "mean_predicted_probability": float(
                    np.mean(probabilities[mask])
                ),
                "observed_churn_rate": float(
                    np.mean(y_true[mask])
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SPEARMAN RANK CORRELATION
# =============================================================================

def ranking_correlation(
    original: np.ndarray,
    calibrated: np.ndarray,
) -> float | None:

    try:
        correlation = spearmanr(
            original,
            calibrated,
        ).statistic

        return safe_float(correlation)

    except Exception:
        return None


# =============================================================================
# FRESH TUNED-XGB OOF PREDICTIONS
# =============================================================================

def generate_tuned_xgb_oof(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    parameters: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print()
    print(
        "Generating fresh 5-fold OOF predictions "
        "for exact tuned XGB..."
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probability = np.full(
        len(X_dev),
        np.nan,
        dtype=float,
    )

    fold_rows = []

    X_values = X_dev.to_numpy(dtype=float)
    y_values = y_dev.to_numpy(dtype=int)

    for fold_number, (
        train_index,
        oof_index,
    ) in enumerate(
        splitter.split(
            X_values,
            y_values,
        ),
        start=1,
    ):

        print(
            f"  Fold {fold_number}/{N_SPLITS}"
        )

        X_train = X_values[train_index]
        y_train = y_values[train_index]

        X_oof = X_values[oof_index]
        y_oof = y_values[oof_index]

        model = build_tuned_xgb(
            parameters
        )

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_oof
        )[:, 1]

        probabilities = clip_probabilities(
            probabilities
        )

        oof_probability[oof_index] = probabilities

        metrics = calculate_metrics(
            y_oof,
            probabilities,
        )

        fold_rows.append(
            {
                "fold": fold_number,
                "train_rows": len(train_index),
                "oof_rows": len(oof_index),
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "ece": metrics["ece"],
                "calibration_slope": metrics[
                    "calibration_slope"
                ],
                "calibration_intercept": metrics[
                    "calibration_intercept"
                ],
            }
        )

    if np.isnan(oof_probability).any():
        missing_count = int(
            np.isnan(oof_probability).sum()
        )

        raise ValueError(
            "OOF completeness failed. "
            f"Missing predictions: {missing_count}"
        )

    if np.any(
        (oof_probability < 0)
        | (oof_probability > 1)
    ):
        raise ValueError(
            "OOF probability bounds failed."
        )

    oof_df = pd.DataFrame(
        {
            "development_row_id": np.arange(
                len(X_dev)
            ),
            "actual_churn": y_values,
            "tuned_xgb_oof_probability": oof_probability,
        }
    )

    fold_df = pd.DataFrame(
        fold_rows
    )

    return oof_df, fold_df


# =============================================================================
# CALIBRATION METHOD COMPARISON
# =============================================================================

def compare_calibration_methods(
    probabilities: np.ndarray,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print()
    print(
        "Comparing calibration methods using "
        "development OOF data..."
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []

    for method in CALIBRATION_METHODS:

        print(
            f"  Method: {method}"
        )

        for fold_number, (
            train_index,
            test_index,
        ) in enumerate(
            splitter.split(
                probabilities.reshape(-1, 1),
                y,
            ),
            start=1,
        ):

            p_train = probabilities[train_index]
            y_train = y[train_index]

            p_test = probabilities[test_index]
            y_test = y[test_index]

            calibrator = fit_calibrator(
                method,
                p_train,
                y_train,
            )

            calibrated_test = apply_calibrator(
                method,
                calibrator,
                p_test,
            )

            metrics = calculate_metrics(
                y_test,
                calibrated_test,
            )

            fold_rows.append(
                {
                    "method": method,
                    "fold": fold_number,
                    **metrics,
                }
            )

    fold_df = pd.DataFrame(
        fold_rows
    )

    summary_rows = []

    for method in CALIBRATION_METHODS:

        method_df = fold_df[
            fold_df["method"] == method
        ]

        summary_rows.append(
            {
                "method": method,
                "brier_mean": method_df[
                    "brier_score"
                ].mean(),
                "brier_std": method_df[
                    "brier_score"
                ].std(ddof=1),
                "log_loss_mean": method_df[
                    "log_loss"
                ].mean(),
                "log_loss_std": method_df[
                    "log_loss"
                ].std(ddof=1),
                "ece_mean": method_df[
                    "ece"
                ].mean(),
                "ece_std": method_df[
                    "ece"
                ].std(ddof=1),
                "pr_auc_mean": method_df[
                    "pr_auc"
                ].mean(),
                "pr_auc_std": method_df[
                    "pr_auc"
                ].std(ddof=1),
                "roc_auc_mean": method_df[
                    "roc_auc"
                ].mean(),
                "roc_auc_std": method_df[
                    "roc_auc"
                ].std(ddof=1),
                "calibration_slope_mean": method_df[
                    "calibration_slope"
                ].mean(),
                "calibration_intercept_mean": method_df[
                    "calibration_intercept"
                ].mean(),
            }
        )

    comparison_df = pd.DataFrame(
        summary_rows
    )

    # -------------------------------------------------------------------------
    # Selection hierarchy
    # -------------------------------------------------------------------------
    #
    # Primary:
    #   Brier score
    #
    # Secondary:
    #   Log loss
    #
    # Tertiary:
    #   ECE
    #
    # Lower is better for all three.
    #
    comparison_df = comparison_df.sort_values(
        by=[
            "brier_mean",
            "log_loss_mean",
            "ece_mean",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    comparison_df[
        "selection_rank"
    ] = np.arange(
        1,
        len(comparison_df) + 1,
    )

    return fold_df, comparison_df


# =============================================================================
# VALIDATION PREDICTION LOADING
# =============================================================================

def identify_validation_columns(
    validation_df: pd.DataFrame,
) -> tuple[str, str, str]:

    normalized = normalize_columns(
        validation_df
    )

    columns = set(
        normalized.columns
    )

    model_candidates = [
        "model",
        "model_name",
    ]

    probability_candidates = [
        "predicted_churn_probability",
        "predicted_probability",
        "churn_probability",
        "probability",
    ]

    target_candidates = [
        "actual_churn",
        "churn_probability",
        "target",
        "y_true",
    ]

    model_col = next(
        (
            col
            for col in model_candidates
            if col in columns
        ),
        None,
    )

    probability_col = next(
        (
            col
            for col in probability_candidates
            if col in columns
        ),
        None,
    )

    target_col = next(
        (
            col
            for col in target_candidates
            if col in columns
        ),
        None,
    )

    if model_col is None:
        raise ValueError(
            "Could not identify the model column in "
            f"{VALIDATION_PREDICTIONS_PATH}.\n"
            f"Available columns: {sorted(columns)}"
        )

    if probability_col is None:
        raise ValueError(
            "Could not identify the probability column in "
            f"{VALIDATION_PREDICTIONS_PATH}.\n"
            f"Available columns: {sorted(columns)}"
        )

    if target_col is None:
        raise ValueError(
            "Could not identify the target column in "
            f"{VALIDATION_PREDICTIONS_PATH}.\n"
            f"Available columns: {sorted(columns)}"
        )

    return (
        model_col,
        probability_col,
        target_col,
    )


def load_untouched_validation_predictions():
    """
    Load Step 7.9 predictions.

    IMPORTANT:
    The actual Step 7.9 file uses:
        model = "XGBoost Tuned"

    while Step 7.8 uses:
        selected_candidate = "XGB_02_SLOWER_MORE_TREES"

    This function explicitly separates those two identifiers.
    """

    if not VALIDATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Missing Step 7.9 validation predictions:\n"
            f"{VALIDATION_PREDICTIONS_PATH}"
        )

    original_df = pd.read_csv(
        VALIDATION_PREDICTIONS_PATH
    )

    normalized_df = normalize_columns(
        original_df
    )

    (
        model_col,
        probability_col,
        target_col,
    ) = identify_validation_columns(
        original_df
    )

    print(
        f"Model column       : {model_col}"
    )
    print(
        f"Probability column : {probability_col}"
    )
    print(
        f"Target column      : {target_col}"
    )

    model_values = (
        normalized_df[model_col]
        .astype(str)
        .str.strip()
    )

    available_models = sorted(
        model_values.unique().tolist()
    )

    print()
    print(
        "Validation model labels found:"
    )

    for value in available_models:
        print(
            f"  - {value}"
        )

    # -------------------------------------------------------------------------
    # Explicit Step 7.9 model mapping.
    # -------------------------------------------------------------------------

    selected_mask = (
        model_values
        == VALIDATION_MODEL_NAME.lower()
    )

    selected_df = normalized_df[
        selected_mask
    ].copy()

    if selected_df.empty:

        # Robust fallback:
        # case-insensitive exact comparison.
        selected_df = normalized_df[
            model_values.str.casefold()
            == VALIDATION_MODEL_NAME.casefold()
        ].copy()

    if selected_df.empty:
        raise ValueError(
            "Could not find the Step 7.9 tuned XGB validation "
            "predictions.\n\n"
            f"Expected Step 7.9 model label:\n"
            f"  {VALIDATION_MODEL_NAME}\n\n"
            "Available model labels:\n"
            + "\n".join(
                f"  - {value}"
                for value in available_models
            )
        )

    if len(selected_df) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            "Unexpected number of Step 7.9 tuned XGB "
            "validation rows.\n"
            f"Expected: {EXPECTED_VALIDATION_ROWS}\n"
            f"Found:    {len(selected_df)}"
        )

    probabilities = pd.to_numeric(
        selected_df[probability_col],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    y = pd.to_numeric(
        selected_df[target_col],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if np.isnan(probabilities).any():
        raise ValueError(
            "Step 7.9 validation probabilities contain "
            "missing/non-numeric values."
        )

    if np.isnan(y).any():
        raise ValueError(
            "Step 7.9 validation target contains "
            "missing/non-numeric values."
        )

    y = y.astype(int)

    assert_binary_target(
        pd.Series(y),
        "Step 7.9 validation target",
    )

    if np.any(
        (probabilities < 0)
        | (probabilities > 1)
    ):
        raise ValueError(
            "Step 7.9 validation probabilities are outside "
            "[0, 1]."
        )

    result_df = selected_df.copy()

    result_df["uncalibrated_probability"] = (
        clip_probabilities(
            probabilities
        )
    )

    result_df["actual_churn"] = y

    return (
        result_df,
        probabilities,
        y,
        model_col,
        probability_col,
        target_col,
    )


# =============================================================================
# VALIDATION COMPARISON
# =============================================================================

def build_validation_comparison(
    y_validation: np.ndarray,
    uncalibrated: np.ndarray,
    calibrated: np.ndarray,
    selected_method: str,
) -> pd.DataFrame:

    uncalibrated_metrics = calculate_metrics(
        y_validation,
        uncalibrated,
    )

    calibrated_metrics = calculate_metrics(
        y_validation,
        calibrated,
    )

    ranking_corr = ranking_correlation(
        uncalibrated,
        calibrated,
    )

    rows = []

    rows.append(
        {
            "stage": "before_calibration",
            "method": "uncalibrated",
            **uncalibrated_metrics,
            "ranking_spearman_vs_uncalibrated": 1.0,
            "selected_method": selected_method,
        }
    )

    rows.append(
        {
            "stage": "after_calibration",
            "method": selected_method,
            **calibrated_metrics,
            "ranking_spearman_vs_uncalibrated": ranking_corr,
            "selected_method": selected_method,
        }
    )

    comparison_df = pd.DataFrame(
        rows
    )

    return comparison_df


# =============================================================================
# METADATA
# =============================================================================

def build_metadata(
    parameters: dict,
    selected_method: str,
    calibration_comparison: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    oof_df: pd.DataFrame,
) -> dict:

    selected_row = calibration_comparison[
        calibration_comparison["method"]
        == selected_method
    ].iloc[0]

    validation_after = validation_comparison[
        validation_comparison["stage"]
        == "after_calibration"
    ].iloc[0]

    validation_before = validation_comparison[
        validation_comparison["stage"]
        == "before_calibration"
    ].iloc[0]

    return {
        "step": "7.11",
        "title": (
            "Probability Calibration & Reliability Analysis"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "expected_validation_rows": EXPECTED_VALIDATION_ROWS,

        "selected_model": "XGBoost",
        "selected_candidate": FINAL_MODEL_NAME,
        "validation_model_label": VALIDATION_MODEL_NAME,

        "calibration_methods_evaluated": (
            CALIBRATION_METHODS
        ),

        "calibration_selection_metric_priority": [
            "Brier score",
            "Log loss",
            "ECE",
        ],

        "selected_calibration_method": selected_method,

        "selected_method_oof_brier": safe_float(
            selected_row["brier_mean"]
        ),
        "selected_method_oof_log_loss": safe_float(
            selected_row["log_loss_mean"]
        ),
        "selected_method_oof_ece": safe_float(
            selected_row["ece_mean"]
        ),

        "validation_brier_before": safe_float(
            validation_before["brier_score"]
        ),
        "validation_brier_after": safe_float(
            validation_after["brier_score"]
        ),

        "validation_log_loss_before": safe_float(
            validation_before["log_loss"]
        ),
        "validation_log_loss_after": safe_float(
            validation_after["log_loss"]
        ),

        "validation_ece_before": safe_float(
            validation_before["ece"]
        ),
        "validation_ece_after": safe_float(
            validation_after["ece"]
        ),

        "validation_pr_auc_before": safe_float(
            validation_before["pr_auc"]
        ),
        "validation_pr_auc_after": safe_float(
            validation_after["pr_auc"]
        ),

        "validation_roc_auc_before": safe_float(
            validation_before["roc_auc"]
        ),
        "validation_roc_auc_after": safe_float(
            validation_after["roc_auc"]
        ),

        "ranking_spearman_after_calibration": safe_float(
            validation_after[
                "ranking_spearman_vs_uncalibrated"
            ]
        ),

        "oof_rows": int(len(oof_df)),

        "governance": {
            "feature_freeze_active": True,
            "fresh_oof_for_exact_tuned_model": True,
            "validation_used_for_method_selection": False,
            "validation_used_for_calibrator_fitting": False,
            "test_data_loaded": False,
            "threshold_optimization": False,
            "feature_selection": False,
            "hyperparameter_tuning": False,
            "resampling": False,
            "class_weighting": False,
            "shap_analysis": False,
        },

        "xgb_parameters": parameters,
    }


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    metadata: dict,
    calibration_comparison: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    oof_fold_results: pd.DataFrame,
) -> None:

    selected_method = metadata[
        "selected_calibration_method"
    ]

    validation_before = validation_comparison[
        validation_comparison["stage"]
        == "before_calibration"
    ].iloc[0]

    validation_after = validation_comparison[
        validation_comparison["stage"]
        == "after_calibration"
    ].iloc[0]

    lines = []

    lines.append(
        "# Step 7.11 — Probability Calibration & Reliability Analysis"
    )
    lines.append("")
    lines.append(
        "## Objective"
    )
    lines.append(
        "Assess whether the selected tuned XGBoost model produces "
        "probabilities that are sufficiently reliable for downstream "
        "customer-risk interpretation."
    )
    lines.append("")

    lines.append(
        "## Selected Model"
    )
    lines.append(
        f"- Model: XGBoost"
    )
    lines.append(
        f"- Step 7.8 candidate: `{FINAL_MODEL_NAME}`"
    )
    lines.append(
        f"- Step 7.8 development CV PR-AUC: "
        f"{0.776913:.6f}"
    )
    lines.append(
        f"- Step 7.9 validation model label: "
        f"`{VALIDATION_MODEL_NAME}`"
    )
    lines.append("")

    lines.append(
        "## Methodology"
    )
    lines.append(
        "- Generated fresh 5-fold OOF predictions for the exact "
        "Step 7.8 tuned XGBoost configuration."
    )
    lines.append(
        "- Calibration-method selection used development OOF data only."
    )
    lines.append(
        "- Compared uncalibrated, sigmoid and isotonic calibration."
    )
    lines.append(
        "- Primary calibration metrics: Brier score, log loss and ECE."
    )
    lines.append(
        "- PR-AUC and ROC-AUC were monitored to ensure predictive ranking "
        "was not materially degraded."
    )
    lines.append(
        "- The Step 7.9 validation set remained untouched until final "
        "confirmation."
    )
    lines.append("")

    lines.append(
        "## Calibration Method Selection"
    )

    for _, row in calibration_comparison.iterrows():

        lines.append(
            f"### {row['method'].title()}"
        )
        lines.append(
            f"- Mean Brier score: {row['brier_mean']:.6f}"
        )
        lines.append(
            f"- Mean log loss: {row['log_loss_mean']:.6f}"
        )
        lines.append(
            f"- Mean ECE: {row['ece_mean']:.6f}"
        )
        lines.append(
            f"- Mean PR-AUC: {row['pr_auc_mean']:.6f}"
        )
        lines.append(
            f"- Mean ROC-AUC: {row['roc_auc_mean']:.6f}"
        )
        lines.append(
            f"- Selection rank: {int(row['selection_rank'])}"
        )
        lines.append("")

    lines.append(
        f"**Selected calibration method: `{selected_method}`**"
    )
    lines.append("")

    lines.append(
        "## Validation Reliability Results"
    )

    lines.append(
        f"| Metric | Before | After |"
    )
    lines.append(
        "|---|---:|---:|"
    )
    lines.append(
        f"| Brier score | "
        f"{validation_before['brier_score']:.6f} | "
        f"{validation_after['brier_score']:.6f} |"
    )
    lines.append(
        f"| Log loss | "
        f"{validation_before['log_loss']:.6f} | "
        f"{validation_after['log_loss']:.6f} |"
    )
    lines.append(
        f"| ECE | "
        f"{validation_before['ece']:.6f} | "
        f"{validation_after['ece']:.6f} |"
    )
    lines.append(
        f"| PR-AUC | "
        f"{validation_before['pr_auc']:.6f} | "
        f"{validation_after['pr_auc']:.6f} |"
    )
    lines.append(
        f"| ROC-AUC | "
        f"{validation_before['roc_auc']:.6f} | "
        f"{validation_after['roc_auc']:.6f} |"
    )
    lines.append(
        f"| Calibration slope | "
        f"{validation_before['calibration_slope']:.6f} | "
        f"{validation_after['calibration_slope']:.6f} |"
    )
    lines.append(
        f"| Calibration intercept | "
        f"{validation_before['calibration_intercept']:.6f} | "
        f"{validation_after['calibration_intercept']:.6f} |"
    )
    lines.append("")

    lines.append(
        "## Ranking Preservation"
    )

    ranking = validation_after[
        "ranking_spearman_vs_uncalibrated"
    ]

    lines.append(
        f"- Spearman correlation between uncalibrated and calibrated "
        f"validation probabilities: `{ranking:.6f}`"
    )
    lines.append(
        "Calibration should preserve customer-risk ordering as much as "
        "possible because thresholding and Top-K prioritization depend "
        "on ranking as well as probability quality."
    )
    lines.append("")

    lines.append(
        "## Governance Controls"
    )
    lines.append(
        "- Frozen 169-feature schema: PASS"
    )
    lines.append(
        "- Fresh OOF predictions for exact tuned model: PASS"
    )
    lines.append(
        "- Validation used for calibration-method selection: NO"
    )
    lines.append(
        "- Validation used for calibrator fitting: NO"
    )
    lines.append(
        "- Test data loaded: NO"
    )
    lines.append(
        "- Additional hyperparameter tuning: NO"
    )
    lines.append(
        "- Threshold optimization: NO"
    )
    lines.append(
        "- Feature selection: NO"
    )
    lines.append(
        "- Resampling: NO"
    )
    lines.append(
        "- Class weighting: NO"
    )
    lines.append(
        "- SHAP analysis: NO"
    )
    lines.append("")

    lines.append(
        "## Interpretation"
    )
    lines.append(
        f"The selected calibration method is `{selected_method}` based "
        "on development-only calibration evidence. Validation results "
        "are treated as final confirmation rather than as a method "
        "selection mechanism."
    )
    lines.append("")
    lines.append(
        "Calibration does not change the underlying churn label. It "
        "attempts to make predicted probabilities better aligned with "
        "observed outcome frequencies."
    )
    lines.append("")
    lines.append(
        "Probability calibration is intentionally separated from "
        "threshold optimization. The operating threshold will be "
        "evaluated independently in Step 7.12."
    )
    lines.append("")

    lines.append(
        "## Artifacts"
    )
    lines.append(
        f"- `{OOF_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{OOF_FOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{CALIBRATION_METHOD_FOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{CALIBRATION_METHOD_COMPARISON_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{VALIDATION_COMPARISON_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{RELIABILITY_CURVE_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{CALIBRATED_VALIDATION_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{CALIBRATOR_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )
    lines.append(
        f"- `{METADATA_PATH.relative_to(PROJECT_ROOT)}"
        "`"
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# QUALITY GATE
# =============================================================================

def run_quality_gate(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    oof_df: pd.DataFrame,
    oof_fold_results: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    calibrated_validation: np.ndarray,
) -> list[str]:

    checks = []

    # -------------------------------------------------------------------------
    # Feature count
    # -------------------------------------------------------------------------

    if X_dev.shape[1] == EXPECTED_FEATURE_COUNT:
        checks.append(
            "Frozen 169 features"
        )
    else:
        raise ValueError(
            "Feature-count quality gate failed."
        )

    # -------------------------------------------------------------------------
    # Development rows
    # -------------------------------------------------------------------------

    if len(X_dev) == 55999:
        checks.append(
            "Development row count"
        )
    else:
        raise ValueError(
            "Development row-count quality gate failed."
        )

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    assert_binary_target(
        y_dev,
        "Development target",
    )

    checks.append(
        "Development target binary"
    )

    # -------------------------------------------------------------------------
    # OOF
    # -------------------------------------------------------------------------

    if len(oof_df) != len(X_dev):
        raise ValueError(
            "OOF row count does not match development data."
        )

    if oof_df[
        "tuned_xgb_oof_probability"
    ].isna().any():
        raise ValueError(
            "OOF completeness failed."
        )

    checks.append(
        "Fresh tuned-XGB OOF completeness"
    )

    # -------------------------------------------------------------------------
    # Fold count
    # -------------------------------------------------------------------------

    if (
        len(oof_fold_results)
        == N_SPLITS
    ):
        checks.append(
            "Exactly 5 OOF folds"
        )
    else:
        raise ValueError(
            "OOF fold-count quality gate failed."
        )

    # -------------------------------------------------------------------------
    # Calibration methods
    # -------------------------------------------------------------------------

    methods = set(
        calibration_comparison[
            "method"
        ].tolist()
    )

    if methods == set(
        CALIBRATION_METHODS
    ):
        checks.append(
            "All calibration methods evaluated"
        )
    else:
        raise ValueError(
            "Calibration-method coverage failed."
        )

    # -------------------------------------------------------------------------
    # Selected method
    # -------------------------------------------------------------------------

    selected_rows = calibration_comparison[
        calibration_comparison[
            "selection_rank"
        ]
        == 1
    ]

    if len(selected_rows) != 1:
        raise ValueError(
            "Calibration selection did not produce "
            "exactly one winner."
        )

    checks.append(
        "Single calibration winner"
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if len(validation_comparison) != 2:
        raise ValueError(
            "Validation comparison should contain exactly "
            "before/after rows."
        )

    checks.append(
        "Validation comparison complete"
    )

    # -------------------------------------------------------------------------
    # Probability bounds
    # -------------------------------------------------------------------------

    if np.any(
        (calibrated_validation < 0)
        | (calibrated_validation > 1)
    ):
        raise ValueError(
            "Calibrated validation probabilities are outside [0, 1]."
        )

    checks.append(
        "Calibrated probability bounds"
    )

    # -------------------------------------------------------------------------
    # Finite metrics
    # -------------------------------------------------------------------------

    metric_columns = [
        "brier_mean",
        "log_loss_mean",
        "ece_mean",
        "pr_auc_mean",
        "roc_auc_mean",
    ]

    if calibration_comparison[
        metric_columns
    ].apply(
        lambda column: np.isfinite(
            column.astype(float)
        ).all()
    ).all():
        checks.append(
            "Calibration metrics finite"
        )
    else:
        raise ValueError(
            "Non-finite calibration metric detected."
        )

    return checks


# =============================================================================
# MAIN
# =============================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    ensure_output_directories()

    print_header(
        "STEP 7.11 — PROBABILITY CALIBRATION & RELIABILITY ANALYSIS"
    )

    # =========================================================================
    # 1. DEVELOPMENT DATA
    # =========================================================================

    print()
    print(
        "[1/10] Loading frozen development dataset..."
    )

    if not X_DEV_PATH.exists():
        raise FileNotFoundError(
            f"Missing X_dev:\n{X_DEV_PATH}"
        )

    if not Y_DEV_PATH.exists():
        raise FileNotFoundError(
            f"Missing y_dev:\n{Y_DEV_PATH}"
        )

    X_dev = pd.read_csv(
        X_DEV_PATH
    )

    y_dev = pd.read_csv(
        Y_DEV_PATH
    ).squeeze("columns")

    print(
        f"X_dev shape : {X_dev.shape}"
    )
    print(
        f"y_dev shape : {y_dev.shape}"
    )

    if X_dev.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {X_dev.shape[1]}."
        )

    print(
        "Frozen 169 features : PASS"
    )

    assert_binary_target(
        y_dev,
        "Development target",
    )

    print(
        "Binary target        : PASS"
    )

    assert_no_missing_or_infinite(
        X_dev,
        "X_dev",
    )

    print(
        "Missing values       : PASS"
    )
    print(
        "Infinite values      : PASS"
    )

    # =========================================================================
    # 2. PARAMETERS
    # =========================================================================

    print()
    print(
        "[2/10] Loading exact Step 7.8 selected XGB parameters..."
    )

    parameters = load_selected_xgb_parameters()

    print(
        f"Selected candidate : {FINAL_MODEL_NAME}"
    )

    if "learning_rate" in parameters:
        print(
            f"Step 7.8 CV PR-AUC  : 0.776913"
        )

    print(
        "Parameters:"
    )

    for key in [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]:
        print(
            f"  {key}: {parameters[key]}"
        )

    print(
        "Parameter artifact validation : PASS"
    )

    # =========================================================================
    # 3. FRESH OOF
    # =========================================================================

    print()
    print(
        "[3/10] Generating tuned XGB 5-fold OOF predictions..."
    )

    oof_df, oof_fold_results = (
        generate_tuned_xgb_oof(
            X_dev,
            y_dev,
            parameters,
        )
    )

    print(
        "OOF completeness : PASS"
    )

    print(
        "OOF probability range : "
        f"{oof_df['tuned_xgb_oof_probability'].min():.6f} — "
        f"{oof_df['tuned_xgb_oof_probability'].max():.6f}"
    )

    # =========================================================================
    # SAVE OOF
    # =========================================================================

    oof_df.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    oof_fold_results.to_csv(
        OOF_FOLD_RESULTS_PATH,
        index=False,
    )

    # =========================================================================
    # 4. CALIBRATION METHODS
    # =========================================================================

    print()
    print(
        "[4/10] Comparing calibration methods using development OOF data..."
    )

    oof_probability = (
        oof_df[
            "tuned_xgb_oof_probability"
        ]
        .to_numpy(dtype=float)
    )

    y_oof = (
        oof_df[
            "actual_churn"
        ]
        .to_numpy(dtype=int)
    )

    (
        calibration_method_fold_results,
        calibration_comparison,
    ) = compare_calibration_methods(
        oof_probability,
        y_oof,
    )

    calibration_method_fold_results.to_csv(
        CALIBRATION_METHOD_FOLD_RESULTS_PATH,
        index=False,
    )

    calibration_comparison.to_csv(
        CALIBRATION_METHOD_COMPARISON_PATH,
        index=False,
    )

    print()
    print(
        "Calibration method comparison:"
    )

    display_columns = [
        "method",
        "brier_mean",
        "log_loss_mean",
        "ece_mean",
        "pr_auc_mean",
        "roc_auc_mean",
        "calibration_slope_mean",
        "calibration_intercept_mean",
        "selection_rank",
    ]

    print(
        calibration_comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    selected_method = (
        calibration_comparison.iloc[0][
            "method"
        ]
    )

    print()
    print(
        f"Selected calibration method : {selected_method}"
    )

    # =========================================================================
    # 5. FIT FINAL CALIBRATOR
    # =========================================================================

    print()
    print(
        "[5/10] Fitting selected calibrator on all development OOF predictions..."
    )

    final_calibrator = fit_calibrator(
        selected_method,
        oof_probability,
        y_oof,
    )

    if selected_method == "uncalibrated":
        calibrator_artifact = {
            "method": "uncalibrated",
            "calibrator": None,
        }
    else:
        calibrator_artifact = {
            "method": selected_method,
            "calibrator": final_calibrator,
        }

    joblib.dump(
        calibrator_artifact,
        CALIBRATOR_PATH,
    )

    if not CALIBRATOR_PATH.exists():
        raise RuntimeError(
            "Calibrator artifact was not created."
        )

    print(
        "Calibrator artifact : PASS"
    )

    # =========================================================================
    # 6. LOAD UNTOUCHED STEP 7.9 VALIDATION
    # =========================================================================

    print()
    print(
        "[6/10] Loading untouched Step 7.9 validation predictions..."
    )

    (
        validation_df,
        validation_probability,
        y_validation,
        model_col,
        probability_col,
        target_col,
    ) = load_untouched_validation_predictions()

    print()
    print(
        f"Matched Step 7.9 model : {VALIDATION_MODEL_NAME}"
    )

    print(
        f"Validation rows        : {len(validation_df)}"
    )

    print(
        "Validation model match : PASS"
    )

    # =========================================================================
    # 7. APPLY CALIBRATION
    # =========================================================================

    print()
    print(
        "[7/10] Applying selected calibration to validation probabilities..."
    )

    calibrated_validation = apply_calibrator(
        selected_method,
        final_calibrator,
        validation_probability,
    )

    if len(calibrated_validation) != len(
        validation_probability
    ):
        raise ValueError(
            "Calibrated validation row count mismatch."
        )

    if np.any(
        (calibrated_validation < 0)
        | (calibrated_validation > 1)
    ):
        raise ValueError(
            "Calibrated validation probabilities outside [0,1]."
        )

    print(
        "Calibrated probability bounds : PASS"
    )

    validation_df[
        "calibrated_churn_probability"
    ] = calibrated_validation

    validation_df[
        "selected_calibration_method"
    ] = selected_method

    # =========================================================================
    # 8. VALIDATION COMPARISON
    # =========================================================================

    print()
    print(
        "[8/10] Evaluating validation calibration before vs after..."
    )

    validation_comparison = (
        build_validation_comparison(
            y_validation,
            validation_probability,
            calibrated_validation,
            selected_method,
        )
    )

    validation_comparison.to_csv(
        VALIDATION_COMPARISON_PATH,
        index=False,
    )

    ranking_corr = ranking_correlation(
        validation_probability,
        calibrated_validation,
    )

    print()
    print(
        "Validation calibration comparison:"
    )

    print(
        validation_comparison[
            [
                "stage",
                "method",
                "brier_score",
                "log_loss",
                "ece",
                "pr_auc",
                "roc_auc",
                "calibration_slope",
                "calibration_intercept",
                "ranking_spearman_vs_uncalibrated",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(
        "Ranking Spearman correlation : "
        f"{ranking_corr:.6f}"
        if ranking_corr is not None
        else "Ranking Spearman correlation : unavailable"
    )

    # =========================================================================
    # 9. RELIABILITY CURVE
    # =========================================================================

    print()
    print(
        "[9/10] Building reliability-curve data..."
    )

    reliability_before = reliability_curve_dataframe(
        y_validation,
        validation_probability,
        "uncalibrated",
    )

    reliability_after = reliability_curve_dataframe(
        y_validation,
        calibrated_validation,
        selected_method,
    )

    reliability_df = pd.concat(
        [
            reliability_before,
            reliability_after,
        ],
        ignore_index=True,
    )

    reliability_df.to_csv(
        RELIABILITY_CURVE_PATH,
        index=False,
    )

    validation_df[
        [
            "uncalibrated_probability",
            "calibrated_churn_probability",
            "actual_churn",
            "selected_calibration_method",
        ]
    ].to_csv(
        CALIBRATED_VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    print(
        "Reliability data : PASS"
    )

    # =========================================================================
    # 10. METADATA + REPORT + QUALITY GATE
    # =========================================================================

    print()
    print(
        "[10/10] Finalizing Step 7.11 artifacts and quality gate..."
    )

    metadata = build_metadata(
        parameters,
        selected_method,
        calibration_comparison,
        validation_comparison,
        oof_df,
    )

    quality_checks = run_quality_gate(
        X_dev,
        y_dev,
        oof_df,
        oof_fold_results,
        calibration_comparison,
        validation_comparison,
        calibrated_validation,
    )

    metadata[
        "quality_gate_checks"
    ] = quality_checks

    metadata[
        "quality_gate"
    ] = "PASS"

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    write_report(
        metadata,
        calibration_comparison,
        validation_comparison,
        oof_fold_results,
    )

    print()
    print(
        "Quality gate checks:"
    )

    for check in quality_checks:
        print(
            f"  - {check} : PASS"
        )

    print()
    print(
        "Artifacts:"
    )

    artifact_paths = [
        OOF_PREDICTIONS_PATH,
        OOF_FOLD_RESULTS_PATH,
        CALIBRATION_METHOD_FOLD_RESULTS_PATH,
        CALIBRATION_METHOD_COMPARISON_PATH,
        VALIDATION_COMPARISON_PATH,
        RELIABILITY_CURVE_PATH,
        CALIBRATED_VALIDATION_PREDICTIONS_PATH,
        CALIBRATOR_PATH,
        METADATA_PATH,
        REPORT_PATH,
    ]

    for artifact in artifact_paths:
        print(
            f"  - {artifact.relative_to(PROJECT_ROOT)}"
        )

    print()
    print(
        "=" * 80
    )
    print(
        "STEP 7.11 COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        f"Selected calibration method : {selected_method}"
    )

    print(
        "Validation Brier score       : "
        f"{validation_comparison.iloc[1]['brier_score']:.6f}"
    )

    print(
        "Validation Log Loss          : "
        f"{validation_comparison.iloc[1]['log_loss']:.6f}"
    )

    print(
        "Validation ECE               : "
        f"{validation_comparison.iloc[1]['ece']:.6f}"
    )

    print(
        "Validation PR-AUC            : "
        f"{validation_comparison.iloc[1]['pr_auc']:.6f}"
    )

    print(
        "Validation ROC-AUC           : "
        f"{validation_comparison.iloc[1]['roc_auc']:.6f}"
    )

    if ranking_corr is not None:
        print(
            "Ranking Spearman             : "
            f"{ranking_corr:.6f}"
        )

    print()
    print(
        "Governance checks:"
    )

    print(
        "  - Fresh OOF for tuned XGB             : YES"
    )
    print(
        "  - Validation used for method selection: NO"
    )
    print(
        "  - Test data loaded                    : NO"
    )
    print(
        "  - Threshold optimization              : NO"
    )
    print(
        "  - Additional tuning                   : NO"
    )
    print(
        "  - Feature selection                   : NO"
    )
    print(
        "  - SHAP analysis                       : NO"
    )
    print(
        "  - Feature freeze                      : ACTIVE"
    )

    print()
    print(
        "QUALITY GATE: PASS"
    )


if __name__ == "__main__":
    main()