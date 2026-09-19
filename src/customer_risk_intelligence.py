from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =============================================================================
# ChurnIQ — Phase 11
# Customer Risk Intelligence
#
# Purpose:
# Build the complete 69,999-customer risk intelligence dataset using:
#
#   Tuned XGBoost
#        ↓
#   Sigmoid-calibrated churn probability
#        ↓
#   Risk score
#        ↓
#   Risk level
#        ↓
#   Primary threshold flag
#        ↓
#   Full-universe risk ranking
#
# Governance:
# - No model retraining
# - No tuning
# - No recalibration
# - No feature selection
# - No test data
# - Exact Step 7.11 sigmoid calibrator reused
# - Exact Step 7.12 primary threshold reused
# - 69,999-customer universe
#
# Phase boundaries:
# - Customer value / revenue exposure → Phase 12
# - Retention prioritization / ROI → Phase 13
#
# Important distinction:
# - calibrated_churn_probability = model-estimated churn probability
# - actual_churn = observed historical outcome
#
# Risk score is NOT a separate model.
# It is calibrated probability × 100 for communication.
# =============================================================================


warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_TRAIN_FILE = (
    ROOT
    / "data"
    / "raw"
    / "train.csv"
)

OOF_FILE = (
    ROOT
    / "reports"
    / "07_11_tuned_xgb_oof_predictions.csv"
)

VALIDATION_FILE = (
    ROOT
    / "reports"
    / "07_11_calibrated_validation_predictions.csv"
)

CALIBRATOR_FILE = (
    ROOT
    / "models"
    / "xgb_probability_calibrator.joblib"
)

THRESHOLD_FILE = (
    ROOT
    / "reports"
    / "07_12_threshold_selection.json"
)

X_DEV_FILE = (
    ROOT
    / "data"
    / "processed"
    / "X_dev.csv"
)

Y_DEV_FILE = (
    ROOT
    / "data"
    / "processed"
    / "y_dev.csv"
)

Y_VALIDATION_FILE = (
    ROOT
    / "data"
    / "processed"
    / "y_validation.csv"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    ROOT
    / "reports"
)


# =============================================================================
# 2. CONSTANTS
# =============================================================================

EXPECTED_TOTAL_CUSTOMERS = 69999
EXPECTED_DEVELOPMENT_CUSTOMERS = 55999
EXPECTED_VALIDATION_CUSTOMERS = 14000

PRIMARY_THRESHOLD_EXPECTED = 0.10

VERY_HIGH_THRESHOLD = 0.50

EXPECTED_RISK_LEVELS = {
    "Below Primary Threshold",
    "High",
    "Very High",
}

EXPECTED_CORE_COLUMNS = [
    "risk_rank",
    "risk_percentile",
    "id",
    "calibrated_churn_probability",
    "risk_score",
    "risk_level",
    "primary_threshold",
    "above_primary_threshold",
    "scoring_population",
    "actual_churn",
]


# =============================================================================
# 3. UTILITY FUNCTIONS
# =============================================================================

def check(name, status, detail=""):
    message = f"{name:<48} {status}"

    if detail:
        message += f" — {detail}"

    print(message)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def validate_probability(values, name):
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if (
        (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError(
            f"{name} contains values outside [0, 1]."
        )


def apply_sigmoid_calibration(raw_probability, calibrator):
    """
    Reproduce the exact Step 7.11 sigmoid calibration.

    Step 7.11 fitted logistic regression on:

        logit(raw_probability)

    and used predict_proba() to obtain the calibrated
    churn probability.
    """

    p = np.asarray(
        raw_probability,
        dtype=float
    )

    validate_probability(
        p,
        "Raw probability"
    )

    # Protect against log(0) and log(infinity).
    p = np.clip(
        p,
        1e-15,
        1 - 1e-15
    )

    logit_p = np.log(
        p / (1 - p)
    )

    calibrated = (
        calibrator
        .predict_proba(
            logit_p.reshape(-1, 1)
        )[:, 1]
    )

    validate_probability(
        calibrated,
        "Calibrated probability"
    )

    return calibrated


def assign_risk_level(probability):
    """
    Locked Phase 11 risk framework.

    Very High:
        >= 0.50

    High:
        0.10 to <0.50

    Below Primary Threshold:
        <0.10
    """

    if probability >= VERY_HIGH_THRESHOLD:
        return "Very High"

    if probability >= PRIMARY_THRESHOLD_EXPECTED:
        return "High"

    return "Below Primary Threshold"


def calculate_risk_percentile(probabilities):
    """
    Full-universe percentile.

    Highest-risk customer receives approximately 100.
    Lowest-risk customer receives approximately 0.

    Percentile is descriptive only.
    It does not change the risk probability or threshold.
    """

    probabilities = pd.Series(
        probabilities,
        dtype=float
    )

    percentile = (
        probabilities
        .rank(
            method="average",
            pct=True
        )
        * 100
    )

    return percentile.to_numpy()


def validate_required_columns(
    dataframe,
    required_columns,
    dataset_name
):
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing}"
        )


# =============================================================================
# 4. MAIN WORKFLOW
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 11 — CUSTOMER RISK INTELLIGENCE")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — REQUIRED ARTIFACT CHECKS
    # =========================================================================

    print("\n[1/14] Checking required artifacts...")

    required_files = [
        (RAW_TRAIN_FILE, "Raw train dataset"),
        (OOF_FILE, "Tuned XGBoost OOF predictions"),
        (VALIDATION_FILE, "Calibrated validation predictions"),
        (CALIBRATOR_FILE, "Sigmoid probability calibrator"),
        (THRESHOLD_FILE, "Threshold selection artifact"),
        (X_DEV_FILE, "Development feature matrix"),
        (Y_DEV_FILE, "Development target"),
        (Y_VALIDATION_FILE, "Validation target"),
    ]

    for path, description in required_files:
        require_file(
            path,
            description
        )

    check(
        "Required artifacts",
        "PASS",
        f"{len(required_files)} files available"
    )

    # =========================================================================
    # STEP 2 — LOAD SOURCE ARTIFACTS
    # =========================================================================

    print("\n[2/14] Loading source artifacts...")

    raw_train = pd.read_csv(
        RAW_TRAIN_FILE
    )

    oof = pd.read_csv(
        OOF_FILE
    )

    validation_predictions = pd.read_csv(
        VALIDATION_FILE
    )

    X_dev = pd.read_csv(
        X_DEV_FILE
    )

    y_dev = pd.read_csv(
        Y_DEV_FILE
    )

    y_validation = pd.read_csv(
        Y_VALIDATION_FILE
    )

    check(
        "Raw train shape",
        "PASS",
        str(raw_train.shape)
    )

    check(
        "OOF shape",
        "PASS",
        str(oof.shape)
    )

    check(
        "Validation prediction shape",
        "PASS",
        str(validation_predictions.shape)
    )

    check(
        "X_dev shape",
        "PASS",
        str(X_dev.shape)
    )

    # =========================================================================
    # STEP 3 — VALIDATE CUSTOMER UNIVERSE
    # =========================================================================

    print("\n[3/14] Validating complete customer universe...")

    if len(raw_train) != EXPECTED_TOTAL_CUSTOMERS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_CUSTOMERS:,} customers, "
            f"found {len(raw_train):,}."
        )

    validate_required_columns(
        raw_train,
        [
            "id",
            "churn_probability",
        ],
        "Raw train dataset"
    )

    if raw_train["id"].isna().any():
        raise ValueError(
            "Missing customer IDs detected."
        )

    if raw_train["id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    raw_target = (
        raw_train["churn_probability"]
        .astype(int)
    )

    if not set(raw_target.unique()).issubset({0, 1}):
        raise ValueError(
            "churn_probability contains values other than 0 and 1."
        )

    check(
        "Customer universe",
        "PASS",
        "69,999 unique customers"
    )

    check(
        "Historical target validity",
        "PASS",
        "Binary 0/1"
    )

    # =========================================================================
    # STEP 4 — VALIDATE DEVELOPMENT OOF PREDICTIONS
    # =========================================================================

    print("\n[4/14] Validating development OOF predictions...")

    required_oof_columns = [
        "development_row_id",
        "actual_churn",
        "tuned_xgb_oof_probability",
    ]

    validate_required_columns(
        oof,
        required_oof_columns,
        "OOF prediction file"
    )

    if len(oof) != len(X_dev):
        raise ValueError(
            "OOF row count does not match X_dev."
        )

    if len(oof) != len(y_dev):
        raise ValueError(
            "OOF row count does not match y_dev."
        )

    if oof["development_row_id"].isna().any():
        raise ValueError(
            "OOF development_row_id contains missing values."
        )

    if oof["development_row_id"].duplicated().any():
        raise ValueError(
            "Duplicate development_row_id values detected in OOF."
        )

    oof_probability = (
        oof[
            "tuned_xgb_oof_probability"
        ]
        .to_numpy(dtype=float)
    )

    validate_probability(
        oof_probability,
        "OOF tuned XGBoost probability"
    )

    oof_actual = (
        oof["actual_churn"]
        .astype(int)
        .to_numpy()
    )

    y_dev_values = (
        y_dev.iloc[:, 0]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        oof_actual,
        y_dev_values
    ):
        raise ValueError(
            "OOF actual_churn does not align with y_dev."
        )

    check(
        "OOF target alignment",
        "PASS",
        f"{len(oof):,} development rows"
    )

    check(
        "OOF row identifiers",
        "PASS",
        "Unique development_row_id values"
    )

    check(
        "OOF probability bounds",
        "PASS",
        "[0, 1]"
    )

    # =========================================================================
    # STEP 5 — RECONSTRUCT FROZEN DEVELOPMENT / VALIDATION SPLIT
    # =========================================================================

    print(
        "\n[5/14] Reconstructing frozen development/validation split..."
    )

    target = (
        raw_train[
            "churn_probability"
        ]
        .astype(int)
    )

    all_indices = np.arange(
        len(raw_train)
    )

    (
        development_indices,
        validation_indices
    ) = train_test_split(
        all_indices,
        test_size=0.20,
        stratify=target,
        random_state=42
    )

    if len(development_indices) != EXPECTED_DEVELOPMENT_CUSTOMERS:
        raise ValueError(
            "Unexpected development split size."
        )

    if len(validation_indices) != EXPECTED_VALIDATION_CUSTOMERS:
        raise ValueError(
            "Unexpected validation split size."
        )

    if set(development_indices).intersection(
        set(validation_indices)
    ):
        raise ValueError(
            "Development/validation overlap detected."
        )

    check(
        "Development split",
        "PASS",
        f"{len(development_indices):,} customers"
    )

    check(
        "Validation split",
        "PASS",
        f"{len(validation_indices):,} customers"
    )

    check(
        "Split overlap",
        "PASS",
        "0 rows"
    )

    # =========================================================================
    # STEP 6 — VALIDATE RECORDED VALIDATION PREDICTIONS
    # =========================================================================

    print(
        "\n[6/14] Validating recorded validation predictions..."
    )

    required_validation_columns = [
        "uncalibrated_probability",
        "calibrated_churn_probability",
        "actual_churn",
    ]

    validate_required_columns(
        validation_predictions,
        required_validation_columns,
        "Validation prediction file"
    )

    if len(validation_predictions) != EXPECTED_VALIDATION_CUSTOMERS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_CUSTOMERS:,} "
            f"validation predictions, found "
            f"{len(validation_predictions):,}."
        )

    if len(y_validation) != EXPECTED_VALIDATION_CUSTOMERS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_CUSTOMERS:,} "
            f"validation targets, found "
            f"{len(y_validation):,}."
        )

    validation_actual = (
        validation_predictions[
            "actual_churn"
        ]
        .astype(int)
        .to_numpy()
    )

    y_validation_values = (
        y_validation.iloc[:, 0]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        validation_actual,
        y_validation_values
    ):
        raise ValueError(
            "Validation actual_churn does not align "
            "with y_validation."
        )

    validate_probability(
        validation_predictions[
            "uncalibrated_probability"
        ].to_numpy(dtype=float),
        "Validation uncalibrated probability"
    )

    validate_probability(
        validation_predictions[
            "calibrated_churn_probability"
        ].to_numpy(dtype=float),
        "Validation calibrated probability"
    )

    check(
        "Validation target alignment",
        "PASS",
        f"{EXPECTED_VALIDATION_CUSTOMERS:,} rows"
    )

    check(
        "Validation probabilities",
        "PASS",
        "[0, 1]"
    )

    # =========================================================================
    # STEP 7 — LOAD EXACT SIGMOID CALIBRATOR
    # =========================================================================

    print(
        "\n[7/14] Loading exact Step 7.11 sigmoid calibrator..."
    )

    calibrator_artifact = joblib.load(
        CALIBRATOR_FILE
    )

    if not isinstance(
        calibrator_artifact,
        dict
    ):
        raise ValueError(
            "Unexpected calibrator artifact structure."
        )

    if calibrator_artifact.get("method") != "sigmoid":
        raise ValueError(
            "Expected sigmoid calibration method."
        )

    calibrator = (
        calibrator_artifact.get("calibrator")
    )

    if calibrator is None:
        raise ValueError(
            "Sigmoid calibrator object not found."
        )

    check(
        "Calibration method",
        "PASS",
        "sigmoid"
    )

    check(
        "Calibrator source",
        "PASS",
        "Step 7.11 artifact"
    )

    # =========================================================================
    # STEP 8 — REPRODUCE OOF CALIBRATION
    # =========================================================================

    print(
        "\n[8/14] Applying exact sigmoid calibration to OOF predictions..."
    )

    oof_calibrated_probability = (
        apply_sigmoid_calibration(
            oof_probability,
            calibrator
        )
    )

    check(
        "OOF calibrated probabilities",
        "PASS",
        f"{len(oof_calibrated_probability):,} rows"
    )

    # =========================================================================
    # STEP 9 — VERIFY VALIDATION CALIBRATION
    # =========================================================================

    print(
        "\n[9/14] Verifying Step 7.11 validation calibration..."
    )

    validation_uncalibrated = (
        validation_predictions[
            "uncalibrated_probability"
        ]
        .to_numpy(dtype=float)
    )

    reproduced_validation_calibrated = (
        apply_sigmoid_calibration(
            validation_uncalibrated,
            calibrator
        )
    )

    recorded_validation_calibrated = (
        validation_predictions[
            "calibrated_churn_probability"
        ]
        .to_numpy(dtype=float)
    )

    calibration_difference = np.max(
        np.abs(
            reproduced_validation_calibrated
            - recorded_validation_calibrated
        )
    )

    if calibration_difference > 1e-10:
        raise ValueError(
            "Reproduced validation calibration does not "
            "match the recorded Step 7.11 artifact."
        )

    check(
        "Calibration reproduction",
        "PASS",
        f"max difference = {calibration_difference:.2e}"
    )

    # =========================================================================
    # STEP 10 — LOAD LOCKED PRIMARY THRESHOLD
    # =========================================================================

    print(
        "\n[10/14] Loading locked Step 7.12 primary threshold..."
    )

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        threshold_config = json.load(file)

    if "primary_operating_point" not in threshold_config:
        raise ValueError(
            "Threshold artifact does not contain "
            "'primary_operating_point'."
        )

    primary_operating_point = (
        threshold_config[
            "primary_operating_point"
        ]
    )

    if not isinstance(
        primary_operating_point,
        dict
    ):
        raise ValueError(
            "primary_operating_point is not a dictionary."
        )

    threshold_key_candidates = [
        "threshold",
        "selected_threshold",
        "primary_threshold",
    ]

    primary_threshold = None

    for key in threshold_key_candidates:

        if key in primary_operating_point:

            primary_threshold = float(
                primary_operating_point[key]
            )

            break

    if primary_threshold is None:
        raise ValueError(
            "Could not identify the selected threshold."
        )

    if not (
        0 < primary_threshold < 1
    ):
        raise ValueError(
            f"Invalid primary threshold: "
            f"{primary_threshold}"
        )

    if not np.isclose(
        primary_threshold,
        PRIMARY_THRESHOLD_EXPECTED,
        atol=1e-12
    ):
        raise ValueError(
            f"Locked threshold mismatch. "
            f"Expected {PRIMARY_THRESHOLD_EXPECTED}, "
            f"found {primary_threshold}."
        )

    if (
        threshold_config.get(
            "validation_used_for_threshold_selection"
        )
        is not False
    ):
        raise ValueError(
            "Threshold artifact indicates validation "
            "was used for threshold selection."
        )

    if (
        threshold_config.get("test_used")
        is not False
    ):
        raise ValueError(
            "Threshold artifact indicates test data was used."
        )

    check(
        "Primary threshold",
        "PASS",
        f"{primary_threshold:.6f}"
    )

    check(
        "Threshold governance",
        "PASS",
        "Validation/test restrictions satisfied"
    )

    # =========================================================================
    # STEP 11 — BUILD DEVELOPMENT CUSTOMER RISK DATASET
    # =========================================================================

    print(
        "\n[11/14] Building development customer risk dataset..."
    )

    development_customers = (
        raw_train
        .iloc[
            development_indices
        ]
        .reset_index(drop=True)
        .copy()
    )

    if len(development_customers) != len(
        oof_calibrated_probability
    ):
        raise ValueError(
            "Development customer count does not "
            "match OOF probability count."
        )

    development_customers[
        "calibrated_churn_probability"
    ] = oof_calibrated_probability

    # Explicitly label the observed historical outcome.
    development_customers[
        "actual_churn"
    ] = (
        development_customers[
            "churn_probability"
        ]
        .astype(int)
    )

    # Population label is assigned BEFORE concatenation.
    development_customers[
        "scoring_population"
    ] = "Development OOF"

    check(
        "Development customer mapping",
        "PASS",
        f"{len(development_customers):,} customers"
    )

    # =========================================================================
    # STEP 12 — BUILD VALIDATION CUSTOMER RISK DATASET
    # =========================================================================

    print(
        "\n[12/14] Building validation customer risk dataset..."
    )

    validation_customers = (
        raw_train
        .iloc[
            validation_indices
        ]
        .reset_index(drop=True)
        .copy()
    )

    if len(validation_customers) != len(
        recorded_validation_calibrated
    ):
        raise ValueError(
            "Validation customer count does not "
            "match validation probability count."
        )

    validation_customers[
        "calibrated_churn_probability"
    ] = recorded_validation_calibrated

    validation_customers[
        "actual_churn"
    ] = (
        validation_customers[
            "churn_probability"
        ]
        .astype(int)
    )

    # Population label is assigned BEFORE concatenation.
    validation_customers[
        "scoring_population"
    ] = "Validation"

    check(
        "Validation customer mapping",
        "PASS",
        f"{len(validation_customers):,} customers"
    )

    # =========================================================================
    # STEP 13 — COMBINE FULL 69,999-CUSTOMER UNIVERSE
    # =========================================================================

    print(
        "\n[13/14] Combining complete customer risk universe..."
    )

    customer_risk = pd.concat(
        [
            development_customers,
            validation_customers,
        ],
        ignore_index=True
    )

    if len(customer_risk) != EXPECTED_TOTAL_CUSTOMERS:
        raise ValueError(
            "Combined customer universe does not contain "
            "69,999 customers."
        )

    if customer_risk["id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs found after combining "
            "development and validation."
        )

    if customer_risk["id"].isna().any():
        raise ValueError(
            "Missing customer IDs found after combining."
        )

    validate_probability(
        customer_risk[
            "calibrated_churn_probability"
        ].to_numpy(),
        "Final calibrated probability"
    )

    # -------------------------------------------------------------------------
    # Risk score
    # -------------------------------------------------------------------------

    customer_risk[
        "risk_score"
    ] = (
        customer_risk[
            "calibrated_churn_probability"
        ]
        * 100
    )

    # -------------------------------------------------------------------------
    # Risk level
    # -------------------------------------------------------------------------

    customer_risk[
        "risk_level"
    ] = (
        customer_risk[
            "calibrated_churn_probability"
        ]
        .apply(assign_risk_level)
    )

    # -------------------------------------------------------------------------
    # Locked primary threshold
    # -------------------------------------------------------------------------

    customer_risk[
        "primary_threshold"
    ] = primary_threshold

    customer_risk[
        "above_primary_threshold"
    ] = (
        customer_risk[
            "calibrated_churn_probability"
        ]
        >= primary_threshold
    )

    # -------------------------------------------------------------------------
    # Deterministic full-universe ranking
    # -------------------------------------------------------------------------

    customer_risk = (
        customer_risk
        .sort_values(
            [
                "calibrated_churn_probability",
                "id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # Risk rank
    # -------------------------------------------------------------------------

    customer_risk[
        "risk_rank"
    ] = (
        np.arange(
            1,
            len(customer_risk) + 1
        )
    )

    # -------------------------------------------------------------------------
    # Full-universe risk percentile
    # -------------------------------------------------------------------------

    customer_risk[
        "risk_percentile"
    ] = calculate_risk_percentile(
        customer_risk[
            "calibrated_churn_probability"
        ]
    )

    # -------------------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------------------

    customer_risk_output = (
        customer_risk[
            EXPECTED_CORE_COLUMNS
        ]
        .copy()
    )

    check(
        "Complete customer universe",
        "PASS",
        f"{len(customer_risk_output):,} customers"
    )

    check(
        "Risk scoring",
        "PASS",
        "Calibrated probability → 0–100 risk score"
    )

    check(
        "Risk classification",
        "PASS",
        "3 locked risk levels"
    )

    check(
        "Risk ranking",
        "PASS",
        "Probability + ID deterministic ordering"
    )

    check(
        "Risk percentile",
        "PASS",
        "Full-universe descriptive percentile"
    )

    # =========================================================================
    # STEP 14 — RISK DISTRIBUTION + QUALITY GATE + OUTPUTS
    # =========================================================================

    print(
        "\n[14/14] Building risk distribution and quality reports..."
    )

    total_customers = len(
        customer_risk_output
    )

    threshold_customers = int(
        customer_risk_output[
            "above_primary_threshold"
        ].sum()
    )

    # -------------------------------------------------------------------------
    # Risk distribution
    # -------------------------------------------------------------------------

    risk_order = [
        "Below Primary Threshold",
        "High",
        "Very High",
    ]

    risk_distribution = (
        customer_risk_output
        .groupby(
            "risk_level",
            observed=False
        )
        .agg(
            customer_count=(
                "id",
                "count"
            ),
            customer_share=(
                "id",
                lambda x:
                len(x) / total_customers
            ),
            mean_churn_probability=(
                "calibrated_churn_probability",
                "mean"
            ),
            median_churn_probability=(
                "calibrated_churn_probability",
                "median"
            ),
            actual_churn_count=(
                "actual_churn",
                "sum"
            ),
            actual_churn_rate=(
                "actual_churn",
                "mean"
            ),
        )
        .reset_index()
    )

    risk_distribution[
        "risk_level"
    ] = pd.Categorical(
        risk_distribution[
            "risk_level"
        ],
        categories=risk_order,
        ordered=True
    )

    risk_distribution = (
        risk_distribution
        .sort_values("risk_level")
        .reset_index(drop=True)
    )

    total_actual_churners = int(
        customer_risk_output[
            "actual_churn"
        ].sum()
    )

    if total_actual_churners > 0:

        risk_distribution[
            "actual_churn_capture_share"
        ] = (
            risk_distribution[
                "actual_churn_count"
            ]
            / total_actual_churners
        )

    else:

        risk_distribution[
            "actual_churn_capture_share"
        ] = 0.0

    # -------------------------------------------------------------------------
    # Threshold retrospective metrics
    # -------------------------------------------------------------------------

    threshold_actual_churners = int(
        customer_risk_output.loc[
            customer_risk_output[
                "above_primary_threshold"
            ],
            "actual_churn"
        ].sum()
    )

    if total_actual_churners > 0:

        threshold_actual_churn_capture = (
            threshold_actual_churners
            / total_actual_churners
        )

    else:

        threshold_actual_churn_capture = 0.0

    if threshold_customers > 0:

        threshold_observed_churn_rate = (
            threshold_actual_churners
            / threshold_customers
        )

    else:

        threshold_observed_churn_rate = 0.0

    # -------------------------------------------------------------------------
    # Risk-level summary
    # -------------------------------------------------------------------------

    risk_level_counts = (
        customer_risk_output[
            "risk_level"
        ]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0
        )
    )

    risk_level_shares = (
        risk_level_counts
        / total_customers
    )

    # -------------------------------------------------------------------------
    # Quality checks
    # -------------------------------------------------------------------------

    quality_checks = []

    def quality(
        check_name,
        status,
        detail
    ):
        quality_checks.append(
            {
                "check": check_name,
                "status": status,
                "detail": detail,
            }
        )

    # 1. Full universe
    quality(
        "Customer universe",
        "PASS"
        if len(customer_risk_output)
        == EXPECTED_TOTAL_CUSTOMERS
        else "FAIL",
        f"{len(customer_risk_output):,} rows"
    )

    # 2. Unique IDs
    quality(
        "Unique customer IDs",
        "PASS"
        if not customer_risk_output[
            "id"
        ].duplicated().any()
        else "FAIL",
        "No duplicates"
    )

    # 3. Missing IDs
    quality(
        "Missing customer IDs",
        "PASS"
        if not customer_risk_output[
            "id"
        ].isna().any()
        else "FAIL",
        "No missing IDs"
    )

    # 4. Probability validity
    probabilities = (
        customer_risk_output[
            "calibrated_churn_probability"
        ]
        .to_numpy(dtype=float)
    )

    probability_valid = (
        np.isfinite(probabilities).all()
        and
        (probabilities >= 0).all()
        and
        (probabilities <= 1).all()
    )

    quality(
        "Calibrated probabilities",
        "PASS"
        if probability_valid
        else "FAIL",
        "[0, 1] and finite"
    )

    # 5. Risk score validity
    risk_scores = (
        customer_risk_output[
            "risk_score"
        ]
        .to_numpy(dtype=float)
    )

    risk_score_valid = (
        np.isfinite(risk_scores).all()
        and
        (risk_scores >= 0).all()
        and
        (risk_scores <= 100).all()
    )

    quality(
        "Risk scores",
        "PASS"
        if risk_score_valid
        else "FAIL",
        "0 to 100"
    )

    # 6. Risk percentile validity
    risk_percentiles = (
        customer_risk_output[
            "risk_percentile"
        ]
        .to_numpy(dtype=float)
    )

    risk_percentile_valid = (
        np.isfinite(risk_percentiles).all()
        and
        (risk_percentiles >= 0).all()
        and
        (risk_percentiles <= 100).all()
    )

    quality(
        "Risk percentiles",
        "PASS"
        if risk_percentile_valid
        else "FAIL",
        "0 to 100 and finite"
    )

    # 7. Risk level validity
    observed_levels = set(
        customer_risk_output[
            "risk_level"
        ].dropna().unique()
    )

    risk_levels_valid = (
        observed_levels == EXPECTED_RISK_LEVELS
    )

    quality(
        "Risk levels",
        "PASS"
        if risk_levels_valid
        else "FAIL",
        "All 3 locked risk levels present"
    )

    # 8. Risk classification completeness
    risk_level_missing = (
        customer_risk_output[
            "risk_level"
        ].isna().any()
    )

    quality(
        "Risk classification completeness",
        "PASS"
        if not risk_level_missing
        else "FAIL",
        "Every customer classified exactly once"
    )

    # 9. Threshold validity
    threshold_values = (
        customer_risk_output[
            "primary_threshold"
        ].unique()
    )

    threshold_valid = (
        len(threshold_values) == 1
        and
        np.isclose(
            threshold_values[0],
            primary_threshold,
            atol=1e-12
        )
    )

    quality(
        "Primary threshold",
        "PASS"
        if threshold_valid
        else "FAIL",
        f"{primary_threshold:.6f}"
    )

    # 10. Threshold logic
    expected_flags = (
        customer_risk_output[
            "calibrated_churn_probability"
        ]
        >= primary_threshold
    )

    threshold_logic_valid = (
        expected_flags
        .equals(
            customer_risk_output[
                "above_primary_threshold"
            ]
        )
    )

    quality(
        "Threshold logic",
        "PASS"
        if threshold_logic_valid
        else "FAIL",
        "Probability >= threshold"
    )

    # 11. Risk classification logic
    expected_levels = (
        customer_risk_output[
            "calibrated_churn_probability"
        ]
        .apply(assign_risk_level)
    )

    risk_logic_valid = (
        expected_levels
        .equals(
            customer_risk_output[
                "risk_level"
            ]
        )
    )

    quality(
        "Risk classification logic",
        "PASS"
        if risk_logic_valid
        else "FAIL",
        "Locked Phase 11 rules"
    )

    # 12. Ranking
    ranking_valid = (
        customer_risk_output[
            "calibrated_churn_probability"
        ]
        .is_monotonic_decreasing
    )

    quality(
        "Risk ranking",
        "PASS"
        if ranking_valid
        else "FAIL",
        "Descending calibrated probability"
    )

    # 13. Risk rank validity
    expected_ranks = np.arange(
        1,
        EXPECTED_TOTAL_CUSTOMERS + 1
    )

    actual_ranks = (
        customer_risk_output[
            "risk_rank"
        ]
        .to_numpy()
    )

    rank_valid = np.array_equal(
        actual_ranks,
        expected_ranks
    )

    quality(
        "Risk rank integrity",
        "PASS"
        if rank_valid
        else "FAIL",
        "Exactly 1 to 69,999"
    )

    # 14. Development count
    development_count = int(
        (
            customer_risk_output[
                "scoring_population"
            ]
            == "Development OOF"
        ).sum()
    )

    quality(
        "Development population",
        "PASS"
        if development_count
        == EXPECTED_DEVELOPMENT_CUSTOMERS
        else "FAIL",
        f"{development_count:,} rows"
    )

    # 15. Validation count
    validation_count = int(
        (
            customer_risk_output[
                "scoring_population"
            ]
            == "Validation"
        ).sum()
    )

    quality(
        "Validation population",
        "PASS"
        if validation_count
        == EXPECTED_VALIDATION_CUSTOMERS
        else "FAIL",
        f"{validation_count:,} rows"
    )

    # 16. Population completeness
    population_count_valid = (
        development_count
        + validation_count
        == EXPECTED_TOTAL_CUSTOMERS
    )

    quality(
        "Scoring population completeness",
        "PASS"
        if population_count_valid
        else "FAIL",
        "Development + Validation = full universe"
    )

    # 17. Actual outcome validity
    actual_churn_values = set(
        customer_risk_output[
            "actual_churn"
        ].dropna().astype(int).unique()
    )

    actual_churn_valid = (
        actual_churn_values.issubset({0, 1})
        and
        not customer_risk_output[
            "actual_churn"
        ].isna().any()
    )

    quality(
        "Actual churn outcome",
        "PASS"
        if actual_churn_valid
        else "FAIL",
        "Observed historical outcome kept separate"
    )

    # 18. Output schema
    output_schema_valid = (
        list(
            customer_risk_output.columns
        )
        == EXPECTED_CORE_COLUMNS
    )

    quality(
        "Output schema",
        "PASS"
        if output_schema_valid
        else "FAIL",
        "Locked Phase 11 customer-level schema"
    )

    # 19. No test data
    quality(
        "Test data loaded",
        "PASS",
        "NO"
    )

    # 20. Model retraining
    quality(
        "Model retraining",
        "PASS",
        "NO"
    )

    # 21. Additional tuning
    quality(
        "Additional tuning",
        "PASS",
        "NO"
    )

    # 22. Additional calibration
    quality(
        "Additional calibration",
        "PASS",
        "NO"
    )

    # 23. Feature selection
    quality(
        "Feature selection",
        "PASS",
        "NO"
    )

    # -------------------------------------------------------------------------
    # Overall status
    # -------------------------------------------------------------------------

    quality_checks_df = pd.DataFrame(
        quality_checks
    )

    overall_status = (
        "PASS"
        if (
            quality_checks_df[
                "status"
            ]
            == "PASS"
        ).all()
        else "FAIL"
    )

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    customer_risk_file = (
        PROCESSED_DIR
        / "customer_risk_intelligence.csv"
    )

    top_risk_file = (
        PROCESSED_DIR
        / "top_100_highest_risk_customers.csv"
    )

    risk_distribution_file = (
        REPORT_DIR
        / "11_customer_risk_distribution.csv"
    )

    quality_file = (
        REPORT_DIR
        / "11_customer_risk_quality_gate.csv"
    )

    metadata_file = (
        REPORT_DIR
        / "11_customer_risk_intelligence_metadata.json"
    )

    report_file = (
        REPORT_DIR
        / "11_customer_risk_intelligence_report.md"
    )

    # =========================================================================
    # SAVE FULL CUSTOMER RISK DATASET
    # =========================================================================

    customer_risk_output.to_csv(
        customer_risk_file,
        index=False
    )

    # =========================================================================
    # SAVE TOP 100 FULL-UNIVERSE RISK CUSTOMERS
    # =========================================================================

    top_100_risk = (
        customer_risk_output
        .head(100)
        .copy()
    )

    top_100_risk.to_csv(
        top_risk_file,
        index=False
    )

    # =========================================================================
    # SAVE RISK DISTRIBUTION
    # =========================================================================

    risk_distribution.to_csv(
        risk_distribution_file,
        index=False
    )

    # =========================================================================
    # SAVE QUALITY GATE
    # =========================================================================

    quality_checks_df.to_csv(
        quality_file,
        index=False
    )

    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {

        "phase":
            "11",

        "purpose":
            "Customer Risk Intelligence",

        "customer_universe":
            EXPECTED_TOTAL_CUSTOMERS,

        "development_customers":
            EXPECTED_DEVELOPMENT_CUSTOMERS,

        "validation_customers":
            EXPECTED_VALIDATION_CUSTOMERS,

        "risk_probability_source":
            "Tuned XGBoost OOF predictions + recorded validation predictions",

        "calibration_method":
            "sigmoid",

        "calibrator_source":
            "models/xgb_probability_calibrator.joblib",

        "threshold_source":
            "reports/07_12_threshold_selection.json",

        "primary_threshold":
            primary_threshold,

        "very_high_threshold":
            VERY_HIGH_THRESHOLD,

        "risk_level_definition":
            {
                "Very High":
                    ">= 0.50",

                "High":
                    ">= 0.10 and < 0.50",

                "Below Primary Threshold":
                    "< 0.10",
            },

        "risk_score_definition":
            "calibrated churn probability multiplied by 100",

        "risk_percentile_definition":
            "Percentile rank of calibrated churn probability across the full 69,999-customer universe; descriptive only",

        "risk_rank_definition":
            "Deterministic descending rank by calibrated churn probability, then customer ID",

        "actual_churn_definition":
            "Observed historical churn outcome from the original churn_probability target; not a model prediction",

        "top_100_definition":
            "100 highest calibrated churn probability customers across the full 69,999-customer universe",

        "threshold_metrics_definition":
            "Retrospective descriptive metrics comparing the locked threshold against observed historical churn; not future guarantees",

        "risk_level_counts":
            {
                "Below Primary Threshold":
                    int(
                        risk_level_counts[
                            "Below Primary Threshold"
                        ]
                    ),

                "High":
                    int(
                        risk_level_counts[
                            "High"
                        ]
                    ),

                "Very High":
                    int(
                        risk_level_counts[
                            "Very High"
                        ]
                    ),
            },

        "risk_level_shares":
            {
                "Below Primary Threshold":
                    float(
                        risk_level_shares[
                            "Below Primary Threshold"
                        ]
                    ),

                "High":
                    float(
                        risk_level_shares[
                            "High"
                        ]
                    ),

                "Very High":
                    float(
                        risk_level_shares[
                            "Very High"
                        ]
                    ),
            },

        "primary_threshold_population":
            {
                "customer_count":
                    threshold_customers,

                "customer_share":
                    threshold_customers
                    / total_customers,

                "observed_actual_churners":
                    threshold_actual_churners,

                "observed_churn_rate_within_threshold_population":
                    threshold_observed_churn_rate,

                "observed_churn_capture_share":
                    threshold_actual_churn_capture,
            },

        "governance":
            {
                "model_retraining":
                    False,

                "additional_tuning":
                    False,

                "additional_calibration":
                    False,

                "feature_selection":
                    False,

                "test_data_loaded":
                    False,

                "feature_freeze":
                    True,
            },

        "phase_boundary":
            {
                "customer_value":
                    "Phase 12",

                "revenue_exposure":
                    "Phase 12",

                "retention_prioritization":
                    "Phase 13",

                "roi":
                    "Phase 13",
            },

        "quality_gate":
            overall_status,
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    # =========================================================================
    # MARKDOWN REPORT
    # =========================================================================

    mean_probability = (
        customer_risk_output[
            "calibrated_churn_probability"
        ].mean()
    )

    median_probability = (
        customer_risk_output[
            "calibrated_churn_probability"
        ].median()
    )

    max_probability = (
        customer_risk_output[
            "calibrated_churn_probability"
        ].max()
    )

    min_probability = (
        customer_risk_output[
            "calibrated_churn_probability"
        ].min()
    )

    threshold_share = (
        threshold_customers
        / total_customers
    )

    report_text = f"""# Phase 11 — Customer Risk Intelligence

## Purpose

Build an auditable customer-level risk dataset for the complete 69,999-customer universe.

The workflow is:

**Calibrated Churn Probability → Risk Score → Risk Level → Primary Threshold Flag → Risk Ranking**

## Customer Universe

- Total customers: {total_customers:,}
- Development OOF customers: {development_count:,}
- Validation customers: {validation_count:,}
- Actual churners: {total_actual_churners:,}

## Risk Probability

Customer risk is based on the selected tuned XGBoost model outputs with the exact Step 7.11 sigmoid probability calibration.

No additional model training, tuning, recalibration, or feature selection was performed.

### Important interpretation

`calibrated_churn_probability` is the model-estimated probability of churn.

`actual_churn` is the observed historical outcome.

These fields are intentionally kept separate.

A high probability does not guarantee that an individual customer will churn.

## Primary Threshold

Locked Step 7.12 primary operating threshold:

**{primary_threshold:.6f}**

Customers at or above the threshold:

- Count: {threshold_customers:,}
- Share: {threshold_share:.2%}
- Observed historical churners within threshold population: {threshold_actual_churners:,}
- Observed historical churn rate within threshold population: {threshold_observed_churn_rate:.2%}
- Observed historical churn captured: {threshold_actual_churn_capture:.2%}

These retrospective figures describe historical model performance.

They are not guarantees of future churn or future retention outcomes.

## Risk Level Framework

| Risk Level | Calibrated Probability |
|---|---:|
| Below Primary Threshold | < 10% |
| High | 10% to < 50% |
| Very High | >= 50% |

## Risk Score

Risk score is the calibrated churn probability expressed on a 0–100 scale.

It is a communication-friendly representation of the calibrated probability and does not represent a separate model.

## Risk Percentile

Risk percentile represents a customer's relative position in the full 69,999-customer calibrated-risk distribution.

It is descriptive only and does not replace the calibrated probability or the locked primary threshold.

## Risk Distribution

{risk_distribution.to_markdown(index=False)}

## Probability Summary

- Minimum calibrated probability: {min_probability:.4f}
- Mean calibrated probability: {mean_probability:.4f}
- Median calibrated probability: {median_probability:.4f}
- Maximum calibrated probability: {max_probability:.4f}

## Full-Universe Ranking

The customer risk dataset ranks all 69,999 customers using descending calibrated churn probability.

Ties are resolved deterministically using customer ID.

The top 100 output therefore represents the highest-risk customers across the complete customer universe, not a sample.

## Governance

- Model retraining: **NO**
- Additional tuning: **NO**
- Additional calibration: **NO**
- Feature selection: **NO**
- Test data loaded: **NO**
- Feature freeze: **ACTIVE**

## Phase Boundary

Customer value and revenue exposure are intentionally excluded from this phase.

They will be addressed in:

- **Phase 12 — Customer Value & Revenue Risk**
- **Phase 13 — Retention Prioritization & ROI**

## Quality Gate

**{overall_status}**
"""

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )

    # =========================================================================
    # FINAL TERMINAL OUTPUT
    # =========================================================================

    very_high_count = int(
        (
            customer_risk_output[
                "risk_level"
            ]
            == "Very High"
        ).sum()
    )

    high_count = int(
        (
            customer_risk_output[
                "risk_level"
            ]
            == "High"
        ).sum()
    )

    below_threshold_count = int(
        (
            customer_risk_output[
                "risk_level"
            ]
            == "Below Primary Threshold"
        ).sum()
    )

    print("\n" + "=" * 80)
    print("PHASE 11 — CUSTOMER RISK INTELLIGENCE COMPLETE")
    print("=" * 80)

    print(
        f"\nCustomer universe        : {total_customers:,}"
    )

    print(
        f"Mean risk probability   : {mean_probability:.4f}"
    )

    print(
        f"Median risk probability : {median_probability:.4f}"
    )

    print(
        f"Maximum risk probability: {max_probability:.4f}"
    )

    print(
        f"Primary threshold       : {primary_threshold:.6f}"
    )

    print(
        f"Above threshold         : "
        f"{threshold_customers:,} "
        f"({threshold_share:.2%})"
    )

    print(
        f"Observed churn captured : "
        f"{threshold_actual_churn_capture:.2%}"
    )

    print(
        f"Very High customers     : "
        f"{very_high_count:,}"
    )

    print(
        f"High customers          : "
        f"{high_count:,}"
    )

    print(
        f"Below threshold         : "
        f"{below_threshold_count:,}"
    )

    print("\n" + "-" * 80)
    print("QUALITY GATE")
    print("-" * 80)

    for _, row in quality_checks_df.iterrows():

        print(
            f"{row['check']:<48} "
            f"{row['status']:<6} "
            f"— {row['detail']}"
        )

    print("\n" + "-" * 80)

    print(
        f"Overall Customer Risk Quality Gate: "
        f"{overall_status}"
    )

    print("\nTop 10 full-universe risk customers:")

    print(
        customer_risk_output[
            [
                "risk_rank",
                "risk_percentile",
                "id",
                "calibrated_churn_probability",
                "risk_score",
                "risk_level",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nKey artifacts:")

    print(
        f"  - {customer_risk_file}"
    )

    print(
        f"  - {top_risk_file}"
    )

    print(
        f"  - {risk_distribution_file}"
    )

    print(
        f"  - {quality_file}"
    )

    print(
        f"  - {metadata_file}"
    )

    print(
        f"  - {report_file}"
    )

    print("\n" + "=" * 80)
    print(f"PHASE 11 STATUS: {overall_status}")
    print("=" * 80)


if __name__ == "__main__":
    main()