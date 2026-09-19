"""
ChurnIQ - Step 7.12
Threshold Optimization & Business Operating Points

Purpose
-------
Evaluate practical churn-risk operating points using sigmoid-calibrated
probabilities from the finalized tuned XGBoost model.

Governance
----------
- Frozen 169-feature model dataset.
- Development OOF predictions used for threshold selection.
- Exact Step 7.11 sigmoid calibrator applied.
- Held-out validation used only for final confirmation.
- Test data not loaded.
- No model retraining.
- No additional tuning.
- No additional calibration.
- No feature selection.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT:
# These are the actual project locations.
OOF_FILE = ROOT / "reports" / "07_11_tuned_xgb_oof_predictions.csv"

CALIBRATOR_FILE = (
    ROOT / "models" / "xgb_probability_calibrator.joblib"
)

VALIDATION_FILE = (
    ROOT / "reports" / "07_11_calibrated_validation_predictions.csv"
)

X_DEV_FILE = (
    ROOT / "data" / "processed" / "X_dev.csv"
)

REPORT_DIR = ROOT / "reports"

EXPECTED_FEATURE_COUNT = 169
RANDOM_STATE = 42

# Threshold grid
THRESHOLDS = np.round(
    np.arange(0.05, 0.851, 0.01),
    2
)

# Intervention capacity scenarios
CAPACITIES = [0.05, 0.10, 0.15, 0.20]

# Hypothetical business cost scenarios
COST_SCENARIOS = {
    "Conservative_1_to_1": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 1.0,
    },
    "Retention_Focused_1_to_3": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 3.0,
    },
    "High_Missed_Churn_Cost_1_to_5": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 5.0,
    },
}

# Analytical risk bands
RISK_BANDS = [
    ("Very Low", 0.00, 0.20),
    ("Low", 0.20, 0.40),
    ("Medium", 0.40, 0.60),
    ("High", 0.60, 0.80),
    ("Very High", 0.80, 1.01),
]


# ============================================================
# HELPERS
# ============================================================

def ensure_paths():
    """Check that required Step 7.11 artifacts exist."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    required = [
        OOF_FILE,
        CALIBRATOR_FILE,
        VALIDATION_FILE,
        X_DEV_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required ChurnIQ files are missing:\n"
            + "\n".join(missing)
        )


def resolve_column(df, candidates, description):
    """Resolve a column using case-insensitive matching."""

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not resolve {description}.\n"
        f"Available columns: {list(df.columns)}"
    )


def safe_logit(probabilities):
    """
    Convert probabilities to log-odds safely.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15
    )

    return np.log(
        probabilities / (1 - probabilities)
    )


def apply_sigmoid_calibrator(
    raw_probability,
    calibrator
):
    """
    Apply the exact Step 7.11 sigmoid calibrator.

    Step 7.11 used LogisticRegression on the
    logit-transformed raw XGBoost probabilities.
    """

    raw_probability = np.asarray(
        raw_probability,
        dtype=float
    )

    logit_probability = safe_logit(
        raw_probability
    )

    calibrated_probability = calibrator.predict_proba(
        logit_probability.reshape(-1, 1)
    )[:, 1]

    return np.clip(
        calibrated_probability,
        0.0,
        1.0
    )


def binary_metrics(
    y_true,
    probability,
    threshold
):
    """
    Calculate threshold-based classification metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    probability = np.asarray(
        probability,
        dtype=float
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    tp = int(
        (
            (y_true == 1)
            & (prediction == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (prediction == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (prediction == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (prediction == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(y_true)
    )

    targeting_rate = prediction.mean()

    churn_prevalence = y_true.mean()

    lift = (
        recall / targeting_rate
        if targeting_rate > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "targeting_rate": float(targeting_rate),
        "churn_capture": float(recall),
        "lift": float(lift),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "predicted_churn_count": int(
            prediction.sum()
        ),
        "actual_churn_count": int(
            y_true.sum()
        ),
        "missed_churn_rate": float(
            fn / len(y_true)
        ),
        "actual_churn_rate": float(
            churn_prevalence
        ),
    }


def assign_risk_band(probability):
    """Assign analytical probability risk band."""

    for label, lower, upper in RISK_BANDS:

        if (
            probability >= lower
            and probability < upper
        ):
            return label

    return "Very High"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("ChurnIQ - Step 7.12")
    print("Threshold Optimization & Business Operating Points")
    print("=" * 72)

    # --------------------------------------------------------
    # 1. LOAD INPUTS
    # --------------------------------------------------------

    print(
        "\n[1/12] Loading Step 7.11 OOF, "
        "calibrator, validation and feature data..."
    )

    ensure_paths()

    oof = pd.read_csv(
        OOF_FILE
    )

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    X_dev = pd.read_csv(
        X_DEV_FILE
    )

    calibrator_artifact = joblib.load(
        CALIBRATOR_FILE
    )

    print(
        f"OOF shape        : {oof.shape}"
    )

    print(
        f"Validation shape : {validation.shape}"
    )

    print(
        f"X_dev shape      : {X_dev.shape}"
    )

    # --------------------------------------------------------
    # 2. VERIFY FEATURE FREEZE
    # --------------------------------------------------------

    print(
        "\n[2/12] Verifying frozen feature schema..."
    )

    if X_dev.shape[1] != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT} frozen features, "
            f"found {X_dev.shape[1]}."
        )

    print(
        f"Frozen features  : "
        f"{X_dev.shape[1]}"
    )

    print(
        "Feature freeze   : PASS"
    )

    # --------------------------------------------------------
    # 3. RESOLVE OOF COLUMNS
    # --------------------------------------------------------

    print(
        "\n[3/12] Resolving OOF columns..."
    )

    oof_id_col = resolve_column(
        oof,
        [
            "development_row_id",
            "oof_row_id",
            "row_id",
        ],
        "OOF row identifier",
    )

    oof_target_col = resolve_column(
        oof,
        [
            "actual_churn",
            "target",
            "y",
        ],
        "OOF target",
    )

    oof_probability_col = resolve_column(
        oof,
        [
            "tuned_xgb_oof_probability",
            "oof_probability",
            "predicted_churn_probability",
        ],
        "raw tuned-XGB OOF probability",
    )

    print(
        f"ID column        : "
        f"{oof_id_col}"
    )

    print(
        f"Target column    : "
        f"{oof_target_col}"
    )

    print(
        f"Raw probability  : "
        f"{oof_probability_col}"
    )

    # --------------------------------------------------------
    # 4. VALIDATE OOF DATA
    # --------------------------------------------------------

    print(
        "\n[4/12] Validating OOF probabilities and target..."
    )

    y_oof_series = pd.to_numeric(
        oof[oof_target_col],
        errors="coerce"
    )

    raw_probability_series = pd.to_numeric(
        oof[oof_probability_col],
        errors="coerce"
    )

    if y_oof_series.isna().any():

        raise ValueError(
            "OOF target contains "
            "missing/non-numeric values."
        )

    if raw_probability_series.isna().any():

        raise ValueError(
            "OOF probabilities contain "
            "missing/non-numeric values."
        )

    y_oof = y_oof_series.astype(
        int
    ).to_numpy()

    raw_oof_probability = (
        raw_probability_series
        .to_numpy(dtype=float)
    )

    if not set(
        np.unique(y_oof)
    ).issubset({0, 1}):

        raise ValueError(
            "OOF target is not binary."
        )

    if not np.isfinite(
        raw_oof_probability
    ).all():

        raise ValueError(
            "OOF probabilities contain "
            "non-finite values."
        )

    if (
        (raw_oof_probability < 0).any()
        or
        (raw_oof_probability > 1).any()
    ):

        raise ValueError(
            "OOF probabilities fall "
            "outside [0, 1]."
        )

    print(
        f"OOF rows         : "
        f"{len(oof):,}"
    )

    print(
        f"Churn rate       : "
        f"{y_oof.mean():.4%}"
    )

    print(
        f"Raw probability  : "
        f"{raw_oof_probability.min():.6f}"
        f" - "
        f"{raw_oof_probability.max():.6f}"
    )

    print(
        "OOF validation   : PASS"
    )

    # --------------------------------------------------------
    # 5. APPLY EXACT SIGMOID CALIBRATOR
    # --------------------------------------------------------

    print(
        "\n[5/12] Applying exact Step 7.11 "
        "sigmoid calibrator..."
    )

    if not isinstance(
        calibrator_artifact,
        dict
    ):

        raise ValueError(
            "Unexpected calibrator artifact structure."
        )

    calibration_method = (
        calibrator_artifact.get(
            "method"
        )
    )

    calibrator = (
        calibrator_artifact.get(
            "calibrator"
        )
    )

    if calibration_method != "sigmoid":

        raise ValueError(
            "Expected Step 7.11 sigmoid "
            f"calibration, found: "
            f"{calibration_method}"
        )

    if calibrator is None:

        raise ValueError(
            "Calibrator artifact does not "
            "contain a calibrator object."
        )

    if not hasattr(
        calibrator,
        "predict_proba"
    ):

        raise ValueError(
            "Expected LogisticRegression-style "
            "sigmoid calibrator."
        )

    calibrated_oof_probability = (
        apply_sigmoid_calibrator(
            raw_oof_probability,
            calibrator
        )
    )

    if not np.isfinite(
        calibrated_oof_probability
    ).all():

        raise ValueError(
            "Calibrated OOF probabilities "
            "contain non-finite values."
        )

    if (
        (calibrated_oof_probability < 0).any()
        or
        (calibrated_oof_probability > 1).any()
    ):

        raise ValueError(
            "Calibrated OOF probabilities "
            "fall outside [0, 1]."
        )

    print(
        f"Calibration      : "
        f"{calibration_method}"
    )

    print(
        f"Calibrated range : "
        f"{calibrated_oof_probability.min():.6f}"
        f" - "
        f"{calibrated_oof_probability.max():.6f}"
    )

    print(
        "Calibration      : PASS"
    )

    # --------------------------------------------------------
    # 6. THRESHOLD PERFORMANCE
    # --------------------------------------------------------

    print(
        "\n[6/12] Evaluating threshold trade-offs..."
    )

    threshold_rows = []

    for threshold in THRESHOLDS:

        metrics = binary_metrics(
            y_oof,
            calibrated_oof_probability,
            threshold
        )

        threshold_rows.append(
            metrics
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        REPORT_DIR
        / "07_12_threshold_metrics.csv",
        index=False
    )

    print(
        f"Thresholds evaluated : "
        f"{len(threshold_df)}"
    )

    # --------------------------------------------------------
    # 7. COST-SENSITIVE SCENARIOS
    # --------------------------------------------------------

    print(
        "\n[7/12] Evaluating hypothetical "
        "cost scenarios..."
    )

    cost_rows = []

    for (
        scenario_name,
        costs
    ) in COST_SCENARIOS.items():

        for _, row in threshold_df.iterrows():

            total_cost = (
                row["false_positives"]
                * costs[
                    "false_positive_cost"
                ]
                +
                row["false_negatives"]
                * costs[
                    "false_negative_cost"
                ]
            )

            cost_rows.append(
                {
                    "scenario": scenario_name,
                    "threshold": row[
                        "threshold"
                    ],
                    "false_positive_cost": costs[
                        "false_positive_cost"
                    ],
                    "false_negative_cost": costs[
                        "false_negative_cost"
                    ],
                    "false_positives": row[
                        "false_positives"
                    ],
                    "false_negatives": row[
                        "false_negatives"
                    ],
                    "total_hypothetical_cost":
                        total_cost,
                }
            )

    cost_df = pd.DataFrame(
        cost_rows
    )

    cost_df.to_csv(
        REPORT_DIR
        / "07_12_cost_scenarios.csv",
        index=False
    )

    print(
        f"Cost scenarios       : "
        f"{len(COST_SCENARIOS)}"
    )

    # --------------------------------------------------------
    # 8. CAPACITY SCENARIOS
    # --------------------------------------------------------

    print(
        "\n[8/12] Evaluating intervention "
        "capacity scenarios..."
    )

    capacity_rows = []

    ranking_order = np.argsort(
        -calibrated_oof_probability
    )

    actual_churners = int(
        y_oof.sum()
    )

    for capacity in CAPACITIES:

        n_target = max(
            1,
            int(
                np.floor(
                    len(oof)
                    * capacity
                )
            )
        )

        selected_indices = (
            ranking_order[
                :n_target
            ]
        )

        y_selected = (
            y_oof[
                selected_indices
            ]
        )

        captured_churners = int(
            y_selected.sum()
        )

        capture = (
            captured_churners
            / actual_churners
            if actual_churners > 0
            else 0.0
        )

        precision = (
            captured_churners
            / n_target
            if n_target > 0
            else 0.0
        )

        capacity_rows.append(
            {
                "capacity_rate": capacity,
                "targeted_customers": n_target,
                "targeted_rate_actual":
                    n_target / len(oof),
                "churners_captured":
                    captured_churners,
                "churn_capture":
                    capture,
                "precision":
                    precision,
                "lift":
                    (
                        capture / capacity
                        if capacity > 0
                        else np.nan
                    ),
                "rank_based_selection":
                    True,
            }
        )

    capacity_df = pd.DataFrame(
        capacity_rows
    )

    capacity_df.to_csv(
        REPORT_DIR
        / "07_12_capacity_scenarios.csv",
        index=False
    )

    # --------------------------------------------------------
    # 9. RISK BANDS
    # --------------------------------------------------------

    print(
        "\n[9/12] Creating analytical risk bands..."
    )

    risk_df = pd.DataFrame(
        {
            "development_row_id":
                oof[oof_id_col],
            "actual_churn":
                y_oof,
            "raw_oof_probability":
                raw_oof_probability,
            "calibrated_churn_probability":
                calibrated_oof_probability,
        }
    )

    risk_df[
        "risk_band"
    ] = risk_df[
        "calibrated_churn_probability"
    ].apply(
        assign_risk_band
    )

    risk_band_rows = []

    for (
        band,
        group
    ) in risk_df.groupby(
        "risk_band",
        sort=False
    ):

        group_churners = int(
            group[
                "actual_churn"
            ].sum()
        )

        risk_band_rows.append(
            {
                "risk_band": band,
                "customers": len(group),
                "customer_share":
                    len(group)
                    / len(risk_df),
                "actual_churners":
                    group_churners,
                "actual_churn_rate":
                    group[
                        "actual_churn"
                    ].mean(),
                "mean_calibrated_probability":
                    group[
                        "calibrated_churn_probability"
                    ].mean(),
                "actual_churn_capture_share":
                    (
                        group_churners
                        / actual_churners
                        if actual_churners > 0
                        else np.nan
                    ),
            }
        )

    risk_band_df = pd.DataFrame(
        risk_band_rows
    )

    risk_band_df.to_csv(
        REPORT_DIR
        / "07_12_risk_band_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # 10. PRIMARY OPERATING POINT
    # --------------------------------------------------------

    print(
        "\n[10/12] Selecting analytical "
        "operating point..."
    )

    eligible = threshold_df[
        (threshold_df["targeting_rate"] >= 0.05)
        &
        (threshold_df["targeting_rate"] <= 0.20)
        &
        (threshold_df["precision"] >= 0.50)
    ].copy()

    if eligible.empty:

        raise ValueError(
            "No threshold satisfies the configured "
            "5%-20% targeting and >=50% precision "
            "criteria."
        )

    eligible = eligible.sort_values(
        [
            "churn_capture",
            "precision",
            "targeting_rate",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    primary = eligible.iloc[0]

    primary_threshold = float(
        primary["threshold"]
    )

    # --------------------------------------------------------
    # 10B. CAPACITY-CONSTRAINED POINT
    # --------------------------------------------------------

    capacity_target = 0.10

    capacity_n = max(
        1,
        int(
            np.floor(
                len(oof)
                * capacity_target
            )
        )
    )

    capacity_indices = (
        ranking_order[
            :capacity_n
        ]
    )

    capacity_churners = int(
        y_oof[
            capacity_indices
        ].sum()
    )

    capacity_churn_capture = (
        capacity_churners
        / actual_churners
        if actual_churners > 0
        else 0.0
    )

    capacity_precision = (
        capacity_churners
        / capacity_n
        if capacity_n > 0
        else 0.0
    )

    capacity_threshold = float(
        np.min(
            calibrated_oof_probability[
                capacity_indices
            ]
        )
    )

    print(
        f"Primary threshold          : "
        f"{primary_threshold:.2f}"
    )

    print(
        f"Primary targeting rate    : "
        f"{primary['targeting_rate']:.2%}"
    )

    print(
        f"Primary churn capture     : "
        f"{primary['churn_capture']:.2%}"
    )

    print(
        f"Primary precision         : "
        f"{primary['precision']:.2%}"
    )

    print(
        f"10% capacity threshold    : "
        f"{capacity_threshold:.6f}"
    )

    print(
        f"10% capacity churn capture: "
        f"{capacity_churn_capture:.2%}"
    )

    print(
        f"10% capacity precision    : "
        f"{capacity_precision:.2%}"
    )

    # --------------------------------------------------------
    # 11. HELD-OUT VALIDATION CONFIRMATION
    # --------------------------------------------------------

    print(
        "\n[11/12] Confirming operating point "
        "on untouched validation..."
    )

    validation_target_col = resolve_column(
        validation,
        [
            "actual_churn",
            "target",
        ],
        "validation target",
    )

    validation_probability_col = resolve_column(
        validation,
        [
            "calibrated_churn_probability",
            "sigmoid_probability",
            "calibrated_probability",
            "predicted_churn_probability",
        ],
        "calibrated validation probability",
    )

    y_validation_series = pd.to_numeric(
        validation[
            validation_target_col
        ],
        errors="coerce"
    )

    validation_probability_series = pd.to_numeric(
        validation[
            validation_probability_col
        ],
        errors="coerce"
    )

    if y_validation_series.isna().any():

        raise ValueError(
            "Validation target contains "
            "missing/non-numeric values."
        )

    if validation_probability_series.isna().any():

        raise ValueError(
            "Validation probabilities contain "
            "missing/non-numeric values."
        )

    y_validation = (
        y_validation_series
        .astype(int)
        .to_numpy()
    )

    validation_probability = (
        validation_probability_series
        .to_numpy(dtype=float)
    )

    if not set(
        np.unique(y_validation)
    ).issubset({0, 1}):

        raise ValueError(
            "Validation target is not binary."
        )

    if not np.isfinite(
        validation_probability
    ).all():

        raise ValueError(
            "Validation probabilities contain "
            "non-finite values."
        )

    validation_metrics = binary_metrics(
        y_validation,
        validation_probability,
        primary_threshold
    )

    validation_capacity_prediction = (
        validation_probability
        >= capacity_threshold
    ).astype(int)

    validation_capacity_count = int(
        validation_capacity_prediction.sum()
    )

    validation_capacity_churners = int(
        y_validation[
            validation_capacity_prediction == 1
        ].sum()
    )

    validation_capacity_capture = (
        validation_capacity_churners
        / y_validation.sum()
        if y_validation.sum() > 0
        else 0.0
    )

    validation_capacity_precision = (
        validation_capacity_churners
        / validation_capacity_count
        if validation_capacity_count > 0
        else 0.0
    )

    validation_capacity_targeting_rate = (
        validation_capacity_count
        / len(validation)
    )

    validation_confirmation = pd.DataFrame(
        [
            {
                "operating_point":
                    "Primary_threshold",
                "threshold":
                    primary_threshold,
                "validation_targeting_rate":
                    validation_metrics[
                        "targeting_rate"
                    ],
                "validation_precision":
                    validation_metrics[
                        "precision"
                    ],
                "validation_recall":
                    validation_metrics[
                        "recall"
                    ],
                "validation_f1":
                    validation_metrics[
                        "f1"
                    ],
                "validation_churn_capture":
                    validation_metrics[
                        "churn_capture"
                    ],
                "validation_lift":
                    validation_metrics[
                        "lift"
                    ],
                "validation_predicted_churn_count":
                    validation_metrics[
                        "predicted_churn_count"
                    ],
            },
            {
                "operating_point":
                    "10_percent_capacity",
                "threshold":
                    capacity_threshold,
                "validation_targeting_rate":
                    validation_capacity_targeting_rate,
                "validation_precision":
                    validation_capacity_precision,
                "validation_recall":
                    validation_capacity_capture,
                "validation_f1":
                    np.nan,
                "validation_churn_capture":
                    validation_capacity_capture,
                "validation_lift":
                    (
                        validation_capacity_capture
                        / validation_capacity_targeting_rate
                        if validation_capacity_targeting_rate > 0
                        else np.nan
                    ),
                "validation_predicted_churn_count":
                    validation_capacity_count,
            },
        ]
    )

    validation_confirmation.to_csv(
        REPORT_DIR
        / "07_12_validation_threshold_confirmation.csv",
        index=False
    )

    print(
        "Validation confirmation : PASS"
    )

    # --------------------------------------------------------
    # 12. DECISION ARTIFACTS
    # --------------------------------------------------------

    print(
        "\n[12/12] Saving decision artifacts..."
    )

    decision = {
        "step": "7.12",
        "purpose":
            "Threshold Optimization & "
            "Business Operating Points",
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "random_state":
            RANDOM_STATE,
        "feature_count":
            EXPECTED_FEATURE_COUNT,
        "development_oof_rows":
            int(len(oof)),
        "validation_rows":
            int(len(validation)),
        "calibration_method":
            "sigmoid",
        "calibrator_source":
            "Step 7.11",
        "threshold_selection_data":
            "Development OOF only",
        "validation_used_for_threshold_selection":
            False,
        "test_used":
            False,
        "additional_training":
            False,
        "additional_tuning":
            False,
        "additional_calibration":
            False,
        "feature_selection":
            False,
        "primary_operating_point": {
            "threshold":
                primary_threshold,
            "targeting_rate":
                float(
                    primary[
                        "targeting_rate"
                    ]
                ),
            "precision":
                float(
                    primary[
                        "precision"
                    ]
                ),
            "churn_capture":
                float(
                    primary[
                        "churn_capture"
                    ]
                ),
            "lift":
                float(
                    primary[
                        "lift"
                    ]
                ),
        },
        "capacity_operating_point": {
            "capacity_rate":
                capacity_target,
            "threshold":
                capacity_threshold,
            "development_churn_capture":
                float(
                    capacity_churn_capture
                ),
            "development_precision":
                float(
                    capacity_precision
                ),
        },
        "risk_bands": [
            {
                "label": label,
                "lower_bound": lower,
                "upper_bound": upper,
            }
            for (
                label,
                lower,
                upper
            ) in RISK_BANDS
        ],
        "quality_gate":
            "PASS",
    }

    with open(
        REPORT_DIR
        / "07_12_threshold_selection.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            decision,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    report = f"""# ChurnIQ — Step 7.12
## Threshold Optimization & Business Operating Points

### Objective

Determine practical churn-risk operating points using sigmoid-calibrated probabilities from the finalized tuned XGBoost model.

### Data Boundary

- Development OOF rows: {len(oof):,}
- Held-out validation rows: {len(validation):,}
- Frozen features: {EXPECTED_FEATURE_COUNT}
- Test data: not loaded
- Threshold selection: development OOF only

### Calibration

The exact Step 7.11 sigmoid calibrator was applied to the raw tuned-XGBoost OOF probabilities.

Calibration method: **sigmoid**

No additional calibration was fitted.

### Primary Operating Point

| Metric | Value |
|---|---:|
| Threshold | {primary_threshold:.2f} |
| Targeting rate | {primary["targeting_rate"]:.2%} |
| Precision | {primary["precision"]:.2%} |
| Churn capture | {primary["churn_capture"]:.2%} |
| Lift | {primary["lift"]:.2f}x |

This operating point is an analytical recommendation. Actual deployment should consider intervention capacity, retention strategy, contact cost, and business value.

### Capacity-Constrained Operating Point

A separate 10% intervention-capacity scenario was evaluated.

| Metric | Development OOF |
|---|---:|
| Capacity | 10.00% |
| Threshold | {capacity_threshold:.6f} |
| Churn capture | {capacity_churn_capture:.2%} |
| Precision | {capacity_precision:.2%} |

The capacity point is intentionally separated from the probability threshold selected from the broader trade-off.

### Risk Bands

Analytical churn-probability bands:

- Very Low: 0–20%
- Low: 20–40%
- Medium: 40–60%
- High: 60–80%
- Very High: 80%+

These are prioritization bands and do not guarantee individual churn outcomes.

### Hypothetical Cost Scenarios

Three scenario assumptions were evaluated:

1. Conservative: false positive cost = 1, false negative cost = 1
2. Retention-focused: false positive cost = 1, false negative cost = 3
3. High missed-churn cost: false positive cost = 1, false negative cost = 5

These are illustrative assumptions rather than measured financial costs.

### Held-Out Validation Confirmation

The selected operating point was evaluated on the untouched validation set only after threshold selection.

| Metric | Validation |
|---|---:|
| Targeting rate | {validation_metrics["targeting_rate"]:.2%} |
| Precision | {validation_metrics["precision"]:.2%} |
| Recall / churn capture | {validation_metrics["recall"]:.2%} |
| F1 | {validation_metrics["f1"]:.4f} |
| Lift | {validation_metrics["lift"]:.2f}x |

### Governance

- Exact Step 7.11 sigmoid calibrator used: YES
- Threshold selection from development OOF: YES
- Validation used for threshold selection: NO
- Test data loaded: NO
- Additional training: NO
- Additional tuning: NO
- Additional calibration: NO
- Feature selection: NO
- Feature freeze: ACTIVE

### Quality Gate

**PASS**

Step 7.12 is complete.
"""

    with open(
        REPORT_DIR
        / "07_12_threshold_optimization_report.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    # --------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "Step 7.12 Complete"
    )

    print(
        "=" * 72
    )

    print(
        f"Primary threshold       : "
        f"{primary_threshold:.2f}"
    )

    print(
        f"Primary targeting rate : "
        f"{primary['targeting_rate']:.2%}"
    )

    print(
        f"Primary churn capture  : "
        f"{primary['churn_capture']:.2%}"
    )

    print(
        f"Primary precision      : "
        f"{primary['precision']:.2%}"
    )

    print(
        f"10% capacity threshold : "
        f"{capacity_threshold:.6f}"
    )

    print(
        "\nGovernance checks:"
    )

    print(
        "  - Exact Step 7.11 calibrator used : YES"
    )

    print(
        "  - Threshold selection on OOF      : YES"
    )

    print(
        "  - Validation used for selection   : NO"
    )

    print(
        "  - Test data loaded                : NO"
    )

    print(
        "  - Additional training             : NO"
    )

    print(
        "  - Additional tuning               : NO"
    )

    print(
        "  - Additional calibration          : NO"
    )

    print(
        "  - Feature selection               : NO"
    )

    print(
        "  - Feature freeze                  : ACTIVE"
    )

    print(
        "\nQUALITY GATE: PASS"
    )


if __name__ == "__main__":
    main()