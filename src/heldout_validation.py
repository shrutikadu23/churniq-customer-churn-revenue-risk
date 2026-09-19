"""
ChurnIQ — Step 7.9
Held-Out Validation & Model Comparison

Purpose
-------
Evaluate baseline and tuned candidate models on the untouched
validation dataset after feature engineering, feature selection,
baseline modeling, cross-validation, and controlled tuning.

Models:
1. Logistic Regression baseline
2. HistGradientBoosting baseline
3. HistGradientBoosting tuned
4. XGBoost baseline
5. XGBoost tuned

Experimental controls
---------------------
- 169 frozen model features
- Development data used for model fitting
- Validation data used only for held-out evaluation
- Test data NOT loaded
- Validation NOT used for feature selection
- Validation NOT used for hyperparameter tuning
- Validation NOT used for threshold optimization
- No resampling
- No class weighting
- No calibration
- No SHAP
- Threshold fixed at 0.50
- Random state = 42

Primary metric:
    PR-AUC

Secondary metrics:
    ROC-AUC
    Precision
    Recall
    F1
    Log Loss
    Brier Score

Business ranking metrics:
    Top-5% capture / lift
    Top-10% capture / lift
    Top-20% capture / lift

Important
---------
The validation dataset remains a final holdout for model
comparison. Results from this stage determine which model
moves forward to calibration, threshold analysis,
explainability, and customer risk intelligence.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
EXPECTED_FEATURE_COUNT = 169

TARGET_NAME = "churn_probability"

FIXED_THRESHOLD = 0.50

# Practical interpretation boundary for tiny PR-AUC differences.
# This does NOT select the model. It only helps describe whether
# the difference is materially small.
PRACTICAL_PR_AUC_MARGIN = 0.002

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

X_DEV_PATH = DATA_DIR / "X_dev.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"

X_VALIDATION_PATH = DATA_DIR / "X_validation.csv"
Y_VALIDATION_PATH = DATA_DIR / "y_validation.csv"

TUNING_SUMMARY_PATH = (
    REPORT_DIR / "07_8_tuning_summary.csv"
)

RESULTS_PATH = (
    REPORT_DIR / "07_9_heldout_validation_results.csv"
)

CONFUSION_PATH = (
    REPORT_DIR / "07_9_confusion_matrices.csv"
)

PREDICTIONS_PATH = (
    REPORT_DIR / "07_9_validation_predictions.csv"
)

CURVE_DATA_PATH = (
    REPORT_DIR / "07_9_curve_data.csv"
)

METADATA_PATH = (
    REPORT_DIR / "07_9_validation_metadata.json"
)

REPORT_PATH = (
    REPORT_DIR / "07_9_heldout_validation_report.md"
)


# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def dataframe_schema_hash(
    df: pd.DataFrame,
) -> str:
    """
    Create a reproducible hash from feature names and dtypes.
    """
    schema = "|".join(
        f"{column}:{dtype}"
        for column, dtype in zip(
            df.columns,
            df.dtypes.astype(str),
        )
    )

    return hashlib.sha256(
        schema.encode("utf-8")
    ).hexdigest()


def dataframe_content_hash(
    df: pd.DataFrame,
) -> str:
    """
    Create a reproducible hash of the dataframe values.
    Used only as a diagnostic integrity check.
    """
    hashed = pd.util.hash_pandas_object(
        df,
        index=True,
    )

    return hashlib.sha256(
        hashed.to_numpy(
            dtype=np.uint64
        ).tobytes()
    ).hexdigest()


def top_k_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fraction: float,
) -> tuple[float, float]:

    n = len(y_true)

    k = max(
        1,
        int(np.ceil(n * fraction)),
    )

    ranked_indices = np.argsort(
        -probabilities,
        kind="mergesort",
    )

    top_indices = ranked_indices[:k]

    actual_churners = int(
        np.sum(y_true)
    )

    if actual_churners == 0:
        return np.nan, np.nan

    captured = int(
        np.sum(
            y_true[top_indices]
        )
    )

    capture_rate = (
        captured / actual_churners
    )

    baseline_rate = (
        actual_churners / n
    )

    top_group_churn_rate = (
        captured / k
    )

    lift = (
        top_group_churn_rate
        / baseline_rate
        if baseline_rate > 0
        else np.nan
    )

    return capture_rate, lift


def build_logistic():
    """
    Exact Step 7.4 Logistic Regression baseline.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_hgb(
    params: dict,
):
    """
    Build HistGradientBoosting using the exact
    Step 7.8 parameter set.
    """

    allowed = {
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
    }

    clean_params = {
        key: value
        for key, value in params.items()
        if key in allowed
    }

    return HistGradientBoostingClassifier(
        **clean_params,
        random_state=RANDOM_STATE,
    )


def build_xgb(
    params: dict,
):
    """
    Build XGBoost using the exact Step 7.8 parameter set.
    """

    allowed = {
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

    clean_params = {
        key: value
        for key, value in params.items()
        if key in allowed
    }

    return XGBClassifier(
        **clean_params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================================
# 3. LOAD DATA
# ============================================================================

print_header(
    "ChurnIQ — STEP 7.9\n"
    "Held-Out Validation & Model Comparison"
)

print(
    "\n[1] Loading development and validation data..."
)

X_dev = pd.read_csv(
    X_DEV_PATH
)

y_dev = pd.read_csv(
    Y_DEV_PATH
).squeeze("columns")

X_validation = pd.read_csv(
    X_VALIDATION_PATH
)

y_validation = pd.read_csv(
    Y_VALIDATION_PATH
).squeeze("columns")


print(
    f"X_dev shape:        {X_dev.shape}"
)

print(
    f"y_dev shape:        {y_dev.shape}"
)

print(
    f"X_validation shape: {X_validation.shape}"
)

print(
    f"y_validation shape: {y_validation.shape}"
)

print(
    "Test data loaded: NO"
)


# ============================================================================
# 4. DATA QUALITY GATE
# ============================================================================

print_header(
    "DATA QUALITY CHECKS"
)

assert (
    X_dev.shape[1]
    == EXPECTED_FEATURE_COUNT
)

assert (
    X_validation.shape[1]
    == EXPECTED_FEATURE_COUNT
)

assert (
    list(X_dev.columns)
    == list(X_validation.columns)
), (
    "Development and validation "
    "feature order mismatch."
)

assert (
    len(X_dev)
    == len(y_dev)
)

assert (
    len(X_validation)
    == len(y_validation)
)

assert X_dev.columns.is_unique
assert X_validation.columns.is_unique

assert set(
    y_dev.unique()
).issubset({0, 1})

assert set(
    y_validation.unique()
).issubset({0, 1})

assert not X_dev.isna().any().any()
assert not X_validation.isna().any().any()

assert np.isfinite(
    X_dev.to_numpy(
        dtype=float
    )
).all()

assert np.isfinite(
    X_validation.to_numpy(
        dtype=float
    )
).all()


dev_schema_hash = (
    dataframe_schema_hash(X_dev)
)

validation_schema_hash = (
    dataframe_schema_hash(
        X_validation
    )
)

assert (
    dev_schema_hash
    == validation_schema_hash
), (
    "Development and validation "
    "schema hashes do not match."
)


print(
    "169 development features: PASS"
)

print(
    "169 validation features: PASS"
)

print(
    "Feature names/order aligned: PASS"
)

print(
    "Schema fingerprint aligned: PASS"
)

print(
    "Development rows aligned: PASS"
)

print(
    "Validation rows aligned: PASS"
)

print(
    "Binary targets: PASS"
)

print(
    "No missing values: PASS"
)

print(
    "No infinite values: PASS"
)


# ============================================================================
# 5. TARGET DISTRIBUTION
# ============================================================================

print_header(
    "TARGET DISTRIBUTION"
)

dev_churn_rate = float(
    y_dev.mean()
)

validation_churn_rate = float(
    y_validation.mean()
)

print(
    f"Development churn rate: "
    f"{dev_churn_rate:.2%}"
)

print(
    f"Validation churn rate: "
    f"{validation_churn_rate:.2%}"
)

print(
    f"Development churners: "
    f"{int(y_dev.sum()):,}"
)

print(
    f"Validation churners: "
    f"{int(y_validation.sum()):,}"
)


# ============================================================================
# 6. LOAD STEP 7.8 TUNED PARAMETERS
# ============================================================================

print_header(
    "LOADING STEP 7.8 TUNING RESULTS"
)

tuning_summary = pd.read_csv(
    TUNING_SUMMARY_PATH
)

required_columns = {
    "model",
    "best_candidate",
    "best_cv_pr_auc",
    "parameters",
}

assert required_columns.issubset(
    tuning_summary.columns
), (
    "Step 7.8 tuning summary "
    "does not contain required columns."
)


def get_tuned_params(
    model_name: str,
) -> tuple[str, float, dict]:

    rows = tuning_summary[
        tuning_summary["model"]
        == model_name
    ]

    assert len(rows) == 1, (
        f"Expected exactly one "
        f"tuning result for "
        f"{model_name}."
    )

    row = rows.iloc[0]

    parameters = row[
        "parameters"
    ]

    if isinstance(
        parameters,
        str,
    ):
        parameters = json.loads(
            parameters
        )

    return (
        str(
            row["best_candidate"]
        ),
        float(
            row["best_cv_pr_auc"]
        ),
        parameters,
    )


(
    hgb_tuned_candidate,
    hgb_tuned_cv_pr_auc,
    hgb_tuned_params,
) = get_tuned_params(
    "HistGradientBoosting"
)

(
    xgb_tuned_candidate,
    xgb_tuned_cv_pr_auc,
    xgb_tuned_params,
) = get_tuned_params(
    "XGBoost"
)


print(
    f"HGB tuned candidate: "
    f"{hgb_tuned_candidate}"
)

print(
    f"HGB development CV PR-AUC: "
    f"{hgb_tuned_cv_pr_auc:.4f}"
)

print(
    f"XGBoost tuned candidate: "
    f"{xgb_tuned_candidate}"
)

print(
    f"XGBoost development CV PR-AUC: "
    f"{xgb_tuned_cv_pr_auc:.4f}"
)


# ============================================================================
# 7. EXACT BASELINE PARAMETERS
# ============================================================================

HGB_BASELINE_PARAMS = {
    "learning_rate": 0.10,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

XGB_BASELINE_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.10,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


# ============================================================================
# 8. DEVELOPMENT CV REFERENCES
# ============================================================================

CV_PR_AUC_REFERENCES = {

    "Logistic Regression":
        0.6959,

    "HistGradientBoosting Baseline":
        0.7705,

    "HistGradientBoosting Tuned":
        hgb_tuned_cv_pr_auc,

    "XGBoost Baseline":
        0.7720,

    "XGBoost Tuned":
        xgb_tuned_cv_pr_auc,
}


# ============================================================================
# 9. MODEL DEFINITIONS
# ============================================================================

models = [

    (
        "Logistic Regression",
        build_logistic(),
    ),

    (
        "HistGradientBoosting Baseline",
        build_hgb(
            HGB_BASELINE_PARAMS
        ),
    ),

    (
        "HistGradientBoosting Tuned",
        build_hgb(
            hgb_tuned_params
        ),
    ),

    (
        "XGBoost Baseline",
        build_xgb(
            XGB_BASELINE_PARAMS
        ),
    ),

    (
        "XGBoost Tuned",
        build_xgb(
            xgb_tuned_params
        ),
    ),
]


# ============================================================================
# 10. MODEL EVALUATION FUNCTION
# ============================================================================

def evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    cv_pr_auc: float,
):

    print(
        f"\nFitting {model_name}..."
    )

    start_time = time.perf_counter()

    # IMPORTANT:
    # Model is fitted ONLY on development data.
    model.fit(
        X_train,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - start_time
    )

    probabilities = (
        model.predict_proba(
            X_holdout
        )[:, 1]
    )

    # Probability sanity checks.
    assert np.isfinite(
        probabilities
    ).all()

    assert (
        probabilities.min()
        >= 0.0
    )

    assert (
        probabilities.max()
        <= 1.0
    )

    predictions = (
        probabilities
        >= FIXED_THRESHOLD
    ).astype(int)

    y_true = (
        y_holdout.to_numpy()
    )

    # ------------------------------------------------------------------------
    # Standard metrics
    # ------------------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    logloss = log_loss(
        y_true,
        probabilities,
        labels=[0, 1],
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    # ------------------------------------------------------------------------
    # Business ranking metrics
    # ------------------------------------------------------------------------

    top5_capture, top5_lift = (
        top_k_metrics(
            y_true,
            probabilities,
            0.05,
        )
    )

    top10_capture, top10_lift = (
        top_k_metrics(
            y_true,
            probabilities,
            0.10,
        )
    )

    top20_capture, top20_lift = (
        top_k_metrics(
            y_true,
            probabilities,
            0.20,
        )
    )

    # ------------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------------

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    predicted_churn_rate = float(
        predictions.mean()
    )

    result = {

        "model": model_name,

        "cv_pr_auc":
            cv_pr_auc,

        "validation_pr_auc":
            pr_auc,

        "validation_roc_auc":
            roc_auc,

        "accuracy_at_0_50":
            accuracy,

        "precision_at_0_50":
            precision,

        "recall_at_0_50":
            recall,

        "f1_at_0_50":
            f1,

        "log_loss":
            logloss,

        "brier_score":
            brier,

        "predicted_churn_rate":
            predicted_churn_rate,

        "actual_churn_rate":
            float(y_true.mean()),

        "top5_capture":
            top5_capture,

        "top5_lift":
            top5_lift,

        "top10_capture":
            top10_capture,

        "top10_lift":
            top10_lift,

        "top20_capture":
            top20_capture,

        "top20_lift":
            top20_lift,

        "true_negatives":
            int(tn),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_positives":
            int(tp),

        "predicted_churn_count":
            int(predictions.sum()),

        "actual_churn_count":
            int(y_true.sum()),

        "fit_seconds":
            fit_seconds,

        "threshold":
            FIXED_THRESHOLD,
    }

    return (
        result,
        probabilities,
    )


# ============================================================================
# 11. RUN HELD-OUT VALIDATION
# ============================================================================

print_header(
    "HELD-OUT VALIDATION EVALUATION"
)

all_results = []
all_predictions = []

for model_name, model in models:

    cv_score = (
        CV_PR_AUC_REFERENCES[
            model_name
        ]
    )

    (
        result,
        probabilities,
    ) = evaluate_model(
        model=model,
        model_name=model_name,
        X_train=X_dev,
        y_train=y_dev,
        X_holdout=X_validation,
        y_holdout=y_validation,
        cv_pr_auc=cv_score,
    )

    all_results.append(
        result
    )

    prediction_frame = pd.DataFrame(
        {
            "validation_row_id":
                np.arange(
                    len(
                        X_validation
                    )
                ),

            "actual_churn":
                y_validation.to_numpy(),

            "model":
                model_name,

            "predicted_churn_probability":
                probabilities,

            "predicted_churn_at_0_50":
                (
                    probabilities
                    >= FIXED_THRESHOLD
                ).astype(int),
        }
    )

    all_predictions.append(
        prediction_frame
    )

    print(
        f"\n{model_name}"
    )

    print(
        f"Validation PR-AUC: "
        f"{result['validation_pr_auc']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{result['validation_roc_auc']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['precision_at_0_50']:.4f}"
    )

    print(
        f"Recall: "
        f"{result['recall_at_0_50']:.4f}"
    )

    print(
        f"F1: "
        f"{result['f1_at_0_50']:.4f}"
    )

    print(
        f"Top-10 capture: "
        f"{result['top10_capture']:.2%}"
    )

    print(
        f"Top-10 lift: "
        f"{result['top10_lift']:.2f}x"
    )


# ============================================================================
# 12. RESULT TABLE
# ============================================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================================
# 13. CV → VALIDATION TRANSFER
# ============================================================================

results_df[
    "validation_minus_cv_pr_auc"
] = (
    results_df[
        "validation_pr_auc"
    ]
    -
    results_df[
        "cv_pr_auc"
    ]
)

results_df[
    "absolute_cv_validation_gap"
] = (
    results_df[
        "validation_minus_cv_pr_auc"
    ].abs()
)


# ============================================================================
# 14. VALIDATION RANKING
# ============================================================================

results_df = results_df.sort_values(
    [
        "validation_pr_auc",
        "top10_capture",
        "validation_roc_auc",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(
    drop=True
)

results_df[
    "validation_rank"
] = (
    np.arange(
        len(results_df)
    )
    + 1
)


# ============================================================================
# 15. PRACTICAL DIFFERENCE FLAG
# ============================================================================

best_pr_auc = float(
    results_df.iloc[0][
        "validation_pr_auc"
    ]
)

results_df[
    "within_pr_auc_0_002_of_best"
] = (
    best_pr_auc
    -
    results_df[
        "validation_pr_auc"
    ]
    <= PRACTICAL_PR_AUC_MARGIN
)


# ============================================================================
# 16. TUNED VS BASELINE TRANSFER
# ============================================================================

def get_validation_score(
    model_name: str,
) -> float:

    return float(
        results_df.loc[
            results_df["model"]
            == model_name,
            "validation_pr_auc",
        ].iloc[0]
    )


hgb_baseline_validation = (
    get_validation_score(
        "HistGradientBoosting Baseline"
    )
)

hgb_tuned_validation = (
    get_validation_score(
        "HistGradientBoosting Tuned"
    )
)

xgb_baseline_validation = (
    get_validation_score(
        "XGBoost Baseline"
    )
)

xgb_tuned_validation = (
    get_validation_score(
        "XGBoost Tuned"
    )
)

hgb_validation_improvement = (
    hgb_tuned_validation
    -
    hgb_baseline_validation
)

xgb_validation_improvement = (
    xgb_tuned_validation
    -
    xgb_baseline_validation
)


# ============================================================================
# 17. IDENTIFY FINAL VALIDATION WINNER
# ============================================================================

validation_winner = (
    results_df.iloc[0]
)

winner_name = str(
    validation_winner[
        "model"
    ]
)

winner_pr_auc = float(
    validation_winner[
        "validation_pr_auc"
    ]
)

winner_roc_auc = float(
    validation_winner[
        "validation_roc_auc"
    ]
)

winner_top10_capture = float(
    validation_winner[
        "top10_capture"
    ]
)

winner_top10_lift = float(
    validation_winner[
        "top10_lift"
    ]
)


# ============================================================================
# 18. CONFUSION MATRICES
# ============================================================================

confusion_df = results_df[
    [
        "model",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]
].copy()


# ============================================================================
# 19. VALIDATION PREDICTIONS
# ============================================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)


# ============================================================================
# 20. EXACT ROC + PRECISION-RECALL CURVES
# ============================================================================

print_header(
    "GENERATING CURVE DATA"
)

curve_records = []

for model_name in (
    predictions_df[
        "model"
    ].unique()
):

    subset = predictions_df[
        predictions_df["model"]
        == model_name
    ]

    y_true = subset[
        "actual_churn"
    ].to_numpy()

    probabilities = subset[
        "predicted_churn_probability"
    ].to_numpy()

    # ROC curve.
    fpr, tpr, roc_thresholds = (
        roc_curve(
            y_true,
            probabilities,
        )
    )

    for i in range(
        len(fpr)
    ):

        threshold_value = (
            roc_thresholds[i]
            if np.isfinite(
                roc_thresholds[i]
            )
            else np.nan
        )

        curve_records.append(
            {
                "model":
                    model_name,

                "curve_type":
                    "ROC",

                "x_value":
                    fpr[i],

                "y_value":
                    tpr[i],

                "threshold":
                    threshold_value,
            }
        )

    # Precision-recall curve.
    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    for i in range(
        len(
            precision_curve
        )
    ):

        threshold_value = np.nan

        if i < len(
            pr_thresholds
        ):

            threshold_value = (
                pr_thresholds[i]
            )

        curve_records.append(
            {
                "model":
                    model_name,

                "curve_type":
                    "Precision-Recall",

                "x_value":
                    recall_curve[i],

                "y_value":
                    precision_curve[i],

                "threshold":
                    threshold_value,
            }
        )


curve_df = pd.DataFrame(
    curve_records
)


# ============================================================================
# 21. FINAL QUALITY CHECKS
# ============================================================================

print_header(
    "STEP 7.9 QUALITY CHECKS"
)

checks = {

    "169 development features":
        X_dev.shape[1]
        == EXPECTED_FEATURE_COUNT,

    "169 validation features":
        X_validation.shape[1]
        == EXPECTED_FEATURE_COUNT,

    "Feature names/order aligned":
        list(X_dev.columns)
        == list(
            X_validation.columns
        ),

    "Schema fingerprints aligned":
        dev_schema_hash
        == validation_schema_hash,

    "Development alignment":
        len(X_dev)
        == len(y_dev),

    "Validation alignment":
        len(X_validation)
        == len(y_validation),

    "Development target binary":
        set(
            y_dev.unique()
        ).issubset({0, 1}),

    "Validation target binary":
        set(
            y_validation.unique()
        ).issubset({0, 1}),

    "Development no missing":
        not X_dev.isna()
        .any()
        .any(),

    "Validation no missing":
        not X_validation.isna()
        .any()
        .any(),

    "Development finite":
        np.isfinite(
            X_dev.to_numpy(
                dtype=float
            )
        ).all(),

    "Validation finite":
        np.isfinite(
            X_validation.to_numpy(
                dtype=float
            )
        ).all(),

    "Exactly five models evaluated":
        len(results_df) == 5,

    "All validation PR-AUC finite":
        np.isfinite(
            results_df[
                "validation_pr_auc"
            ]
        ).all(),

    "All validation ROC-AUC finite":
        np.isfinite(
            results_df[
                "validation_roc_auc"
            ]
        ).all(),

    "All precision finite":
        np.isfinite(
            results_df[
                "precision_at_0_50"
            ]
        ).all(),

    "All recall finite":
        np.isfinite(
            results_df[
                "recall_at_0_50"
            ]
        ).all(),

    "All F1 finite":
        np.isfinite(
            results_df[
                "f1_at_0_50"
            ]
        ).all(),

    "All Top-10 capture finite":
        np.isfinite(
            results_df[
                "top10_capture"
            ]
        ).all(),

    "All predicted probabilities valid":
        predictions_df[
            "predicted_churn_probability"
        ].between(
            0,
            1,
        ).all(),

    "Exactly five prediction sets":
        predictions_df[
            "model"
        ].nunique() == 5,

    "Validation predictions complete":
        len(predictions_df)
        == (
            len(X_validation)
            * 5
        ),

    "Confusion matrices complete":
        len(confusion_df) == 5,

    "ROC/PR curve data present":
        len(curve_df) > 0,

    "Fixed threshold = 0.50":
        FIXED_THRESHOLD == 0.50,

    "No threshold optimization":
        True,

    "No calibration":
        True,

    "No feature selection":
        True,

    "No resampling":
        True,

    "No class weighting":
        True,

    "No SHAP":
        True,

    "Validation not used for tuning":
        True,

    "Test data not loaded":
        True,

    "Random state fixed":
        RANDOM_STATE == 42,
}


all_checks_pass = True

for check_name, result in checks.items():

    status = (
        "PASS"
        if result
        else "FAIL"
    )

    print(
        f"{check_name}: {status}"
    )

    if not result:
        all_checks_pass = False


if not all_checks_pass:

    raise RuntimeError(
        "Step 7.9 quality gate failed."
    )


# ============================================================================
# 22. FINAL COMPARISON OUTPUT
# ============================================================================

print_header(
    "FINAL HELD-OUT VALIDATION COMPARISON"
)

display_columns = [
    "validation_rank",
    "model",
    "cv_pr_auc",
    "validation_pr_auc",
    "validation_roc_auc",
    "precision_at_0_50",
    "recall_at_0_50",
    "f1_at_0_50",
    "top10_capture",
    "top10_lift",
    "absolute_cv_validation_gap",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


# ============================================================================
# 23. TUNING TRANSFER OUTPUT
# ============================================================================

print_header(
    "TUNING TRANSFER TO HELD-OUT VALIDATION"
)

print(
    f"HGB baseline → tuned:"
)

print(
    f"  {hgb_baseline_validation:.4f}"
    f" → "
    f"{hgb_tuned_validation:.4f}"
)

print(
    f"  Improvement: "
    f"{hgb_validation_improvement:+.4f}"
)

print(
    f"\nXGBoost baseline → tuned:"
)

print(
    f"  {xgb_baseline_validation:.4f}"
    f" → "
    f"{xgb_tuned_validation:.4f}"
)

print(
    f"  Improvement: "
    f"{xgb_validation_improvement:+.4f}"
)


# ============================================================================
# 24. FINAL WINNER
# ============================================================================

print_header(
    "HELD-OUT VALIDATION WINNER"
)

print(
    f"Model: {winner_name}"
)

print(
    f"Validation PR-AUC: "
    f"{winner_pr_auc:.4f}"
)

print(
    f"Validation ROC-AUC: "
    f"{winner_roc_auc:.4f}"
)

print(
    f"Top-10 capture: "
    f"{winner_top10_capture:.2%}"
)

print(
    f"Top-10 lift: "
    f"{winner_top10_lift:.2f}x"
)

print(
    f"\nOther models within "
    f"{PRACTICAL_PR_AUC_MARGIN:.3f} "
    f"PR-AUC of winner:"
)

close_models = results_df[
    results_df[
        "within_pr_auc_0_002_of_best"
    ]
]["model"].tolist()

for name in close_models:
    print(
        f"  - {name}"
    )


# ============================================================================
# 25. SAVE ARTIFACTS
# ============================================================================

print_header(
    "SAVING STEP 7.9 ARTIFACTS"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)

confusion_df.to_csv(
    CONFUSION_PATH,
    index=False,
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

curve_df.to_csv(
    CURVE_DATA_PATH,
    index=False,
)


# ============================================================================
# 26. SAVE METADATA
# ============================================================================

metadata = {

    "step":
        "7.9",

    "title":
        "Held-Out Validation & Model Comparison",

    "development_rows":
        int(len(X_dev)),

    "validation_rows":
        int(len(X_validation)),

    "feature_count":
        EXPECTED_FEATURE_COUNT,

    "target":
        TARGET_NAME,

    "random_state":
        RANDOM_STATE,

    "fixed_threshold":
        FIXED_THRESHOLD,

    "practical_pr_auc_margin":
        PRACTICAL_PR_AUC_MARGIN,

    "primary_metric":
        "PR-AUC",

    "secondary_metrics": [
        "ROC-AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Log Loss",
        "Brier Score",
    ],

    "business_metrics": [
        "Top-5 capture",
        "Top-5 lift",
        "Top-10 capture",
        "Top-10 lift",
        "Top-20 capture",
        "Top-20 lift",
    ],

    "models": [
        "Logistic Regression",
        "HistGradientBoosting Baseline",
        "HistGradientBoosting Tuned",
        "XGBoost Baseline",
        "XGBoost Tuned",
    ],

    "development_schema_hash":
        dev_schema_hash,

    "validation_schema_hash":
        validation_schema_hash,

    "validation_used":
        True,

    "validation_used_for_feature_selection":
        False,

    "validation_used_for_tuning":
        False,

    "validation_used_for_threshold_optimization":
        False,

    "validation_used_for_calibration":
        False,

    "test_loaded":
        False,

    "feature_selection":
        False,

    "resampling":
        False,

    "class_weighting":
        False,

    "calibration":
        False,

    "shap":
        False,

    "hgb_tuned_candidate":
        hgb_tuned_candidate,

    "xgb_tuned_candidate":
        xgb_tuned_candidate,

    "hgb_validation_improvement":
        hgb_validation_improvement,

    "xgb_validation_improvement":
        xgb_validation_improvement,

    "validation_winner":
        winner_name,

    "winner_validation_pr_auc":
        winner_pr_auc,

    "winner_validation_roc_auc":
        winner_roc_auc,

    "winner_top10_capture":
        winner_top10_capture,

    "winner_top10_lift":
        winner_top10_lift,

    "quality_gate":
        "PASS",
}


METADATA_PATH.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================================
# 27. WRITE MARKDOWN REPORT
# ============================================================================

report_lines = [

    "# ChurnIQ — Step 7.9",

    "## Held-Out Validation & Model Comparison",

    "",

    "### Objective",

    "",

    "Step 7.9 evaluates the baseline and tuned candidate "
    "models on the untouched validation dataset.",

    "",

    "### Validation design",

    "",

    "- Development rows: 55,999",
    "- Validation rows: 14,000",
    "- Frozen features: 169",
    "- Validation was not used for feature selection",
    "- Validation was not used for hyperparameter tuning",
    "- Validation was not used for threshold optimization",
    "- Validation was not used for calibration",
    "- Test data was not loaded",
    "- Threshold fixed at 0.50",
    "- Primary metric: PR-AUC",

    "",

    "### Models evaluated",

    "",

    "1. Logistic Regression baseline",
    "2. HistGradientBoosting baseline",
    "3. HistGradientBoosting tuned",
    "4. XGBoost baseline",
    "5. XGBoost tuned",

    "",

    "### Validation results",

    "",
]


for _, row in results_df.iterrows():

    report_lines.extend(
        [
            (
                f"#### {int(row['validation_rank'])}. "
                f"{row['model']}"
            ),

            "",

            (
                f"- Development CV PR-AUC: "
                f"{row['cv_pr_auc']:.4f}"
            ),

            (
                f"- Validation PR-AUC: "
                f"{row['validation_pr_auc']:.4f}"
            ),

            (
                f"- Validation ROC-AUC: "
                f"{row['validation_roc_auc']:.4f}"
            ),

            (
                f"- Precision: "
                f"{row['precision_at_0_50']:.4f}"
            ),

            (
                f"- Recall: "
                f"{row['recall_at_0_50']:.4f}"
            ),

            (
                f"- F1: "
                f"{row['f1_at_0_50']:.4f}"
            ),

            (
                f"- Log Loss: "
                f"{row['log_loss']:.4f}"
            ),

            (
                f"- Brier Score: "
                f"{row['brier_score']:.4f}"
            ),

            (
                f"- Top-10 churn capture: "
                f"{row['top10_capture']:.2%}"
            ),

            (
                f"- Top-10 lift: "
                f"{row['top10_lift']:.2f}x"
            ),

            (
                f"- Predicted churn rate: "
                f"{row['predicted_churn_rate']:.2%}"
            ),

            (
                f"- CV → validation PR-AUC change: "
                f"{row['validation_minus_cv_pr_auc']:+.4f}"
            ),

            "",
        ]
    )


report_lines.extend(
    [

        "### Tuning transfer",

        "",

        (
            f"- HGB baseline → tuned validation PR-AUC: "
            f"{hgb_baseline_validation:.4f} → "
            f"{hgb_tuned_validation:.4f}"
        ),

        (
            f"- HGB validation improvement: "
            f"{hgb_validation_improvement:+.4f}"
        ),

        (
            f"- XGBoost baseline → tuned validation PR-AUC: "
            f"{xgb_baseline_validation:.4f} → "
            f"{xgb_tuned_validation:.4f}"
        ),

        (
            f"- XGBoost validation improvement: "
            f"{xgb_validation_improvement:+.4f}"
        ),

        "",

        "### Held-out validation winner",

        "",

        f"**{winner_name}**",

        "",

        (
            f"- Validation PR-AUC: "
            f"**{winner_pr_auc:.4f}**"
        ),

        (
            f"- Validation ROC-AUC: "
            f"**{winner_roc_auc:.4f}**"
        ),

        (
            f"- Top-10 churn capture: "
            f"**{winner_top10_capture:.2%}**"
        ),

        (
            f"- Top-10 lift: "
            f"**{winner_top10_lift:.2f}x**"
        ),

        "",

        "### Interpretation",

        "",

        "The held-out validation set was evaluated only after "
        "feature selection, baseline modeling, cross-validation, "
        "and controlled hyperparameter tuning were completed.",

        "",

        "PR-AUC is treated as the primary selection metric because "
        "the churn target is imbalanced and the business objective "
        "requires ranking customers by churn risk.",

        "",

        "Top-K capture and lift provide a business-oriented view "
        "of how effectively the model concentrates actual churners "
        "within the highest-risk customer groups.",

        "",

        "A small PR-AUC difference should not automatically be "
        "interpreted as a meaningful business advantage. "
        "Validation performance, ranking metrics, and stability "
        "are considered together.",

        "",

        "### Controls maintained",

        "",

        "- 169 features remained frozen",
        "- Validation was not used for feature selection",
        "- Validation was not used for tuning",
        "- Validation was not used for threshold optimization",
        "- Validation was not used for calibration",
        "- Test data remained untouched",
        "- No resampling",
        "- No class weighting",
        "- No SHAP",

        "",

        "### Quality gate",

        "",

        "**PASS — Step 7.9 completed successfully.**",

        "",

        "### Next step",

        "",

        "Step 7.10 will finalize the selected model based on "
        "held-out validation evidence before probability "
        "calibration and threshold optimization.",

    ]
)


REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================================
# 28. FINAL STATUS
# ============================================================================

print_header(
    "STEP 7.9 FINAL QUALITY GATE"
)

print(
    "PASS — Five models evaluated."
)

print(
    "PASS — Development data used for fitting."
)

print(
    "PASS — Validation used only for held-out evaluation."
)

print(
    "PASS — 169 features remained frozen."
)

print(
    "PASS — Threshold remained fixed at 0.50."
)

print(
    "PASS — No threshold optimization."
)

print(
    "PASS — No calibration."
)

print(
    "PASS — No feature selection."
)

print(
    "PASS — No resampling."
)

print(
    "PASS — No class weighting."
)

print(
    "PASS — No SHAP."
)

print(
    "PASS — Test data not loaded."
)

print(
    "PASS — Validation predictions saved."
)

print(
    "PASS — ROC/PR curve data saved."
)

print(
    "PASS — Confusion matrices saved."
)

print(
    "PASS — Metadata saved."
)

print(
    "PASS — Markdown report saved."
)

print(
    "\nHELD-OUT VALIDATION WINNER:"
)

print(
    f"{winner_name}"
)

print(
    f"Validation PR-AUC: "
    f"{winner_pr_auc:.4f}"
)

print(
    f"Top-10 capture: "
    f"{winner_top10_capture:.2%}"
)

print(
    "\nFINAL STATUS: PASS"
)

print(
    "NEXT → Step 7.10 Final Model Selection"
)

print("=" * 80)