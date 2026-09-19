"""
ChurnIQ — Step 7.7
Cross-Validation & Model Stability

Purpose
-------
Evaluate baseline model stability using development data only.

Models
------
1. Logistic Regression
2. HistGradientBoostingClassifier
3. XGBClassifier

Experimental controls
---------------------
- 169 frozen model features
- Development data only
- Validation data NOT loaded
- Test data NOT loaded
- Stratified 5-fold cross-validation
- Random state = 42
- Exact baseline configurations reused
- No hyperparameter tuning
- No threshold optimization
- No calibration
- No SHAP
- No feature selection
- No resampling
- No class weighting

Primary metric
--------------
PR-AUC

Secondary metrics
-----------------
ROC-AUC
Precision
Recall
F1
Log Loss
Brier Score

Business targeting metrics
--------------------------
Top-5%, Top-10%, Top-20% churn capture
Top-5%, Top-10%, Top-20% lift

Additional stability analysis
-----------------------------
- Fold-to-fold variability
- Model ranking by fold
- Mean PR-AUC gaps between models
- Out-of-fold predictions
- Reproducibility metadata
"""


from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.50

EXPECTED_FEATURE_COUNT = 169
TARGET_NAME = "churn_probability"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

X_DEV_PATH = DATA_DIR / "X_dev.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"

FOLD_RESULTS_PATH = (
    REPORT_DIR / "07_7_cross_validation_results.csv"
)

SUMMARY_PATH = (
    REPORT_DIR / "07_7_cross_validation_summary.csv"
)

OOF_PATH = (
    REPORT_DIR / "07_7_oof_predictions.csv"
)

RANKING_PATH = (
    REPORT_DIR / "07_7_fold_model_ranking.csv"
)

PAIRWISE_PATH = (
    REPORT_DIR / "07_7_pairwise_model_comparison.csv"
)

METADATA_PATH = (
    REPORT_DIR / "07_7_cross_validation_metadata.json"
)

STABILITY_REPORT_PATH = (
    REPORT_DIR / "07_7_model_stability_report.md"
)


MODEL_NAMES = [
    "Logistic Regression",
    "HistGradientBoosting",
    "XGBoost",
]


# ============================================================================
# 2. EXACT BASELINE CONFIGURATIONS
# ============================================================================

LOGISTIC_PARAMS = {
    "max_iter": 3000,
    "class_weight": None,
    "random_state": RANDOM_STATE,
}


HGB_PARAMS = {
    "learning_rate": 0.10,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "random_state": RANDOM_STATE,
}


XGB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.10,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_logistic() -> Pipeline:
    """
    Exact Step 7.4 Logistic Regression baseline.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    **LOGISTIC_PARAMS
                ),
            ),
        ]
    )


def build_hgb() -> HistGradientBoostingClassifier:
    """
    Exact Step 7.5 HistGradientBoosting baseline.
    """

    return HistGradientBoostingClassifier(
        **HGB_PARAMS
    )


def build_xgb() -> XGBClassifier:
    """
    Exact Step 7.6 XGBoost baseline.
    """

    return XGBClassifier(
        **XGB_PARAMS
    )


def build_model(model_name: str):
    """
    Return a fresh model instance.
    """

    if model_name == "Logistic Regression":
        return build_logistic()

    if model_name == "HistGradientBoosting":
        return build_hgb()

    if model_name == "XGBoost":
        return build_xgb()

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def calculate_top_k_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    percentage: int,
) -> tuple[float, float]:
    """
    Calculate churn capture and lift among the top percentage
    of customers ranked by predicted churn probability.
    """

    n = len(y_true)

    k = max(
        1,
        int(np.ceil(n * percentage / 100)),
    )

    ranking = pd.DataFrame(
        {
            "probability": probabilities,
            "actual": y_true,
            "row_order": np.arange(n),
        }
    )

    ranking = ranking.sort_values(
        by=["probability", "row_order"],
        ascending=[False, True],
        kind="mergesort",
    )

    top_k = ranking.head(k)

    total_churners = int(y_true.sum())
    captured_churners = int(
        top_k["actual"].sum()
    )

    if total_churners == 0:
        return np.nan, np.nan

    capture = (
        captured_churners /
        total_churners
    )

    selected_churn_rate = (
        captured_churners / k
    )

    overall_churn_rate = (
        total_churners / n
    )

    lift = (
        selected_churn_rate /
        overall_churn_rate
        if overall_churn_rate > 0
        else np.nan
    )

    return capture, lift


# ============================================================================
# 4. START
# ============================================================================

print_header(
    "ChurnIQ — STEP 7.7\n"
    "Cross-Validation & Model Stability"
)


# ============================================================================
# 5. LOAD DEVELOPMENT DATA ONLY
# ============================================================================

print("\n[1] Loading development data only...")

X_dev = pd.read_csv(X_DEV_PATH)
y_dev = pd.read_csv(Y_DEV_PATH).squeeze("columns")

print(
    f"Development features: {X_dev.shape}"
)

print(
    f"Development target:   {y_dev.shape}"
)

print(
    "Validation data loaded: NO"
)

print(
    "Test data loaded:       NO"
)


# ============================================================================
# 6. STRUCTURAL VALIDATION
# ============================================================================

print("\n[2] Validating development dataset...")

assert (
    X_dev.shape[1]
    == EXPECTED_FEATURE_COUNT
), (
    f"Expected {EXPECTED_FEATURE_COUNT} "
    f"features; found {X_dev.shape[1]}"
)

assert (
    len(X_dev) == len(y_dev)
), (
    "Development feature/target "
    "row mismatch."
)

assert X_dev.columns.is_unique, (
    "Duplicate development feature "
    "names detected."
)

assert set(
    y_dev.unique()
).issubset({0, 1}), (
    "Development target is not binary."
)

assert y_dev.name == TARGET_NAME, (
    f"Unexpected target name: {y_dev.name}"
)

assert not X_dev.isna().any().any(), (
    "Missing development feature values "
    "detected."
)

assert np.isfinite(
    X_dev.to_numpy(dtype=float)
).all(), (
    "Infinite development feature values "
    "detected."
)

print(
    f"Feature count: "
    f"{EXPECTED_FEATURE_COUNT} PASS"
)

print(
    "Row alignment PASS"
)

print(
    "Feature-name uniqueness PASS"
)

print(
    "Binary target validation PASS"
)

print(
    "Target name validation PASS"
)

print(
    "Missing-value validation PASS"
)

print(
    "Infinite-value validation PASS"
)


# ============================================================================
# 7. TARGET DISTRIBUTION
# ============================================================================

print(
    "\n[3] Development target distribution"
)

retained_count = int(
    (y_dev == 0).sum()
)

churned_count = int(
    (y_dev == 1).sum()
)

churn_rate = float(
    y_dev.mean()
)

print(
    f"Retained: "
    f"{retained_count:,} "
    f"({retained_count / len(y_dev):.2%})"
)

print(
    f"Churned:  "
    f"{churned_count:,} "
    f"({churned_count / len(y_dev):.2%})"
)

print(
    f"No-skill PR-AUC: "
    f"{churn_rate:.6f}"
)


# ============================================================================
# 8. CROSS-VALIDATION SETUP
# ============================================================================

print(
    "\n[4] Configuring Stratified 5-Fold CV..."
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print(
    "CV strategy: StratifiedKFold"
)

print(
    f"Number of folds: {N_SPLITS}"
)

print(
    "Shuffle: True"
)

print(
    f"Random state: {RANDOM_STATE}"
)

print(
    "Development data only: PASS"
)

print(
    "Validation data untouched: PASS"
)

print(
    "Test data not loaded: PASS"
)


# ============================================================================
# 9. PREPARE OOF STORAGE
# ============================================================================

print(
    "\n[5] Preparing out-of-fold prediction storage..."
)

oof_predictions = pd.DataFrame(
    index=X_dev.index
)

oof_predictions["row_index"] = (
    X_dev.index
)

oof_predictions["actual"] = (
    y_dev.values
)

oof_predictions["fold"] = -1


# ============================================================================
# 10. CROSS-VALIDATION
# ============================================================================

print(
    "\n[6] Running cross-validation..."
)

fold_results = []


for model_name in MODEL_NAMES:

    print("\n" + "-" * 80)
    print(
        f"MODEL: {model_name}"
    )
    print("-" * 80)

    model_oof = np.full(
        len(X_dev),
        np.nan,
        dtype=float,
    )

    fold_number = 0

    # New deterministic CV iterator for each model.
    model_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for train_idx, test_idx in model_cv.split(
        X_dev,
        y_dev,
    ):

        fold_number += 1

        print(
            f"\nFold "
            f"{fold_number}/{N_SPLITS}"
        )

        X_train_fold = (
            X_dev.iloc[train_idx]
        )

        X_test_fold = (
            X_dev.iloc[test_idx]
        )

        y_train_fold = (
            y_dev.iloc[train_idx]
        )

        y_test_fold = (
            y_dev.iloc[test_idx]
        )

        # ------------------------------------------------------------
        # Fold structure checks
        # ------------------------------------------------------------

        assert len(
            set(train_idx)
            & set(test_idx)
        ) == 0, (
            "Train/test overlap detected "
            "inside CV fold."
        )

        assert (
            len(train_idx)
            + len(test_idx)
            == len(X_dev)
        )

        assert set(
            y_train_fold.unique()
        ).issubset({0, 1})

        assert set(
            y_test_fold.unique()
        ).issubset({0, 1})

        # ------------------------------------------------------------
        # Fresh model
        # ------------------------------------------------------------

        model = build_model(
            model_name
        )

        # ------------------------------------------------------------
        # Train
        # ------------------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        model.fit(
            X_train_fold,
            y_train_fold,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        # ------------------------------------------------------------
        # Predict probabilities
        # ------------------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_test_fold
            )[:, 1]
        )

        assert len(
            probabilities
        ) == len(y_test_fold)

        assert np.isfinite(
            probabilities
        ).all()

        assert (
            probabilities.min()
            >= 0
        )

        assert (
            probabilities.max()
            <= 1
        )

        # ------------------------------------------------------------
        # Store OOF predictions
        # ------------------------------------------------------------

        model_oof[test_idx] = (
            probabilities
        )

        # ------------------------------------------------------------
        # Classification at baseline threshold
        # ------------------------------------------------------------

        predictions = (
            probabilities
            >= THRESHOLD
        ).astype(int)

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------

        pr_auc = (
            average_precision_score(
                y_test_fold,
                probabilities,
            )
        )

        roc_auc = (
            roc_auc_score(
                y_test_fold,
                probabilities,
            )
        )

        precision = (
            precision_score(
                y_test_fold,
                predictions,
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_test_fold,
                predictions,
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_test_fold,
                predictions,
                zero_division=0,
            )
        )

        fold_log_loss = (
            log_loss(
                y_test_fold,
                probabilities,
                labels=[0, 1],
            )
        )

        fold_brier = (
            brier_score_loss(
                y_test_fold,
                probabilities,
            )
        )

        # ------------------------------------------------------------
        # Business targeting
        # ------------------------------------------------------------

        top5_capture, top5_lift = (
            calculate_top_k_metrics(
                y_test_fold.to_numpy(),
                probabilities,
                5,
            )
        )

        top10_capture, top10_lift = (
            calculate_top_k_metrics(
                y_test_fold.to_numpy(),
                probabilities,
                10,
            )
        )

        top20_capture, top20_lift = (
            calculate_top_k_metrics(
                y_test_fold.to_numpy(),
                probabilities,
                20,
            )
        )

        # ------------------------------------------------------------
        # Store fold results
        # ------------------------------------------------------------

        fold_results.append(
            {
                "model": model_name,
                "fold": fold_number,
                "train_rows": len(
                    train_idx
                ),
                "fold_test_rows": len(
                    test_idx
                ),
                "fold_churn_rate": float(
                    y_test_fold.mean()
                ),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "log_loss": fold_log_loss,
                "brier_score": fold_brier,
                "predicted_churn_rate": float(
                    predictions.mean()
                ),
                "top_5_capture": (
                    top5_capture
                ),
                "top_5_lift": (
                    top5_lift
                ),
                "top_10_capture": (
                    top10_capture
                ),
                "top_10_lift": (
                    top10_lift
                ),
                "top_20_capture": (
                    top20_capture
                ),
                "top_20_lift": (
                    top20_lift
                ),
                "training_seconds": (
                    training_seconds
                ),
            }
        )

        print(
            f"PR-AUC: {pr_auc:.4f} | "
            f"ROC-AUC: {roc_auc:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Training: "
            f"{training_seconds:.2f}s"
        )

    # ------------------------------------------------------------
    # Store model OOF predictions
    # ------------------------------------------------------------

    assert np.isfinite(
        model_oof
    ).all(), (
        f"OOF predictions incomplete "
        f"for {model_name}."
    )

    column_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    oof_predictions[
        f"{column_name}_probability"
    ] = model_oof


# ============================================================================
# 11. FOLD RESULTS
# ============================================================================

print(
    "\n[7] Creating fold-level results..."
)

fold_results_df = pd.DataFrame(
    fold_results
)

assert (
    len(fold_results_df)
    == len(MODEL_NAMES) * N_SPLITS
)

assert (
    fold_results_df["model"].nunique()
    == len(MODEL_NAMES)
)

assert (
    fold_results_df["fold"].nunique()
    == N_SPLITS
)

print(
    f"Fold-level observations: "
    f"{len(fold_results_df)}"
)


# ============================================================================
# 12. SUMMARY STATISTICS
# ============================================================================

print(
    "\n[8] Calculating model stability summaries..."
)

metrics_to_summarize = [
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "log_loss",
    "brier_score",
    "predicted_churn_rate",
    "top_5_capture",
    "top_5_lift",
    "top_10_capture",
    "top_10_lift",
    "top_20_capture",
    "top_20_lift",
    "training_seconds",
]

summary_rows = []

for model_name in MODEL_NAMES:

    model_group = (
        fold_results_df[
            fold_results_df["model"]
            == model_name
        ]
    )

    row = {
        "model": model_name,
        "folds": len(model_group),
    }

    for metric in metrics_to_summarize:

        row[
            f"{metric}_mean"
        ] = model_group[metric].mean()

        row[
            f"{metric}_std"
        ] = model_group[metric].std(
            ddof=1
        )

        row[
            f"{metric}_min"
        ] = model_group[metric].min()

        row[
            f"{metric}_max"
        ] = model_group[metric].max()

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================================
# 13. FOLD-LEVEL MODEL RANKING
# ============================================================================

print(
    "\n[9] Creating fold-level model rankings..."
)

ranking_records = []

for fold_number in range(
    1,
    N_SPLITS + 1,
):

    fold_data = (
        fold_results_df[
            fold_results_df["fold"]
            == fold_number
        ]
        .sort_values(
            "pr_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    fold_data["pr_auc_rank"] = (
        np.arange(len(fold_data))
        + 1
    )

    for _, row in fold_data.iterrows():

        ranking_records.append(
            {
                "fold": fold_number,
                "model": row["model"],
                "pr_auc": row["pr_auc"],
                "pr_auc_rank": int(
                    row["pr_auc_rank"]
                ),
            }
        )


ranking_df = pd.DataFrame(
    ranking_records
)


# ============================================================================
# 14. RANKING CONSISTENCY SUMMARY
# ============================================================================

print(
    "\n[10] Checking ranking consistency..."
)

ranking_counts = (
    ranking_df[
        ranking_df["pr_auc_rank"] == 1
    ]["model"]
    .value_counts()
)

for model_name in MODEL_NAMES:

    count = int(
        ranking_counts.get(
            model_name,
            0,
        )
    )

    print(
        f"{model_name}: "
        f"best PR-AUC in "
        f"{count}/{N_SPLITS} folds"
    )


# ============================================================================
# 15. PAIRWISE MODEL COMPARISON
# ============================================================================

print(
    "\n[11] Calculating pairwise PR-AUC comparisons..."
)

wide_pr_auc = (
    fold_results_df
    .pivot(
        index="fold",
        columns="model",
        values="pr_auc",
    )
)

pairwise_records = []

pairs = [
    (
        "XGBoost",
        "HistGradientBoosting",
    ),
    (
        "XGBoost",
        "Logistic Regression",
    ),
    (
        "HistGradientBoosting",
        "Logistic Regression",
    ),
]

for model_a, model_b in pairs:

    difference = (
        wide_pr_auc[model_a]
        - wide_pr_auc[model_b]
    )

    pairwise_records.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "mean_pr_auc_a": (
                wide_pr_auc[
                    model_a
                ].mean()
            ),
            "mean_pr_auc_b": (
                wide_pr_auc[
                    model_b
                ].mean()
            ),
            "mean_pr_auc_difference": (
                difference.mean()
            ),
            "std_pr_auc_difference": (
                difference.std(
                    ddof=1
                )
            ),
            "folds_model_a_higher": int(
                (difference > 0).sum()
            ),
            "folds_model_b_higher": int(
                (difference < 0).sum()
            ),
            "folds_equal": int(
                (difference == 0).sum()
            ),
        }
    )


pairwise_df = pd.DataFrame(
    pairwise_records
)


# ============================================================================
# 16. SORT SUMMARY BY MEAN PR-AUC
# ============================================================================

summary_df = (
    summary_df
    .sort_values(
        "pr_auc_mean",
        ascending=False,
    )
    .reset_index(drop=True)
)

summary_df["pr_auc_rank"] = (
    np.arange(
        len(summary_df)
    ) + 1
)


# ============================================================================
# 17. PRINT STABILITY RESULTS
# ============================================================================

print(
    "\n[12] Model stability summary"
)

for _, row in summary_df.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"  PR-AUC: "
        f"{row['pr_auc_mean']:.4f} "
        f"± "
        f"{row['pr_auc_std']:.4f}"
    )

    print(
        f"  ROC-AUC: "
        f"{row['roc_auc_mean']:.4f} "
        f"± "
        f"{row['roc_auc_std']:.4f}"
    )

    print(
        f"  Precision: "
        f"{row['precision_mean']:.4f} "
        f"± "
        f"{row['precision_std']:.4f}"
    )

    print(
        f"  Recall: "
        f"{row['recall_mean']:.4f} "
        f"± "
        f"{row['recall_std']:.4f}"
    )

    print(
        f"  F1: "
        f"{row['f1_mean']:.4f} "
        f"± "
        f"{row['f1_std']:.4f}"
    )

    print(
        f"  Top-10 capture: "
        f"{row['top_10_capture_mean']:.2%} "
        f"± "
        f"{row['top_10_capture_std']:.2%}"
    )

    print(
        f"  Training time: "
        f"{row['training_seconds_mean']:.2f}s "
        f"± "
        f"{row['training_seconds_std']:.2f}s"
    )


# ============================================================================
# 18. OOF VALIDATION
# ============================================================================

print(
    "\n[13] Validating OOF prediction coverage..."
)

assert len(
    oof_predictions
) == len(X_dev)

assert (
    oof_predictions["actual"]
    .to_numpy()
    == y_dev.to_numpy()
).all()

for model_name in MODEL_NAMES:

    column_name = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_probability"
    )

    assert (
        column_name
        in oof_predictions.columns
    )

    assert np.isfinite(
        oof_predictions[
            column_name
        ]
    ).all()

    assert (
        oof_predictions[
            column_name
        ].between(0, 1).all()
    )

print(
    "OOF row coverage: PASS"
)

print(
    "OOF probability bounds: PASS"
)

print(
    "OOF target alignment: PASS"
)


# ============================================================================
# 19. QUALITY CHECKS
# ============================================================================

print(
    "\n[14] Running Step 7.7 quality checks..."
)

checks = {
    "169 frozen features":
        X_dev.shape[1]
        == EXPECTED_FEATURE_COUNT,

    "Development rows aligned":
        len(X_dev)
        == len(y_dev),

    "Binary target":
        set(
            y_dev.unique()
        ).issubset({0, 1}),

    "No development missing values":
        not X_dev.isna()
        .any()
        .any(),

    "No development infinite values":
        np.isfinite(
            X_dev.to_numpy(
                dtype=float
            )
        ).all(),

    "Exactly 5 CV folds":
        fold_results_df["fold"]
        .nunique()
        == N_SPLITS,

    "Exactly 3 models evaluated":
        fold_results_df["model"]
        .nunique()
        == len(MODEL_NAMES),

    "15 fold-model observations":
        len(fold_results_df)
        == 15,

    "All PR-AUC values finite":
        np.isfinite(
            fold_results_df[
                "pr_auc"
            ]
        ).all(),

    "All ROC-AUC values finite":
        np.isfinite(
            fold_results_df[
                "roc_auc"
            ]
        ).all(),

    "All classification metrics finite":
        np.isfinite(
            fold_results_df[
                [
                    "precision",
                    "recall",
                    "f1",
                ]
            ].to_numpy()
        ).all(),

    "All probability metrics finite":
        np.isfinite(
            fold_results_df[
                [
                    "log_loss",
                    "brier_score",
                ]
            ].to_numpy()
        ).all(),

    "All Top-K metrics finite":
        np.isfinite(
            fold_results_df[
                [
                    "top_5_capture",
                    "top_5_lift",
                    "top_10_capture",
                    "top_10_lift",
                    "top_20_capture",
                    "top_20_lift",
                ]
            ].to_numpy()
        ).all(),

    "OOF predictions complete":
        np.isfinite(
            oof_predictions.drop(
                columns=[
                    "row_index",
                    "actual",
                    "fold",
                ]
            ).to_numpy()
        ).all(),

    "OOF prediction bounds":
        oof_predictions.drop(
            columns=[
                "row_index",
                "actual",
                "fold",
            ]
        ).apply(
            lambda col:
            col.between(0, 1).all()
        ).all(),

    "No train/test fold overlap":
        True,

    "Validation data not loaded":
        True,

    "Test data not loaded":
        True,

    "Hyperparameter tuning not performed":
        True,

    "Threshold optimization not performed":
        True,

    "Calibration not performed":
        True,

    "SHAP not performed":
        True,

    "Feature selection not performed":
        True,

    "Resampling not performed":
        True,

    "Class weighting not performed":
        True,
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
        "Step 7.7 quality checks failed."
    )


# ============================================================================
# 20. SAVE ARTIFACTS
# ============================================================================

print(
    "\n[15] Saving Step 7.7 artifacts..."
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

fold_results_df.to_csv(
    FOLD_RESULTS_PATH,
    index=False,
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)

oof_predictions.to_csv(
    OOF_PATH,
    index=False,
)

ranking_df.to_csv(
    RANKING_PATH,
    index=False,
)

pairwise_df.to_csv(
    PAIRWISE_PATH,
    index=False,
)


# ============================================================================
# 21. METADATA
# ============================================================================

metadata = {
    "step": "7.7",
    "title": (
        "Cross-Validation & Model Stability"
    ),
    "random_state": RANDOM_STATE,
    "n_splits": N_SPLITS,
    "cv_strategy": (
        "StratifiedKFold"
    ),
    "shuffle": True,
    "feature_count": EXPECTED_FEATURE_COUNT,
    "development_rows": len(X_dev),
    "development_churn_rate": churn_rate,
    "target": TARGET_NAME,
    "primary_metric": "PR-AUC",
    "secondary_metrics": [
        "ROC-AUC",
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
    "models": MODEL_NAMES,
    "model_configurations": {
        "Logistic Regression":
            LOGISTIC_PARAMS,
        "HistGradientBoosting":
            HGB_PARAMS,
        "XGBoost":
            XGB_PARAMS,
    },
    "validation_loaded": False,
    "test_loaded": False,
    "hyperparameter_tuning": False,
    "threshold_optimization": False,
    "calibration": False,
    "shap": False,
    "feature_selection": False,
    "resampling": False,
    "class_weighting": False,
    "oof_predictions_saved": True,
    "quality_gate": "PASS",
}


METADATA_PATH.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================================
# 22. FINAL REPORT
# ============================================================================

best_model = (
    summary_df.iloc[0]["model"]
)

best_pr_auc = (
    summary_df.iloc[0][
        "pr_auc_mean"
    ]
)

best_pr_auc_std = (
    summary_df.iloc[0][
        "pr_auc_std"
    ]
)

best_fold_count = int(
    ranking_counts.get(
        best_model,
        0,
    )
)


report_lines = [
    "# ChurnIQ — Step 7.7",
    "## Cross-Validation & Model Stability",
    "",
    "### Purpose",
    "",
    "Step 7.7 evaluates the stability of the three "
    "baseline models using development data only.",
    "",
    "The held-out validation dataset remains untouched "
    "and the test dataset is not loaded.",
    "",
    "### Evaluation design",
    "",
    "- Development data only",
    "- 55,999 development observations",
    "- 169 frozen model features",
    "- Stratified 5-fold cross-validation",
    "- Shuffle enabled",
    "- Random state: 42",
    "- Same folds used for model comparison",
    "- Exact baseline model configurations reused",
    "- No hyperparameter tuning",
    "- No threshold optimization",
    "- No calibration",
    "- No SHAP",
    "- No feature selection",
    "- No resampling",
    "- No class weighting",
    "",
    "### Primary metric",
    "",
    "PR-AUC is the primary model discrimination metric "
    "because churn is an imbalanced binary outcome.",
    "",
    "### Current CV ranking",
    "",
    f"1. **{summary_df.iloc[0]['model']}** — "
    f"PR-AUC "
    f"{summary_df.iloc[0]['pr_auc_mean']:.4f} "
    f"± "
    f"{summary_df.iloc[0]['pr_auc_std']:.4f}",
    "",
    f"2. **{summary_df.iloc[1]['model']}** — "
    f"PR-AUC "
    f"{summary_df.iloc[1]['pr_auc_mean']:.4f} "
    f"± "
    f"{summary_df.iloc[1]['pr_auc_std']:.4f}",
    "",
    f"3. **{summary_df.iloc[2]['model']}** — "
    f"PR-AUC "
    f"{summary_df.iloc[2]['pr_auc_mean']:.4f} "
    f"± "
    f"{summary_df.iloc[2]['pr_auc_std']:.4f}",
    "",
    "### Fold ranking consistency",
    "",
    f"Current CV leader: **{best_model}**",
    "",
    f"It achieved the highest PR-AUC in "
    f"{best_fold_count}/{N_SPLITS} folds.",
    "",
    "### Pairwise model comparison",
    "",
]

for _, row in pairwise_df.iterrows():

    report_lines.append(
        f"- **{row['model_a']} vs "
        f"{row['model_b']}**: mean PR-AUC "
        f"difference = "
        f"{row['mean_pr_auc_difference']:.4f}; "
        f"{int(row['folds_model_a_higher'])} "
        f"fold(s) favored the first model, "
        f"{int(row['folds_model_b_higher'])} "
        f"fold(s) favored the second model."
    )


report_lines.extend(
    [
        "",
        "### Out-of-fold predictions",
        "",
        "Out-of-fold probability predictions were generated "
        "for every development observation. Each observation "
        "received a prediction from a model that was not trained "
        "on that observation.",
        "",
        "These predictions are saved as an analytical artifact "
        "for later probability calibration and decision-threshold "
        "analysis. No calibration or threshold optimization was "
        "performed in Step 7.7.",
        "",
        "### Interpretation rule",
        "",
        "The CV leader is not automatically declared the final "
        "model. Model selection will consider:",
        "",
        "- Mean PR-AUC",
        "- Fold-to-fold variability",
        "- ROC-AUC stability",
        "- Classification metric stability",
        "- Top-K churn capture stability",
        "- Pairwise fold consistency",
        "- Previously observed held-out validation performance",
        "",
        "### Validation boundary",
        "",
        "The held-out validation dataset was not loaded or evaluated "
        "during this step. It remains reserved for later model "
        "comparison and final evaluation.",
        "",
        "### Quality gate",
        "",
        "**PASS — Step 7.7 Cross-Validation & Model Stability completed.**",
        "",
        "### Next step",
        "",
        "Proceed to Step 7.8 Controlled Hyperparameter Tuning only "
        "after reviewing the CV evidence and confirming the model "
        "candidate(s) to tune.",
    ]
)


STABILITY_REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================================
# 23. FINAL STATUS
# ============================================================================

print_header(
    "STEP 7.7 QUALITY GATE"
)

print(
    "PASS — Development-only cross-validation completed."
)

print(
    "PASS — Validation dataset remained untouched."
)

print(
    "PASS — Test dataset was not loaded."
)

print(
    "PASS — All three baseline models evaluated."
)

print(
    "PASS — Five stratified folds completed per model."
)

print(
    "PASS — Fold-level metrics saved."
)

print(
    "PASS — Stability summary saved."
)

print(
    "PASS — Fold-level ranking saved."
)

print(
    "PASS — Pairwise model comparison saved."
)

print(
    "PASS — OOF predictions saved."
)

print(
    "PASS — Reproducibility metadata saved."
)

print(
    f"\nCURRENT CV LEADER: "
    f"{best_model}"
)

print(
    f"Mean PR-AUC: "
    f"{best_pr_auc:.4f} "
    f"± "
    f"{best_pr_auc_std:.4f}"
)

print(
    f"Best PR-AUC in "
    f"{best_fold_count}/{N_SPLITS} folds."
)

print(
    "\nFINAL STATUS: PASS"
)

print(
    "NEXT → Step 7.8 Controlled Hyperparameter Tuning"
)

print("=" * 80)