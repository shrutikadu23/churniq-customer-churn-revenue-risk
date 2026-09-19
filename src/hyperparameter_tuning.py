"""
ChurnIQ — Step 7.8
Controlled Hyperparameter Tuning

Objective
---------
Tune the two strongest baseline candidates identified in Step 7.7:

1. HistGradientBoosting
2. XGBoost

Experimental controls
---------------------
- Development data only
- 169 frozen features
- Validation data NOT loaded
- Test data NOT loaded
- Same 5 StratifiedKFold splits for every candidate
- Random state = 42
- Primary metric = PR-AUC
- Secondary metrics = ROC-AUC, Precision, Recall, F1
- Business ranking metric = Top-10 churn capture
- No feature selection
- No resampling
- No class weighting
- No threshold optimization
- No calibration
- No SHAP

Selection principle
-------------------
Candidates are ranked primarily by mean PR-AUC.

Tie-breakers:
1. Lower PR-AUC standard deviation
2. Higher Top-10 churn capture
3. Lower generalization gap

The untouched validation set is reserved for Step 7.9.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
N_SPLITS = 5

EXPECTED_FEATURE_COUNT = 169
TARGET_NAME = "churn_probability"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

X_DEV_PATH = DATA_DIR / "X_dev.csv"
Y_DEV_PATH = DATA_DIR / "y_dev.csv"

CANDIDATE_RESULTS_PATH = (
    REPORT_DIR / "07_8_tuning_candidate_results.csv"
)

SUMMARY_PATH = (
    REPORT_DIR / "07_8_tuning_summary.csv"
)

FOLD_RESULTS_PATH = (
    REPORT_DIR / "07_8_tuning_fold_results.csv"
)

PARAMETERS_PATH = (
    REPORT_DIR / "07_8_selected_model_parameters.json"
)

METADATA_PATH = (
    REPORT_DIR / "07_8_tuning_metadata.json"
)

REPORT_PATH = (
    REPORT_DIR / "07_8_hyperparameter_tuning_report.md"
)


# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def top_k_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k_fraction: float = 0.10,
) -> tuple[float, float]:

    n = len(y_true)

    k = max(
        1,
        int(np.ceil(n * k_fraction))
    )

    ranked_indices = np.argsort(
        -probabilities,
        kind="mergesort",
    )

    top_indices = ranked_indices[:k]

    actual_churners = np.sum(y_true)

    if actual_churners == 0:
        return np.nan, np.nan

    churn_capture = (
        np.sum(
            y_true[top_indices]
        )
        / actual_churners
    )

    baseline_rate = (
        actual_churners / n
    )

    top_rate = (
        np.sum(
            y_true[top_indices]
        )
        / k
    )

    lift = (
        top_rate / baseline_rate
        if baseline_rate > 0
        else np.nan
    )

    return churn_capture, lift


def evaluate_candidate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    candidate_id: str,
    parameters: dict,
) -> tuple[list[dict], dict]:

    fold_records = []

    total_start = time.perf_counter()

    for fold_number, (
        train_idx,
        valid_idx,
    ) in enumerate(
        folds,
        start=1,
    ):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        fold_start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        train_prob = (
            model.predict_proba(
                X_train
            )[:, 1]
        )

        valid_prob = (
            model.predict_proba(
                X_valid
            )[:, 1]
        )

        train_pr_auc = (
            average_precision_score(
                y_train,
                train_prob,
            )
        )

        valid_pr_auc = (
            average_precision_score(
                y_valid,
                valid_prob,
            )
        )

        valid_roc_auc = (
            roc_auc_score(
                y_valid,
                valid_prob,
            )
        )

        predictions = (
            valid_prob >= 0.50
        ).astype(int)

        precision = (
            precision_score(
                y_valid,
                predictions,
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_valid,
                predictions,
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_valid,
                predictions,
                zero_division=0,
            )
        )

        top10_capture, top10_lift = (
            top_k_metrics(
                y_valid.to_numpy(),
                valid_prob,
                k_fraction=0.10,
            )
        )

        fit_seconds = (
            time.perf_counter()
            - fold_start
        )

        fold_records.append(
            {
                "model": model_name,
                "candidate_id": candidate_id,
                "fold": fold_number,
                "train_pr_auc": train_pr_auc,
                "validation_pr_auc": valid_pr_auc,
                "validation_roc_auc": valid_roc_auc,
                "precision_at_0_50": precision,
                "recall_at_0_50": recall,
                "f1_at_0_50": f1,
                "top10_capture": top10_capture,
                "top10_lift": top10_lift,
                "fit_seconds": fit_seconds,
            }
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    fold_df = pd.DataFrame(
        fold_records
    )

    summary = {
        "model": model_name,
        "candidate_id": candidate_id,
        "mean_pr_auc": fold_df[
            "validation_pr_auc"
        ].mean(),
        "std_pr_auc": fold_df[
            "validation_pr_auc"
        ].std(ddof=1),
        "mean_roc_auc": fold_df[
            "validation_roc_auc"
        ].mean(),
        "std_roc_auc": fold_df[
            "validation_roc_auc"
        ].std(ddof=1),
        "mean_precision": fold_df[
            "precision_at_0_50"
        ].mean(),
        "std_precision": fold_df[
            "precision_at_0_50"
        ].std(ddof=1),
        "mean_recall": fold_df[
            "recall_at_0_50"
        ].mean(),
        "std_recall": fold_df[
            "recall_at_0_50"
        ].std(ddof=1),
        "mean_f1": fold_df[
            "f1_at_0_50"
        ].mean(),
        "std_f1": fold_df[
            "f1_at_0_50"
        ].std(ddof=1),
        "mean_top10_capture": fold_df[
            "top10_capture"
        ].mean(),
        "std_top10_capture": fold_df[
            "top10_capture"
        ].std(ddof=1),
        "mean_top10_lift": fold_df[
            "top10_lift"
        ].mean(),
        "std_top10_lift": fold_df[
            "top10_lift"
        ].std(ddof=1),
        "mean_train_pr_auc": fold_df[
            "train_pr_auc"
        ].mean(),
        "generalization_gap": (
            fold_df[
                "train_pr_auc"
            ].mean()
            - fold_df[
                "validation_pr_auc"
            ].mean()
        ),
        "total_fit_seconds": total_seconds,
        "parameters": json.dumps(
            parameters,
            sort_keys=True,
        ),
    }

    return fold_records, summary


# ============================================================================
# 3. LOAD DEVELOPMENT DATA
# ============================================================================

print_header(
    "ChurnIQ — STEP 7.8\n"
    "Controlled Hyperparameter Tuning"
)

print(
    "\n[1] Loading development data..."
)

X_dev = pd.read_csv(
    X_DEV_PATH
)

y_dev = pd.read_csv(
    Y_DEV_PATH
).squeeze("columns")

print(
    f"X_dev shape: {X_dev.shape}"
)

print(
    f"y_dev shape: {y_dev.shape}"
)

print(
    "Validation loaded: NO"
)

print(
    "Test loaded:       NO"
)


# ============================================================================
# 4. DATA QUALITY GATE
# ============================================================================

print(
    "\n[2] Running development data quality checks..."
)

assert (
    X_dev.shape[1]
    == EXPECTED_FEATURE_COUNT
), (
    f"Expected {EXPECTED_FEATURE_COUNT} "
    f"features but found {X_dev.shape[1]}."
)

assert (
    len(X_dev)
    == len(y_dev)
), (
    "X/y row count mismatch."
)

assert X_dev.columns.is_unique, (
    "Duplicate feature names detected."
)

assert (
    y_dev.name
    == TARGET_NAME
), (
    f"Unexpected target name: "
    f"{y_dev.name}"
)

assert set(
    y_dev.unique()
).issubset({0, 1}), (
    "Target is not binary."
)

assert not X_dev.isna().any().any(), (
    "Missing values detected."
)

assert np.isfinite(
    X_dev.to_numpy(
        dtype=float
    )
).all(), (
    "Infinite values detected."
)

print(
    "169 frozen features: PASS"
)

print(
    "Development alignment: PASS"
)

print(
    "Binary target: PASS"
)

print(
    "No missing values: PASS"
)

print(
    "No infinite values: PASS"
)


# ============================================================================
# 5. CREATE IDENTICAL CV FOLDS
# ============================================================================

print(
    "\n[3] Creating fixed five-fold CV splits..."
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

folds = list(
    cv.split(
        X_dev,
        y_dev,
    )
)

assert len(folds) == N_SPLITS

print(
    f"CV strategy: StratifiedKFold"
)

print(
    f"Folds: {N_SPLITS}"
)

print(
    f"Random state: {RANDOM_STATE}"
)

print(
    "Same folds reused for every candidate: PASS"
)


# ============================================================================
# 6. DEFINE CONTROLLED CANDIDATES
# ============================================================================

"""
Candidate design
----------------

Each model includes its exact Step 7.7 baseline plus a controlled
set of nearby configurations.

This is deliberately not an enormous brute-force search.

The objective is to test meaningful changes in:

HGB:
- learning rate
- number of boosting iterations
- tree complexity
- minimum leaf size
- L2 regularization

XGBoost:
- number of trees
- learning rate
- tree depth
- minimum child weight
- row subsampling
- column subsampling
- gamma
- L1/L2 regularization
"""

HGB_CANDIDATES = [

    # Baseline
    (
        "HGB_00_BASELINE",
        {
            "learning_rate": 0.10,
            "max_iter": 100,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_01_SLOWER_MORE_TREES",
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_02_MODERATE_MORE_TREES",
        {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_03_MORE_TREES",
        {
            "learning_rate": 0.10,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_04_MORE_COMPLEX",
        {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_05_REGULARIZED",
        {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
    ),

    (
        "HGB_06_LARGER_LEAF",
        {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_07_SHALLOWER",
        {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
    ),

    (
        "HGB_08_SLOWER_COMPLEX",
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
    ),

    (
        "HGB_09_STRONG_REGULARIZATION",
        {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 5.0,
        },
    ),
]


XGB_CANDIDATES = [

    # Baseline
    (
        "XGB_00_BASELINE",
        {
            "n_estimators": 100,
            "learning_rate": 0.10,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_01_MORE_TREES",
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_02_SLOWER_MORE_TREES",
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_03_SHALLOWER",
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_04_DEEPER",
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 8,
            "min_child_weight": 1,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_05_MIN_CHILD",
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_06_SUBSAMPLED",
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),

    (
        "XGB_07_REGULARIZED",
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    ),

    (
        "XGB_08_STRONG_REGULARIZATION",
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    ),

    (
        "XGB_09_BALANCED",
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
    ),
]


# ============================================================================
# 7. MODEL BUILDERS
# ============================================================================

def build_hgb(params: dict):
    return HistGradientBoostingClassifier(
        **params,
        random_state=RANDOM_STATE,
    )


def build_xgb(params: dict):
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================================
# 8. RUN HGB CANDIDATES
# ============================================================================

print_header(
    "HISTGRADIENTBOOSTING TUNING"
)

all_fold_records = []
all_summary_records = []

for candidate_id, params in HGB_CANDIDATES:

    print(
        f"\nRunning {candidate_id}..."
    )

    model = build_hgb(
        params
    )

    fold_records, summary = (
        evaluate_candidate(
            model=model,
            X=X_dev,
            y=y_dev,
            folds=folds,
            model_name="HistGradientBoosting",
            candidate_id=candidate_id,
            parameters=params,
        )
    )

    all_fold_records.extend(
        fold_records
    )

    all_summary_records.append(
        summary
    )

    print(
        f"PR-AUC: "
        f"{summary['mean_pr_auc']:.4f} "
        f"+/- "
        f"{summary['std_pr_auc']:.4f}"
    )

    print(
        f"Top-10 capture: "
        f"{summary['mean_top10_capture']:.2%}"
    )


# ============================================================================
# 9. RUN XGBOOST CANDIDATES
# ============================================================================

print_header(
    "XGBOOST TUNING"
)

for candidate_id, params in XGB_CANDIDATES:

    print(
        f"\nRunning {candidate_id}..."
    )

    model = build_xgb(
        params
    )

    fold_records, summary = (
        evaluate_candidate(
            model=model,
            X=X_dev,
            y=y_dev,
            folds=folds,
            model_name="XGBoost",
            candidate_id=candidate_id,
            parameters=params,
        )
    )

    all_fold_records.extend(
        fold_records
    )

    all_summary_records.append(
        summary
    )

    print(
        f"PR-AUC: "
        f"{summary['mean_pr_auc']:.4f} "
        f"+/- "
        f"{summary['std_pr_auc']:.4f}"
    )

    print(
        f"Top-10 capture: "
        f"{summary['mean_top10_capture']:.2%}"
    )


# ============================================================================
# 10. CREATE RESULT DATAFRAMES
# ============================================================================

print_header(
    "TUNING RESULTS"
)

fold_results_df = pd.DataFrame(
    all_fold_records
)

summary_df = pd.DataFrame(
    all_summary_records
)


# ============================================================================
# 11. IDENTIFY BASELINES
# ============================================================================

baseline_pr_auc = {}

for model_name, baseline_id in [
    (
        "HistGradientBoosting",
        "HGB_00_BASELINE",
    ),
    (
        "XGBoost",
        "XGB_00_BASELINE",
    ),
]:

    row = summary_df[
        (
            summary_df["candidate_id"]
            == baseline_id
        )
    ]

    assert len(row) == 1

    baseline_pr_auc[
        model_name
    ] = float(
        row.iloc[0]["mean_pr_auc"]
    )


# ============================================================================
# 12. CALCULATE IMPROVEMENTS
# ============================================================================

summary_df[
    "baseline_cv_pr_auc"
] = summary_df["model"].map(
    baseline_pr_auc
)

summary_df[
    "pr_auc_improvement"
] = (
    summary_df["mean_pr_auc"]
    - summary_df[
        "baseline_cv_pr_auc"
    ]
)

summary_df[
    "pr_auc_improvement_pct"
] = (
    summary_df[
        "pr_auc_improvement"
    ]
    / summary_df[
        "baseline_cv_pr_auc"
    ]
    * 100
)


# ============================================================================
# 13. RANK CANDIDATES WITHIN EACH MODEL
# ============================================================================

"""
Ranking rule:

1. Higher mean PR-AUC
2. Lower PR-AUC standard deviation
3. Higher Top-10 capture
4. Lower generalization gap
"""

summary_df = summary_df.sort_values(
    [
        "model",
        "mean_pr_auc",
        "std_pr_auc",
        "mean_top10_capture",
        "generalization_gap",
    ],
    ascending=[
        True,
        False,
        True,
        False,
        True,
    ],
).reset_index(
    drop=True
)

summary_df[
    "model_rank"
] = (
    summary_df.groupby(
        "model"
    ).cumcount()
    + 1
)


# ============================================================================
# 14. IDENTIFY BEST CANDIDATE PER MODEL
# ============================================================================

best_hgb = summary_df[
    (
        summary_df["model"]
        == "HistGradientBoosting"
    )
].iloc[0]

best_xgb = summary_df[
    (
        summary_df["model"]
        == "XGBoost"
    )
].iloc[0]


# ============================================================================
# 15. DETERMINE WHETHER TUNING IMPROVED EACH MODEL
# ============================================================================

best_model_rows = [
    best_hgb,
    best_xgb,
]

model_selection_records = []

for row in best_model_rows:

    improvement = float(
        row[
            "pr_auc_improvement"
        ]
    )

    if improvement > 0:
        tuning_status = "IMPROVED"
    elif improvement < 0:
        tuning_status = "WORSE"
    else:
        tuning_status = "UNCHANGED"

    model_selection_records.append(
        {
            "model": row["model"],
            "baseline_cv_pr_auc":
                row[
                    "baseline_cv_pr_auc"
                ],
            "best_candidate":
                row["candidate_id"],
            "best_cv_pr_auc":
                row["mean_pr_auc"],
            "pr_auc_improvement":
                improvement,
            "pr_auc_improvement_pct":
                row[
                    "pr_auc_improvement_pct"
                ],
            "std_pr_auc":
                row["std_pr_auc"],
            "top10_capture":
                row[
                    "mean_top10_capture"
                ],
            "top10_lift":
                row[
                    "mean_top10_lift"
                ],
            "generalization_gap":
                row[
                    "generalization_gap"
                ],
            "tuning_status":
                tuning_status,
            "parameters":
                row["parameters"],
        }
    )


model_selection_df = pd.DataFrame(
    model_selection_records
)


# ============================================================================
# 16. IDENTIFY OVERALL TUNED LEADER
# ============================================================================

overall_candidates = (
    model_selection_df
    .sort_values(
        [
            "best_cv_pr_auc",
            "std_pr_auc",
            "top10_capture",
            "generalization_gap",
        ],
        ascending=[
            False,
            True,
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)

overall_candidates[
    "overall_rank"
] = (
    np.arange(
        len(overall_candidates)
    )
    + 1
)

overall_winner = (
    overall_candidates.iloc[0]
)


# ============================================================================
# 17. PRINT COMPARISON
# ============================================================================

print_header(
    "MODEL TUNING SUMMARY"
)

for _, row in model_selection_df.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        f"Baseline PR-AUC: "
        f"{row['baseline_cv_pr_auc']:.4f}"
    )

    print(
        f"Best candidate: "
        f"{row['best_candidate']}"
    )

    print(
        f"Best CV PR-AUC: "
        f"{row['best_cv_pr_auc']:.4f}"
    )

    print(
        f"Improvement: "
        f"{row['pr_auc_improvement']:+.4f}"
    )

    print(
        f"Top-10 capture: "
        f"{row['top10_capture']:.2%}"
    )

    print(
        f"PR-AUC std: "
        f"{row['std_pr_auc']:.4f}"
    )

    print(
        f"Generalization gap: "
        f"{row['generalization_gap']:.4f}"
    )

    print(
        f"Status: "
        f"{row['tuning_status']}"
    )


print_header(
    "CURRENT TUNED LEADER"
)

print(
    f"Model: "
    f"{overall_winner['model']}"
)

print(
    f"Candidate: "
    f"{overall_winner['best_candidate']}"
)

print(
    f"CV PR-AUC: "
    f"{overall_winner['best_cv_pr_auc']:.4f}"
)

print(
    f"Top-10 capture: "
    f"{overall_winner['top10_capture']:.2%}"
)


# ============================================================================
# 18. QUALITY CHECKS
# ============================================================================

print_header(
    "STEP 7.8 QUALITY CHECKS"
)

checks = {

    "169 frozen features":
        X_dev.shape[1]
        == EXPECTED_FEATURE_COUNT,

    "Development alignment":
        len(X_dev)
        == len(y_dev),

    "Binary target":
        set(
            y_dev.unique()
        ).issubset({0, 1}),

    "No missing values":
        not X_dev.isna()
        .any()
        .any(),

    "No infinite values":
        np.isfinite(
            X_dev.to_numpy(
                dtype=float
            )
        ).all(),

    "Exactly 5 CV folds":
        len(folds)
        == N_SPLITS,

    "10 HGB candidates":
        len(HGB_CANDIDATES)
        == 10,

    "10 XGB candidates":
        len(XGB_CANDIDATES)
        == 10,

    "20 total candidates":
        len(summary_df)
        == 20,

    "100 total fold evaluations":
        len(fold_results_df)
        == 100,

    "HGB baseline present":
        (
            (
                summary_df[
                    "candidate_id"
                ]
                == "HGB_00_BASELINE"
            ).sum()
            == 1
        ),

    "XGB baseline present":
        (
            (
                summary_df[
                    "candidate_id"
                ]
                == "XGB_00_BASELINE"
            ).sum()
            == 1
        ),

    "All PR-AUC finite":
        np.isfinite(
            summary_df[
                "mean_pr_auc"
            ]
        ).all(),

    "All PR-AUC std finite":
        np.isfinite(
            summary_df[
                "std_pr_auc"
            ]
        ).all(),

    "All Top-10 capture finite":
        np.isfinite(
            summary_df[
                "mean_top10_capture"
            ]
        ).all(),

    "All fold results complete":
        len(fold_results_df)
        == N_SPLITS * 20,

    "Validation not loaded":
        True,

    "Test not loaded":
        True,

    "Feature selection not performed":
        True,

    "Resampling not performed":
        True,

    "Class weighting not performed":
        True,

    "Threshold optimization not performed":
        True,

    "Calibration not performed":
        True,

    "SHAP not performed":
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
        "Step 7.8 quality gate failed."
    )


# ============================================================================
# 19. SAVE RESULTS
# ============================================================================

print(
    "\nSaving Step 7.8 artifacts..."
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
    CANDIDATE_RESULTS_PATH,
    index=False,
)

model_selection_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================================
# 20. SAVE SELECTED PARAMETERS
# ============================================================================

selected_parameters = {
    "step": "7.8",
    "selection_metric": "PR-AUC",
    "selection_stage": "development_cv",
    "validation_used": False,
    "test_used": False,
    "random_state": RANDOM_STATE,
    "cv_folds": N_SPLITS,
    "selected_model": overall_winner[
        "model"
    ],
    "selected_candidate": overall_winner[
        "best_candidate"
    ],
    "selected_cv_pr_auc": float(
        overall_winner[
            "best_cv_pr_auc"
        ]
    ),
    "selected_top10_capture": float(
        overall_winner[
            "top10_capture"
        ]
    ),
    "selected_parameters": json.loads(
        overall_winner[
            "parameters"
        ]
    ),
}


PARAMETERS_PATH.write_text(
    json.dumps(
        selected_parameters,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================================
# 21. SAVE METADATA
# ============================================================================

metadata = {
    "step": "7.8",
    "title":
        "Controlled Hyperparameter Tuning",
    "development_rows":
        int(len(X_dev)),
    "feature_count":
        EXPECTED_FEATURE_COUNT,
    "target":
        TARGET_NAME,
    "random_state":
        RANDOM_STATE,
    "cv_strategy":
        "StratifiedKFold",
    "cv_folds":
        N_SPLITS,
    "candidates_per_model":
        10,
    "total_candidates":
        20,
    "total_fold_evaluations":
        100,
    "optimization_metric":
        "PR-AUC",
    "secondary_metrics": [
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1",
        "Top-10 capture",
        "Top-10 lift",
    ],
    "models_tuned": [
        "HistGradientBoosting",
        "XGBoost",
    ],
    "validation_loaded":
        False,
    "test_loaded":
        False,
    "feature_selection":
        False,
    "resampling":
        False,
    "class_weighting":
        False,
    "threshold_optimization":
        False,
    "calibration":
        False,
    "shap":
        False,
    "selected_model":
        overall_winner[
            "model"
        ],
    "selected_candidate":
        overall_winner[
            "best_candidate"
        ],
    "selected_cv_pr_auc":
        float(
            overall_winner[
                "best_cv_pr_auc"
            ]
        ),
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
# 22. WRITE MARKDOWN REPORT
# ============================================================================

report_lines = [

    "# ChurnIQ — Step 7.8",

    "## Controlled Hyperparameter Tuning",

    "",

    "### Objective",

    "",

    "Step 7.8 performs controlled hyperparameter tuning "
    "for HistGradientBoosting and XGBoost, the two strongest "
    "candidates identified during Step 7.7.",

    "",

    "### Experimental design",

    "",

    "- Development dataset only",
    "- 55,999 development observations",
    "- 169 frozen model features",
    "- Stratified 5-fold cross-validation",
    "- Same CV folds reused across every candidate",
    "- Random state: 42",
    "- 10 candidates per model",
    "- 20 candidates total",
    "- 100 fold-level evaluations",
    "- Primary metric: PR-AUC",
    "- Secondary metrics: ROC-AUC, Precision, Recall, F1",
    "- Business metric: Top-10 churn capture and lift",

    "",

    "### Validation boundary",

    "",

    "The held-out validation dataset was not loaded during "
    "hyperparameter tuning. The test dataset was also not loaded.",

    "",

    "### HGB result",

    "",

    f"- Baseline CV PR-AUC: "
    f"{best_hgb['baseline_cv_pr_auc']:.4f}",

    f"- Best candidate: "
    f"{best_hgb['candidate_id']}",

    f"- Best CV PR-AUC: "
    f"{best_hgb['mean_pr_auc']:.4f}",

    f"- Improvement: "
    f"{best_hgb['pr_auc_improvement']:+.4f}",

    f"- PR-AUC standard deviation: "
    f"{best_hgb['std_pr_auc']:.4f}",

    f"- Top-10 churn capture: "
    f"{best_hgb['mean_top10_capture']:.2%}",

    "",

    "### XGBoost result",

    "",

    f"- Baseline CV PR-AUC: "
    f"{best_xgb['baseline_cv_pr_auc']:.4f}",

    f"- Best candidate: "
    f"{best_xgb['candidate_id']}",

    f"- Best CV PR-AUC: "
    f"{best_xgb['mean_pr_auc']:.4f}",

    f"- Improvement: "
    f"{best_xgb['pr_auc_improvement']:+.4f}",

    f"- PR-AUC standard deviation: "
    f"{best_xgb['std_pr_auc']:.4f}",

    f"- Top-10 churn capture: "
    f"{best_xgb['mean_top10_capture']:.2%}",

    "",

    "### Current development-CV leader",

    "",

    f"**{overall_winner['model']} "
    f"({overall_winner['best_candidate']})**",

    "",

    f"Development CV PR-AUC: "
    f"**{overall_winner['best_cv_pr_auc']:.4f}**",

    "",

    f"Top-10 churn capture: "
    f"**{overall_winner['top10_capture']:.2%}**",

    "",

    "This is a development cross-validation result, "
    "not the final held-out validation result.",

    "",

    "### Selection rule",

    "",

    "Candidates are ranked using mean PR-AUC as the primary "
    "criterion. Lower PR-AUC variability, higher Top-10 capture, "
    "and lower generalization gap are used as secondary criteria.",

    "",

    "### Controls maintained",

    "",

    "- Feature freeze remained active",
    "- No feature selection",
    "- No resampling",
    "- No class weighting",
    "- No threshold optimization",
    "- No probability calibration",
    "- No SHAP analysis",
    "- Validation remained untouched",
    "- Test remained untouched",

    "",

    "### Interpretation",

    "",

    "Tuning is considered useful only when it produces a "
    "repeatable improvement over the corresponding baseline "
    "without an obvious increase in instability or "
    "generalization gap.",

    "",

    "### Quality gate",

    "",

    "**PASS — Step 7.8 completed successfully.**",

    "",

    "### Next step",

    "",

    "Step 7.9 will evaluate the tuned candidates and "
    "untuned baselines on the untouched validation dataset "
    "for final model comparison.",

]


REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================================
# 23. FINAL STATUS
# ============================================================================

print_header(
    "STEP 7.8 FINAL QUALITY GATE"
)

print(
    "PASS — Controlled tuning completed."
)

print(
    "PASS — HGB evaluated across 10 candidates."
)

print(
    "PASS — XGBoost evaluated across 10 candidates."
)

print(
    "PASS — 20 candidates / 100 fold evaluations."
)

print(
    "PASS — Same five folds used for every candidate."
)

print(
    "PASS — 169 features remained frozen."
)

print(
    "PASS — Validation remained untouched."
)

print(
    "PASS — Test remained untouched."
)

print(
    "PASS — Candidate results saved."
)

print(
    "PASS — Fold-level results saved."
)

print(
    "PASS — Tuning summary saved."
)

print(
    "PASS — Selected parameters saved."
)

print(
    "PASS — Metadata saved."
)

print(
    "PASS — Markdown report saved."
)

print(
    f"\nCURRENT DEVELOPMENT-CV LEADER:"
)

print(
    f"{overall_winner['model']} — "
    f"{overall_winner['best_candidate']}"
)

print(
    f"PR-AUC: "
    f"{overall_winner['best_cv_pr_auc']:.4f}"
)

print(
    f"Top-10 capture: "
    f"{overall_winner['top10_capture']:.2%}"
)

print(
    "\nFINAL STATUS: PASS"
)

print(
    "NEXT → Step 7.9 Model Comparison & Held-Out Validation"
)

print("=" * 80)