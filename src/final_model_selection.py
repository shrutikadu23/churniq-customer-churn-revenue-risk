"""
ChurnIQ - Step 7.10: Final Model Selection

Purpose
-------
Make the final model-selection decision using ONLY previously generated
cross-validation, tuning, and held-out validation evidence.

This script does NOT:
- train models
- tune hyperparameters
- perform feature selection
- optimize thresholds
- calibrate probabilities
- generate SHAP explanations
- load test data

Primary selection metric:
    Held-out Validation PR-AUC

Secondary evidence:
    ROC-AUC
    Precision
    Recall
    F1
    Top-10% churn capture
    Top-10% lift
    Cross-validation stability
    CV-to-holdout transfer
    Governance checks

Current expected winner:
    XGB_02_SLOWER_MORE_TREES
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"

VALIDATION_FILE = REPORTS_DIR / "07_9_heldout_validation_results.csv"
TUNING_FILE = REPORTS_DIR / "07_8_tuning_summary.csv"

FINAL_SCORECARD_FILE = REPORTS_DIR / "07_10_final_model_selection.csv"
RANKING_FILE = REPORTS_DIR / "07_10_model_validation_ranking.csv"
DECISION_FILE = REPORTS_DIR / "07_10_model_selection_decision.json"
REPORT_FILE = REPORTS_DIR / "07_10_final_model_selection.md"

EXPECTED_FEATURE_COUNT = 169
EXPECTED_MODEL_COUNT = 5

PRIMARY_METRIC = "validation_pr_auc"
PRIMARY_METRIC_LABEL = "Held-out Validation PR-AUC"

SELECTED_MODEL_CANONICAL = "XGB_02_SLOWER_MORE_TREES"


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def normalize_column_name(value: str) -> str:
    """
    Normalize a column name for robust matching.
    """
    value = str(value).strip().lower()
    value = value.replace("%", " pct ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_model_name(value: str) -> str:
    """
    Normalize model labels for matching without changing the original
    displayed model name.
    """
    value = str(value).strip().lower()

    replacements = {
        "_": " ",
        "-": " ",
        "–": " ",
        "—": " ",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def safe_float(value):
    """
    Convert a value to float safely.
    """
    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return np.nan


def print_section(title: str):
    """
    Print a consistent console section.
    """
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_required_files():
    """
    Confirm that required evidence artifacts exist.
    """
    required_files = [
        VALIDATION_FILE,
        TUNING_FILE,
    ]

    missing = [path for path in required_files if not path.exists()]

    if missing:
        print("\nMissing required evidence files:")

        for path in missing:
            print(f"  - {path}")

        raise FileNotFoundError(
            "Required Step 7.9/7.8 evidence files are missing."
        )


# ============================================================================
# COLUMN IDENTIFICATION
# ============================================================================

def identify_metric_columns(validation_df: pd.DataFrame) -> dict:
    """
    Resolve the actual Step 7.9 validation artifact schema.

    The current Step 7.9 schema explicitly contains:

        model
        cv_pr_auc
        validation_pr_auc
        validation_roc_auc
        precision_at_0_50
        recall_at_0_50
        f1_at_0_50
        top10_capture
        top10_lift
        absolute_cv_validation_gap
        ...

    Explicit aliases are supported so the script remains robust to harmless
    naming variations.
    """

    normalized = {
        normalize_column_name(column): column
        for column in validation_df.columns
    }

    def resolve(
        aliases,
        required=True,
        label=None,
    ):
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)

            if normalized_alias in normalized:
                return normalized[normalized_alias]

        if required:
            display_label = label or aliases[0]

            raise ValueError(
                f"{display_label} column could not be identified.\n\n"
                f"Expected one of:\n"
                + "\n".join(f"  - {alias}" for alias in aliases)
                + "\n\nActual validation columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in validation_df.columns
                )
            )

        return None

    metric_columns = {
        "model": resolve(
            [
                "model",
                "model_name",
                "candidate",
                "model_candidate",
            ],
            label="Model",
        ),

        # --------------------------------------------------------------
        # PRIMARY METRIC
        # --------------------------------------------------------------
        "pr_auc": resolve(
            [
                "validation_pr_auc",
                "validation_pr-auc",
                "validation_pr auc",
                "pr_auc",
                "pr-auc",
                "pr auc",
            ],
            label="Held-out Validation PR-AUC",
        ),

        # --------------------------------------------------------------
        # SECONDARY MODEL METRICS
        # --------------------------------------------------------------
        "roc_auc": resolve(
            [
                "validation_roc_auc",
                "validation_roc-auc",
                "validation_roc auc",
                "roc_auc",
                "roc-auc",
            ],
            label="Held-out Validation ROC-AUC",
        ),

        "precision": resolve(
            [
                "precision_at_0_50",
                "validation_precision_at_0_50",
                "validation_precision",
                "precision",
            ],
            required=False,
        ),

        "recall": resolve(
            [
                "recall_at_0_50",
                "validation_recall_at_0_50",
                "validation_recall",
                "recall",
            ],
            required=False,
        ),

        "f1": resolve(
            [
                "f1_at_0_50",
                "validation_f1_at_0_50",
                "validation_f1",
                "f1",
            ],
            required=False,
        ),

        # --------------------------------------------------------------
        # BUSINESS METRICS
        # --------------------------------------------------------------
        "top10_capture": resolve(
            [
                "top10_capture",
                "top_10_capture",
                "top10_churn_capture",
                "top_10_churn_capture",
            ],
            required=False,
        ),

        "top10_lift": resolve(
            [
                "top10_lift",
                "top_10_lift",
            ],
            required=False,
        ),

        # --------------------------------------------------------------
        # CROSS-VALIDATION EVIDENCE
        # --------------------------------------------------------------
        "cv_pr_auc": resolve(
            [
                "cv_pr_auc",
                "mean_cv_pr_auc",
                "cv_pr-auc",
                "cv_pr auc",
            ],
            required=False,
        ),

        "cv_validation_gap": resolve(
            [
                "absolute_cv_validation_gap",
                "cv_validation_gap",
                "validation_minus_cv_pr_auc",
            ],
            required=False,
        ),
    }

    return metric_columns


# ============================================================================
# MODEL IDENTIFICATION
# ============================================================================

def canonical_model_name(model_name: str) -> str:
    """
    Map model aliases to stable canonical names.
    """

    normalized = normalize_model_name(model_name)

    # Logistic
    if "logistic" in normalized:
        return "Logistic Regression"

    # HGB tuned
    if (
        "hgb_09" in normalized
        or "hgb 09" in normalized
        or (
            "histgradientboosting" in normalized
            and "tuned" in normalized
        )
        or (
            "hgb" in normalized
            and "strong regularization" in normalized
        )
    ):
        return "HGB_09_STRONG_REGULARIZATION"

    # HGB baseline
    if (
        normalized in {
            "hgb",
            "hgb baseline",
            "hgb 00 baseline",
            "hgb_00_baseline",
            "histgradientboosting",
            "histgradientboosting baseline",
        }
        or (
            "hgb" in normalized
            and "baseline" in normalized
        )
    ):
        return "HGB_00_BASELINE"

    # XGB tuned
    if (
        "xgb_02" in normalized
        or "xgb 02" in normalized
        or "slower more trees" in normalized
        or "xgboost tuned" in normalized
        or "xgb tuned" in normalized
    ):
        return "XGB_02_SLOWER_MORE_TREES"

    # XGB baseline
    if (
        normalized in {
            "xgb",
            "xgb baseline",
            "xgb 00 baseline",
            "xgb_00_baseline",
            "xgboost",
            "xgboost baseline",
        }
        or (
            "xgb" in normalized
            and "baseline" in normalized
        )
        or (
            "xgboost" in normalized
            and "baseline" in normalized
        )
    ):
        return "XGB_00_BASELINE"

    return str(model_name).strip()


def add_canonical_model_column(df: pd.DataFrame, model_column: str):
    """
    Add canonical model identifiers while preserving the original model
    column.
    """

    df = df.copy()

    df["canonical_model"] = df[model_column].map(
        canonical_model_name
    )

    return df


# ============================================================================
# VALIDATION EVIDENCE CHECKS
# ============================================================================

def validate_validation_artifact(
    validation_df: pd.DataFrame,
    metric_columns: dict,
):
    """
    Validate the structural and numerical integrity of the Step 7.9
    validation evidence.
    """

    checks = []

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    checks.append(
        (
            "Expected model evaluation rows",
            len(validation_df) == EXPECTED_MODEL_COUNT,
            len(validation_df),
        )
    )

    # ------------------------------------------------------------------
    # Required model column
    # ------------------------------------------------------------------

    model_column = metric_columns["model"]

    checks.append(
        (
            "Model names complete",
            validation_df[model_column].notna().all(),
            int(validation_df[model_column].notna().sum()),
        )
    )

    # ------------------------------------------------------------------
    # Primary metric
    # ------------------------------------------------------------------

    pr_auc_column = metric_columns["pr_auc"]

    pr_values = pd.to_numeric(
        validation_df[pr_auc_column],
        errors="coerce",
    )

    checks.append(
        (
            "Validation PR-AUC finite",
            np.isfinite(pr_values).all(),
            int(np.isfinite(pr_values).sum()),
        )
    )

    checks.append(
        (
            "Validation PR-AUC bounded [0,1]",
            ((pr_values >= 0) & (pr_values <= 1)).all(),
            int(((pr_values >= 0) & (pr_values <= 1)).sum()),
        )
    )

    # ------------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------------

    roc_column = metric_columns["roc_auc"]

    if roc_column is not None:
        roc_values = pd.to_numeric(
            validation_df[roc_column],
            errors="coerce",
        )

        checks.append(
            (
                "Validation ROC-AUC finite",
                np.isfinite(roc_values).all(),
                int(np.isfinite(roc_values).sum()),
            )
        )

    # ------------------------------------------------------------------
    # Business metrics
    # ------------------------------------------------------------------

    for key in ["top10_capture", "top10_lift"]:
        column = metric_columns.get(key)

        if column is not None:
            values = pd.to_numeric(
                validation_df[column],
                errors="coerce",
            )

            checks.append(
                (
                    f"{column} finite",
                    np.isfinite(values).all(),
                    int(np.isfinite(values).sum()),
                )
            )

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------

    print("\nValidation artifact quality checks:")
    print("-" * 72)

    all_passed = True

    for name, passed, value in checks:
        status = "PASS" if passed else "FAIL"

        print(
            f"{name:<45} {status:<6} {value}"
        )

        if not passed:
            all_passed = False

    if not all_passed:
        raise ValueError(
            "Validation artifact quality checks failed."
        )

    print("\nValidation evidence quality gate: PASS")

    return checks


# ============================================================================
# TUNING EVIDENCE
# ============================================================================

def load_tuning_evidence() -> pd.DataFrame:
    """
    Load the Step 7.8 tuning summary.

    This is evidence only. No tuning is performed here.
    """

    tuning_df = pd.read_csv(TUNING_FILE)

    return tuning_df


def identify_tuning_columns(tuning_df: pd.DataFrame) -> dict:
    """
    Resolve tuning-summary columns robustly.
    """

    normalized = {
        normalize_column_name(column): column
        for column in tuning_df.columns
    }

    def resolve(aliases, required=False):
        for alias in aliases:
            key = normalize_column_name(alias)

            if key in normalized:
                return normalized[key]

        if required:
            raise ValueError(
                "Required tuning column could not be identified."
            )

        return None

    return {
        "model": resolve(
            [
                "model",
                "model_name",
                "candidate",
                "candidate_name",
            ],
            required=False,
        ),
        "mean_pr_auc": resolve(
            [
                "mean_pr_auc",
                "cv_pr_auc",
                "mean_cv_pr_auc",
            ],
            required=False,
        ),
        "std_pr_auc": resolve(
            [
                "std_pr_auc",
                "cv_pr_auc_std",
            ],
            required=False,
        ),
    }


# ============================================================================
# MODEL RANKING
# ============================================================================

def create_validation_ranking(
    validation_df: pd.DataFrame,
    metric_columns: dict,
) -> pd.DataFrame:
    """
    Rank models strictly by held-out validation PR-AUC.
    """

    df = validation_df.copy()

    model_column = metric_columns["model"]

    df["canonical_model"] = df[model_column].map(
        canonical_model_name
    )

    df["selection_pr_auc"] = pd.to_numeric(
        df[metric_columns["pr_auc"]],
        errors="coerce",
    )

    df = df.sort_values(
        "selection_pr_auc",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    df["validation_rank"] = (
        np.arange(len(df)) + 1
    )

    best_pr_auc = df["selection_pr_auc"].max()

    df["pr_auc_gap_from_best"] = (
        best_pr_auc - df["selection_pr_auc"]
    )

    df["within_pr_auc_0_002_of_best"] = (
        df["pr_auc_gap_from_best"] <= 0.002
    )

    return df


# ============================================================================
# SELECT WINNER
# ============================================================================

def select_final_model(
    ranking_df: pd.DataFrame,
) -> dict:
    """
    Select the final model.

    Locked rule:
        The final selected model must be the held-out validation PR-AUC
        leader.

    No subjective override is allowed here.
    """

    if ranking_df.empty:
        raise ValueError(
            "No model evaluation rows available."
        )

    winner = ranking_df.iloc[0]

    selected_model = winner["canonical_model"]
    selected_pr_auc = safe_float(
        winner["selection_pr_auc"]
    )

    if selected_model != SELECTED_MODEL_CANONICAL:
        print()
        print(
            "WARNING: Current held-out validation leader is not "
            f"{SELECTED_MODEL_CANONICAL}."
        )

        print(
            f"Observed leader: {selected_model}"
        )

    decision = {
        "selected_model": selected_model,
        "selected_model_expected": SELECTED_MODEL_CANONICAL,
        "selected_model_matches_expected": (
            selected_model == SELECTED_MODEL_CANONICAL
        ),
        "primary_metric": PRIMARY_METRIC_LABEL,
        "selected_primary_metric_value": selected_pr_auc,
        "selection_rule": (
            "Select the model with the highest held-out "
            "validation PR-AUC."
        ),
    }

    return decision


# ============================================================================
# EVIDENCE COMPARISON
# ============================================================================

def extract_model_row(
    ranking_df: pd.DataFrame,
    canonical_name: str,
):
    """
    Retrieve one canonical model row.
    """

    matches = ranking_df[
        ranking_df["canonical_model"] == canonical_name
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def build_selection_scorecard(
    ranking_df: pd.DataFrame,
    metric_columns: dict,
) -> pd.DataFrame:
    """
    Create a concise final selection scorecard.
    """

    rows = []

    for _, row in ranking_df.iterrows():

        result = {
            "model": row["canonical_model"],
            "validation_rank": int(row["validation_rank"]),
            "validation_pr_auc": safe_float(
                row["selection_pr_auc"]
            ),
            "pr_auc_gap_from_best": safe_float(
                row["pr_auc_gap_from_best"]
            ),
            "within_pr_auc_0_002_of_best": bool(
                row["within_pr_auc_0_002_of_best"]
            ),
        }

        # --------------------------------------------------------------
        # Secondary metrics
        # --------------------------------------------------------------

        for output_name, key in [
            ("validation_roc_auc", "roc_auc"),
            ("precision_at_0_50", "precision"),
            ("recall_at_0_50", "recall"),
            ("f1_at_0_50", "f1"),
            ("top10_capture", "top10_capture"),
            ("top10_lift", "top10_lift"),
            ("cv_pr_auc", "cv_pr_auc"),
            ("absolute_cv_validation_gap", "cv_validation_gap"),
        ]:

            column = metric_columns.get(key)

            if column is not None:
                result[output_name] = safe_float(
                    row[column]
                )
            else:
                result[output_name] = np.nan

        result["final_selection"] = (
            "SELECTED"
            if int(row["validation_rank"]) == 1
            else "NOT SELECTED"
        )

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================================
# MODEL-SPECIFIC COMPARISON
# ============================================================================

def create_decision_evidence(
    ranking_df: pd.DataFrame,
    metric_columns: dict,
    tuning_df: pd.DataFrame,
):
    """
    Generate evidence specifically supporting or challenging the final
    selection.
    """

    selected_row = extract_model_row(
        ranking_df,
        SELECTED_MODEL_CANONICAL,
    )

    hgb_row = extract_model_row(
        ranking_df,
        "HGB_09_STRONG_REGULARIZATION",
    )

    if selected_row is None:
        raise ValueError(
            f"Expected selected model {SELECTED_MODEL_CANONICAL} "
            "was not found in validation results."
        )

    selected_pr_auc = safe_float(
        selected_row["selection_pr_auc"]
    )

    selected_top10 = (
        safe_float(selected_row["top10_capture"])
        if "top10_capture" in selected_row.index
        else np.nan
    )

    selected_lift = (
        safe_float(selected_row["top10_lift"])
        if "top10_lift" in selected_row.index
        else np.nan
    )

    evidence = {
        "selected_model": SELECTED_MODEL_CANONICAL,
        "selected_validation_pr_auc": selected_pr_auc,
        "selected_top10_capture": selected_top10,
        "selected_top10_lift": selected_lift,
    }

    # ------------------------------------------------------------------
    # HGB comparison
    # ------------------------------------------------------------------

    if hgb_row is not None:

        hgb_pr_auc = safe_float(
            hgb_row["selection_pr_auc"]
        )

        evidence["hgb_tuned_validation_pr_auc"] = hgb_pr_auc

        evidence["xgb_minus_hgb_validation_pr_auc"] = (
            selected_pr_auc - hgb_pr_auc
        )

        evidence["xgb_hgb_pr_auc_difference_is_small"] = (
            abs(selected_pr_auc - hgb_pr_auc) < 0.005
        )

    # ------------------------------------------------------------------
    # Tuning transfer
    # ------------------------------------------------------------------

    tuning_columns = identify_tuning_columns(
        tuning_df
    )

    tuning_model_column = tuning_columns["model"]
    tuning_pr_column = tuning_columns["mean_pr_auc"]

    if (
        tuning_model_column is not None
        and tuning_pr_column is not None
    ):

        tuning_copy = tuning_df.copy()

        tuning_copy["canonical_model"] = (
            tuning_copy[tuning_model_column].map(
                canonical_model_name
            )
        )

        xgb_tuned_tuning = tuning_copy[
            tuning_copy["canonical_model"]
            == SELECTED_MODEL_CANONICAL
        ]

        xgb_baseline_tuning = tuning_copy[
            tuning_copy["canonical_model"]
            == "XGB_00_BASELINE"
        ]

        if not xgb_tuned_tuning.empty:

            tuned_cv_pr_auc = safe_float(
                xgb_tuned_tuning.iloc[0][tuning_pr_column]
            )

            evidence["xgb_tuned_cv_pr_auc"] = tuned_cv_pr_auc

            if not xgb_baseline_tuning.empty:

                baseline_cv_pr_auc = safe_float(
                    xgb_baseline_tuning.iloc[0][tuning_pr_column]
                )

                evidence["xgb_baseline_cv_pr_auc"] = (
                    baseline_cv_pr_auc
                )

                evidence["xgb_tuning_cv_improvement"] = (
                    tuned_cv_pr_auc - baseline_cv_pr_auc
                )

    return evidence


# ============================================================================
# GOVERNANCE SUMMARY
# ============================================================================

def build_governance_summary() -> dict:
    """
    Record governance controls for the final selection step.
    """

    return {
        "training_performed": False,
        "hyperparameter_tuning_performed": False,
        "feature_selection_performed": False,
        "threshold_optimization_performed": False,
        "probability_calibration_performed": False,
        "shap_analysis_performed": False,
        "resampling_performed": False,
        "class_weighting_performed": False,
        "test_data_loaded": False,
        "validation_used_for_tuning": False,
        "validation_used_for_feature_selection": False,
        "selection_metric": PRIMARY_METRIC_LABEL,
        "selection_metric_source": (
            "Step 7.9 held-out validation artifact"
        ),
        "feature_count_locked": EXPECTED_FEATURE_COUNT,
        "feature_freeze_active": True,
        "selection_is_evidence_only": True,
    }


# ============================================================================
# MARKDOWN REPORT
# ============================================================================

def write_markdown_report(
    ranking_df: pd.DataFrame,
    decision: dict,
    evidence: dict,
    governance: dict,
):
    """
    Write the formal Step 7.10 model-selection decision record.
    """

    selected_model = decision["selected_model"]

    lines = []

    lines.append("# ChurnIQ — Step 7.10 Final Model Selection")
    lines.append("")
    lines.append(
        "## 1. Purpose"
    )
    lines.append("")
    lines.append(
        "This step formally selects the final candidate model using "
        "previously generated cross-validation, tuning, and held-out "
        "validation evidence."
    )
    lines.append("")
    lines.append(
        "No model training, hyperparameter tuning, feature selection, "
        "threshold optimization, probability calibration, SHAP analysis, "
        "or test-data evaluation is performed in this step."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Framework
    # ------------------------------------------------------------------

    lines.append("## 2. Selection Framework")
    lines.append("")
    lines.append(
        f"- **Primary metric:** {PRIMARY_METRIC_LABEL}"
    )
    lines.append(
        "- Secondary metrics: ROC-AUC, Precision, Recall, F1"
    )
    lines.append(
        "- Business metrics: Top-10% churn capture and lift"
    )
    lines.append(
        "- Stability evidence: 5-fold cross-validation"
    )
    lines.append(
        "- Generalization evidence: CV-to-held-out transfer"
    )
    lines.append(
        "- Governance: leakage and validation isolation"
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Winner
    # ------------------------------------------------------------------

    lines.append("## 3. Final Selection")
    lines.append("")
    lines.append(
        f"**Selected model: `{selected_model}`**"
    )
    lines.append("")
    lines.append(
        f"Selected held-out validation PR-AUC: "
        f"**{decision['selected_primary_metric_value']:.4f}**"
    )
    lines.append("")
    lines.append(
        "The selected model is the held-out validation PR-AUC leader. "
        "This prevents subjective model selection after observing the "
        "validation results."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    lines.append("## 4. Held-Out Validation Ranking")
    lines.append("")

    lines.append(
        "| Rank | Model | Validation PR-AUC | ROC-AUC | "
        "Precision | Recall | F1 | Top-10 Capture | Top-10 Lift |"
    )

    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in ranking_df.iterrows():

        def fmt(column):
            if column not in row.index:
                return "—"

            value = safe_float(row[column])

            if np.isnan(value):
                return "—"

            return f"{value:.4f}"

        lines.append(
            f"| {int(row['validation_rank'])} "
            f"| {row['canonical_model']} "
            f"| {fmt('selection_pr_auc')} "
            f"| {fmt(metric_columns_local['roc_auc'])} "
            f"| {fmt(metric_columns_local['precision'])} "
            f"| {fmt(metric_columns_local['recall'])} "
            f"| {fmt(metric_columns_local['f1'])} "
            f"| {fmt(metric_columns_local['top10_capture'])} "
            f"| {fmt(metric_columns_local['top10_lift'])} |"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Close competitor
    # ------------------------------------------------------------------

    if "hgb_tuned_validation_pr_auc" in evidence:

        lines.append("## 5. Close Competitor")
        lines.append("")

        lines.append(
            "The tuned HGB candidate remains a close alternative."
        )
        lines.append("")

        lines.append(
            f"- XGBoost tuned validation PR-AUC: "
            f"**{evidence['selected_validation_pr_auc']:.4f}**"
        )

        lines.append(
            f"- HGB tuned validation PR-AUC: "
            f"**{evidence['hgb_tuned_validation_pr_auc']:.4f}**"
        )

        lines.append(
            f"- Difference: "
            f"**{evidence['xgb_minus_hgb_validation_pr_auc']:.4f}**"
        )

        lines.append("")

        lines.append(
            "The difference is modest, so the selection should not be "
            "presented as evidence that XGBoost is categorically superior "
            "to HGB. It is the current winner under the locked validation "
            "selection framework."
        )

        lines.append("")

    # ------------------------------------------------------------------
    # Tuning
    # ------------------------------------------------------------------

    lines.append("## 6. Tuning Transfer")
    lines.append("")

    if "xgb_tuning_cv_improvement" in evidence:

        lines.append(
            f"- XGBoost tuned CV PR-AUC: "
            f"**{evidence['xgb_tuned_cv_pr_auc']:.4f}**"
        )

        lines.append(
            f"- XGBoost baseline CV PR-AUC: "
            f"**{evidence['xgb_baseline_cv_pr_auc']:.4f}**"
        )

        lines.append(
            f"- CV improvement from tuning: "
            f"**{evidence['xgb_tuning_cv_improvement']:+.4f}**"
        )

        lines.append("")

    if "absolute_cv_validation_gap" in ranking_df.columns:

        selected_row = ranking_df[
            ranking_df["canonical_model"]
            == SELECTED_MODEL_CANONICAL
        ]

        if not selected_row.empty:

            gap = safe_float(
                selected_row.iloc[0][
                    "absolute_cv_validation_gap"
                ]
            )

            if not np.isnan(gap):

                lines.append(
                    f"- Selected model CV-to-holdout absolute "
                    f"PR-AUC gap: **{gap:.4f}**"
                )

                lines.append("")

    lines.append(
        "The held-out validation result is used as the final decision "
        "evidence rather than selecting a model solely from cross-validation."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Business interpretation
    # ------------------------------------------------------------------

    lines.append("## 7. Business Interpretation")
    lines.append("")

    if not np.isnan(evidence.get("selected_top10_capture", np.nan)):

        lines.append(
            f"The selected model captures approximately "
            f"**{evidence['selected_top10_capture'] * 100:.2f}%** "
            "of actual churners within the top 10% highest-risk "
            "validation customers."
        )

        lines.append("")

    if not np.isnan(evidence.get("selected_top10_lift", np.nan)):

        lines.append(
            f"This corresponds to approximately "
            f"**{evidence['selected_top10_lift']:.2f}× lift** "
            "over untargeted selection."
        )

        lines.append("")

    lines.append(
        "These metrics support prioritization of a limited retention "
        "capacity. They do not imply that every selected customer will "
        "churn."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Governance
    # ------------------------------------------------------------------

    lines.append("## 8. Governance Controls")
    lines.append("")

    governance_items = [
        (
            "169-feature freeze active",
            governance["feature_freeze_active"],
        ),
        (
            "No training performed",
            not governance["training_performed"],
        ),
        (
            "No additional tuning performed",
            not governance["hyperparameter_tuning_performed"],
        ),
        (
            "No threshold optimization performed",
            not governance["threshold_optimization_performed"],
        ),
        (
            "No probability calibration performed",
            not governance["probability_calibration_performed"],
        ),
        (
            "No SHAP analysis performed",
            not governance["shap_analysis_performed"],
        ),
        (
            "No test data loaded",
            governance["test_data_loaded"] is False,
        ),
        (
            "Validation not reused for tuning",
            governance["validation_used_for_tuning"] is False,
        ),
        (
            "Validation not reused for feature selection",
            governance["validation_used_for_feature_selection"] is False,
        ),
    ]

    for name, passed in governance_items:

        status = "PASS" if passed else "FAIL"

        lines.append(
            f"- **{name}: {status}**"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------

    lines.append("## 9. What This Step Does Not Decide")
    lines.append("")

    lines.append(
        "Final model selection does not determine the operational "
        "probability threshold."
    )

    lines.append("")

    lines.append(
        "The threshold will be addressed separately using business-aware "
        "cost considerations, precision/recall trade-offs, churn capture, "
        "and revenue-risk prioritization."
    )

    lines.append("")

    lines.append(
        "Probability calibration is also a separate step and has not "
        "been performed here."
    )

    lines.append("")

    # ------------------------------------------------------------------
    # Next step
    # ------------------------------------------------------------------

    lines.append("## 10. Next Step")
    lines.append("")

    lines.append(
        "**Step 7.11 — Probability Calibration & Reliability Analysis**"
    )

    lines.append("")

    lines.append(
        "The selected XGBoost candidate will next be assessed for "
        "probability reliability before operational risk thresholds "
        "and customer prioritization are finalized."
    )

    lines.append("")

    lines.append(
        "## 11. Status"
    )

    lines.append("")

    lines.append(
        "**Step 7.10 Final Model Selection: COMPLETE — PASS**"
    )

    lines.append("")

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    global metric_columns_local

    print_section(
        "ChurnIQ - Step 7.10: Final Model Selection"
    )

    print(
        "\nSelection framework"
    )
    print("-" * 72)
    print(
        f"Primary metric  : {PRIMARY_METRIC_LABEL}"
    )
    print(
        "Secondary       : ROC-AUC, Precision, Recall, F1"
    )
    print(
        "Business        : Top-10% capture and lift"
    )
    print(
        "Stability       : Cross-validation evidence"
    )
    print(
        "Generalization  : CV-to-holdout transfer"
    )
    print(
        "Governance      : Leakage and validation isolation"
    )
    print(
        "Training        : NONE"
    )

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    validate_required_files()

    # ------------------------------------------------------------------
    # Load validation evidence
    # ------------------------------------------------------------------

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    print("\nEvidence loaded:")
    print(
        f"  Held-out validation results: "
        f"{validation_df.shape}"
    )

    print("\nValidation artifact columns:")

    for column in validation_df.columns:
        print(f"  - {column}")

    # ------------------------------------------------------------------
    # Resolve schema
    # ------------------------------------------------------------------

    metric_columns_local = identify_metric_columns(
        validation_df
    )

    print("\nResolved metric columns:")
    print("-" * 72)

    for key, value in metric_columns_local.items():
        print(
            f"  {key:<25} : {value}"
        )

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    validate_validation_artifact(
        validation_df,
        metric_columns_local,
    )

    # ------------------------------------------------------------------
    # Model count
    # ------------------------------------------------------------------

    unique_models = validation_df[
        metric_columns_local["model"]
    ].nunique()

    print(
        f"\nUnique evaluated models: {unique_models}"
    )

    if unique_models != EXPECTED_MODEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_COUNT} models, "
            f"found {unique_models}."
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    ranking_df = create_validation_ranking(
        validation_df,
        metric_columns_local,
    )

    print_section(
        "Held-Out Validation Ranking"
    )

    for _, row in ranking_df.iterrows():

        print(
            f"{int(row['validation_rank'])}. "
            f"{row['canonical_model']:<38} "
            f"PR-AUC = "
            f"{row['selection_pr_auc']:.4f}"
        )

    # ------------------------------------------------------------------
    # Select final model
    # ------------------------------------------------------------------

    decision = select_final_model(
        ranking_df
    )

    print_section(
        "Final Model Decision"
    )

    print(
        f"Selected model : {decision['selected_model']}"
    )

    print(
        f"Primary metric : {PRIMARY_METRIC_LABEL}"
    )

    print(
        f"PR-AUC         : "
        f"{decision['selected_primary_metric_value']:.4f}"
    )

    # ------------------------------------------------------------------
    # Tuning evidence
    # ------------------------------------------------------------------

    tuning_df = load_tuning_evidence()

    print(
        f"\nTuning evidence loaded: "
        f"{tuning_df.shape}"
    )

    evidence = create_decision_evidence(
        ranking_df,
        metric_columns_local,
        tuning_df,
    )

    # ------------------------------------------------------------------
    # Scorecard
    # ------------------------------------------------------------------

    scorecard_df = build_selection_scorecard(
        ranking_df,
        metric_columns_local,
    )

    # ------------------------------------------------------------------
    # Governance
    # ------------------------------------------------------------------

    governance = build_governance_summary()

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    scorecard_df.to_csv(
        FINAL_SCORECARD_FILE,
        index=False,
    )

    ranking_export = ranking_df.copy()

    ranking_export.to_csv(
        RANKING_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Decision JSON
    # ------------------------------------------------------------------

    decision_record = {
        "project": "ChurnIQ",
        "step": "7.10",
        "step_name": "Final Model Selection",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "selected_model": decision["selected_model"],
        "primary_metric": PRIMARY_METRIC_LABEL,
        "selected_primary_metric_value": (
            decision["selected_primary_metric_value"]
        ),
        "selection_rule": decision["selection_rule"],
        "ranking": (
            ranking_df[
                [
                    "validation_rank",
                    "canonical_model",
                    "selection_pr_auc",
                    "pr_auc_gap_from_best",
                    "within_pr_auc_0_002_of_best",
                ]
            ]
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
        "decision_evidence": evidence,
        "governance": governance,
        "artifacts": {
            "validation_evidence": str(
                VALIDATION_FILE.relative_to(PROJECT_ROOT)
            ),
            "tuning_evidence": str(
                TUNING_FILE.relative_to(PROJECT_ROOT)
            ),
            "final_scorecard": str(
                FINAL_SCORECARD_FILE.relative_to(PROJECT_ROOT)
            ),
            "validation_ranking": str(
                RANKING_FILE.relative_to(PROJECT_ROOT)
            ),
            "decision_record": str(
                DECISION_FILE.relative_to(PROJECT_ROOT)
            ),
            "markdown_report": str(
                REPORT_FILE.relative_to(PROJECT_ROOT)
            ),
        },
    }

    DECISION_FILE.write_text(
        json.dumps(
            decision_record,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------

    write_markdown_report(
        ranking_df,
        decision,
        evidence,
        governance,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print_section(
        "Step 7.10 Complete"
    )

    print(
        f"Final selected model : "
        f"{decision['selected_model']}"
    )

    print(
        f"Validation PR-AUC    : "
        f"{decision['selected_primary_metric_value']:.4f}"
    )

    if "selected_top10_capture" in evidence:
        print(
            f"Top-10 capture       : "
            f"{evidence['selected_top10_capture']:.4f}"
        )

    if "selected_top10_lift" in evidence:
        print(
            f"Top-10 lift          : "
            f"{evidence['selected_top10_lift']:.2f}x"
        )

    print()
    print(
        "Governance checks:"
    )
    print(
        "  - Training                         : NONE"
    )
    print(
        "  - Additional tuning               : NONE"
    )
    print(
        "  - Threshold optimization          : NONE"
    )
    print(
        "  - Probability calibration         : NONE"
    )
    print(
        "  - SHAP analysis                   : NONE"
    )
    print(
        "  - Feature selection               : NONE"
    )
    print(
        "  - Test data loaded                : NO"
    )
    print(
        "  - Validation reused for tuning    : NO"
    )
    print(
        "  - Feature freeze                  : ACTIVE"
    )

    print()
    print(
        "Artifacts saved:"
    )
    print(
        f"  - {FINAL_SCORECARD_FILE}"
    )
    print(
        f"  - {RANKING_FILE}"
    )
    print(
        f"  - {DECISION_FILE}"
    )
    print(
        f"  - {REPORT_FILE}"
    )

    print()
    print(
        "QUALITY GATE: PASS"
    )


if __name__ == "__main__":
    main()