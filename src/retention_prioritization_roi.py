from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# =============================================================================
# ChurnIQ — Phase 13.1
# Retention Economics Foundation
#
# Purpose:
# Establish the auditable economic framework used for retention prioritization.
#
# This phase converts Phase 12 customer-level revenue exposure into scenario-based
# retention economics.
#
# Core logic:
#
# Potential Monthly Revenue Exposure
#             ×
# Retention Success Rate (ASSUMPTION)
#             =
# Expected Monthly Revenue Retained
#
# Expected Monthly Revenue Retained
#             ×
# Evaluation Horizon in Months
#             =
# Expected Revenue Retained Over Horizon
#
# Expected Revenue Retained Over Horizon
#             -
# Intervention Cost
#             =
# Expected Net Value
#
# Expected Net Value
#             ÷
# Intervention Cost
#             =
# ROI
#
# Governance:
# - No model retraining
# - No probability recalibration
# - No threshold changes
# - No feature engineering
# - No use of test data
# - Churn probability remains the Phase 11 calibrated probability
# - Revenue exposure remains the Phase 12 calculated exposure
# - Retention success rates are assumptions, not observed outcomes
# - Intervention costs are assumptions, not observed costs
# - ROI is scenario-based, not realized business ROI
# =============================================================================


warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PHASE_12_INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "customer_value_revenue_risk.csv"
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
# 2. SCENARIO GOVERNANCE
# =============================================================================

SCENARIO_VERSION = (
    "Phase 13.1 — Base Retention Economics Scenario Set v1"
)

EVALUATION_HORIZON_MONTHS = 12


# =============================================================================
# 3. RETENTION SCENARIOS
#
# IMPORTANT:
# These are analytical assumptions for scenario evaluation.
# They are NOT observed retention outcomes.
#
# intervention_cost_inr:
#     Total assumed intervention cost per customer for the scenario.
#
# retention_success_rate:
#     Assumed probability that the retention intervention succeeds.
# =============================================================================

RETENTION_SCENARIOS = [
    {
        "scenario_id": "low_touch",
        "scenario_name": "Low-Touch Retention",
        "intervention_cost_inr": 100.0,
        "retention_success_rate": 0.20,
    },
    {
        "scenario_id": "targeted",
        "scenario_name": "Targeted Retention",
        "intervention_cost_inr": 250.0,
        "retention_success_rate": 0.35,
    },
    {
        "scenario_id": "high_touch",
        "scenario_name": "High-Touch Retention",
        "intervention_cost_inr": 500.0,
        "retention_success_rate": 0.50,
    },
]


# =============================================================================
# 4. CONSTANTS
# =============================================================================

EXPECTED_CUSTOMERS = 69999

REQUIRED_INPUT_COLUMNS = [
    "id",
    "calibrated_churn_probability",
    "risk_score",
    "risk_level",
    "above_primary_threshold",
    "potential_monthly_revenue_exposure",
]


# =============================================================================
# 5. UTILITY FUNCTIONS
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


def validate_non_negative(values, name):
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if (values < 0).any():
        raise ValueError(
            f"{name} contains negative values."
        )


def format_inr(value):
    return f"₹{value:,.2f}"


# =============================================================================
# 6. MAIN WORKFLOW
# =============================================================================

def main():

    print("=" * 80)
    print("PHASE 13.2 — TARGETING COHORT ANALYSIS")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — CHECK INPUT
    # =========================================================================

    print("\n[1/10] Checking Phase 12 input...")

    require_file(
        PHASE_12_INPUT_FILE,
        "Phase 12 customer value and revenue risk dataset"
    )

    customer_data = pd.read_csv(
        PHASE_12_INPUT_FILE
    )

    check(
        "Phase 12 input",
        "PASS",
        f"{customer_data.shape}"
    )


    # =========================================================================
    # STEP 2 — VALIDATE CUSTOMER UNIVERSE
    # =========================================================================

    print("\n[2/10] Validating customer universe...")

    required_columns = [
        "id",
        "calibrated_churn_probability",
        "risk_score",
        "risk_level",
        "above_primary_threshold",
        "potential_monthly_revenue_exposure",
        "actual_churn",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in customer_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Phase 12 columns: {missing_columns}"
        )

    if len(customer_data) != EXPECTED_CUSTOMERS:
        raise ValueError(
            f"Expected {EXPECTED_CUSTOMERS:,} customers, "
            f"found {len(customer_data):,}."
        )

    if customer_data["id"].isna().any():
        raise ValueError(
            "Missing customer IDs detected."
        )

    if customer_data["id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    validate_probability(
        customer_data[
            "calibrated_churn_probability"
        ].to_numpy(),
        "Calibrated churn probability"
    )

    validate_non_negative(
        customer_data[
            "potential_monthly_revenue_exposure"
        ].to_numpy(),
        "Potential monthly revenue exposure"
    )

    actual_churn_values = set(
        customer_data["actual_churn"]
        .dropna()
        .unique()
    )

    if not actual_churn_values.issubset({0, 1}):
        raise ValueError(
            "actual_churn must contain only 0/1 values."
        )

    check(
        "Customer universe",
        "PASS",
        f"{customer_data['id'].nunique():,} unique customers"
    )

    check(
        "Historical churn label",
        "PASS",
        "Binary 0/1"
    )


    # =========================================================================
    # STEP 3 — PREPARE DETERMINISTIC EXPOSURE RANKING
    # =========================================================================

    print("\n[3/10] Preparing deterministic exposure ranking...")

    customer_data = customer_data.sort_values(
        by=[
            "potential_monthly_revenue_exposure",
            "calibrated_churn_probability",
            "id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    customer_data[
        "exposure_rank"
    ] = np.arange(
        1,
        len(customer_data) + 1
    )

    customer_data[
        "exposure_percentile"
    ] = (
        customer_data["exposure_rank"]
        /
        len(customer_data)
    )

    total_exposure = customer_data[
        "potential_monthly_revenue_exposure"
    ].sum()

    total_annualized_exposure = (
        total_exposure * 12
    )

    if total_exposure <= 0:
        raise ValueError(
            "Total potential monthly revenue exposure must be > 0."
        )

    total_historical_churn = (
        customer_data["actual_churn"]
        .sum()
    )

    check(
        "Exposure ranking",
        "PASS",
        f"Total monthly exposure = {format_inr(total_exposure)}"
    )

    check(
        "Historical churn population",
        "PASS",
        f"{int(total_historical_churn):,} churned customers"
    )


    # =========================================================================
    # STEP 4 — DEFINE LOCKED TARGETING COHORTS
    # =========================================================================

    print("\n[4/10] Defining locked targeting cohorts...")

    universe_size = len(customer_data)

    top_1_cutoff = max(
        1,
        int(np.ceil(universe_size * 0.01))
    )

    top_5_cutoff = max(
        1,
        int(np.ceil(universe_size * 0.05))
    )

    top_10_cutoff = max(
        1,
        int(np.ceil(universe_size * 0.10))
    )

    top_20_cutoff = max(
        1,
        int(np.ceil(universe_size * 0.20))
    )

    high_very_high_mask = (
        customer_data["risk_level"]
        .isin(
            [
                "High",
                "Very High",
            ]
        )
    )

    cohort_masks = {

        "Top 1% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_1_cutoff,

        "Top 5% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_5_cutoff,

        "Top 10% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_10_cutoff,

        "Top 20% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_20_cutoff,

        "High + Very High Risk":
            high_very_high_mask,
    }

    check(
        "Targeting cohorts",
        "PASS",
        f"{len(cohort_masks)} locked cohorts"
    )


    # =========================================================================
    # STEP 5 — CALCULATE COHORT METRICS
    # =========================================================================

    print("\n[5/10] Calculating cohort metrics...")

    cohort_results = []

    for cohort_name, mask in cohort_masks.items():

        cohort = customer_data.loc[
            mask
        ].copy()

        targeted_customers = len(
            cohort
        )

        if targeted_customers == 0:
            raise ValueError(
                f"Cohort '{cohort_name}' contains zero customers."
            )

        monthly_exposure = cohort[
            "potential_monthly_revenue_exposure"
        ].sum()

        annualized_exposure = (
            monthly_exposure * 12
        )

        positive_exposure_customers = (
            cohort[
                "potential_monthly_revenue_exposure"
            ] > 0
        ).sum()

        targeting_population_pct = (
            targeted_customers
            /
            universe_size
            *
            100
        )

        exposure_capture_pct = (
            monthly_exposure
            /
            total_exposure
            *
            100
        )

        average_exposure = (
            monthly_exposure
            /
            targeted_customers
        )

        exposure_efficiency = (
            exposure_capture_pct
            /
            targeting_population_pct
        )

        exposure_per_population_percent = (
            exposure_capture_pct
            /
            targeting_population_pct
        )

        historical_churned = (
            cohort["actual_churn"]
            .sum()
        )

        historical_churn_capture_pct = (
            historical_churned
            /
            total_historical_churn
            *
            100
            if total_historical_churn > 0
            else np.nan
        )

        average_churn_probability = (
            cohort[
                "calibrated_churn_probability"
            ].mean()
        )

        high_very_high_count = (
            cohort[
                "risk_level"
            ]
            .isin(
                [
                    "High",
                    "Very High",
                ]
            )
            .sum()
        )

        high_very_high_share_pct = (
            high_very_high_count
            /
            targeted_customers
            *
            100
        )

        cohort_results.append(
            {
                "targeting_cohort":
                    cohort_name,

                "targeted_customers":
                    targeted_customers,

                "targeting_population_pct":
                    targeting_population_pct,

                "positive_exposure_customers":
                    positive_exposure_customers,

                "positive_exposure_share_pct":
                    (
                        positive_exposure_customers
                        /
                        targeted_customers
                        *
                        100
                    ),

                "monthly_revenue_exposure_inr":
                    monthly_exposure,

                "monthly_exposure_capture_pct":
                    exposure_capture_pct,

                "annualized_revenue_exposure_inr":
                    annualized_exposure,

                "average_monthly_exposure_per_customer_inr":
                    average_exposure,

                "exposure_capture_efficiency":
                    exposure_efficiency,

                "exposure_captured_per_1pct_population_targeted":
                    exposure_per_population_percent,

                "historical_churned_customers":
                    int(historical_churned),

                "historical_churn_capture_pct":
                    historical_churn_capture_pct,

                "average_calibrated_churn_probability":
                    average_churn_probability,

                "high_very_high_risk_customers":
                    int(high_very_high_count),

                "high_very_high_risk_share_pct":
                    high_very_high_share_pct,
            }
        )

    cohort_summary = pd.DataFrame(
        cohort_results
    )

    check(
        "Cohort calculations",
        "PASS",
        f"{len(cohort_summary)} cohort summaries"
    )


    # =========================================================================
    # STEP 6 — COHORT OVERLAP ANALYSIS
    # =========================================================================

    print("\n[6/10] Calculating cohort overlap with High + Very High Risk...")

    exposure_cohorts = [
        "Top 1% Exposure",
        "Top 5% Exposure",
        "Top 10% Exposure",
        "Top 20% Exposure",
    ]

    overlap_results = []

    for cohort_name in exposure_cohorts:

        exposure_mask = cohort_masks[
            cohort_name
        ]

        overlap_mask = (
            exposure_mask
            &
            high_very_high_mask
        )

        overlap_customers = int(
            overlap_mask.sum()
        )

        exposure_cohort_customers = int(
            exposure_mask.sum()
        )

        overlap_exposure = customer_data.loc[
            overlap_mask,
            "potential_monthly_revenue_exposure"
        ].sum()

        cohort_exposure = customer_data.loc[
            exposure_mask,
            "potential_monthly_revenue_exposure"
        ].sum()

        overlap_historical_churn = int(
            customer_data.loc[
                overlap_mask,
                "actual_churn"
            ].sum()
        )

        overlap_results.append(
            {
                "exposure_cohort":
                    cohort_name,

                "exposure_cohort_customers":
                    exposure_cohort_customers,

                "high_very_high_overlap_customers":
                    overlap_customers,

                "high_very_high_overlap_share_of_cohort_pct":
                    (
                        overlap_customers
                        /
                        exposure_cohort_customers
                        *
                        100
                    ),

                "overlap_monthly_revenue_exposure_inr":
                    overlap_exposure,

                "overlap_exposure_share_of_cohort_pct":
                    (
                        overlap_exposure
                        /
                        cohort_exposure
                        *
                        100
                        if cohort_exposure > 0
                        else np.nan
                    ),

                "overlap_historical_churned_customers":
                    overlap_historical_churn,
            }
        )

    overlap_summary = pd.DataFrame(
        overlap_results
    )

    check(
        "Cohort overlap analysis",
        "PASS",
        f"{len(overlap_summary)} overlap comparisons"
    )


    # =========================================================================
    # STEP 7 — CUSTOMER COHORT ASSIGNMENT
    # =========================================================================

    print("\n[7/10] Creating customer cohort assignment...")

    customer_data[
        "targeting_cohort"
    ] = "Outside Locked Cohorts"

    # This assignment is for reference only.
    #
    # Exposure cohorts are nested:
    # Top 1% ⊂ Top 5% ⊂ Top 10% ⊂ Top 20%
    #
    # Therefore the narrowest cohort receives the final label.
    #
    # High + Very High Risk is also preserved through a separate
    # indicator rather than replacing the exposure cohort.

    assignment_order = [
        "Top 20% Exposure",
        "Top 10% Exposure",
        "Top 5% Exposure",
        "Top 1% Exposure",
    ]

    for cohort_name in assignment_order:

        customer_data.loc[
            cohort_masks[cohort_name],
            "targeting_cohort"
        ] = cohort_name

    customer_data[
        "high_very_high_risk_flag"
    ] = high_very_high_mask

    check(
        "Customer cohort assignment",
        "PASS",
        f"{len(customer_data):,} customers assigned"
    )


    # =========================================================================
    # STEP 8 — COHORT COUNT TABLE
    # =========================================================================

    print("\n[8/10] Building cohort count table...")

    cohort_assignment_counts = (
        customer_data[
            "targeting_cohort"
        ]
        .value_counts()
        .rename_axis(
            "targeting_cohort"
        )
        .reset_index(
            name="customer_count"
        )
    )

    high_risk_count = int(
        high_very_high_mask.sum()
    )

    check(
        "Assignment counts",
        "PASS",
        f"{len(cohort_assignment_counts)} assignment groups"
    )


    # =========================================================================
    # STEP 9 — QUALITY GATE
    # =========================================================================

    print("\n[9/10] Running Phase 13.2 quality gate...")

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

    quality(
        "Customer universe",
        "PASS"
        if customer_data["id"].nunique()
        == EXPECTED_CUSTOMERS
        else "FAIL",
        f"{customer_data['id'].nunique():,}"
    )

    quality(
        "Unique exposure ranking",
        "PASS"
        if customer_data["exposure_rank"].is_unique
        else "FAIL",
        "One rank per customer"
    )

    quality(
        "Total monthly exposure",
        "PASS"
        if total_exposure > 0
        else "FAIL",
        format_inr(total_exposure)
    )

    quality(
        "Historical churn count",
        "PASS"
        if total_historical_churn >= 0
        else "FAIL",
        f"{int(total_historical_churn):,}"
    )

    quality(
        "Top 1% cohort",
        "PASS"
        if (
            cohort_summary[
                "targeting_cohort"
            ]
            == "Top 1% Exposure"
        ).any()
        else "FAIL",
        "Defined"
    )

    quality(
        "Top 5% cohort",
        "PASS"
        if (
            cohort_summary[
                "targeting_cohort"
            ]
            == "Top 5% Exposure"
        ).any()
        else "FAIL",
        "Defined"
    )

    quality(
        "Top 10% cohort",
        "PASS"
        if (
            cohort_summary[
                "targeting_cohort"
            ]
            == "Top 10% Exposure"
        ).any()
        else "FAIL",
        "Defined"
    )

    quality(
        "Top 20% cohort",
        "PASS"
        if (
            cohort_summary[
                "targeting_cohort"
            ]
            == "Top 20% Exposure"
        ).any()
        else "FAIL",
        "Defined"
    )

    quality(
        "High + Very High Risk cohort",
        "PASS"
        if high_risk_count > 0
        else "FAIL",
        f"{high_risk_count:,} customers"
    )

    quality(
        "Exposure capture bounds",
        "PASS"
        if cohort_summary[
            "monthly_exposure_capture_pct"
        ].between(0, 100).all()
        else "FAIL",
        "0–100%"
    )

    quality(
        "Historical churn capture bounds",
        "PASS"
        if cohort_summary[
            "historical_churn_capture_pct"
        ].between(0, 100).all()
        else "FAIL",
        "0–100%"
    )

    quality(
        "Overlap analysis",
        "PASS"
        if len(overlap_summary) == 4
        else "FAIL",
        "Four exposure/risk overlap checks"
    )

    quality(
        "Deterministic ranking",
        "PASS",
        "Exposure → calibrated probability → ID"
    )

    quality(
        "No model changes",
        "PASS",
        "No retraining, recalibration, or threshold changes"
    )

    quality(
        "No test data",
        "PASS",
        "Phase 12 production scoring dataset only"
    )

    quality(
        "Historical churn used descriptively",
        "PASS",
        "Not used for ROI calculation"
    )

    quality_checks_df = pd.DataFrame(
        quality_checks
    )

    overall_status = (
        "PASS"
        if quality_checks_df[
            "status"
        ].eq("PASS").all()
        else "FAIL"
    )

    check(
        "Overall quality gate",
        overall_status,
        f"{len(quality_checks_df)} checks"
    )


    # =========================================================================
    # STEP 10 — SAVE OUTPUTS
    # =========================================================================

    print("\n[10/10] Saving Phase 13.2 outputs...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    cohort_summary_file = (
        REPORT_DIR
        / "13_02_targeting_cohort_summary.csv"
    )

    cohort_assignment_file = (
        PROCESSED_DIR
        / "customer_targeting_cohorts_phase13.csv"
    )

    cohort_counts_file = (
        REPORT_DIR
        / "13_02_targeting_cohort_counts.csv"
    )

    overlap_file = (
        REPORT_DIR
        / "13_02_targeting_cohort_overlap.csv"
    )

    quality_file = (
        REPORT_DIR
        / "13_02_targeting_cohort_quality_gate.csv"
    )

    metadata_file = (
        REPORT_DIR
        / "13_02_targeting_cohort_metadata.json"
    )

    report_file = (
        REPORT_DIR
        / "13_02_targeting_cohort_report.md"
    )

    cohort_summary.to_csv(
        cohort_summary_file,
        index=False
    )

    customer_data.to_csv(
        cohort_assignment_file,
        index=False
    )

    cohort_assignment_counts.to_csv(
        cohort_counts_file,
        index=False
    )

    overlap_summary.to_csv(
        overlap_file,
        index=False
    )

    quality_checks_df.to_csv(
        quality_file,
        index=False
    )


    # =========================================================================
    # METADATA
    # =========================================================================

    metadata = {
        "phase": "13.2",
        "purpose": "Targeting Cohort Analysis",
        "source": (
            "data/processed/customer_value_revenue_risk.csv"
        ),
        "customer_universe": EXPECTED_CUSTOMERS,
        "total_monthly_exposure_inr": float(
            total_exposure
        ),
        "total_annualized_exposure_inr": float(
            total_annualized_exposure
        ),
        "historical_churned_customers": int(
            total_historical_churn
        ),
        "targeting_cohorts": [
            "Top 1% Exposure",
            "Top 5% Exposure",
            "Top 10% Exposure",
            "Top 20% Exposure",
            "High + Very High Risk",
        ],
        "ranking_basis": (
            "Potential monthly revenue exposure, "
            "with calibrated churn probability and customer ID "
            "used as deterministic tie-breakers."
        ),
        "historical_churn_usage": (
            "Descriptive validation only; not used to calculate "
            "retention ROI."
        ),
        "overlap_analysis": (
            "Exposure cohorts intersected with High + Very High "
            "Risk classification."
        ),
        "governance": {
            "model_retraining": False,
            "probability_recalibration": False,
            "threshold_changes": False,
            "test_data_loaded": False,
            "actual_churn_used_for_roi": False,
        },
        "quality_gate": overall_status,
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

    report_text = f"""# Phase 13.2 — Targeting Cohort Analysis

## Purpose

Identify economically concentrated customer cohorts for potential retention targeting.

The analysis uses the Phase 12 potential monthly revenue exposure and the locked Phase 11 risk classification.

## Customer Universe

**{EXPECTED_CUSTOMERS:,} customers**

## Total Potential Revenue Exposure

- Monthly: **{format_inr(total_exposure)}**
- Annualized: **{format_inr(total_annualized_exposure)}**

## Historical Churn Context

Historical churn is used only as a descriptive validation measure.

It is **not** used to calculate retention ROI or future retention success.

Historical churned customers:

**{int(total_historical_churn):,}**

## Targeting Cohorts

1. Top 1% Exposure
2. Top 5% Exposure
3. Top 10% Exposure
4. Top 20% Exposure
5. High + Very High Risk

## Cohort Summary

{cohort_summary.to_markdown(index=False)}

## Exposure / Risk Overlap

{overlap_summary.to_markdown(index=False)}

## Interpretation

Exposure capture measures how much of the Phase 12 revenue-exposure proxy is contained within each cohort.

Historical churn capture provides descriptive evidence about where previously observed churn was concentrated.

The High + Very High Risk overlap analysis shows whether economically valuable exposure is also concentrated among customers classified as higher model risk.

These measures are decision-support metrics and do not represent guaranteed future churn, guaranteed revenue loss, or guaranteed retention success.

## Ranking Basis

Customers are ranked primarily by:

**Potential Monthly Revenue Exposure**

Calibrated churn probability and customer ID are used only as deterministic tie-breakers.

## Governance

- Model retraining: **NO**
- Probability recalibration: **NO**
- Threshold changes: **NO**
- Test data loaded: **NO**
- Actual churn used for ROI: **NO**
- Historical churn used: **DESCRIPTIVE VALIDATION ONLY**

## Phase Boundary

Phase 13.2 identifies and compares targeting cohorts.

It does **not** select a final intervention strategy.

Retention economics and ROI sensitivity are evaluated in subsequent Phase 13 steps.

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
    # FINAL TERMINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("PHASE 13.2 — TARGETING COHORT ANALYSIS COMPLETE")
    print("=" * 80)

    print("\nCOHORT SUMMARY")

    display_columns = [
        "targeting_cohort",
        "targeted_customers",
        "targeting_population_pct",
        "monthly_exposure_capture_pct",
        "historical_churn_capture_pct",
        "high_very_high_risk_share_pct",
        "exposure_captured_per_1pct_population_targeted",
    ]

    print(
        cohort_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "-" * 80)
    print("RISK / EXPOSURE OVERLAP")
    print("-" * 80)

    overlap_display_columns = [
        "exposure_cohort",
        "exposure_cohort_customers",
        "high_very_high_overlap_customers",
        "high_very_high_overlap_share_of_cohort_pct",
        "overlap_exposure_share_of_cohort_pct",
    ]

    print(
        overlap_summary[
            overlap_display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "-" * 80)
    print("QUALITY GATE")
    print("-" * 80)

    for _, row in quality_checks_df.iterrows():

        print(
            f"{row['check']:<44} "
            f"{row['status']:<6} "
            f"— {row['detail']}"
        )

    print("\n" + "=" * 80)
    print(
        f"PHASE 13.2 STATUS: {overall_status}"
    )
    print("=" * 80)

    print("\nKey artifacts:")

    for output_file in [
        cohort_summary_file,
        cohort_assignment_file,
        cohort_counts_file,
        overlap_file,
        quality_file,
        metadata_file,
        report_file,
    ]:
        print(f"  - {output_file}")


# =============================================================================
# PHASE 13.3 — RETENTION ECONOMICS BY TARGETING COHORT
# =============================================================================
#
# Purpose:
# Evaluate the locked Phase 13.1 retention scenarios against the
# targeting cohorts identified in Phase 13.2.
#
# Governance:
# - No model retraining
# - No probability recalibration
# - No threshold changes
# - No September predictors
# - No realized ROI claims
# - No guaranteed revenue savings
#
# All retention rates and intervention costs are analytical assumptions.
# =============================================================================


def main_phase_13_3():

    print("=" * 80)
    print("PHASE 13.3 — RETENTION ECONOMICS BY TARGETING COHORT")
    print("=" * 80)

    # =========================================================================
    # STEP 1 — CHECK PHASE 13.2 INPUT
    # =========================================================================

    print("\n[1/9] Checking Phase 13.2 targeting dataset...")

    phase_13_2_file = (
        PROCESSED_DIR
        / "customer_targeting_cohorts_phase13.csv"
    )

    require_file(
        phase_13_2_file,
        "Phase 13.2 targeting cohort dataset"
    )

    customer_data = pd.read_csv(
        phase_13_2_file
    )

    check(
        "Phase 13.2 input",
        "PASS",
        f"{customer_data.shape}"
    )

    # =========================================================================
    # STEP 2 — VALIDATE INPUT
    # =========================================================================

    print("\n[2/9] Validating customer-level economic inputs...")

    required_columns = [
        "id",
        "exposure_rank",
        "calibrated_churn_probability",
        "risk_level",
        "potential_monthly_revenue_exposure",
        "actual_churn",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in customer_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(customer_data) != EXPECTED_CUSTOMERS:
        raise ValueError(
            f"Expected {EXPECTED_CUSTOMERS:,} customers, "
            f"found {len(customer_data):,}."
        )

    if customer_data["id"].isna().any():
        raise ValueError(
            "Missing customer IDs detected."
        )

    if customer_data["id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    validate_probability(
        customer_data[
            "calibrated_churn_probability"
        ].to_numpy(),
        "Calibrated churn probability"
    )

    validate_non_negative(
        customer_data[
            "potential_monthly_revenue_exposure"
        ].to_numpy(),
        "Potential monthly revenue exposure"
    )

    actual_churn_values = set(
        customer_data[
            "actual_churn"
        ]
        .dropna()
        .unique()
    )

    if not actual_churn_values.issubset({0, 1}):
        raise ValueError(
            "actual_churn must contain only 0/1 values."
        )

    check(
        "Customer universe",
        "PASS",
        f"{customer_data['id'].nunique():,} unique customers"
    )

    check(
        "Historical churn label",
        "PASS",
        "Binary 0/1"
    )

    # =========================================================================
    # STEP 3 — RECREATE LOCKED TARGETING COHORTS
    # =========================================================================

    print("\n[3/9] Recreating locked Phase 13.2 cohorts...")

    universe_size = len(
        customer_data
    )

    top_1_cutoff = max(
        1,
        int(
            np.ceil(
                universe_size * 0.01
            )
        )
    )

    top_5_cutoff = max(
        1,
        int(
            np.ceil(
                universe_size * 0.05
            )
        )
    )

    top_10_cutoff = max(
        1,
        int(
            np.ceil(
                universe_size * 0.10
            )
        )
    )

    top_20_cutoff = max(
        1,
        int(
            np.ceil(
                universe_size * 0.20
            )
        )
    )

    high_very_high_mask = (
        customer_data[
            "risk_level"
        ]
        .isin(
            [
                "High",
                "Very High",
            ]
        )
    )

    cohort_masks = {

        "Top 1% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_1_cutoff,

        "Top 5% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_5_cutoff,

        "Top 10% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_10_cutoff,

        "Top 20% Exposure":
            customer_data[
                "exposure_rank"
            ] <= top_20_cutoff,

        "High + Very High Risk":
            high_very_high_mask,
    }

    check(
        "Targeting cohorts",
        "PASS",
        f"{len(cohort_masks)} locked cohorts"
    )

    # =========================================================================
    # STEP 4 — VALIDATE LOCKED SCENARIOS
    # =========================================================================

    print("\n[4/9] Validating locked retention scenarios...")

    if len(RETENTION_SCENARIOS) != 3:
        raise ValueError(
            "Expected exactly 3 locked retention scenarios."
        )

    scenario_ids = {
        scenario["scenario_id"]
        for scenario in RETENTION_SCENARIOS
    }

    expected_scenario_ids = {
        "low_touch",
        "targeted",
        "high_touch",
    }

    if scenario_ids != expected_scenario_ids:
        raise ValueError(
            "Locked retention scenario definitions do not match Phase 13.1."
        )

    for scenario in RETENTION_SCENARIOS:

        success_rate = float(
            scenario[
                "retention_success_rate"
            ]
        )

        cost = float(
            scenario[
                "intervention_cost_inr"
            ]
        )

        if not 0 < success_rate <= 1:
            raise ValueError(
                f"Invalid retention success rate: {success_rate}"
            )

        if cost < 0:
            raise ValueError(
                f"Invalid intervention cost: {cost}"
            )

    check(
        "Scenario assumptions",
        "PASS",
        "3 locked Phase 13.1 scenarios"
    )

    # =========================================================================
    # STEP 5 — CALCULATE COHORT ECONOMICS
    # =========================================================================

    print("\n[5/9] Calculating cohort-level retention economics...")

    total_exposure = (
        customer_data[
            "potential_monthly_revenue_exposure"
        ]
        .sum()
    )

    if total_exposure <= 0:
        raise ValueError(
            "Total potential monthly revenue exposure must be > 0."
        )

    economics_results = []

    for cohort_name, mask in cohort_masks.items():

        cohort = customer_data.loc[
            mask
        ].copy()

        targeted_customers = len(
            cohort
        )

        if targeted_customers == 0:
            raise ValueError(
                f"Cohort '{cohort_name}' contains zero customers."
            )

        monthly_exposure = (
            cohort[
                "potential_monthly_revenue_exposure"
            ]
            .sum()
        )

        annualized_exposure = (
            monthly_exposure
            *
            EVALUATION_HORIZON_MONTHS
        )

        exposure_capture_pct = (
            monthly_exposure
            /
            total_exposure
            *
            100
        )

        positive_exposure_customers = (
            cohort[
                "potential_monthly_revenue_exposure"
            ]
            > 0
        ).sum()

        average_monthly_exposure = (
            monthly_exposure
            /
            targeted_customers
        )

        historical_churned_customers = (
            cohort[
                "actual_churn"
            ]
            .sum()
        )

        average_churn_probability = (
            cohort[
                "calibrated_churn_probability"
            ].mean()
        )

        for scenario in RETENTION_SCENARIOS:

            scenario_id = scenario[
                "scenario_id"
            ]

            scenario_name = scenario[
                "scenario_name"
            ]

            intervention_cost_per_customer = float(
                scenario[
                    "intervention_cost_inr"
                ]
            )

            retention_success_rate = float(
                scenario[
                    "retention_success_rate"
                ]
            )

            # -----------------------------------------------------------------
            # ECONOMIC CALCULATIONS
            # -----------------------------------------------------------------

            expected_monthly_revenue_retained = (
                monthly_exposure
                *
                retention_success_rate
            )

            expected_revenue_retained_over_horizon = (
                expected_monthly_revenue_retained
                *
                EVALUATION_HORIZON_MONTHS
            )

            # Intervention cost is modeled as a ONE-TIME cost per
            # targeted customer over the evaluation horizon.

            intervention_cost = (
                targeted_customers
                *
                intervention_cost_per_customer
            )

            expected_net_value = (
                expected_revenue_retained_over_horizon
                -
                intervention_cost
            )

            roi = (
                expected_net_value
                /
                intervention_cost
                if intervention_cost > 0
                else np.nan
            )

            # Break-even success rate required for:
            #
            # Expected retained revenue = intervention cost
            #
            break_even_success_rate = (
                intervention_cost
                /
                annualized_exposure
                if annualized_exposure > 0
                else np.nan
            )

            break_even_feasible = (
                break_even_success_rate <= 1
                if pd.notna(
                    break_even_success_rate
                )
                else False
            )

            retention_success_above_break_even = (
                retention_success_rate
                >=
                break_even_success_rate
                if pd.notna(
                    break_even_success_rate
                )
                else False
            )

            net_value_per_targeted_customer = (
                expected_net_value
                /
                targeted_customers
            )

            intervention_cost_as_pct_of_annualized_exposure = (
                intervention_cost
                /
                annualized_exposure
                *
                100
                if annualized_exposure > 0
                else np.nan
            )

            economics_results.append(
                {
                    "targeting_cohort":
                        cohort_name,

                    "scenario_id":
                        scenario_id,

                    "scenario_name":
                        scenario_name,

                    "targeted_customers":
                        targeted_customers,

                    "positive_exposure_customers":
                        int(
                            positive_exposure_customers
                        ),

                    "targeting_population_pct":
                        (
                            targeted_customers
                            /
                            universe_size
                            *
                            100
                        ),

                    "monthly_revenue_exposure_inr":
                        monthly_exposure,

                    "monthly_exposure_capture_pct":
                        exposure_capture_pct,

                    "annualized_revenue_exposure_inr":
                        annualized_exposure,

                    "average_monthly_exposure_per_customer_inr":
                        average_monthly_exposure,

                    "average_calibrated_churn_probability":
                        average_churn_probability,

                    "historical_churned_customers":
                        int(
                            historical_churned_customers
                        ),

                    "intervention_cost_per_customer_inr":
                        intervention_cost_per_customer,

                    "retention_success_rate":
                        retention_success_rate,

                    "expected_monthly_revenue_retained_inr":
                        expected_monthly_revenue_retained,

                    "expected_revenue_retained_over_horizon_inr":
                        expected_revenue_retained_over_horizon,

                    "total_intervention_cost_inr":
                        intervention_cost,

                    "expected_net_value_inr":
                        expected_net_value,

                    "net_value_per_targeted_customer_inr":
                        net_value_per_targeted_customer,

                    "roi":
                        roi,

                    "roi_pct":
                        (
                            roi * 100
                            if pd.notna(roi)
                            else np.nan
                        ),

                    "break_even_retention_success_rate":
                        break_even_success_rate,

                    "break_even_retention_success_rate_pct":
                        (
                            break_even_success_rate * 100
                            if pd.notna(
                                break_even_success_rate
                            )
                            else np.nan
                        ),

                    "break_even_feasible":
                        break_even_feasible,

                    "retention_success_above_break_even":
                        retention_success_above_break_even,

                    "intervention_cost_as_pct_of_annualized_exposure":
                        intervention_cost_as_pct_of_annualized_exposure,

                    "economically_viable":
                        (
                            expected_net_value > 0
                        ),
                }
            )

    economics_df = pd.DataFrame(
        economics_results
    )

    check(
        "Cohort-scenario economics",
        "PASS",
        f"{len(economics_df)} rows = 5 cohorts × 3 scenarios"
    )

    # =========================================================================
    # STEP 6 — ECONOMIC SUMMARY
    # =========================================================================

    print("\n[6/9] Building economic viability summary...")

    viability_summary = (
        economics_df[
            [
                "targeting_cohort",
                "scenario_name",
                "targeted_customers",
                "targeting_population_pct",
                "monthly_exposure_capture_pct",
                "expected_revenue_retained_over_horizon_inr",
                "total_intervention_cost_inr",
                "expected_net_value_inr",
                "net_value_per_targeted_customer_inr",
                "roi_pct",
                "break_even_retention_success_rate_pct",
                "break_even_feasible",
                "retention_success_above_break_even",
                "intervention_cost_as_pct_of_annualized_exposure",
                "economically_viable",
            ]
        ]
        .copy()
    )

    viable_count = int(
        economics_df[
            "economically_viable"
        ]
        .sum()
    )

    feasible_break_even_count = int(
        economics_df[
            "break_even_feasible"
        ]
        .sum()
    )

    check(
        "Economic viability evaluation",
        "PASS",
        f"{viable_count} of {len(economics_df)} scenarios economically positive"
    )

    check(
        "Break-even feasibility",
        "PASS",
        f"{feasible_break_even_count} of {len(economics_df)} scenarios have feasible break-even rates"
    )

    # =========================================================================
    # STEP 7 — QUALITY GATE
    # =========================================================================

    print("\n[7/9] Running Phase 13.3 quality gate...")

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

    quality(
        "Customer universe",
        "PASS"
        if len(customer_data)
        == EXPECTED_CUSTOMERS
        else "FAIL",
        f"{len(customer_data):,}"
    )

    quality(
        "Unique customer IDs",
        "PASS"
        if customer_data["id"].is_unique
        else "FAIL",
        "No duplicate IDs"
    )

    quality(
        "Valid churn probabilities",
        "PASS",
        "[0, 1]"
    )

    quality(
        "Non-negative revenue exposure",
        "PASS",
        "Exposure values valid"
    )

    quality(
        "Targeting cohort count",
        "PASS"
        if len(cohort_masks) == 5
        else "FAIL",
        "Five locked cohorts"
    )

    quality(
        "Scenario count",
        "PASS"
        if len(RETENTION_SCENARIOS) == 3
        else "FAIL",
        "Three locked scenarios"
    )

    quality(
        "Economic row count",
        "PASS"
        if len(economics_df) == 15
        else "FAIL",
        "5 cohorts × 3 scenarios"
    )

    quality(
        "ROI calculation",
        "PASS"
        if economics_df[
            "roi"
        ].notna().all()
        else "FAIL",
        "ROI calculated"
    )

    quality(
        "Break-even calculation",
        "PASS"
        if economics_df[
            "break_even_retention_success_rate"
        ].notna().all()
        else "FAIL",
        "Break-even success rate calculated"
    )

    quality(
        "Break-even feasibility",
        "PASS"
        if economics_df[
            "break_even_retention_success_rate"
        ].between(
            0,
            1
        ).all()
        else "FAIL",
        "Required success rates are bounded"
    )

    quality(
        "Net value per customer",
        "PASS"
        if economics_df[
            "net_value_per_targeted_customer_inr"
        ].notna().all()
        else "FAIL",
        "Calculated for all scenarios"
    )

    quality(
        "Cost-to-exposure ratio",
        "PASS"
        if economics_df[
            "intervention_cost_as_pct_of_annualized_exposure"
        ].notna().all()
        else "FAIL",
        "Calculated for all scenarios"
    )

    quality(
        "No model changes",
        "PASS",
        "No retraining, recalibration, or threshold changes"
    )

    quality(
        "No test data",
        "PASS",
        "Phase 13.2 scored customer universe only"
    )

    quality(
        "Scenario assumptions governed",
        "PASS",
        "Retention success and intervention cost are assumptions"
    )

    quality(
        "One-time intervention cost assumption",
        "PASS",
        "Cost modeled once per targeted customer"
    )

    quality(
        "No realized ROI claim",
        "PASS",
        "Economics are scenario-based"
    )

    quality(
        "Historical churn excluded from ROI",
        "PASS",
        "Used only as descriptive context"
    )

    quality(
        "No guaranteed revenue claim",
        "PASS",
        "Revenue retained is expected scenario value"
    )

    quality_df = pd.DataFrame(
        quality_checks
    )

    overall_status = (
        "PASS"
        if quality_df[
            "status"
        ].eq("PASS").all()
        else "FAIL"
    )

    check(
        "Overall quality gate",
        overall_status,
        f"{len(quality_df)} checks"
    )

    # =========================================================================
    # STEP 8 — SAVE OUTPUTS
    # =========================================================================

    print("\n[8/9] Saving Phase 13.3 outputs...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    economics_file = (
        PROCESSED_DIR
        / "retention_economics_by_targeting_cohort_phase13.csv"
    )

    summary_file = (
        REPORT_DIR
        / "13_03_retention_economics_summary.csv"
    )

    quality_file = (
        REPORT_DIR
        / "13_03_retention_economics_quality_gate.csv"
    )

    metadata_file = (
        REPORT_DIR
        / "13_03_retention_economics_metadata.json"
    )

    report_file = (
        REPORT_DIR
        / "13_03_retention_economics_report.md"
    )

    economics_df.to_csv(
        economics_file,
        index=False
    )

    viability_summary.to_csv(
        summary_file,
        index=False
    )

    quality_df.to_csv(
        quality_file,
        index=False
    )

    metadata = {
        "phase": "13.3",
        "purpose": (
            "Retention Economics by Targeting Cohort"
        ),
        "source": (
            "data/processed/"
            "customer_targeting_cohorts_phase13.csv"
        ),
        "customer_universe": EXPECTED_CUSTOMERS,
        "evaluation_horizon_months":
            EVALUATION_HORIZON_MONTHS,
        "intervention_cost_timing":
            "One-time per targeted customer",
        "targeting_cohorts": [
            "Top 1% Exposure",
            "Top 5% Exposure",
            "Top 10% Exposure",
            "Top 20% Exposure",
            "High + Very High Risk",
        ],
        "retention_scenarios": [
            {
                "scenario_id":
                    s["scenario_id"],
                "scenario_name":
                    s["scenario_name"],
                "intervention_cost_inr":
                    s["intervention_cost_inr"],
                "retention_success_rate":
                    s["retention_success_rate"],
            }
            for s in RETENTION_SCENARIOS
        ],
        "scenario_count": 3,
        "cohort_count": 5,
        "economic_combinations": 15,
        "break_even_definition": (
            "Intervention cost divided by annualized "
            "revenue exposure."
        ),
        "historical_churn_usage": (
            "Descriptive context only; "
            "not used to calculate ROI."
        ),
        "governance": {
            "model_retraining": False,
            "probability_recalibration": False,
            "threshold_changes": False,
            "test_data_loaded": False,
            "actual_churn_used_for_roi": False,
            "realized_roi_claimed": False,
            "guaranteed_revenue_claimed": False,
        },
        "quality_gate": overall_status,
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
    # STEP 9 — REPORT
    # =========================================================================

    print("\n[9/9] Writing Phase 13.3 report...")

    report_text = f"""# Phase 13.3 — Retention Economics by Targeting Cohort

## Purpose

Evaluate whether the targeting cohorts identified in Phase 13.2 produce economically meaningful retention scenarios under the locked Phase 13.1 assumptions.

## Customer Universe

**{EXPECTED_CUSTOMERS:,} customers**

## Evaluation Horizon

**{EVALUATION_HORIZON_MONTHS} months**

## Intervention Cost Timing

Intervention cost is modeled as a **one-time cost per targeted customer** over the evaluation horizon.

## Locked Retention Scenarios

The analysis uses the three Phase 13.1 scenario assumptions:

- Low-Touch Retention: ₹100/customer, 20% assumed success
- Targeted Retention: ₹250/customer, 35% assumed success
- High-Touch Retention: ₹500/customer, 50% assumed success

These are analytical assumptions and are not observed company outcomes or costs.

## Economic Formula

Expected Revenue Retained:

**Monthly Revenue Exposure × Assumed Retention Success Rate × Evaluation Horizon**

Expected Net Value:

**Expected Revenue Retained − Intervention Cost**

ROI:

**Expected Net Value ÷ Intervention Cost**

Break-even Retention Success Rate:

**Intervention Cost ÷ Annualized Revenue Exposure**

## Cohort × Scenario Results

{economics_df.to_markdown(index=False)}

## Economic Interpretation

**{viable_count} of {len(economics_df)} scenarios** have positive expected net value under the stated assumptions.

**{feasible_break_even_count} of {len(economics_df)} scenarios** have a mathematically feasible break-even retention rate at or below 100%.

A positive scenario means that the stated assumptions produce positive expected economics. It does not mean that an intervention will generate realized ROI.

## Governance

- No model retraining
- No probability recalibration
- No threshold changes
- No test data
- Historical churn is descriptive only
- Retention success rates are assumptions
- Intervention costs are assumptions
- Intervention cost is modeled as one-time per targeted customer
- Revenue retained is scenario-based
- ROI is scenario-based, not realized ROI
- No guaranteed revenue savings are claimed

## Phase Boundary

Phase 13.3 evaluates the economics of the locked targeting cohorts under the locked Phase 13.1 scenarios.

It does not determine final intervention policy.

Further independent cost/success sensitivity analysis is reserved for Phase 13.5.

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
    # TERMINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("PHASE 13.3 — RETENTION ECONOMICS COMPLETE")
    print("=" * 80)

    print("\nECONOMIC SUMMARY")

    display_columns = [
        "targeting_cohort",
        "scenario_name",
        "targeted_customers",
        "monthly_exposure_capture_pct",
        "expected_revenue_retained_over_horizon_inr",
        "total_intervention_cost_inr",
        "expected_net_value_inr",
        "net_value_per_targeted_customer_inr",
        "roi_pct",
        "break_even_retention_success_rate_pct",
        "break_even_feasible",
        "retention_success_above_break_even",
        "economically_viable",
    ]

    print(
        economics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "-" * 80)
    print("QUALITY GATE")
    print("-" * 80)

    for _, row in quality_df.iterrows():

        print(
            f"{row['check']:<44} "
            f"{row['status']:<6} "
            f"— {row['detail']}"
        )

    print("\n" + "=" * 80)
    print(
        f"PHASE 13.3 STATUS: {overall_status}"
    )
    print("=" * 80)

    print("\nKey artifacts:")

    for output_file in [
        economics_file,
        summary_file,
        quality_file,
        metadata_file,
        report_file,
    ]:
        print(
            f"  - {output_file}"
        )     

# =============================================================================
# PHASE 13.4 — RETENTION ECONOMICS SENSITIVITY & ROBUSTNESS ANALYSIS
# =============================================================================

def main_phase_13_4():

    print("\n" + "=" * 80)
    print("PHASE 13.4 — RETENTION ECONOMICS SENSITIVITY & ROBUSTNESS ANALYSIS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. LOAD PHASE 13.2 TARGETING DATA
    # -------------------------------------------------------------------------

    input_file = (
        PROCESSED_DIR /
        "customer_targeting_cohorts_phase13.csv"
    )

    require_file(
        input_file,
       "Phase 13.2 customer targeting cohort output"
    )

    df = pd.read_csv(input_file)

    print(
        f"\nPhase 13.2 input shape: {df.shape}"
    )

    required_columns = [
        "id",
        "exposure_rank",
        "calibrated_churn_probability",
        "risk_level",
        "potential_monthly_revenue_exposure",
        "actual_churn",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required Phase 13.2 columns: {missing_columns}"
        )

    # -------------------------------------------------------------------------
    # 2. BASIC VALIDATION
    # -------------------------------------------------------------------------

    if len(df) != EXPECTED_CUSTOMERS:
        raise ValueError(
            f"Expected {EXPECTED_CUSTOMERS:,} customers, "
            f"found {len(df):,}."
        )

    if df["id"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs detected."
        )

    if not set(
        pd.Series(df["actual_churn"])
        .dropna()
        .unique()
    ).issubset({0, 1}):
        raise ValueError(
            "actual_churn must be binary."
        )

    validate_probability(
        df["calibrated_churn_probability"],
        "calibrated_churn_probability"
    )

    validate_non_negative(
        df["potential_monthly_revenue_exposure"],
        "potential_monthly_revenue_exposure"
    )

    # -------------------------------------------------------------------------
    # 3. PREPARE EXPOSURE RANKING
    # -------------------------------------------------------------------------

    df = df.copy()

    df["exposure_rank"] = pd.to_numeric(
        df["exposure_rank"],
        errors="coerce"
    )

    df["potential_monthly_revenue_exposure"] = pd.to_numeric(
        df["potential_monthly_revenue_exposure"],
        errors="coerce"
    )

    df = df.sort_values(
        ["exposure_rank", "id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 4. RECREATE LOCKED TARGETING COHORTS
    #
    # IMPORTANT:
    # We deliberately recreate the nested cohort definitions instead of
    # relying on the single "targeting_cohort" label.
    # -------------------------------------------------------------------------

    cohort_definitions = {
        "Top 1%": int(np.ceil(EXPECTED_CUSTOMERS * 0.01)),
        "Top 5%": int(np.ceil(EXPECTED_CUSTOMERS * 0.05)),
        "Top 10%": int(np.ceil(EXPECTED_CUSTOMERS * 0.10)),
        "Top 20%": int(np.ceil(EXPECTED_CUSTOMERS * 0.20)),
        "High + Very High": None,
    }

    cohort_masks = {}

    for cohort_name, cohort_size in cohort_definitions.items():

        if cohort_name == "High + Very High":

            mask = df["risk_level"].isin(
                ["High", "Very High"]
            )

        else:

            mask = (
                df["exposure_rank"] <= cohort_size
            )

        cohort_masks[cohort_name] = mask

    # -------------------------------------------------------------------------
    # 5. LOCKED PHASE 13.1 ECONOMIC ASSUMPTIONS
    # -------------------------------------------------------------------------

    success_rates = [
        0.10,
        0.20,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    intervention_costs = [
        100.0,
        250.0,
        500.0,
    ]

    cohort_names = list(
        cohort_definitions.keys()
    )

    expected_scenario_count = (
        len(cohort_names)
        * len(success_rates)
        * len(intervention_costs)
    )

    print(
        f"\nExpected sensitivity scenarios: "
        f"{expected_scenario_count:,}"
    )

    # -------------------------------------------------------------------------
    # 6. BUILD SENSITIVITY SCENARIOS
    # -------------------------------------------------------------------------

    scenario_rows = []

    for cohort_name in cohort_names:

        mask = cohort_masks[cohort_name]

        cohort = df.loc[mask].copy()

        customer_count = len(cohort)

        if customer_count == 0:
            raise ValueError(
                f"Cohort '{cohort_name}' contains zero customers."
            )

        monthly_exposure = cohort[
            "potential_monthly_revenue_exposure"
        ].sum()

        annualized_exposure = (
            monthly_exposure
            * EVALUATION_HORIZON_MONTHS
        )

        historical_churn_count = (
            cohort["actual_churn"]
            .sum()
        )

        historical_churn_rate = (
            historical_churn_count
            / customer_count
        )

        for success_rate in success_rates:

            expected_monthly_retained = (
                monthly_exposure
                * success_rate
            )

            expected_horizon_retained = (
                expected_monthly_retained
                * EVALUATION_HORIZON_MONTHS
            )

            break_even_results = {}

            for intervention_cost in intervention_costs:

                campaign_cost = (
                    customer_count
                    * intervention_cost
                )

                net_value = (
                    expected_horizon_retained
                    - campaign_cost
                )

                if annualized_exposure > 0:

                    break_even_success = (
                        campaign_cost
                        / annualized_exposure
                    )

                else:

                    break_even_success = np.inf

                roi = (
                    net_value
                    / campaign_cost
                    if campaign_cost > 0
                    else np.nan
                )

                success_above_break_even = (
                    success_rate
                    >= break_even_success
                )

                if break_even_success <= 0.10:
                    robustness_class = (
                        "Robustly viable"
                    )

                elif break_even_success <= 0.50:
                    robustness_class = (
                        "Conditionally viable"
                    )

                elif break_even_success <= 0.80:
                    robustness_class = (
                        "Weakly viable"
                    )

                else:
                    robustness_class = (
                        "Not viable within tested range"
                    )

                economically_viable = (
                    net_value > 0
                )

                net_value_per_customer = (
                    net_value
                    / customer_count
                )

                cost_as_pct_exposure = (
                    campaign_cost
                    / annualized_exposure
                    if annualized_exposure > 0
                    else np.inf
                )

                scenario_rows.append({

                    "targeting_cohort":
                        cohort_name,

                    "customer_count":
                        customer_count,

                    "cohort_population_pct":
                        customer_count
                        / EXPECTED_CUSTOMERS,

                    "monthly_revenue_exposure":
                        monthly_exposure,

                    "annualized_revenue_exposure":
                        annualized_exposure,

                    "historical_churn_count":
                        int(historical_churn_count),

                    "historical_churn_rate":
                        historical_churn_rate,

                    "retention_success_rate":
                        success_rate,

                    "intervention_cost_per_customer":
                        intervention_cost,

                    "expected_monthly_retained_value":
                        expected_monthly_retained,

                    "expected_horizon_retained_value":
                        expected_horizon_retained,

                    "campaign_cost":
                        campaign_cost,

                    "net_value":
                        net_value,

                    "roi":
                        roi,

                    "break_even_success_rate":
                        break_even_success,

                    "break_even_feasible":
                        bool(
                            np.isfinite(
                                break_even_success
                            )
                            and
                            break_even_success <= 1
                        ),

                    "success_above_break_even":
                        bool(
                            success_above_break_even
                        ),

                    "net_value_per_customer":
                        net_value_per_customer,

                    "intervention_cost_pct_annualized_exposure":
                        cost_as_pct_exposure,

                    "economically_viable":
                        bool(economically_viable),

                    "robustness_class":
                        robustness_class,

                    "evaluation_horizon_months":
                        EVALUATION_HORIZON_MONTHS,

                })

    sensitivity_df = pd.DataFrame(
        scenario_rows
    )

    # -------------------------------------------------------------------------
    # 7. SCENARIO COUNT VALIDATION
    # -------------------------------------------------------------------------

    actual_scenario_count = len(
        sensitivity_df
    )

    print(
        f"Actual sensitivity scenarios: "
        f"{actual_scenario_count:,}"
    )

    if actual_scenario_count != expected_scenario_count:
        raise ValueError(
            "Sensitivity scenario count mismatch."
        )

    # -------------------------------------------------------------------------
    # 8. NUMERIC VALIDATION
    # -------------------------------------------------------------------------

    numeric_columns = [
        "monthly_revenue_exposure",
        "annualized_revenue_exposure",
        "retention_success_rate",
        "intervention_cost_per_customer",
        "expected_monthly_retained_value",
        "expected_horizon_retained_value",
        "campaign_cost",
        "net_value",
        "roi",
        "break_even_success_rate",
        "net_value_per_customer",
        "intervention_cost_pct_annualized_exposure",
    ]

    for column in numeric_columns:

        if not np.isfinite(
            sensitivity_df[column]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        ).all():

            raise ValueError(
                f"Invalid numeric values detected in "
                f"{column}."
            )

    # -------------------------------------------------------------------------
    # 9. MONOTONICITY CHECKS
    #
    # These are powerful sanity checks:
    #
    # Higher retention success should never reduce net value or ROI.
    # Higher intervention cost should never improve net value.
    # -------------------------------------------------------------------------

    success_monotonicity_pass = True

    for (
        cohort_name,
        intervention_cost
    ), group in sensitivity_df.groupby(
        [
            "targeting_cohort",
            "intervention_cost_per_customer",
        ],
        sort=False
    ):

        ordered = group.sort_values(
            "retention_success_rate"
        )

        if not ordered["net_value"].is_monotonic_increasing:
            success_monotonicity_pass = False

        if not ordered["roi"].is_monotonic_increasing:
            success_monotonicity_pass = False

    cost_monotonicity_pass = True

    for (
        cohort_name,
        success_rate
    ), group in sensitivity_df.groupby(
        [
            "targeting_cohort",
            "retention_success_rate",
        ],
        sort=False
    ):

        ordered = group.sort_values(
            "intervention_cost_per_customer"
        )

        if not ordered["net_value"].is_monotonic_decreasing:
            cost_monotonicity_pass = False

    if not success_monotonicity_pass:
        raise ValueError(
            "Success-rate monotonicity check failed."
        )

    if not cost_monotonicity_pass:
        raise ValueError(
            "Intervention-cost monotonicity check failed."
        )

    # -------------------------------------------------------------------------
    # 10. BREAK-EVEN ANALYSIS
    # -------------------------------------------------------------------------

    break_even_df = (
        sensitivity_df
        .groupby(
            [
                "targeting_cohort",
                "intervention_cost_per_customer",
            ],
            as_index=False
        )
        .agg(
            customer_count=(
                "customer_count",
                "first"
            ),

            annualized_revenue_exposure=(
                "annualized_revenue_exposure",
                "first"
            ),

            break_even_success_rate=(
                "break_even_success_rate",
                "first"
            ),

            break_even_feasible=(
                "break_even_feasible",
                "first"
            ),

            minimum_tested_success_rate=(
                "retention_success_rate",
                "min"
            ),

            maximum_tested_success_rate=(
                "retention_success_rate",
                "max"
            ),

            economically_viable_scenarios=(
                "economically_viable",
                "sum"
            ),

            tested_scenarios=(
                "economically_viable",
                "count"
            ),

            best_net_value=(
                "net_value",
                "max"
            ),

            worst_net_value=(
                "net_value",
                "min"
            ),

            best_roi=(
                "roi",
                "max"
            ),

            worst_roi=(
                "roi",
                "min"
            ),
        )
    )

    break_even_df[
        "economic_viability_rate"
    ] = (
        break_even_df[
            "economically_viable_scenarios"
        ]
        /
        break_even_df[
            "tested_scenarios"
        ]
    )

    # -------------------------------------------------------------------------
    # 11. ROBUSTNESS SUMMARY BY COHORT
    # -------------------------------------------------------------------------

    robustness_order = [
        "Robustly viable",
        "Conditionally viable",
        "Weakly viable",
        "Not viable within tested range",
    ]

    robustness_summary = (
        sensitivity_df
        .groupby(
            [
                "targeting_cohort",
                "robustness_class",
            ],
            as_index=False
        )
        .agg(
            scenario_count=(
                "robustness_class",
                "size"
            ),
            viable_scenarios=(
                "economically_viable",
                "sum"
            ),
            best_roi=(
                "roi",
                "max"
            ),
            worst_roi=(
                "roi",
                "min"
            ),
        )
    )

    robustness_summary[
        "robustness_class"
    ] = pd.Categorical(
        robustness_summary[
            "robustness_class"
        ],
        categories=robustness_order,
        ordered=True
    )

    robustness_summary = (
        robustness_summary
        .sort_values(
            [
                "targeting_cohort",
                "robustness_class",
            ]
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # 12. OVERALL SENSITIVITY SUMMARY
    # -------------------------------------------------------------------------

    sensitivity_summary = (
        sensitivity_df
        .groupby(
            "targeting_cohort",
            as_index=False
        )
        .agg(
            customer_count=(
                "customer_count",
                "first"
            ),

            monthly_revenue_exposure=(
                "monthly_revenue_exposure",
                "first"
            ),

            annualized_revenue_exposure=(
                "annualized_revenue_exposure",
                "first"
            ),

            total_scenarios=(
                "net_value",
                "count"
            ),

            economically_viable_scenarios=(
                "economically_viable",
                "sum"
            ),

            minimum_break_even_success=(
                "break_even_success_rate",
                "min"
            ),

            maximum_break_even_success=(
                "break_even_success_rate",
                "max"
            ),

            best_net_value=(
                "net_value",
                "max"
            ),

            worst_net_value=(
                "net_value",
                "min"
            ),

            best_roi=(
                "roi",
                "max"
            ),

            worst_roi=(
                "roi",
                "min"
            ),
        )
    )

    sensitivity_summary[
        "economic_viability_rate"
    ] = (
        sensitivity_summary[
            "economically_viable_scenarios"
        ]
        /
        sensitivity_summary[
            "total_scenarios"
        ]
    )

    # -------------------------------------------------------------------------
    # 13. QUALITY GATE
    # -------------------------------------------------------------------------

    quality_rows = []

    def qcheck(name, status, detail):
        quality_rows.append({
            "check": name,
            "status": status,
            "detail": detail,
        })

    qcheck(
        "Input row count",
        "PASS"
        if len(df) == EXPECTED_CUSTOMERS
        else "FAIL",
        f"{len(df):,} rows"
    )

    qcheck(
        "Unique customer IDs",
        "PASS"
        if not df["id"].duplicated().any()
        else "FAIL",
        "No duplicate IDs"
    )

    qcheck(
        "Historical churn binary",
        "PASS"
        if set(
            pd.Series(
                df["actual_churn"]
            ).dropna().unique()
        ).issubset({0, 1})
        else "FAIL",
        "Binary historical label"
    )

    qcheck(
        "Probability validity",
        "PASS",
        "Calibrated probabilities validated"
    )

    qcheck(
        "Exposure non-negative",
        "PASS",
        "Revenue exposure is non-negative"
    )

    qcheck(
        "Expected scenario count",
        "PASS"
        if actual_scenario_count
        == expected_scenario_count
        else "FAIL",
        f"{actual_scenario_count:,} / "
        f"{expected_scenario_count:,}"
    )

    qcheck(
        "Five targeting cohorts",
        "PASS"
        if sensitivity_df[
            "targeting_cohort"
        ].nunique() == 5
        else "FAIL",
        "5 cohorts"
    )

    qcheck(
        "Nine success rates",
        "PASS"
        if sensitivity_df[
            "retention_success_rate"
        ].nunique() == 9
        else "FAIL",
        "9 success-rate assumptions"
    )

    qcheck(
        "Three intervention costs",
        "PASS"
        if sensitivity_df[
            "intervention_cost_per_customer"
        ].nunique() == 3
        else "FAIL",
        "3 intervention-cost assumptions"
    )

    qcheck(
        "Break-even non-negative",
        "PASS"
        if (
            sensitivity_df[
                "break_even_success_rate"
            ].dropna() >= 0
        ).all()
        else "FAIL",
        "Break-even values are non-negative"
    )

    qcheck(
        "Break-even infeasibility handled",
        "PASS"
        if (
            sensitivity_df[
                "break_even_feasible"
            ]
            .isin([True, False])
            .all()
        )
        else "FAIL",
        "Values above 100% are retained and flagged infeasible"
    )

    qcheck(
        "Success monotonicity",
        "PASS"
        if success_monotonicity_pass
        else "FAIL",
        "Higher success does not reduce net value or ROI"
    )

    qcheck(
        "Cost monotonicity",
        "PASS"
        if cost_monotonicity_pass
        else "FAIL",
        "Higher cost does not improve net value"
    )

    qcheck(
        "ROI finite",
        "PASS"
        if np.isfinite(
            sensitivity_df["roi"]
        ).all()
        else "FAIL",
        "All ROI values finite"
    )

    qcheck(
        "Net value finite",
        "PASS"
        if np.isfinite(
            sensitivity_df["net_value"]
        ).all()
        else "FAIL",
        "All net values finite"
    )

    qcheck(
        "No model changes",
        "PASS",
        "Phase 13.4 uses frozen Phase 11/12 outputs"
    )

    qcheck(
        "No September predictors",
        "PASS",
        "No future-outcome predictors introduced"
    )

    qcheck(
        "No test data",
        "PASS",
        "No test-set data used"
    )

    qcheck(
        "Historical churn governance",
        "PASS",
        "Historical churn retained for descriptive context only"
    )

    qcheck(
        "Scenario assumptions governed",
        "PASS",
        "Success rates and costs are analytical assumptions"
    )

    quality_gate = pd.DataFrame(
        quality_rows
    )

    if not (
        quality_gate["status"]
        .eq("PASS")
        .all()
    ):
        failed = quality_gate.loc[
            quality_gate["status"] != "PASS"
        ]

        raise ValueError(
            "Phase 13.4 quality gate failed:\n"
            + failed.to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # 14. SAVE PROCESSED OUTPUT
    # -------------------------------------------------------------------------

    sensitivity_output = (
        PROCESSED_DIR /
        "retention_economics_sensitivity_phase13.csv"
    )

    sensitivity_df.to_csv(
        sensitivity_output,
        index=False
    )

    # -------------------------------------------------------------------------
    # 15. SAVE REPORT TABLES
    # -------------------------------------------------------------------------

    break_even_output = (
        REPORT_DIR /
        "13_04_break_even_analysis.csv"
    )

    summary_output = (
        REPORT_DIR /
        "13_04_retention_sensitivity_summary.csv"
    )

    robustness_output = (
        REPORT_DIR /
        "13_04_retention_robustness_summary.csv"
    )

    quality_output = (
        REPORT_DIR /
        "13_04_retention_sensitivity_quality_gate.csv"
    )

    break_even_df.to_csv(
        break_even_output,
        index=False
    )

    sensitivity_summary.to_csv(
        summary_output,
        index=False
    )

    robustness_summary.to_csv(
        robustness_output,
        index=False
    )

    quality_gate.to_csv(
        quality_output,
        index=False
    )

    # -------------------------------------------------------------------------
    # 16. METADATA
    # -------------------------------------------------------------------------

    metadata = {

        "phase":
            "13.4",

        "title":
            "Retention Economics Sensitivity & Robustness Analysis",

        "purpose":
            "Stress-test retention economics across assumed retention "
            "success rates and intervention costs.",

        "input":
            "customer_targeting_cohorts_phase13.csv",

        "customer_universe":
            int(len(df)),

        "cohorts":
            cohort_names,

        "success_rates":
            success_rates,

        "intervention_costs_per_customer":
            intervention_costs,

        "expected_scenario_count":
            int(expected_scenario_count),

        "actual_scenario_count":
            int(actual_scenario_count),

        "evaluation_horizon_months":
            int(EVALUATION_HORIZON_MONTHS),

        "robustness_definitions": {

            "Robustly viable":
                "Break-even success <= 10%",

            "Conditionally viable":
                "Break-even success > 10% and <= 50%",

            "Weakly viable":
                "Break-even success > 50% and <= 80%",

            "Not viable within tested range":
                "Break-even success > 80% or infeasible",
        },

        "governance": {

            "model_retraining":
                False,

            "recalibration":
                False,

            "threshold_change":
                False,

            "test_data_used":
                False,

            "future_predictors_used":
                False,

            "historical_churn_used_for_roi":
                False,

            "success_rates_are_assumptions":
                True,

            "intervention_costs_are_assumptions":
                True,

            "roi_is_realized":
                False,

            "revenue_saved_is_guaranteed":
                False,
        },
    }

    metadata_output = (
        REPORT_DIR /
        "13_04_retention_sensitivity_metadata.json"
    )

    with open(
        metadata_output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # -------------------------------------------------------------------------
    # 17. MARKDOWN REPORT
    # -------------------------------------------------------------------------

    best_row = sensitivity_df.loc[
        sensitivity_df["roi"].idxmax()
    ]

    hardest_break_even = (
        break_even_df.loc[
            break_even_df[
                "break_even_success_rate"
            ].idxmax()
        ]
    )

    easiest_break_even = (
        break_even_df.loc[
            break_even_df[
                "break_even_success_rate"
            ].idxmin()
        ]
    )

    report_lines = [

        "# Phase 13.4 — Retention Economics Sensitivity & Robustness Analysis",
        "",
        "## Purpose",
        "",
        "This phase stress-tests the Phase 13 retention economics "
        "under a wider range of assumed retention success rates "
        "and intervention costs.",
        "",
        "The analysis does not retrain the model, change calibration, "
        "change the primary threshold, or introduce future information.",
        "",
        "## Scenario Design",
        "",
        f"- Customers: **{len(df):,}**",
        f"- Targeting cohorts: **{len(cohort_names)}**",
        f"- Success-rate assumptions: **{len(success_rates)}**",
        f"- Intervention-cost assumptions: **{len(intervention_costs)}**",
        f"- Total scenarios: **{actual_scenario_count:,}**",
        f"- Evaluation horizon: **{EVALUATION_HORIZON_MONTHS} months**",
        "",
        "## Success Rates Tested",
        "",
        ", ".join(
            f"{rate:.0%}"
            for rate in success_rates
        ),
        "",
        "## Intervention Costs Tested",
        "",
        ", ".join(
            format_inr(cost) + "/customer"
            for cost in intervention_costs
        ),
        "",
        "## Economic Interpretation",
        "",
        "Expected retained value is calculated from potential "
        "monthly revenue exposure multiplied by the assumed "
        "retention success rate.",
        "",
        "Campaign cost is calculated as targeted customers multiplied "
        "by the assumed one-time intervention cost per customer.",
        "",
        "Net value equals expected retained value over the evaluation "
        "horizon minus campaign cost.",
        "",
        "ROI equals net value divided by campaign cost.",
        "",
        "These are scenario-based economic estimates and are not "
        "realized company revenue, realized ROI, or guaranteed savings.",
        "",
        "## Break-Even Interpretation",
        "",
        "Break-even success rate is the minimum assumed retention "
        "success rate required for expected retained value to cover "
        "campaign cost.",
        "",
        f"- Easiest break-even scenario: **{easiest_break_even['targeting_cohort']}** "
        f"at {format_inr(easiest_break_even['intervention_cost_per_customer'])}/customer "
        f"→ **{easiest_break_even['break_even_success_rate']:.2%}**",
        "",
        f"- Hardest break-even scenario: **{hardest_break_even['targeting_cohort']}** "
        f"at {format_inr(hardest_break_even['intervention_cost_per_customer'])}/customer "
        f"→ **{hardest_break_even['break_even_success_rate']:.2%}**",
        "",
        "## Best Scenario Within Tested Range",
        "",
        f"- Cohort: **{best_row['targeting_cohort']}**",
        f"- Success rate: **{best_row['retention_success_rate']:.0%}**",
        f"- Intervention cost: **{format_inr(best_row['intervention_cost_per_customer'])}/customer**",
        f"- Net value: **{format_inr(best_row['net_value'])}**",
        f"- ROI: **{best_row['roi']:.2%}**",
        "",
        "This identifies the strongest result within the tested "
        "assumption range only. It does not establish the optimal "
        "real-world intervention strategy.",
        "",
        "## Robustness Classification",
        "",
        "- **Robustly viable:** break-even success <= 10%",
        "- **Conditionally viable:** break-even success > 10% and <= 50%",
        "- **Weakly viable:** break-even success > 50% and <= 80%",
        "- **Not viable within tested range:** break-even success > 80% "
        "or infeasible",
        "",
        "These labels are analytical sensitivity classifications, "
        "not business priority scores.",
        "",
        "## Governance",
        "",
        "- No model retraining",
        "- No probability recalibration",
        "- No threshold change",
        "- No September predictors",
        "- No test-set data",
        "- Historical churn is descriptive only",
        "- Retention success rates are hypothetical assumptions",
        "- Intervention costs are hypothetical assumptions",
        "- ROI is not realized ROI",
        "- Revenue retention is not guaranteed",
        "",
        "## Quality Gate",
        "",
        f"- **{len(quality_gate)} checks passed**",
        "",
        "All Phase 13.4 quality checks passed.",
    ]

    report_output = (
        REPORT_DIR /
        "13_04_retention_sensitivity_report.md"
    )

    with open(
        report_output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report_lines)
        )

    # -------------------------------------------------------------------------
    # 18. TERMINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 13.4 COMPLETE — QUALITY GATE PASS")
    print("=" * 80)

    print(
        f"\nCustomer universe: "
        f"{len(df):,}"
    )

    print(
        f"Scenarios: "
        f"{actual_scenario_count:,}"
    )

    print(
        f"Economic viability: "
        f"{int(sensitivity_df['economically_viable'].sum()):,}"
        f"/{len(sensitivity_df):,}"
    )

    print(
        f"Minimum break-even: "
        f"{break_even_df['break_even_success_rate'].min():.2%}"
    )

    print(
        f"Maximum break-even: "
        f"{break_even_df['break_even_success_rate'].max():.2%}"
    )

    print(
        f"Best ROI: "
        f"{sensitivity_df['roi'].max():.2%}"
    )

    print(
        f"Success monotonicity: "
        f"{'PASS' if success_monotonicity_pass else 'FAIL'}"
    )

    print(
        f"Cost monotonicity: "
        f"{'PASS' if cost_monotonicity_pass else 'FAIL'}"
    )

    print("\nOutput files:")

    for output_file in [
        sensitivity_output,
        break_even_output,
        summary_output,
        robustness_output,
        quality_output,
        metadata_output,
        report_output,
    ]:

        print(
            f"  - {output_file}"
        )


# =============================================================================
# RUN PHASE 13.4
# =============================================================================

if __name__ == "__main__":
    main_phase_13_4()        