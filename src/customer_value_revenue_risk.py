"""
ChurnIQ — Phase 12
Customer Value & Revenue Risk

Purpose
-------
Connect the LOCKED Phase 11 calibrated churn probabilities with
August customer-value economics to quantify scenario-based
potential revenue exposure.

Core framework
--------------
Calibrated Churn Risk
        ↓
August Customer Value Proxy
        ↓
Risk × Value
        ↓
Potential Revenue Exposure
        ↓
Exposure Concentration

Governance
----------
- Phase 11 calibrated probabilities are consumed as locked inputs.
- August ARPU (arpu_8) is the customer-value proxy.
- Raw ARPU is preserved.
- Negative ARPU is clipped to zero ONLY for exposure calculations.
- September predictors are not used.
- No retraining.
- No tuning.
- No recalibration.
- No feature selection.
- No threshold modification.
- No retention actions.
- No retention-cost assumptions.
- No ROI.
- No revenue-saved claims.

Revenue exposure is scenario-based potential exposure,
not guaranteed revenue loss.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_risk_intelligence.csv"
)

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = (
    OUTPUT_DIR / "customer_value_revenue_risk.csv"
)

TOP_EXPOSURE_FILE = (
    OUTPUT_DIR / "top_100_revenue_exposure_customers_phase12.csv"
)

VALUE_SUMMARY_FILE = (
    REPORT_DIR / "12_customer_value_summary.csv"
)

RISK_SUMMARY_FILE = (
    REPORT_DIR / "12_revenue_risk_by_risk_level.csv"
)

VALUE_TIER_FILE = (
    REPORT_DIR / "12_revenue_risk_by_value_tier.csv"
)

RISK_VALUE_MATRIX_FILE = (
    REPORT_DIR / "12_risk_value_matrix.csv"
)

CONCENTRATION_FILE = (
    REPORT_DIR / "12_revenue_risk_concentration.csv"
)

QUALITY_GATE_FILE = (
    REPORT_DIR / "12_customer_value_revenue_risk_quality_gate.csv"
)

METADATA_FILE = (
    REPORT_DIR / "12_customer_value_revenue_risk_metadata.json"
)

REPORT_FILE = (
    REPORT_DIR / "12_customer_value_revenue_risk_report.md"
)


# ============================================================
# 2. GOVERNANCE CONSTANTS
# ============================================================

ANNUALIZATION_MONTHS = 12

EXPECTED_PRIMARY_THRESHOLD = 0.10

EXPECTED_RISK_LEVELS = {
    "Below Primary Threshold",
    "High",
    "Very High",
}


# ============================================================
# 3. HELPERS
# ============================================================

def ensure_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def fail_if(condition, message):
    if condition:
        raise ValueError(message)


def fmt_money(value):
    if pd.isna(value):
        return "NA"

    return f"₹{value:,.2f}"


def pct(value):
    if pd.isna(value):
        return "NA"

    return f"{value:.2f}%"


# ============================================================
# 4. LOAD LOCKED INPUTS
# ============================================================

def load_inputs():

    print("=" * 80)
    print("PHASE 12 — CUSTOMER VALUE & REVENUE RISK")
    print("=" * 80)

    fail_if(
        not RISK_FILE.exists(),
        f"Missing Phase 11 risk file: {RISK_FILE}"
    )

    fail_if(
        not RAW_FILE.exists(),
        f"Missing raw training file: {RAW_FILE}"
    )

    risk = pd.read_csv(RISK_FILE)

    raw = pd.read_csv(
        RAW_FILE,
        usecols=[
            "id",
            "arpu_8",
        ],
    )

    print(
        f"Phase 11 risk input shape: {risk.shape}"
    )

    print(
        f"Raw customer-value input shape: {raw.shape}"
    )

    return risk, raw


# ============================================================
# 5. INPUT VALIDATION
# ============================================================

def validate_inputs(risk, raw):

    required_risk_columns = {
        "id",
        "calibrated_churn_probability",
        "risk_score",
        "risk_level",
        "primary_threshold",
    }

    required_raw_columns = {
        "id",
        "arpu_8",
    }

    fail_if(
        not required_risk_columns.issubset(
            risk.columns
        ),
        "Required Phase 11 columns are missing."
    )

    fail_if(
        not required_raw_columns.issubset(
            raw.columns
        ),
        "Required raw customer-value columns are missing."
    )

    # --------------------------------------------------------
    # ID integrity
    # --------------------------------------------------------

    fail_if(
        risk["id"].duplicated().any(),
        "Duplicate customer IDs found in Phase 11 risk data."
    )

    fail_if(
        raw["id"].duplicated().any(),
        "Duplicate customer IDs found in raw data."
    )

    risk_ids = set(risk["id"])
    raw_ids = set(raw["id"])

    fail_if(
        risk_ids != raw_ids,
        "Phase 11 and raw customer universes do not match."
    )

    # --------------------------------------------------------
    # Probability integrity
    # --------------------------------------------------------

    probabilities = risk[
        "calibrated_churn_probability"
    ]

    fail_if(
        probabilities.isna().any(),
        "Missing calibrated churn probabilities found."
    )

    fail_if(
        (~probabilities.between(0, 1)).any(),
        "Calibrated probabilities outside [0,1]."
    )

    # --------------------------------------------------------
    # ARPU integrity
    # --------------------------------------------------------

    fail_if(
        raw["arpu_8"].isna().any(),
        "Missing August ARPU found."
    )

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    actual_levels = set(
        risk["risk_level"].dropna().unique()
    )

    fail_if(
        actual_levels != EXPECTED_RISK_LEVELS,
        (
            "Unexpected Phase 11 risk levels. "
            f"Found: {actual_levels}"
        )
    )

    # --------------------------------------------------------
    # Threshold governance
    # --------------------------------------------------------

    threshold_values = (
        risk["primary_threshold"]
        .dropna()
        .unique()
    )

    fail_if(
        len(threshold_values) != 1,
        "Multiple primary thresholds found."
    )

    actual_threshold = float(
        threshold_values[0]
    )

    fail_if(
        not np.isclose(
            actual_threshold,
            EXPECTED_PRIMARY_THRESHOLD,
        ),
        (
            "Phase 11 primary threshold changed. "
            f"Expected {EXPECTED_PRIMARY_THRESHOLD}, "
            f"found {actual_threshold}."
        )
    )

    print("\nInput validation: PASS")


# ============================================================
# 6. BUILD CUSTOMER VALUE + REVENUE RISK DATASET
# ============================================================

def build_dataset(risk, raw):

    df = risk.merge(
        raw,
        on="id",
        how="inner",
        validate="one_to_one",
    )

    fail_if(
        len(df) != len(risk),
        "Customer population changed during join."
    )

    # --------------------------------------------------------
    # Customer value proxy
    # --------------------------------------------------------

    df["customer_value_proxy_arpu_8"] = (
        df["arpu_8"]
    )

    # --------------------------------------------------------
    # Exposure-eligible value
    #
    # Raw ARPU remains untouched.
    # Negative values become zero ONLY here.
    # --------------------------------------------------------

    df["exposure_eligible_arpu_8"] = (
        df["arpu_8"].clip(lower=0)
    )

    # --------------------------------------------------------
    # Customer value percentile
    # --------------------------------------------------------

    df["customer_value_percentile"] = (
        df["exposure_eligible_arpu_8"]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    # --------------------------------------------------------
    # Customer value tiers
    #
    # Descriptive tiers only.
    # They are NOT retention recommendations.
    # --------------------------------------------------------

    df["customer_value_tier"] = pd.cut(
        df["customer_value_percentile"],
        bins=[
            -np.inf,
            25,
            50,
            75,
            np.inf,
        ],
        labels=[
            "Lower Value",
            "Mid-Lower Value",
            "Mid-Upper Value",
            "Higher Value",
        ],
        include_lowest=True,
    )

    # --------------------------------------------------------
    # Potential monthly revenue exposure
    # --------------------------------------------------------

    df["potential_monthly_revenue_exposure"] = (
        df["calibrated_churn_probability"]
        * df["exposure_eligible_arpu_8"]
    )

    # --------------------------------------------------------
    # Annualized scenario exposure
    # --------------------------------------------------------

    df["annualized_scenario_exposure"] = (
        df["potential_monthly_revenue_exposure"]
        * ANNUALIZATION_MONTHS
    )

    # --------------------------------------------------------
    # Customer contribution to total exposure
    # --------------------------------------------------------

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    if total_exposure > 0:

        df["customer_exposure_share_pct"] = (
            df["potential_monthly_revenue_exposure"]
            / total_exposure
            * 100
        )

    else:

        df["customer_exposure_share_pct"] = 0.0

    # --------------------------------------------------------
    # Revenue exposure percentile
    # --------------------------------------------------------

    df["revenue_exposure_percentile"] = (
        df["potential_monthly_revenue_exposure"]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    # --------------------------------------------------------
    # Deterministic exposure ranking
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "potential_monthly_revenue_exposure",
            "calibrated_churn_probability",
            "exposure_eligible_arpu_8",
            "id",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    df["revenue_exposure_rank"] = (
        np.arange(len(df)) + 1
    )

    return df


# ============================================================
# 7. CUSTOMER VALUE SUMMARY
# ============================================================

def create_value_summary(df):

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "Customer Count",
                "value": len(df),
            },
            {
                "metric": "Customers With August ARPU",
                "value": df["arpu_8"].notna().sum(),
            },
            {
                "metric": "Negative August ARPU Customers",
                "value": (df["arpu_8"] < 0).sum(),
            },
            {
                "metric": "Exposure Eligible ARPU Customers",
                "value": (
                    df["exposure_eligible_arpu_8"] > 0
                ).sum(),
            },
            {
                "metric": "Minimum August ARPU",
                "value": df["arpu_8"].min(),
            },
            {
                "metric": "Maximum August ARPU",
                "value": df["arpu_8"].max(),
            },
            {
                "metric": "Average August ARPU",
                "value": df["arpu_8"].mean(),
            },
            {
                "metric": "Median August ARPU",
                "value": df["arpu_8"].median(),
            },
            {
                "metric": "Total Exposure-Eligible ARPU",
                "value": (
                    df["exposure_eligible_arpu_8"].sum()
                ),
            },
            {
                "metric": "Total Potential Monthly Exposure",
                "value": total_exposure,
            },
            {
                "metric": "Total Annualized Scenario Exposure",
                "value": (
                    df["annualized_scenario_exposure"]
                    .sum()
                ),
            },
        ]
    )

    summary.to_csv(
        VALUE_SUMMARY_FILE,
        index=False,
    )

    return summary


# ============================================================
# 8. REVENUE RISK BY RISK LEVEL
# ============================================================

def create_risk_summary(df):

    result = (
        df.groupby(
            "risk_level",
            observed=True,
        )
        .agg(
            customers=("id", "count"),
            customer_share_pct=("id", "count"),
            avg_churn_probability=(
                "calibrated_churn_probability",
                "mean",
            ),
            avg_arpu=("arpu_8", "mean"),
            total_exposure_eligible_arpu=(
                "exposure_eligible_arpu_8",
                "sum",
            ),
            potential_monthly_revenue_exposure=(
                "potential_monthly_revenue_exposure",
                "sum",
            ),
            annualized_scenario_exposure=(
                "annualized_scenario_exposure",
                "sum",
            ),
        )
        .reset_index()
    )

    total_customers = len(df)

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    result["customer_share_pct"] = (
        result["customers"]
        / total_customers
        * 100
    )

    result["exposure_share_pct"] = np.where(
        total_exposure > 0,
        result[
            "potential_monthly_revenue_exposure"
        ]
        / total_exposure
        * 100,
        0,
    )

    # Difference between customer share and
    # exposure share shows concentration.
    result["exposure_minus_customer_share_pp"] = (
        result["exposure_share_pct"]
        - result["customer_share_pct"]
    )

    risk_order = {
        "Below Primary Threshold": 0,
        "High": 1,
        "Very High": 2,
    }

    result["_order"] = (
        result["risk_level"]
        .map(risk_order)
    )

    result = (
        result
        .sort_values("_order")
        .drop(columns="_order")
    )

    result.to_csv(
        RISK_SUMMARY_FILE,
        index=False,
    )

    return result


# ============================================================
# 9. REVENUE RISK BY CUSTOMER VALUE TIER
# ============================================================

def create_value_tier_summary(df):

    result = (
        df.groupby(
            "customer_value_tier",
            observed=True,
        )
        .agg(
            customers=("id", "count"),
            avg_arpu=("arpu_8", "mean"),
            avg_churn_probability=(
                "calibrated_churn_probability",
                "mean",
            ),
            potential_monthly_revenue_exposure=(
                "potential_monthly_revenue_exposure",
                "sum",
            ),
            annualized_scenario_exposure=(
                "annualized_scenario_exposure",
                "sum",
            ),
        )
        .reset_index()
    )

    total_customers = len(df)

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    result["customer_share_pct"] = (
        result["customers"]
        / total_customers
        * 100
    )

    result["exposure_share_pct"] = np.where(
        total_exposure > 0,
        result[
            "potential_monthly_revenue_exposure"
        ]
        / total_exposure
        * 100,
        0,
    )

    result.to_csv(
        VALUE_TIER_FILE,
        index=False,
    )

    return result


# ============================================================
# 10. RISK × VALUE MATRIX
# ============================================================

def create_risk_value_matrix(df):

    matrix = (
        df.groupby(
            [
                "risk_level",
                "customer_value_tier",
            ],
            observed=True,
        )
        .agg(
            customers=("id", "count"),
            avg_churn_probability=(
                "calibrated_churn_probability",
                "mean",
            ),
            avg_arpu=("arpu_8", "mean"),
            potential_monthly_revenue_exposure=(
                "potential_monthly_revenue_exposure",
                "sum",
            ),
            annualized_scenario_exposure=(
                "annualized_scenario_exposure",
                "sum",
            ),
        )
        .reset_index()
    )

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    matrix["exposure_share_pct"] = np.where(
        total_exposure > 0,
        matrix[
            "potential_monthly_revenue_exposure"
        ]
        / total_exposure
        * 100,
        0,
    )

    risk_order = {
        "Below Primary Threshold": 0,
        "High": 1,
        "Very High": 2,
    }

    value_order = {
        "Lower Value": 0,
        "Mid-Lower Value": 1,
        "Mid-Upper Value": 2,
        "Higher Value": 3,
    }

    matrix["_risk_order"] = (
        matrix["risk_level"]
        .map(risk_order)
    )

    matrix["_value_order"] = (
        matrix["customer_value_tier"]
        .astype(str)
        .map(value_order)
    )

    matrix = (
        matrix
        .sort_values(
            [
                "_risk_order",
                "_value_order",
            ]
        )
        .drop(
            columns=[
                "_risk_order",
                "_value_order",
            ]
        )
    )

    matrix.to_csv(
        RISK_VALUE_MATRIX_FILE,
        index=False,
    )

    return matrix


# ============================================================
# 11. EXPOSURE CONCENTRATION
# ============================================================

def create_concentration_analysis(df):

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    records = []

    for customer_pct in [
        1,
        5,
        10,
        20,
        50,
    ]:

        customer_count = max(
            1,
            int(
                np.ceil(
                    len(df)
                    * customer_pct
                    / 100
                )
            ),
        )

        top_exposure = (
            df.head(customer_count)
            [
                "potential_monthly_revenue_exposure"
            ]
            .sum()
        )

        exposure_share = (
            top_exposure
            / total_exposure
            * 100
            if total_exposure > 0
            else 0
        )

        records.append(
            {
                "top_customer_percent": customer_pct,
                "customer_count": customer_count,
                "monthly_exposure": top_exposure,
                "exposure_share_pct": exposure_share,
            }
        )

    concentration = pd.DataFrame(
        records
    )

    concentration.to_csv(
        CONCENTRATION_FILE,
        index=False,
    )

    return concentration


# ============================================================
# 12. TOP REVENUE-EXPOSURE CUSTOMERS
# ============================================================

def create_top_exposure(df):

    columns = [
        "revenue_exposure_rank",
        "id",
        "calibrated_churn_probability",
        "risk_score",
        "risk_level",
        "arpu_8",
        "exposure_eligible_arpu_8",
        "customer_value_percentile",
        "customer_value_tier",
        "potential_monthly_revenue_exposure",
        "annualized_scenario_exposure",
        "customer_exposure_share_pct",
        "revenue_exposure_percentile",
    ]

    top_100 = (
        df[columns]
        .head(100)
        .copy()
    )

    top_100.to_csv(
        TOP_EXPOSURE_FILE,
        index=False,
    )

    return top_100


# ============================================================
# 13. QUALITY GATE
# ============================================================

def create_quality_gate(df):

    expected_monthly_exposure = (
        df["calibrated_churn_probability"]
        * df["exposure_eligible_arpu_8"]
    )

    total_exposure = (
        df["potential_monthly_revenue_exposure"]
        .sum()
    )

    checks = []

    def add_check(
        name,
        result,
        detail,
    ):

        checks.append(
            {
                "check": name,
                "status": (
                    "PASS"
                    if result
                    else "FAIL"
                ),
                "detail": detail,
            }
        )

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    add_check(
        "Customer universe",
        len(df) == len(
            pd.read_csv(
                RISK_FILE,
                usecols=["id"],
            )
        ),
        f"{len(df):,} customers",
    )

    add_check(
        "Unique customer IDs",
        df["id"].nunique()
        == len(df),
        f"{df['id'].nunique():,} unique IDs",
    )

    add_check(
        "No missing customer IDs",
        df["id"].notna().all(),
        "0 missing IDs",
    )

    # --------------------------------------------------------
    # Risk integrity
    # --------------------------------------------------------

    add_check(
        "Calibrated probabilities present",
        df[
            "calibrated_churn_probability"
        ].notna().all(),
        "0 missing probabilities",
    )

    add_check(
        "Calibrated probabilities valid",
        df[
            "calibrated_churn_probability"
        ].between(0, 1).all(),
        "All probabilities within [0,1]",
    )

    add_check(
        "Phase 11 risk levels preserved",
        set(
            df["risk_level"].unique()
        ) == EXPECTED_RISK_LEVELS,
        "Locked Phase 11 risk levels preserved",
    )

    add_check(
        "Primary threshold preserved",
        np.allclose(
            df["primary_threshold"],
            EXPECTED_PRIMARY_THRESHOLD,
        ),
        "Primary threshold = 0.10",
    )

    # --------------------------------------------------------
    # ARPU integrity
    # --------------------------------------------------------

    add_check(
        "August ARPU complete",
        df["arpu_8"].notna().all(),
        "0 missing August ARPU values",
    )

    add_check(
        "Raw negative ARPU preserved",
        (
            (df["arpu_8"] < 0)
            ==
            (
                df[
                    "exposure_eligible_arpu_8"
                ] == 0
            )
        ).all(),
        "Negative raw ARPU retained; clipped only for exposure",
    )

    # --------------------------------------------------------
    # Formula checks
    # --------------------------------------------------------

    add_check(
        "Monthly exposure formula",
        np.allclose(
            df[
                "potential_monthly_revenue_exposure"
            ],
            expected_monthly_exposure,
            rtol=1e-10,
            atol=1e-10,
        ),
        "Probability × max(ARPU, 0)",
    )

    add_check(
        "Annualized exposure formula",
        np.allclose(
            df[
                "annualized_scenario_exposure"
            ],
            df[
                "potential_monthly_revenue_exposure"
            ]
            * ANNUALIZATION_MONTHS,
            rtol=1e-10,
            atol=1e-10,
        ),
        "Monthly exposure × 12",
    )

    # --------------------------------------------------------
    # Exposure validity
    # --------------------------------------------------------

    add_check(
        "No negative revenue exposure",
        (
            df[
                "potential_monthly_revenue_exposure"
            ] >= 0
        ).all(),
        "All exposure values >= 0",
    )

    add_check(
        "Exposure shares valid",
        (
            df[
                "customer_exposure_share_pct"
            ] >= 0
        ).all()
        and
        (
            df[
                "customer_exposure_share_pct"
            ] <= 100
        ).all(),
        "All customer exposure shares within [0,100]",
    )

    if total_exposure > 0:

        share_sum_valid = np.isclose(
            df[
                "customer_exposure_share_pct"
            ].sum(),
            100,
            atol=1e-8,
        )

    else:

        share_sum_valid = (
            df[
                "customer_exposure_share_pct"
            ].sum()
            == 0
        )

    add_check(
        "Customer exposure shares sum correctly",
        share_sum_valid,
        "Exposure contribution totals 100%",
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    add_check(
        "Revenue exposure ranking",
        (
            df[
                "revenue_exposure_rank"
            ].is_unique
            and
            df[
                "revenue_exposure_rank"
            ].min() == 1
            and
            df[
                "revenue_exposure_rank"
            ].max() == len(df)
        ),
        f"Ranks 1 through {len(df):,}",
    )

    add_check(
        "Exposure sorted descending",
        df[
            "potential_monthly_revenue_exposure"
        ].is_monotonic_decreasing,
        "Exposure ranking is deterministic",
    )

    # --------------------------------------------------------
    # Output completeness
    # --------------------------------------------------------

    required_output_columns = {
        "id",
        "calibrated_churn_probability",
        "risk_level",
        "arpu_8",
        "customer_value_tier",
        "potential_monthly_revenue_exposure",
        "annualized_scenario_exposure",
        "customer_exposure_share_pct",
        "revenue_exposure_rank",
    }

    add_check(
        "Output schema",
        required_output_columns.issubset(
            df.columns
        ),
        "Required Phase 12 fields present",
    )

    quality_gate = pd.DataFrame(
        checks
    )

    quality_gate.to_csv(
        QUALITY_GATE_FILE,
        index=False,
    )

    overall_pass = (
        quality_gate["status"]
        == "PASS"
    ).all()

    return quality_gate, overall_pass


# ============================================================
# 14. METADATA
# ============================================================

def create_metadata(
    df,
    quality_gate,
    overall_pass,
):

    metadata = {

        "phase": "12",

        "phase_name":
            "Customer Value & Revenue Risk",

        "status":
            "PASS"
            if overall_pass
            else "FAIL",

        "customer_universe":
            int(len(df)),

        "risk_source":
            "data/processed/customer_risk_intelligence.csv",

        "customer_value_source":
            "data/raw/train.csv",

        "customer_value_proxy":
            "August ARPU (arpu_8)",

        "monthly_exposure_formula":
            (
                "calibrated_churn_probability "
                "× max(arpu_8, 0)"
            ),

        "annualized_exposure_formula":
            (
                "potential_monthly_revenue_exposure "
                "× 12"
            ),

        "negative_arpu_treatment":
            (
                "Raw arpu_8 preserved. "
                "Negative values clipped to zero "
                "only for exposure calculations."
            ),

        "primary_threshold":
            EXPECTED_PRIMARY_THRESHOLD,

        "risk_levels":
            sorted(
                EXPECTED_RISK_LEVELS
            ),

        "revenue_exposure_definition":
            (
                "Scenario-based potential "
                "revenue exposure; not guaranteed "
                "revenue loss."
            ),

        "september_data_used":
            False,

        "model_retraining":
            False,

        "model_tuning":
            False,

        "recalibration":
            False,

        "feature_selection":
            False,

        "threshold_changed":
            False,

        "retention_actions_included":
            False,

        "retention_cost_assumptions":
            False,

        "roi_calculation":
            False,

        "quality_gate":
            {
                "checks":
                    int(len(quality_gate)),

                "passed":
                    int(
                        (
                            quality_gate[
                                "status"
                            ]
                            == "PASS"
                        ).sum()
                    ),

                "failed":
                    int(
                        (
                            quality_gate[
                                "status"
                            ]
                            == "FAIL"
                        ).sum()
                    ),
            },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    return metadata


# ============================================================
# 15. BUSINESS REPORT
# ============================================================

def create_report(
    df,
    value_summary,
    risk_summary,
    value_tier_summary,
    risk_value_matrix,
    concentration,
    overall_pass,
):

    total_monthly = (
        df[
            "potential_monthly_revenue_exposure"
        ].sum()
    )

    total_annualized = (
        df[
            "annualized_scenario_exposure"
        ].sum()
    )

    high_risk = df[
        df["risk_level"].isin(
            [
                "High",
                "Very High",
            ]
        )
    ]

    high_risk_exposure = (
        high_risk[
            "potential_monthly_revenue_exposure"
        ].sum()
    )

    very_high = df[
        df["risk_level"]
        == "Very High"
    ]

    very_high_exposure = (
        very_high[
            "potential_monthly_revenue_exposure"
        ].sum()
    )

    very_high_exposure_share = (
        very_high_exposure
        / total_monthly
        * 100
        if total_monthly > 0
        else 0
    )

    very_high_customer_share = (
        len(very_high)
        / len(df)
        * 100
    )

    top_10_share = concentration.loc[
        concentration[
            "top_customer_percent"
        ] == 10,
        "exposure_share_pct",
    ].iloc[0]

    report = f"""
# Phase 12 — Customer Value & Revenue Risk

## Status

**{"PASS" if overall_pass else "FAIL"}**

## Purpose

Phase 12 connects the locked Phase 11 calibrated churn probabilities
with August customer-value economics to quantify scenario-based
potential revenue exposure.

The purpose is to understand **where customer-value exposure is
concentrated**, not to prescribe retention actions.

---

## Customer Value

August ARPU (`arpu_8`) is used as the customer-value proxy.

- Customer universe: **{len(df):,}**
- Average August ARPU: **{fmt_money(df["arpu_8"].mean())}**
- Median August ARPU: **{fmt_money(df["arpu_8"].median())}**
- Minimum August ARPU: **{fmt_money(df["arpu_8"].min())}**
- Maximum August ARPU: **{fmt_money(df["arpu_8"].max())}**
- Negative August ARPU customers: **{(df["arpu_8"] < 0).sum():,}**

Raw ARPU is preserved.

For revenue exposure calculations only, negative ARPU values are
clipped to zero.

---

## Revenue Exposure

### Monthly Formula

`Potential Monthly Revenue Exposure =
Calibrated Churn Probability × max(August ARPU, 0)`

### Annualized Formula

`Annualized Scenario Exposure =
Potential Monthly Revenue Exposure × 12`

### Overall Results

- Total potential monthly exposure: **{fmt_money(total_monthly)}**
- Annualized scenario exposure: **{fmt_money(total_annualized)}**
- High + Very High monthly exposure: **{fmt_money(high_risk_exposure)}**
- Very High monthly exposure: **{fmt_money(very_high_exposure)}**

Very High-risk customers represent **{very_high_customer_share:.2f}%**
of the customer universe and account for **{very_high_exposure_share:.2f}%**
of potential monthly exposure.

This comparison describes exposure concentration; it does not establish
that the customers will churn or that the exposure will become realized
revenue loss.

---

## Revenue Risk by Risk Level

{risk_summary.to_markdown(index=False)}

---

## Revenue Risk by Customer Value Tier

{value_tier_summary.to_markdown(index=False)}

---

## Risk × Value Matrix

{risk_value_matrix.to_markdown(index=False)}

---

## Exposure Concentration

{concentration.to_markdown(index=False)}

The top 10% of customers by potential revenue exposure account for
approximately **{top_10_share:.2f}%** of total potential monthly exposure.

---

## Governance

- Phase 11 calibrated churn probabilities used as locked inputs.
- Primary threshold retained at **0.10**.
- August ARPU used as the customer-value proxy.
- Raw negative ARPU values preserved.
- Negative ARPU clipped to zero only for exposure calculations.
- No September predictors used.
- No model retraining.
- No model tuning.
- No recalibration.
- No feature selection.
- No threshold modification.
- No retention actions.
- No retention-cost assumptions.
- No ROI calculations.

---

## Interpretation Boundary

Potential revenue exposure is a **scenario-based financial exposure
measure** calculated from model-estimated churn probability and August
customer-value proxy.

It is **not**:

- guaranteed revenue loss,
- a forecast of booked revenue,
- guaranteed future churn,
- guaranteed revenue that can be saved,
- or a retention recommendation.

---

## Phase 13 Boundary

Retention prioritization, intervention costs, expected retention success,
potential revenue saved, net benefit, ROI, and sensitivity scenarios belong
to **Phase 13 — Retention Prioritization & ROI**.
"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    ensure_directories()

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        risk, raw = load_inputs()

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validate_inputs(
            risk,
            raw,
        )

        # ----------------------------------------------------
        # Build
        # ----------------------------------------------------

        df = build_dataset(
            risk,
            raw,
        )

        print(
            f"\nPhase 12 dataset shape: {df.shape}"
        )

        # ----------------------------------------------------
        # Analytical outputs
        # ----------------------------------------------------

        value_summary = (
            create_value_summary(df)
        )

        risk_summary = (
            create_risk_summary(df)
        )

        value_tier_summary = (
            create_value_tier_summary(df)
        )

        risk_value_matrix = (
            create_risk_value_matrix(df)
        )

        concentration = (
            create_concentration_analysis(df)
        )

        top_100 = (
            create_top_exposure(df)
        )

        # ----------------------------------------------------
        # Quality gate
        # ----------------------------------------------------

        quality_gate, overall_pass = (
            create_quality_gate(df)
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        create_metadata(
            df,
            quality_gate,
            overall_pass,
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        create_report(
            df,
            value_summary,
            risk_summary,
            value_tier_summary,
            risk_value_matrix,
            concentration,
            overall_pass,
        )

        # ----------------------------------------------------
        # Save final dataset
        # ----------------------------------------------------

        df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

               # ----------------------------------------------------
        # Terminal summary
        # ----------------------------------------------------

        total_monthly_exposure = (
            df["potential_monthly_revenue_exposure"].sum()
        )

        very_high = df[
            df["risk_level"] == "Very High"
        ]

        very_high_exposure = (
            very_high[
                "potential_monthly_revenue_exposure"
            ].sum()
        )

        very_high_exposure_share = (
            very_high_exposure
            / total_monthly_exposure
            * 100
            if total_monthly_exposure > 0
            else 0
        )

        top_10_share = concentration.loc[
            concentration[
                "top_customer_percent"
            ] == 10,
            "exposure_share_pct",
        ].iloc[0]

        print("\n" + "=" * 80)
        print("PHASE 12 SUMMARY")
        print("=" * 80)

        print(
            f"Customer universe: "
            f"{len(df):,}"
        )

        print(
            "Mean calibrated churn probability: "
            f"{df['calibrated_churn_probability'].mean():.4f}"
        )

        print(
            "Median calibrated churn probability: "
            f"{df['calibrated_churn_probability'].median():.4f}"
        )

        print(
            "Average August ARPU: "
            f"{fmt_money(df['arpu_8'].mean())}"
        )

        print(
            "Median August ARPU: "
            f"{fmt_money(df['arpu_8'].median())}"
        )

        print(
            "Negative ARPU customers: "
            f"{(df['arpu_8'] < 0).sum():,}"
        )

        print(
            "Potential monthly revenue exposure: "
            f"{fmt_money(total_monthly_exposure)}"
        )

        print(
            "Annualized scenario exposure: "
            f"{fmt_money(df['annualized_scenario_exposure'].sum())}"
        )

        print(
            "Very High-risk customers: "
            f"{len(very_high):,}"
        )

        print(
            "Very High-risk exposure share: "
            f"{very_high_exposure_share:.2f}%"
        )

        print(
            "Top 10% exposure share: "
            f"{top_10_share:.2f}%"
        )
        
        # ----------------------------------------------------
        # Outputs
        # ----------------------------------------------------

        print("\nOUTPUTS")

        print(
            OUTPUT_FILE
        )

        print(
            TOP_EXPOSURE_FILE
        )

        print(
            VALUE_SUMMARY_FILE
        )

        print(
            RISK_SUMMARY_FILE
        )

        print(
            VALUE_TIER_FILE
        )

        print(
            RISK_VALUE_MATRIX_FILE
        )

        print(
            CONCENTRATION_FILE
        )

        print(
            QUALITY_GATE_FILE
        )

        print(
            METADATA_FILE
        )

        print(
            REPORT_FILE
        )

        if not overall_pass:
            sys.exit(1)

    except Exception as exc:

        print(
            "\nPHASE 12 FAILED"
        )

        print(
            f"Error: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()