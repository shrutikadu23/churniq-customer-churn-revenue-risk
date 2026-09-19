"""
ChurnIQ — Customer Value & Revenue Risk

Phase 14.5
----------
Reusable customer-value layer for the Streamlit application.

Governance:
- Reuses the locked Phase 12 customer-value artifact.
- Does not retrain the model.
- Does not recalibrate probabilities.
- Does not change the primary threshold.
- Does not use September or test data.
- Does not calculate retention ROI.
- August ARPU is treated as a customer-value proxy.
- Revenue exposure is an exposure estimate, not guaranteed future loss.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEPLOYMENT_DATA_DIR = (
    PROJECT_ROOT / "deployment_data"
)

PHASE_12_FILE = (
    DEPLOYMENT_DATA_DIR
    / "customer_value.parquet"
)


# =============================================================================
# LOCKED EXPECTATIONS
# =============================================================================

EXPECTED_CUSTOMERS = 69_999

EXPOSURE_TOLERANCE = 1e-6

REQUIRED_COLUMNS = [
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
    "arpu_8",
    "customer_value_proxy_arpu_8",
    "exposure_eligible_arpu_8",
    "customer_value_percentile",
    "customer_value_tier",
    "potential_monthly_revenue_exposure",
    "annualized_scenario_exposure",
    "customer_exposure_share_pct",
    "revenue_exposure_percentile",
    "revenue_exposure_rank",
]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def validate_probability(
    values: pd.Series,
    name: str,
) -> None:

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric.isna().any():

        raise ValueError(
            f"{name} contains missing or non-numeric values."
        )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if (
        (numeric < 0).any()
        or (numeric > 1).any()
    ):

        raise ValueError(
            f"{name} contains values outside [0, 1]."
        )


def validate_non_negative(
    values: pd.Series,
    name: str,
) -> None:

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric.isna().any():

        raise ValueError(
            f"{name} contains missing or non-numeric values."
        )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():

        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if (numeric < 0).any():

        raise ValueError(
            f"{name} contains negative values."
        )


# =============================================================================
# LOAD LOCKED PHASE 12 DATA
# =============================================================================

@lru_cache(maxsize=1)
def load_customer_value_data() -> pd.DataFrame:
    """
    Load and validate the complete locked Phase 12 artifact.

    The result is cached so the CSV is read only once per process.
    """

    require_file(
        PHASE_12_FILE,
        "Phase 12 customer value and revenue risk artifact",
    )

    data = pd.read_parquet(
        PHASE_12_FILE,
        columns=REQUIRED_COLUMNS,
    )

    # -------------------------------------------------------------------------
    # Customer universe
    # -------------------------------------------------------------------------

    if len(data) != EXPECTED_CUSTOMERS:

        raise ValueError(
            f"Expected {EXPECTED_CUSTOMERS:,} customers, "
            f"found {len(data):,}."
        )

    if data["id"].isna().any():

        raise ValueError(
            "Phase 12 artifact contains missing customer IDs."
        )

    if data["id"].duplicated().any():

        raise ValueError(
            "Phase 12 artifact contains duplicate customer IDs."
        )

    # -------------------------------------------------------------------------
    # Probability
    # -------------------------------------------------------------------------

    validate_probability(
        data["calibrated_churn_probability"],
        "Calibrated churn probability",
    )

    # -------------------------------------------------------------------------
    # Raw August ARPU
    #
    # Raw ARPU is intentionally allowed to be negative because Phase 12
    # preserves the original value.
    # -------------------------------------------------------------------------

    data["arpu_8"] = pd.to_numeric(
        data["arpu_8"],
        errors="coerce",
    )

    if data["arpu_8"].isna().any():

        raise ValueError(
            "arpu_8 contains missing or non-numeric values."
        )

    if not np.isfinite(
        data["arpu_8"].to_numpy()
    ).all():

        raise ValueError(
            "arpu_8 contains NaN or infinite values."
        )

    # -------------------------------------------------------------------------
    # Exposure-eligible ARPU
    # -------------------------------------------------------------------------

    validate_non_negative(
        data["exposure_eligible_arpu_8"],
        "Exposure-eligible August ARPU",
    )

    # -------------------------------------------------------------------------
    # Monthly exposure
    # -------------------------------------------------------------------------

    validate_non_negative(
        data["potential_monthly_revenue_exposure"],
        "Potential monthly revenue exposure",
    )

    # -------------------------------------------------------------------------
    # Annualized exposure
    # -------------------------------------------------------------------------

    validate_non_negative(
        data["annualized_scenario_exposure"],
        "Annualized scenario exposure",
    )

    # -------------------------------------------------------------------------
    # Risk levels
    # -------------------------------------------------------------------------

    allowed_risk_levels = {
        "Below Primary Threshold",
        "High",
        "Very High",
    }

    actual_risk_levels = set(
        data["risk_level"]
        .dropna()
        .astype(str)
        .unique()
    )

    unexpected_levels = (
        actual_risk_levels
        - allowed_risk_levels
    )

    if unexpected_levels:

        raise ValueError(
            "Unexpected risk levels detected: "
            f"{sorted(unexpected_levels)}"
        )

    return data


# =============================================================================
# CUSTOMER VALUE LOOKUP
# =============================================================================

def get_customer_value(
    customer_id: int,
) -> dict:
    """
    Return the locked Phase 12 customer-value and revenue-risk
    context for one existing customer.
    """

    customer_id = int(customer_id)

    data = load_customer_value_data()

    customer = data.loc[
        data["id"] == customer_id
    ]

    if customer.empty:

        raise ValueError(
            f"Customer ID {customer_id} was not found "
            "in the ChurnIQ customer universe."
        )

    row = customer.iloc[0]

    # =========================================================================
    # CUSTOMER VALUE
    # =========================================================================

    raw_arpu = float(
        row["arpu_8"]
    )

    customer_value_proxy = float(
        row["customer_value_proxy_arpu_8"]
    )

    exposure_eligible_arpu = float(
        row["exposure_eligible_arpu_8"]
    )

    # =========================================================================
    # LOCKED RISK
    # =========================================================================

    calibrated_probability = float(
        row["calibrated_churn_probability"]
    )

    risk_score = float(
        row["risk_score"]
    )

    risk_level = str(
        row["risk_level"]
    )

    primary_threshold = float(
        row["primary_threshold"]
    )

    above_primary_threshold = bool(
        row["above_primary_threshold"]
    )

    # =========================================================================
    # LOCKED REVENUE EXPOSURE
    # =========================================================================

    monthly_exposure = float(
        row["potential_monthly_revenue_exposure"]
    )

    annualized_exposure = float(
        row["annualized_scenario_exposure"]
    )

    # =========================================================================
    # EXPOSURE FORMULA VALIDATION
    #
    # Phase 12:
    #
    # calibrated churn probability
    # ×
    # exposure-eligible August ARPU
    #
    # This is reconstructed only as a validation check.
    # The stored Phase 12 exposure remains authoritative.
    # =========================================================================

    expected_monthly_exposure = (
        calibrated_probability
        * exposure_eligible_arpu
    )

    exposure_difference = abs(
        monthly_exposure
        - expected_monthly_exposure
    )

    exposure_formula_valid = (
        exposure_difference
        <= EXPOSURE_TOLERANCE
    )

    # =========================================================================
    # RETURN
    # =========================================================================

    return {
        "customer_id":
            customer_id,

        "august_arpu":
            raw_arpu,

        "customer_value_proxy":
            customer_value_proxy,

        "exposure_eligible_arpu":
            exposure_eligible_arpu,

        "calibrated_churn_probability":
            calibrated_probability,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "primary_threshold":
            primary_threshold,

        "above_primary_threshold":
            above_primary_threshold,

        "customer_value_percentile":
            float(
                row["customer_value_percentile"]
            ),

        "customer_value_tier":
            str(
                row["customer_value_tier"]
            ),

        "potential_monthly_revenue_exposure":
            monthly_exposure,

        "annualized_revenue_exposure":
            annualized_exposure,

        "customer_exposure_share_pct":
            float(
                row["customer_exposure_share_pct"]
            ),

        "revenue_exposure_percentile":
            float(
                row["revenue_exposure_percentile"]
            ),

        "revenue_exposure_rank":
            int(
                row["revenue_exposure_rank"]
            ),

        "exposure_validation_difference":
            exposure_difference,

        "exposure_formula_valid":
            exposure_formula_valid,
    }


# =============================================================================
# RESULT QUALITY GATE
# =============================================================================

def validate_customer_value_result(
    result: dict,
) -> None:
    """
    Validate a single customer-value result before it reaches Streamlit.
    """

    required_keys = [
        "customer_id",
        "august_arpu",
        "customer_value_proxy",
        "exposure_eligible_arpu",
        "calibrated_churn_probability",
        "risk_score",
        "risk_level",
        "potential_monthly_revenue_exposure",
        "annualized_revenue_exposure",
        "exposure_formula_valid",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in result
    ]

    if missing_keys:

        raise ValueError(
            f"Customer value result is missing keys: "
            f"{missing_keys}"
        )

    probability = float(
        result["calibrated_churn_probability"]
    )

    risk_score = float(
        result["risk_score"]
    )

    exposure_eligible_arpu = float(
        result["exposure_eligible_arpu"]
    )

    monthly_exposure = float(
        result["potential_monthly_revenue_exposure"]
    )

    annualized_exposure = float(
        result["annualized_revenue_exposure"]
    )

    # -------------------------------------------------------------------------
    # Probability
    # -------------------------------------------------------------------------

    if not 0 <= probability <= 1:

        raise ValueError(
            "Customer churn probability is outside [0, 1]."
        )

    # -------------------------------------------------------------------------
    # Risk score
    # -------------------------------------------------------------------------

    if not 0 <= risk_score <= 100:

        raise ValueError(
            "Customer risk score is outside [0, 100]."
        )

    # -------------------------------------------------------------------------
    # Exposure-eligible value
    # -------------------------------------------------------------------------

    if exposure_eligible_arpu < 0:

        raise ValueError(
            "Exposure-eligible August ARPU cannot be negative."
        )

    # -------------------------------------------------------------------------
    # Monthly exposure
    # -------------------------------------------------------------------------

    if monthly_exposure < 0:

        raise ValueError(
            "Monthly revenue exposure cannot be negative."
        )

    # -------------------------------------------------------------------------
    # Annualization
    # -------------------------------------------------------------------------

    if not np.isclose(
        annualized_exposure,
        monthly_exposure * 12.0,
        rtol=0,
        atol=1e-9,
    ):

        raise ValueError(
            "Annualized exposure does not equal "
            "monthly exposure × 12."
        )

    # -------------------------------------------------------------------------
    # Formula
    # -------------------------------------------------------------------------

    if not result["exposure_formula_valid"]:

        raise ValueError(
            "Stored Phase 12 exposure does not match "
            "calibrated probability × exposure-eligible ARPU."
        )


# =============================================================================
# INR FORMATTER
# =============================================================================

def format_inr(
    value,
) -> str:
    """
    Format a numeric value as Indian Rupees.
    """

    if value is None or pd.isna(value):

        return "Not available"

    return f"₹{float(value):,.2f}"


# =============================================================================
# SMOKE TEST
# =============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "CHURNIQ — CUSTOMER VALUE & REVENUE RISK SMOKE TEST"
    )
    print("=" * 80)

    customer_id = 0

    print(
        f"\nAnalyzing customer: {customer_id}"
    )

    # -------------------------------------------------------------------------
    # STEP 1
    # -------------------------------------------------------------------------

    print(
        "\n[1/3] Loading locked Phase 12 artifact..."
    )

    data = load_customer_value_data()

    print(
        "Phase 12 artifact           : PASS"
    )

    print(
        f"Customer universe           : "
        f"{len(data):,}"
    )

    # -------------------------------------------------------------------------
    # STEP 2
    # -------------------------------------------------------------------------

    print(
        "\n[2/3] Loading customer-value context..."
    )

    result = get_customer_value(
        customer_id
    )

    print(
        "Customer lookup             : PASS"
    )

    # -------------------------------------------------------------------------
    # STEP 3
    # -------------------------------------------------------------------------

    print(
        "\n[3/3] Running customer-value quality gate..."
    )

    validate_customer_value_result(
        result
    )

    print(
        "Customer-value quality gate : PASS"
    )

    # -------------------------------------------------------------------------
    # RESULT
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CUSTOMER VALUE RESULT")
    print("=" * 80)

    print(
        f"Customer ID                    : "
        f"{result['customer_id']}"
    )

    print(
        f"August ARPU / Value Proxy      : "
        f"{format_inr(result['august_arpu'])}"
    )

    print(
        f"Exposure-eligible ARPU         : "
        f"{format_inr(result['exposure_eligible_arpu'])}"
    )

    print(
        f"Customer value tier            : "
        f"{result['customer_value_tier']}"
    )

    print(
        f"Customer value percentile      : "
        f"{result['customer_value_percentile']:.2f}"
    )

    print(
        f"Calibrated churn probability   : "
        f"{result['calibrated_churn_probability']:.6f}"
    )

    print(
        f"Risk score                     : "
        f"{result['risk_score']:.2f} / 100"
    )

    print(
        f"Risk level                     : "
        f"{result['risk_level']}"
    )

    print(
        f"Monthly revenue exposure      : "
        f"{format_inr(result['potential_monthly_revenue_exposure'])}"
    )

    print(
        f"Annualized exposure            : "
        f"{format_inr(result['annualized_revenue_exposure'])}"
    )

    print(
        f"Revenue exposure rank         : "
        f"{result['revenue_exposure_rank']:,}"
    )

    print(
        f"Exposure share                 : "
        f"{result['customer_exposure_share_pct']:.4f}%"
    )

    print(
        f"Exposure formula validation    : "
        f"{'PASS' if result['exposure_formula_valid'] else 'FAIL'}"
    )

    print(
        f"Validation difference          : "
        f"{result['exposure_validation_difference']:.12f}"
    )

    print("\n" + "=" * 80)
    print(
        "CUSTOMER VALUE & REVENUE RISK SMOKE TEST: PASS"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()