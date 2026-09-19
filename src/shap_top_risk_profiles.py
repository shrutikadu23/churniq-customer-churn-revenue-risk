"""
ChurnIQ — Phase 10.4
Top-Risk Customer Explanation Profiles

Purpose
-------
Create a compact, business-readable explanation profile for the
highest-risk customers within the 10,000-customer SHAP explanation sample.

Methodological boundaries
-------------------------
1. This is an EXPLANATION SAMPLE, not the official full-universe ranking.
2. Phase 11 will rank customers across the complete 69,999-customer universe.
3. No new model is trained in this phase.
4. No September information is used.
5. SHAP values describe model contribution, not causal impact.
6. Customer value and revenue exposure are intentionally excluded;
   those belong to Phase 12.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 01. PATHS & CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "reports"
    / "shap"
    / "customer_shap_explanations.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "shap"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "top_risk_explanation_profiles.csv"
)

QUALITY_FILE = (
    OUTPUT_DIR
    / "top_risk_explanation_quality_gate.txt"
)

TOP_N = 50


# ============================================================
# 02. LOAD PHASE 10.3 OUTPUT
# ============================================================

print("=" * 80)
print("PHASE 10.4 — TOP-RISK CUSTOMER EXPLANATION PROFILES")
print("=" * 80)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Required Phase 10.3 output was not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput shape: {df.shape}")


# ============================================================
# 03. CORE COLUMN VALIDATION
# ============================================================

required_columns = [
    "id",
    "churn_probability",
    "risk_level",
    "primary_threshold",
    "above_primary_threshold",
    "actual_churn",
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )


# ============================================================
# 04. IDENTIFY DRIVER RANKS
# ============================================================

positive_driver_cols = [
    col
    for col in df.columns
    if col.startswith("positive_driver_")
    and col.count("_") == 2
    and col.split("_")[-1].isdigit()
]

negative_driver_cols = [
    col
    for col in df.columns
    if col.startswith("negative_driver_")
    and col.count("_") == 2
    and col.split("_")[-1].isdigit()
]


def extract_rank(column_name):
    return int(column_name.rsplit("_", 1)[-1])


positive_driver_cols = sorted(
    positive_driver_cols,
    key=extract_rank
)

negative_driver_cols = sorted(
    negative_driver_cols,
    key=extract_rank
)


print(
    f"Positive driver columns: {len(positive_driver_cols)}"
)

print(
    f"Negative driver columns: {len(negative_driver_cols)}"
)


# ============================================================
# 05. VALIDATE COMPLETE DRIVER STRUCTURE
# ============================================================

expected_positive_ranks = list(
    range(
        1,
        len(positive_driver_cols) + 1
    )
)

expected_negative_ranks = list(
    range(
        1,
        len(negative_driver_cols) + 1
    )
)

actual_positive_ranks = [
    extract_rank(col)
    for col in positive_driver_cols
]

actual_negative_ranks = [
    extract_rank(col)
    for col in negative_driver_cols
]

if actual_positive_ranks != expected_positive_ranks:
    raise ValueError(
        "Positive driver ranks are incomplete or incorrectly ordered."
    )

if actual_negative_ranks != expected_negative_ranks:
    raise ValueError(
        "Negative driver ranks are incomplete or incorrectly ordered."
    )


# Validate every expected companion column.

companion_suffixes = [
    "_shap",
    "_abs_shap",
    "_value",
    "_type",
]

for driver_col in (
    positive_driver_cols
    + negative_driver_cols
):

    for suffix in companion_suffixes:

        companion = (
            driver_col
            + suffix
        )

        if companion not in df.columns:

            raise ValueError(
                f"Missing companion column: {companion}"
            )

print(
    "Driver structure validation: PASS"
)


# ============================================================
# 06. VALIDATE PROBABILITIES
# ============================================================

df["churn_probability"] = pd.to_numeric(
    df["churn_probability"],
    errors="coerce"
)

if df["churn_probability"].isna().any():

    raise ValueError(
        "Missing or invalid churn probabilities detected."
    )

if not df["churn_probability"].between(
    0,
    1
).all():

    raise ValueError(
        "Churn probabilities outside [0, 1] detected."
    )


# ============================================================
# 07. SELECT TOP-RISK CUSTOMERS
# ============================================================

profile_df = df.copy()

profile_df = profile_df.sort_values(
    by=[
        "churn_probability",
        "id"
    ],
    ascending=[
        False,
        True
    ]
)

top_risk = profile_df.head(
    TOP_N
).copy()

print(
    f"\nTop-risk explanation profiles selected: "
    f"{len(top_risk)}"
)


# ============================================================
# 08. DRIVER SUMMARY FUNCTION
# ============================================================

def build_driver_summary(
    row,
    driver_cols
):
    """
    Build a compact business-readable summary using the
    exact driver, SHAP, feature value, and contribution type
    columns produced by Phase 10.3.
    """

    results = []

    for driver_col in driver_cols:

        rank = extract_rank(
            driver_col
        )

        shap_col = (
            f"{driver_col}_shap"
        )

        value_col = (
            f"{driver_col}_value"
        )

        type_col = (
            f"{driver_col}_type"
        )

        driver_name = row.get(
            driver_col
        )

        shap_value = row.get(
            shap_col
        )

        feature_value = row.get(
            value_col
        )

        contribution_type = row.get(
            type_col
        )

        if pd.isna(driver_name):
            continue

        driver_name = str(
            driver_name
        ).strip()

        if not driver_name:
            continue

        # SHAP contribution
        if pd.notna(shap_value):

            try:
                shap_text = (
                    f"{float(shap_value):+.4f}"
                )
            except (
                TypeError,
                ValueError
            ):
                shap_text = str(
                    shap_value
                )

        else:
            shap_text = "N/A"

        # Actual feature value
        if pd.notna(feature_value):

            try:
                value_text = (
                    f"{float(feature_value):.4f}"
                )
            except (
                TypeError,
                ValueError
            ):
                value_text = str(
                    feature_value
                )

        else:
            value_text = "N/A"

        # Contribution type
        if pd.notna(contribution_type):

            type_text = str(
                contribution_type
            ).strip()

        else:

            type_text = "N/A"

        results.append(
            f"{rank}. {driver_name} "
            f"(SHAP {shap_text}; "
            f"value {value_text}; "
            f"{type_text})"
        )

    return " | ".join(
        results
    )


# ============================================================
# 09. CREATE BUSINESS-READABLE PROFILES
# ============================================================

top_risk["top_positive_drivers"] = top_risk.apply(
    lambda row: build_driver_summary(
        row,
        positive_driver_cols
    ),
    axis=1
)

top_risk["top_negative_drivers"] = top_risk.apply(
    lambda row: build_driver_summary(
        row,
        negative_driver_cols
    ),
    axis=1
)


# ============================================================
# 10. ADD BUSINESS CONTEXT
# ============================================================

top_risk["explanation_scope"] = (
    "Top-risk explanation sample — "
    "not full-universe ranking"
)

top_risk["interpretation_note"] = (
    "Positive SHAP drivers indicate model contributions "
    "toward higher churn risk; negative SHAP drivers indicate "
    "model contributions toward lower churn risk. "
    "SHAP contributions are model explanations, not causal claims."
)


# ============================================================
# 11. FINAL OUTPUT COLUMNS
# ============================================================

final_columns = [
    "id",
    "churn_probability",
    "risk_level",
    "primary_threshold",
    "above_primary_threshold",
    "actual_churn",
    "top_positive_drivers",
    "top_negative_drivers",
    "explanation_scope",
    "interpretation_note",
]

missing_final = [
    col
    for col in final_columns
    if col not in top_risk.columns
]

if missing_final:

    raise ValueError(
        f"Missing final output columns: {missing_final}"
    )

output = top_risk[
    final_columns
].copy()


# ============================================================
# 12. ADD EXPLANATION RANK
# ============================================================

output.insert(
    0,
    "explanation_rank",
    range(
        1,
        len(output) + 1
    )
)


# ============================================================
# 13. FORMAT NUMERIC OUTPUT
# ============================================================

output["churn_probability"] = (
    output["churn_probability"]
    .round(6)
)

output["primary_threshold"] = (
    pd.to_numeric(
        output["primary_threshold"],
        errors="coerce"
    )
    .round(4)
)


# ============================================================
# 14. SAVE PROFILE DATASET
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nProfile dataset saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# 15. QUALITY GATE
# ============================================================

checks = {}

checks["input_exists"] = (
    INPUT_FILE.exists()
)

checks["output_exists"] = (
    OUTPUT_FILE.exists()
)

checks["profile_count_50"] = (
    len(output) == TOP_N
)

checks["unique_customer_ids"] = (
    output["id"].nunique()
    == len(output)
)

checks["probabilities_valid"] = (
    output["churn_probability"].notna().all()
    and
    output["churn_probability"]
    .between(0, 1)
    .all()
)

checks["descending_risk_order"] = (
    output["churn_probability"]
    .is_monotonic_decreasing
)

checks["risk_levels_present"] = (
    output["risk_level"].notna().all()
)

checks["threshold_present"] = (
    output["primary_threshold"].notna().all()
)

checks["threshold_logic_present"] = (
    output["above_primary_threshold"].notna().all()
)

checks["positive_driver_profiles_present"] = (
    output["top_positive_drivers"]
    .fillna("")
    .str.len()
    .gt(0)
    .all()
)

checks["negative_driver_profiles_present"] = (
    output["top_negative_drivers"]
    .fillna("")
    .str.len()
    .gt(0)
    .all()
)

checks["scope_label_present"] = (
    output["explanation_scope"]
    .eq(
        "Top-risk explanation sample — "
        "not full-universe ranking"
    )
    .all()
)

checks["interpretation_note_present"] = (
    output["interpretation_note"]
    .notna()
    .all()
    and
    output["interpretation_note"]
    .str.len()
    .gt(0)
    .all()
)


# ============================================================
# 16. FINAL QUALITY GATE
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "QUALITY GATE"
)

print(
    "=" * 80
)

all_pass = True

for check_name, passed in checks.items():

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{check_name:<45} {status}"
    )

    if not passed:
        all_pass = False


overall_status = (
    "PASS"
    if all_pass
    else "FAIL"
)

print(
    "\n" + "-" * 80
)

print(
    "Overall Top-Risk Explanation Quality Gate: "
    f"{overall_status}"
)


# ============================================================
# 17. SAVE QUALITY REPORT
# ============================================================

with open(
    QUALITY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ChurnIQ — Phase 10.4 "
        "Top-Risk Explanation Quality Gate\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    for check_name, passed in checks.items():

        f.write(
            f"{check_name}: "
            f"{'PASS' if passed else 'FAIL'}\n"
        )

    f.write(
        "\n"
    )

    f.write(
        f"Overall Status: "
        f"{overall_status}\n"
    )


# ============================================================
# 18. PREVIEW — TOP 10
# ============================================================

print(
    "\nTop 10 explanation profiles:\n"
)

preview_columns = [
    "explanation_rank",
    "id",
    "churn_probability",
    "risk_level",
]

print(
    output[
        preview_columns
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 19. TOP-RISK CUSTOMER DETAIL
# ============================================================

if len(output) > 0:

    first_customer = output.iloc[0]

    print(
        "\nTop-risk customer explanation preview:"
    )

    print(
        f"\nCustomer ID: "
        f"{first_customer['id']}"
    )

    print(
        f"Churn Probability: "
        f"{first_customer['churn_probability']:.4f}"
    )

    print(
        f"Risk Level: "
        f"{first_customer['risk_level']}"
    )

    print(
        "\nTop Positive Drivers:"
    )

    print(
        first_customer[
            "top_positive_drivers"
        ]
    )

    print(
        "\nTop Negative Drivers:"
    )

    print(
        first_customer[
            "top_negative_drivers"
        ]
    )


# ============================================================
# 20. COMPLETION STATUS
# ============================================================

print(
    "\n" + "=" * 80
)

if all_pass:

    print(
        "PHASE 10.4 STATUS: PASS"
    )

else:

    print(
        "PHASE 10.4 STATUS: REVIEW REQUIRED"
    )

print(
    "=" * 80
)