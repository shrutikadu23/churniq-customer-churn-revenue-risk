"""
ChurnIQ — Customer Retention Intelligence

Phase 14.5
----------
Streamlit AI application with customer-level SHAP explainability
and customer value / revenue risk analysis.

Current scope:
- Existing customer lookup
- Calibrated churn probability
- Risk score
- Risk level
- Primary threshold status
- Why this prediction? — customer-level SHAP explanation
- Customer value
- Revenue exposure
- Business interpretation

The application uses the locked ChurnIQ inference, explainability,
and customer-value engines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PROFESSIONAL PINK THEME
# =============================================================================

st.markdown(
    """
    <style>
        /* ------------------------------------------------------------------
           Global page
           ------------------------------------------------------------------ */

        html, body, [class*="css"], .stApp {
            color: #2f2529 !important;
        }

        .stApp {
            background: #fffafb !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(
                    180deg,
                    #fff4f8 0px,
                    #fffafb 170px,
                    #ffffff 430px,
                    #ffffff 100%
                ) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 250, 252, 0.96) !important;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 3.25rem;
            padding-bottom: 2.5rem;
        }

        /* ------------------------------------------------------------------
           Sidebar
           ------------------------------------------------------------------ */

        [data-testid="stSidebar"] {
            background: #fff0f5 !important;
            border-right: 1px solid #f2ccd9 !important;
        }

        [data-testid="stSidebar"] * {
            color: #3a2b31 !important;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #6f5962 !important;
        }

        /* ------------------------------------------------------------------
           Typography
           ------------------------------------------------------------------ */

        h1, h2, h3, h4, h5, h6 {
            color: #302329 !important;
            letter-spacing: -0.02em;
        }

        p, li, label {
            color: inherit;
        }

        .section-label {
            color: #a34d73 !important;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0.15rem 0 0.7rem 0;
        }

        /* ------------------------------------------------------------------
           Hero
           ------------------------------------------------------------------ */

        .hero {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(
                    135deg,
                    #f9dce8 0%,
                    #fdebf2 48%,
                    #fff8fb 100%
                );
            border: 1px solid #efbfd1;
            border-radius: 20px;
            padding: 1.45rem 1.6rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 26px rgba(164, 77, 115, 0.08);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 170px;
            height: 170px;
            right: -65px;
            top: -85px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.45);
        }

        .hero-title {
            position: relative;
            z-index: 1;
            color: #302329 !important;
            font-size: 2.05rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .hero-subtitle {
            position: relative;
            z-index: 1;
            color: #684e59 !important;
            font-size: 0.98rem;
            line-height: 1.55;
            max-width: 900px;
        }

        /* ------------------------------------------------------------------
           Streamlit metric cards
           ------------------------------------------------------------------ */

        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #efc9d7 !important;
            border-radius: 16px !important;
            padding: 1rem 1.15rem !important;
            min-height: 108px;
            box-shadow: 0 5px 18px rgba(111, 62, 82, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            color: #765e68 !important;
            opacity: 1 !important;
        }

        div[data-testid="stMetricLabel"] * {
            color: #765e68 !important;
            opacity: 1 !important;
        } 

        div[data-testid="stMetricValue"] {
            color: #302329 !important;
            font-weight: 800 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        div[data-testid="stMetricValue"] > div {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.2 !important;
        }

        /* Keep long business labels readable without making
           numeric metrics unnecessarily large. */
        div[data-testid="stMetric"]:nth-child(n) div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #9c5574 !important;
        }

        /* ------------------------------------------------------------------
           Forms / inputs
           ------------------------------------------------------------------ */

        div[data-testid="stTextInput"] input {
            background: #ffffff !important;
            color: #302329 !important;
            border: 1px solid #e1b6c7 !important;
            border-radius: 10px !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #c96891 !important;
            box-shadow: 0 0 0 2px rgba(201, 104, 145, 0.14) !important;
        }

        /* Primary button */
        div[data-testid="stFormSubmitButton"] > button {
            background: #c85f89 !important;
            color: #ffffff !important;
            border: 1px solid #c85f89 !important;
            border-radius: 10px !important;
            font-weight: 750 !important;
            min-height: 42px;
            box-shadow: 0 5px 14px rgba(200, 95, 137, 0.20);
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #b94f7b !important;
            border-color: #b94f7b !important;
        }

        /* ------------------------------------------------------------------
           Alerts
           ------------------------------------------------------------------ */

        div[data-testid="stAlert"] {
            border-radius: 12px !important;
        }

        /* ------------------------------------------------------------------
           Driver cards
           ------------------------------------------------------------------ */

        .driver-card {
            background: #ffffff;
            border: 1px solid #efd1dc;
            border-left: 4px solid #d36c94;
            border-radius: 12px;
            padding: 0.82rem 0.95rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 3px 12px rgba(111, 62, 82, 0.045);
        }

        .driver-card-negative {
            border-left-color: #6c8a80;
        }

        .driver-name {
            color: #392b31 !important;
            font-weight: 750;
            font-size: 0.94rem;
        }

        .driver-value {
            color: #806b74 !important;
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }

        .impact-positive {
            color: #ad4f77 !important;
            font-weight: 750;
        }

        .impact-negative {
            color: #527268 !important;
            font-weight: 750;
        }

        /* ------------------------------------------------------------------
           Risk badge
           ------------------------------------------------------------------ */

        .risk-badge {
            display: inline-block;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }

        .risk-high {
            color: #8e3d61 !important;
            background: #f9dce8;
            border: 1px solid #eab8cc;
        }

        .risk-very-high {
            color: #7e274f !important;
            background: #f4c9db;
            border: 1px solid #e3a3bd;
        }

        .risk-low {
            color: #4f6d63 !important;
            background: #e3f0eb;
            border: 1px solid #c7ddd5;
        }

        /* ------------------------------------------------------------------
           Customer value / revenue risk cards
           ------------------------------------------------------------------ */

        .business-context {
            background: #fff0f6;
            border: 1px solid #f0c9d8;
            border-left: 5px solid #e75480;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
        }

        .business-context-title {
            color: #7f3154 !important;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .business-context-text {
            color: #654f59 !important;
            line-height: 1.5;
        }

        .value-tier-card {
            background: #ffffff;
            border: 1px solid #efc9d7;
            border-radius: 16px;
            padding: 1rem 1.15rem;
            min-height: 108px;
            box-shadow: 0 5px 18px rgba(111, 62, 82, 0.06);
        }

        .value-tier-label {
            color: #765e68 !important;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.65rem;
        }

        .value-tier-badge {
            display: inline-block;
            padding: 0.38rem 0.78rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 800;
        }

        .tier-high {
            color: #7e274f !important;
            background: #f4c9db;
            border: 1px solid #e3a3bd;
        }

        .tier-medium {
            color: #8e3d61 !important;
            background: #f9dce8;
            border: 1px solid #eab8cc;
        }

        .tier-low {
            color: #5c6870 !important;
            background: #edf0f2;
            border: 1px solid #d7dde1;
        }

        .exposure-context {
            background: #fffafb;
            border: 1px solid #efd1dc;
            border-radius: 13px;
            padding: 0.9rem 1rem;
            margin-top: 0.75rem;
        }

        .exposure-context-title {
            color: #7f3154 !important;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .exposure-context-text {
            color: #654f59 !important;
            font-size: 0.88rem;
            line-height: 1.5;
        }

        /* ------------------------------------------------------------------
           Explanation intro card
           ------------------------------------------------------------------ */

        .explain-note {
            background: #fff3f7;
            border: 1px solid #f0cdd9;
            border-radius: 13px;
            padding: 0.9rem 1rem;
            color: #654f59 !important;
            margin-bottom: 1rem;
        }

        /* ------------------------------------------------------------------
           Expanders
           ------------------------------------------------------------------ */

        div[data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #efd1dc !important;
            border-radius: 13px !important;
        }

        /* ------------------------------------------------------------------
           Footer
           ------------------------------------------------------------------ */

        .footer {
            color: #8a6f79 !important;
            text-align: center;
            font-size: 0.8rem;
            padding-top: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# IMPORT LOCKED ENGINES
# =============================================================================

try:
    from churniq_inference import (
        load_inference_engine,
        score_existing_customer,
    )

    from churniq_explainability import explain_customer

    from churniq_customer_value import (
        get_customer_value,
        validate_customer_value_result,
        format_inr,
    )

except Exception as error:
    st.error("Unable to load the ChurnIQ inference or explainability engine.")
    st.code(str(error))
    st.stop()


# =============================================================================
# CACHED INFERENCE ENGINE
# =============================================================================

@st.cache_resource(show_spinner="Loading ChurnIQ model...")
def get_inference_engine():
    return load_inference_engine()


try:
    (
        model,
        calibrator,
        feature_names,
        imputation_medians,
    ) = get_inference_engine()

except Exception as error:
    st.error("ChurnIQ model artifacts could not be loaded.")
    st.code(str(error))
    st.stop()


# =============================================================================
# CACHED CUSTOMER ID UNIVERSE
# =============================================================================

@st.cache_data(show_spinner=False)
def load_customer_ids():

    feature_file = (
        PROJECT_ROOT
        / "deployment_data"
        / "customer_features.parquet"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Customer feature file not found:\n{feature_file}"
        )

    data = pd.read_parquet(
        feature_file,
        columns=["id"],
    )

    ids = (
        pd.to_numeric(data["id"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    if not ids:
        raise ValueError("No valid customer IDs were found.")

    return ids

try:
    customer_ids = load_customer_ids()

except Exception as error:
    st.error("Unable to load the ChurnIQ customer universe.")
    st.code(str(error))
    st.stop()


CUSTOMER_ID_SET = set(customer_ids)


# =============================================================================
# CACHED CUSTOMER SCORING
# =============================================================================

@st.cache_data(show_spinner="Analyzing customer...")
def analyze_customer(customer_id: int):
    return score_existing_customer(customer_id)


# =============================================================================
# CACHED CUSTOMER SHAP EXPLANATION
# =============================================================================

@st.cache_data(show_spinner="Generating explanation...")
def explain_customer_cached(customer_id: int):
    return explain_customer(customer_id)


# =============================================================================
# CACHED CUSTOMER VALUE
# =============================================================================

@st.cache_data(show_spinner="Loading customer value...")
def get_customer_value_cached(customer_id: int):
    value_result = get_customer_value(customer_id)
    validate_customer_value_result(value_result)
    return value_result


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def humanize_feature_name(feature_name: str) -> str:
    """
    Convert technical feature names into readable labels.
    """
    return (
        str(feature_name)
        .replace("_", " ")
        .replace(" pct ", " % ")
        .strip()
        .title()
    )


def format_feature_value(value) -> str:
    """
    Keep feature values compact and readable.
    """
    if pd.isna(value):
        return "Missing"

    try:
        numeric_value = float(value)

        if abs(numeric_value) >= 100000:
            return f"{numeric_value:,.0f}"

        if abs(numeric_value) >= 100:
            return f"{numeric_value:,.1f}"

        if abs(numeric_value) >= 1:
            return f"{numeric_value:,.2f}"

        return f"{numeric_value:.4f}"

    except (TypeError, ValueError):
        return str(value)


def render_driver_cards(dataframe: pd.DataFrame, positive: bool):
    """
    Render top SHAP drivers as compact business-facing cards.
    """
    if dataframe.empty:
        st.caption("No drivers were identified in this direction.")
        return

    for _, row in dataframe.iterrows():

        feature = humanize_feature_name(row["feature"])
        value = format_feature_value(row["feature_value"])
        shap_value = float(row["shap_value"])

        if positive:
            impact_text = f"+{shap_value:.3f} model impact"
            impact_class = "impact-positive"
            card_class = "driver-card"

        else:
            impact_text = f"{shap_value:.3f} model impact"
            impact_class = "impact-negative"
            card_class = "driver-card driver-card-negative"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="driver-name">{feature}</div>
                <div class="driver-value">
                    Customer value: {value}
                    &nbsp;&nbsp;•&nbsp;&nbsp;
                    <span class="{impact_class}">{impact_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# SESSION STATE
# =============================================================================

if "analyzed_customer_id" not in st.session_state:
    st.session_state.analyzed_customer_id = None


# =============================================================================
# HERO HEADER
# =============================================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">ChurnIQ</div>
        <div class="hero-subtitle">
            Customer Retention Intelligence — predict churn risk,
            explain the signals behind the prediction, quantify value
            exposure, and support retention decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR — CUSTOMER LOOKUP
# =============================================================================

with st.sidebar:

    st.markdown(
        '<div class="section-label">Customer analysis</div>',
        unsafe_allow_html=True,
    )

    st.header("Analyze a Customer")

    st.caption(
        "Enter an existing customer ID from the ChurnIQ customer universe."
    )

    with st.form("customer_analysis_form"):

        customer_input = st.text_input(
            "Customer ID",
            placeholder="Example: 0",
        )

        analyze = st.form_submit_button(
            "Analyze Customer",
            type="primary",
            use_container_width=True,
        )

    if analyze:

        customer_input = customer_input.strip()

        if not customer_input:

            st.session_state.analyzed_customer_id = None
            st.error("Please enter a customer ID.")

        else:

            try:
                customer_id = int(customer_input)

            except ValueError:

                st.session_state.analyzed_customer_id = None
                st.error("Customer ID must be a whole number.")

            else:

                if customer_id not in CUSTOMER_ID_SET:

                    st.session_state.analyzed_customer_id = None

                    st.error(
                        f"Customer ID {customer_id} was not found "
                        "in the ChurnIQ customer universe."
                    )

                else:

                    st.session_state.analyzed_customer_id = customer_id

    st.divider()

    st.markdown("**ChurnIQ model context**")

    st.caption(
        f"Customer universe: {len(customer_ids):,}"
    )

    st.caption(
        f"Frozen model features: {len(feature_names)}"
    )

    st.caption(
        "Primary operating threshold: 10%"
    )


# =============================================================================
# INITIAL STATE
# =============================================================================

if st.session_state.analyzed_customer_id is None:

    st.info(
        "Enter a customer ID and select **Analyze Customer** "
        "to view the customer's ChurnIQ risk assessment."
    )

    st.stop()


customer_id = st.session_state.analyzed_customer_id


# =============================================================================
# SCORE CUSTOMER
# =============================================================================

try:

    result = analyze_customer(customer_id)

except Exception as error:

    st.error("Customer analysis failed.")
    st.code(str(error))
    st.stop()


# =============================================================================
# RESULT VALUES
# =============================================================================

probability = float(
    result["calibrated_churn_probability"]
)

risk_score = float(
    result["risk_score"]
)

risk_level = result["risk_level"]

above_threshold = bool(
    result["above_primary_threshold"]
)

missing_before_imputation = int(
    result["missing_values_before_imputation"]
)

threshold_status = (
    "Above threshold"
    if above_threshold
    else "Below threshold"
)


# =============================================================================
# CUSTOMER RISK
# =============================================================================

st.markdown(
    '<div class="section-label">Customer risk profile</div>',
    unsafe_allow_html=True,
)

st.header(f"Customer {customer_id}")

if risk_level == "Very High":

    st.markdown(
        '<span class="risk-badge risk-very-high">VERY HIGH RISK</span>',
        unsafe_allow_html=True,
    )

elif risk_level == "High":

    st.markdown(
        '<span class="risk-badge risk-high">HIGH RISK</span>',
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        '<span class="risk-badge risk-low">BELOW PRIMARY THRESHOLD</span>',
        unsafe_allow_html=True,
    )


st.subheader("Risk Assessment")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:

    st.metric(
        "Churn Probability",
        f"{probability:.2%}",
    )

with risk_col2:

    st.metric(
        "Risk Score",
        f"{risk_score:.1f} / 100",
    )


risk_col3, risk_col4 = st.columns(2)

with risk_col3:

    st.metric(
        "Risk Level",
        risk_level,
    )

with risk_col4:

    st.metric(
        "Primary Threshold Status",
        threshold_status,
    )


# =============================================================================
# RISK INTERPRETATION
# =============================================================================

st.divider()

st.subheader("Risk Interpretation")

if risk_level == "Very High":

    st.warning(
        "This customer has a very high model-estimated churn probability "
        "and falls within the highest-risk category."
    )

elif risk_level == "High":

    st.warning(
        "This customer is above the primary operating threshold "
        "and is classified as high risk."
    )

else:

    st.success(
        "This customer is below the primary operating threshold "
        "for churn risk."
    )


st.caption(
    "Churn probability is a model-estimated likelihood, "
    "not a guarantee of future churn."
)


if missing_before_imputation > 0:

    st.caption(
        f"{missing_before_imputation} missing frozen feature value(s) "
        "were handled using development-only median imputation."
    )


# =============================================================================
# WHY THIS PREDICTION — SHAP
# =============================================================================

st.divider()

st.markdown(
    '<div class="section-label">Explainability</div>',
    unsafe_allow_html=True,
)

st.header("Why this prediction?")

st.markdown(
    """
    <div class="explain-note">
        The factors below show which customer features were associated
        with a higher or lower XGBoost churn prediction for this customer.
        Positive SHAP values push the model prediction upward, while
        negative SHAP values push it downward.
    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "SHAP values represent model-associated signals, not causal effects. "
    "They are calculated in the model's raw-output/log-odds space."
)


try:

    shap_result = explain_customer_cached(customer_id)

except Exception as error:

    st.error(
        "Customer-level SHAP explanation could not be generated."
    )

    st.code(str(error))
    st.stop()


positive_drivers = shap_result["top_positive"]
negative_drivers = shap_result["top_negative"]


# =============================================================================
# SHAP SUMMARY
# =============================================================================

shap_col1, shap_col2, shap_col3 = st.columns(3)

with shap_col1:

    st.metric(
        "Features explained",
        f"{shap_result['feature_count']}",
    )

with shap_col2:

    st.metric(
        "Risk-increasing factors",
        f"{len(positive_drivers)}",
    )

with shap_col3:

    st.metric(
        "Risk-decreasing factors",
        f"{len(negative_drivers)}",
    )


# =============================================================================
# TOP DRIVERS
# =============================================================================

driver_col1, driver_col2 = st.columns(2)

with driver_col1:

    st.subheader("Increasing churn risk")

    render_driver_cards(
        positive_drivers,
        positive=True,
    )


with driver_col2:

    st.subheader("Decreasing churn risk")

    render_driver_cards(
        negative_drivers,
        positive=False,
    )


# =============================================================================
# SHAP TECHNICAL DETAILS
# =============================================================================

with st.expander("View SHAP technical details"):

    st.write(
        "**Interpretation:** Positive SHAP values push the model's raw "
        "prediction upward; negative values push it downward."
    )

    technical_col1, technical_col2 = st.columns(2)

    with technical_col1:

        st.write(
            f"**SHAP base value:** "
            f"{shap_result['base_value']:.6f}"
        )

        st.write(
            f"**Raw model output:** "
            f"{shap_result['raw_model_output']:.6f}"
        )

        st.write(
            f"**Reconstructed SHAP output:** "
            f"{shap_result['reconstructed_raw_output']:.6f}"
        )

    with technical_col2:

        st.write(
            f"**Additivity error:** "
            f"{shap_result['additivity_error']:.10f}"
        )

        st.write(
            f"**Raw model probability:** "
            f"{shap_result['raw_model_probability']:.4%}"
        )

        st.write(
            "**Customer-facing probability:** "
            "Locked calibrated churn probability"
        )

    st.caption(
        "The SHAP additivity check validates the decomposition against "
        "the XGBoost raw model output. The calibrated probability is "
        "produced separately by the locked ChurnIQ inference pipeline."
    )


# =============================================================================
# CUSTOMER VALUE & REVENUE RISK — PHASE 14.5
# =============================================================================

st.divider()

st.markdown(
    '<div class="section-label">Business impact</div>',
    unsafe_allow_html=True,
)

st.header("Customer Value & Revenue Risk")

st.markdown(
    """
    <div class="business-context">
        <div class="business-context-title">Business Context</div>
        <div class="business-context-text">
            Customer value combines churn risk with August customer value and
            estimated revenue exposure. This helps prioritize retention efforts
            toward customers with higher potential business impact.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:

    value_result = get_customer_value_cached(customer_id)

except Exception as error:

    st.error(
        "Customer value and revenue-risk analysis could not be loaded."
    )

    st.code(str(error))
    st.stop()


# -----------------------------------------------------------------------------
# CUSTOMER VALUE
# -----------------------------------------------------------------------------

st.subheader("💎 Customer Value")

value_col1, value_col2, value_col3 = st.columns(3)

with value_col1:

    st.metric(
        "August ARPU",
        format_inr(value_result["august_arpu"]),
    )


with value_col2:

    tier = str(
        value_result["customer_value_tier"]
    )

    if tier == "High Value":

        tier_class = "tier-high"

    elif tier == "Medium Value":

        tier_class = "tier-medium"

    else:

        tier_class = "tier-low"

    st.markdown(
        f"""
        <div class="value-tier-card">
            <div class="value-tier-label">Value Tier</div>
            <span class="value-tier-badge {tier_class}">{tier}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


with value_col3:

    st.metric(
        "Value Percentile",
        f"{float(value_result['customer_value_percentile']):.1f}%",
    )


# -----------------------------------------------------------------------------
# REVENUE RISK
# -----------------------------------------------------------------------------

st.subheader("💰 Revenue Risk")

revenue_col1, revenue_col2, revenue_col3 = st.columns(3)

with revenue_col1:

    st.metric(
        "Monthly Exposure",
        format_inr(
            value_result["potential_monthly_revenue_exposure"]
        ),
    )


with revenue_col2:

    st.metric(
        "Annualized Exposure",
        format_inr(
            value_result["annualized_revenue_exposure"]
        ),
    )


with revenue_col3:

    st.metric(
        "Exposure Rank",
        f"{int(value_result['revenue_exposure_rank']):,}",
    )


exposure_pct = float(
    value_result["revenue_exposure_percentile"]
)

exposure_pct = max(
    0.0,
    min(exposure_pct, 100.0),
)


st.progress(
    int(round(exposure_pct)),
    text=f"Revenue Exposure Position: {exposure_pct:.1f} percentile",
)


st.markdown(
    f"""
    <div class="exposure-context">
        <div class="exposure-context-title">Revenue Exposure Position</div>
        <div class="exposure-context-text">
            This customer sits at the <b>{exposure_pct:.1f}th percentile</b>
            of the modeled customer exposure distribution. A higher percentile
            indicates greater estimated revenue exposure relative to the
            customer universe.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# BUSINESS INTERPRETATION
# -----------------------------------------------------------------------------

st.subheader("📋 Business Interpretation")

monthly_exposure = float(
    value_result["potential_monthly_revenue_exposure"]
)

annualized_exposure = float(
    value_result["annualized_revenue_exposure"]
)


if above_threshold and monthly_exposure > 0:

    st.warning(
        f"This customer is above the primary churn-risk threshold and is "
        f"classified as {tier}. The modeled monthly revenue exposure is "
        f"{format_inr(monthly_exposure)}, with an annualized scenario "
        f"exposure of {format_inr(annualized_exposure)}."
    )

elif above_threshold and monthly_exposure <= 0:

    st.warning(
        f"This customer is above the primary churn-risk threshold and is "
        f"classified as {tier}, but the current exposure-eligible customer "
        f"value is limited."
    )

elif not above_threshold and monthly_exposure > 0:

    st.info(
        f"This customer is below the primary churn-risk threshold and is "
        f"classified as {tier}. The modeled monthly revenue exposure is "
        f"{format_inr(monthly_exposure)}."
    )

else:

    st.info(
        f"This customer is below the primary churn-risk threshold and has "
        f"limited exposure-eligible customer value in the current scenario."
    )


st.caption(
    "August ARPU is used as a customer-value proxy. Revenue exposure is an "
    "estimated scenario, not guaranteed revenue loss. This section does not "
    "apply ROI assumptions or determine intervention economics."
)


# =============================================================================
# MODEL DETAILS
# =============================================================================

with st.expander("Model & Application Details"):

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:

        st.write("**Model:** Final tuned XGBoost")

        st.write(
            f"**Frozen features:** {len(feature_names)}"
        )

        st.write("**Probability calibration:** Sigmoid")

    with detail_col2:

        st.write("**Primary threshold:** 10%")

        st.write(
            "**Prediction output:** Calibrated churn probability"
        )

        st.write(
            "**Inference preprocessing:** "
            "Development-only median imputation"
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ChurnIQ | Predict → Explain → Quantify Risk → Prioritize → Recommend Action
    </div>
    """,
    unsafe_allow_html=True,
) 