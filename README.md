# ChurnIQ — Customer Churn & Revenue Risk Analytics

> **Predict churn → Explain risk → Quantify revenue exposure → Prioritize retention opportunities**

ChurnIQ is an end-to-end **customer retention analytics and machine learning project** designed to help businesses identify potential churn risk early, understand the behavioral signals behind that risk, quantify customer-value exposure, and evaluate retention opportunities under explicit business assumptions.

The project connects:

**Business Analysis → Prediction → Probability Calibration → Explainability → Customer Risk → Customer Value → Revenue Exposure → Retention Economics → Business Decision Support**

## 🎯 Project at a Glance

| Area | Result |
|---|---:|
| Customers analyzed | **69,999** |
| Final model features | **169** |
| Validation PR-AUC | **0.7614** |
| Validation ROC-AUC | **0.9502** |
| Top-10% churn capture | **71.04%** |
| Top-10% lift | **7.10×** |
| Customers above primary threshold | **12,492** |
| Potential monthly revenue exposure | **₹776,208** |
| Annualized scenario exposure | **₹9.31M** |

> **Note:** Revenue exposure figures are scenario-based estimates using calibrated churn probability and August ARPU as a customer-value proxy. They do not represent guaranteed future revenue loss.

---

## 💡 Why ChurnIQ?

Many churn projects stop at:

> **"Which customers are likely to churn?"**

ChurnIQ extends the analysis to a broader business question:

> **"Which customers may be at risk, what behavioral signals contribute to that risk, what customer-value exposure is associated with it, and how could retention attention be prioritized under stated business assumptions?"**

This makes ChurnIQ a **decision-support framework**, rather than a standalone churn classification model.

---

## 🏢 Business Problem

A business needs to identify potential churn risk early enough to investigate retention opportunities.

The project addresses questions such as:

- Which customers have elevated churn probability?
- What behavioral patterns are associated with higher churn risk?
- Which high-risk customers also represent meaningful customer-value exposure?
- How concentrated is potential revenue exposure?
- Which customer cohorts could be considered for retention analysis?
- Under different retention success and intervention-cost assumptions, where could retention economics become viable?

The analysis uses historical customer behavior available up to the defined prediction point, while keeping future outcome information outside the prediction features.

---

## 🎯 Objective

ChurnIQ follows a business-oriented framework:

```text
Predict
   ↓
Explain
   ↓
Quantify Risk
   ↓
Prioritize
   ↓
Evaluate Retention Opportunities
```

The objective is **not to automate customer decisions**.

Instead, ChurnIQ provides analysts and business stakeholders with evidence that can support more informed retention planning and customer-prioritization decisions.

---

## 🔄 End-to-End Workflow

ChurnIQ was developed as a structured analytics and decision-support pipeline:

```text
Business Problem Definition
        ↓
Dataset & Data Architecture
        ↓
Data Quality & Validation
        ↓
SQL Business Analysis
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Baseline & Model Development
        ↓
Model Evaluation & Selection
        ↓
Probability Calibration
        ↓
Threshold Optimization
        ↓
SHAP Explainability
        ↓
Customer Risk Intelligence
        ↓
Customer Value & Revenue Risk
        ↓
Retention Prioritization & Economics
        ↓
Streamlit AI Application
        ↓
Power BI Executive Dashboard
        ↓
Model Governance & Business Insights
```

The workflow separates **prediction, explanation, customer-value analysis, and retention economics** so that each layer can be evaluated independently.

---

## 📊 Dataset & Prediction Point

The project uses a telecom customer churn case-study dataset containing:

- **69,999 training customers**
- **30,000 test customers**
- **172 raw training columns**
- **171 raw test columns**
- Approximately **10.19% observed churn** in the training population

The prediction framework is designed around a defined **end-of-August 2014 prediction point**.

### Temporal Governance

The project deliberately separates:

- **June + July** → historical behavioral context
- **August** → prediction-point information
- **September** → subsequent outcome period

September outcome information is excluded from model predictors to reduce future-information leakage.

> **Important:** The model estimates churn probability; a probability is not a guaranteed churn outcome.

---

## 🧹 Data Quality & Feature Engineering

The feature-engineering pipeline was designed around reproducibility and temporal eligibility.

### Feature Engineering Scale

```text
170 raw predictors
        ↓
427 engineered candidates
        ↓
169 final frozen model features
```

The engineering process included:

- Structural data-quality validation
- Missingness profiling
- Structural and behavioral missingness treatment
- Development-only imputation
- Multi-month behavioral features
- Recent-vs-baseline behavior
- Temporal change and percentage-change features
- Recency and activity indicators
- Customer-value features
- Correlation and redundancy review
- Predictive stability analysis
- Feature selection

The final **169-feature schema was frozen** before final model development.

Development-only statistics are used for model preprocessing to avoid information leakage from validation data.

---

## 🗄️ SQL Business Analysis

PostgreSQL was used as a supporting business-analysis layer before modeling.

The SQL analysis was organized into **10 business modules**:

1. Data Validation
2. Customer Churn Overview
3. Customer Profile Analysis
4. Usage & Engagement Analysis
5. Recharge & Revenue Analysis
6. Service Behavior Analysis
7. Temporal Behavior Analysis
8. Revenue & Customer-Value Exposure
9. Customer Segmentation
10. Risk & Priority Analysis

The SQL work was used to understand:

- Churn distribution
- Customer characteristics
- Usage and engagement behavior
- Recharge patterns
- Revenue/value exposure
- Segment behavior
- Risk and priority patterns

SQL analysis was kept separate from the machine-learning pipeline so that descriptive business evidence could be evaluated independently.

---

## 🤖 Model Development

Several modeling approaches were evaluated rather than immediately selecting a single algorithm.

### Baseline & Candidate Models

- Dummy classifier
- Logistic Regression
- HistGradientBoosting
- XGBoost

Model evaluation focused on metrics appropriate for an imbalanced churn problem, particularly:

- **PR-AUC**
- **ROC-AUC**
- Precision
- Recall
- F1
- Top-k churn capture
- Lift
- Cross-validation stability

### Final XGBoost Model

The final tuned model was selected after cross-validation and hyperparameter evaluation.

Selected configuration:

- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 6`
- `min_child_weight = 1`
- `subsample = 0.90`
- `colsample_bytree = 0.90`
- `gamma = 0`
- `reg_alpha = 0`
- `reg_lambda = 1`

### Validation Results

| Metric | Result |
|---|---:|
| PR-AUC | **0.7614** |
| ROC-AUC | **0.9502** |
| Precision | **0.7686** |
| Recall | **0.6220** |
| F1 | **0.6876** |
| Top-10% churn capture | **71.04%** |
| Top-10% lift | **7.10×** |

The model was evaluated using a held-out validation population rather than relying only on training performance.

---

## 🎯 Probability Calibration

Raw model probabilities were calibrated before being used for customer-risk interpretation.

The project compared calibration approaches and selected **sigmoid calibration** based on out-of-fold evaluation.

### Calibration Metrics

| Metric | Result |
|---|---:|
| OOF Brier Score | **0.041740** |
| OOF Log Loss | **0.144759** |
| OOF ECE | **0.005859** |
| OOF PR-AUC | **0.776913** |
| OOF ROC-AUC | **0.955720** |

Calibration was applied to the model probabilities before calculating the final customer risk score.

This allows the downstream risk and revenue-exposure layers to work with a probability that has undergone an explicit calibration step rather than treating raw model output as automatically reliable.

## 📈 Model Evaluation & Threshold Optimization

Model selection was based on both predictive performance and business usefulness rather than a single evaluation metric.

Cross-validation showed that the selected XGBoost configuration achieved:

- **OOF PR-AUC: 0.7769**
- Consistent ranking performance across validation folds
- Improved performance over the baseline XGBoost configuration

The final held-out validation evaluation produced:

- **PR-AUC: 0.7614**
- **ROC-AUC: 0.9502**
- **Top-10% churn capture: 71.04%**
- **Top-10% lift: 7.10×**

### Threshold Optimization

A primary probability threshold of **0.10** was selected for the customer-risk framework after evaluating probability thresholds alongside operational-capacity and intervention-cost scenarios.

At this threshold:

- OOF target rate: **17.93%**
- OOF historical churn capture: **88.21%**
- OOF precision: **50.14%**

The threshold is used to structure customer risk classification and should not be interpreted as an automatic intervention rule.

> **Important:** Risk threshold and operational intervention capacity are separate concepts. A business may need to target a smaller population depending on available retention resources.

---

## 🔎 Explainable AI with SHAP

ChurnIQ uses **SHAP (SHapley Additive exPlanations)** to understand the behavioral factors contributing to model predictions.

A reproducible sample of **10,000 customers** was analyzed using the final XGBoost model.

### Top Global Drivers

| Rank | Feature |
|---:|---|
| 1 | `total_activity_8` |
| 2 | `voice_rech_recency_8` |
| 3 | `last_day_rch_amt_8` |
| 4 | `total_ic_mou_8` |
| 5 | `roam_og_mou_8` |
| 6 | `total_activity_pct_change_7_to_8` |
| 7 | `total_ic_mou_min_6_8` |
| 8 | `spl_ic_mou_8` |
| 9 | `aon` |
| 10 | `total_voice_mou_8` |

SHAP analysis was performed at two levels:

**Global explanation**  
→ identifies the behavioral features with the greatest overall contribution to model predictions.

**Individual explanation**  
→ identifies the strongest positive and negative contributors for a specific customer's prediction.

A SHAP additivity validation was also performed to verify that the explanation values were consistent with the model output.

---

## 👥 Customer Risk Intelligence

The calibrated churn probabilities were transformed into a customer-level risk intelligence layer covering the complete **69,999-customer universe**.

### Risk Framework

| Risk Level | Definition |
|---|---|
| Below Primary Threshold | Probability < 0.10 |
| High | 0.10 ≤ Probability < 0.50 |
| Very High | Probability ≥ 0.50 |

### Risk Population

- Customers above primary threshold: **12,492**
- Very High risk: **6,152**
- High risk: **6,340**
- Below primary threshold: **57,507**
- Historical churn captured above threshold: **88.04%**

A normalized **risk score from 0–100** was also created for easier business interpretation.

The risk layer is designed for **decision support and prioritization analysis**; it does not automatically trigger customer actions.

---

## 💰 Customer Value & Revenue Risk

ChurnIQ combines calibrated churn probability with an August customer-value proxy to estimate potential revenue exposure.

### Customer Value Framework

August ARPU is used as a **customer-value proxy**, not booked revenue.

For exposure calculations:

```text
Exposure-Eligible ARPU = max(August ARPU, 0)

Potential Monthly Revenue Exposure
= Calibrated Churn Probability × Exposure-Eligible ARPU

Annualized Scenario Exposure
= Potential Monthly Revenue Exposure × 12
```

### Results

| Metric | Result |
|---|---:|
| Average August ARPU | **₹278.86** |
| Median August ARPU | **₹192.23** |
| Customers with negative August ARPU | **352** |
| Potential monthly exposure | **₹776,208** |
| Annualized scenario exposure | **₹9.31M** |
| Very High risk exposure share | **34.07%** |
| Top-10% exposure share | **77.61%** |

These figures represent **scenario-based customer-value exposure**, not guaranteed revenue loss.

Negative ARPU values are preserved in the underlying customer-value analysis, while negative values are excluded from exposure calculations through an explicit eligibility rule.

---

## 🎯 Retention Prioritization

Risk alone does not determine business priority.

ChurnIQ therefore evaluates targeting cohorts using both:

- **Churn risk**
- **Customer-value / revenue exposure**

Targeting cohorts include:

- Top 1%
- Top 5%
- Top 10%
- Top 20%
- High + Very High risk customers

The cohort analysis examines how potential revenue exposure is concentrated across different targeting populations before evaluating intervention economics.

This creates a decision-support bridge:

```text
Model Risk
    ↓
Customer Value
    ↓
Revenue Exposure
    ↓
Targeting Cohort
    ↓
Retention Economics
```

The purpose is not to declare that every high-risk customer should receive an intervention. Instead, the framework helps identify where further retention analysis may be economically and operationally relevant.

---

## 💼 Retention Economics & Sensitivity Analysis

ChurnIQ evaluates retention economics under **explicit analyst-defined assumptions** rather than presenting a single ROI estimate as a guaranteed business outcome.

The analysis varies:

- Retention success rate
- Intervention cost
- Targeting cohort

### Retention Success Scenarios

Success-rate scenarios include:

**10%, 20%, 30%, 35%, 40%, 50%, 60%, 70%, 80%**

Intervention-cost scenarios include:

**₹100, ₹250, ₹500 per targeted customer**

Targeting cohorts include:

- Top 1%
- Top 5%
- Top 10%
- Top 20%
- High + Very High risk

This produces **135 sensitivity scenarios**.

### Illustrative Retention Economics

Selected scenario results:

| Targeting Cohort | Low Cost | Targeted Cost | High Cost |
|---|---:|---:|---:|
| Top 1% | **752.19% ROI** | **496.53% ROI** | **326.09% ROI** |
| Top 5% | **239.78% ROI** | **137.85% ROI** | **69.89% ROI** |
| Top 10% | **106.54% ROI** | **44.58% ROI** | **3.27% ROI** |
| Top 20% | **17.35% ROI** | **-17.85% ROI** | **-41.32% ROI** |
| High + Very High | **5.94% ROI** | **-25.84% ROI** | **-47.03% ROI** |

> **Important:** These ROI values are modeled scenario outputs based on stated assumptions. They are not historical business results, observed intervention outcomes, or guaranteed future returns.

### Sensitivity Results

Across the 135 sensitivity scenarios:

- **94 / 135 scenarios were economically viable**
- Minimum break-even success rate: **2.35%**
- Maximum break-even success rate: **94.39%**
- Highest modeled ROI: **3,308.75%**
- Sensitivity monotonicity checks: **PASS**

The analysis also categorizes robustness based on the modeled break-even success-rate requirement:

| Break-even Requirement | Interpretation |
|---|---|
| ≤10% | Robust |
| >10% to 50% | Conditional |
| >50% to 80% | Weak |
| >80% or infeasible | Not viable |

These categories are **analytical labels defined within this project** and should not be interpreted as universal business standards.

---

## 🧠 Business Interpretation

The retention-economics layer demonstrates an important decision-support principle:

> **Churn probability alone does not determine whether a customer is economically relevant to target.**

Customer value, potential exposure, operational capacity, intervention cost, and assumed retention effectiveness all affect the modeled economics.

ChurnIQ therefore avoids combining these dimensions into a single arbitrary score.

Instead, it keeps:

**Risk → Value → Exposure → Targeting → Economics**

as separate analytical dimensions that can be evaluated together.

---

## 🖥️ Streamlit AI Application

ChurnIQ includes an interactive **Streamlit application** that brings the analytical pipeline into a customer-level decision-support interface.

### Application Flow

```text
Customer ID
    ↓
Frozen Feature Schema
    ↓
Final XGBoost Model
    ↓
Probability Calibration
    ↓
Risk Classification
    ↓
SHAP Explanation
    ↓
Customer Value
    ↓
Revenue Risk
    ↓
Business Interpretation
```

### Application Sections

**1. Customer Risk**

Displays:

- Calibrated churn probability
- Risk score
- Risk level
- Primary threshold
- Customer risk context

**2. Why This Prediction?**

Provides customer-level SHAP explanations showing the behavioral factors contributing to the prediction.

**3. Customer Value**

Displays:

- August ARPU / customer-value proxy
- Customer-value tier
- Customer-value percentile

**4. Revenue Risk**

Displays:

- Potential monthly revenue exposure
- Annualized scenario exposure
- Revenue-exposure rank
- Exposure position within the customer universe

**5. Business Interpretation**

Connects the model output to customer-risk and value context without automatically prescribing a customer action.

The application uses the **frozen model, calibration artifact, frozen feature schema, and compact deployment artifacts** rather than retraining the model during inference.

---

## 📊 Power BI Executive Dashboard

A supporting **Power BI executive dashboard** provides a higher-level business view of customer risk, customer value, revenue exposure, and retention economics.

The dashboard uses:

- Customer risk and value data
- Targeting cohort analysis
- Retention economics scenarios

Key executive metrics include:

- Total customers
- Churn rate
- Customers above primary threshold
- High-risk population
- Customer-value exposure
- Revenue-risk concentration
- Targeting-cohort context

Power BI serves as the **executive reporting and communication layer**, while Streamlit provides the deeper customer-level AI and explainability experience.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Data processing, feature engineering, ML, explainability |
| **Pandas / NumPy** | Data preparation and analysis |
| **Scikit-learn** | Baselines, preprocessing, calibration, evaluation |
| **XGBoost** | Final churn prediction model |
| **SHAP** | Model explainability |
| **PostgreSQL** | SQL-based business analysis |
| **Streamlit** | Interactive customer-level AI application |
| **Power BI** | Executive dashboard and business communication |
| **Git / GitHub** | Version control and portfolio delivery |

---

## 📁 Project Structure

```text
ChurnIQ/
│
├── app/
│   └── streamlit_app.py
│
├── deployment_data/
│   ├── customer_features.parquet
│   ├── customer_value.parquet
│   └── imputation_medians.csv
│
├── docs/
│   ├── final_business_insights.md
│   ├── limitations.md
│   ├── model_card.md
│   └── monitoring_plan.md
│
├── models/
│   ├── logistic_regression_baseline.joblib
│   ├── tree_based_baseline.joblib
│   ├── xgb_probability_calibrator.joblib
│   ├── xgboost_baseline.joblib
│   ├── xgboost_final_tuned.ubj
│   └── xgboost_final_tuned_metadata.json
│
├── powerbi/
│   └── ChurnIQ_Executive_Dashboard.pbix
│
├── reports/
│   ├── Phase 1–13 analysis reports
│   └── shap/
│       ├── customer_shap_explanations.csv
│       ├── global_shap_feature_importance.csv
│       ├── shap_feature_direction_analysis.csv
│       ├── shap_global_importance_bar.png
│       ├── shap_sample_registry.csv
│       ├── shap_summary_beeswarm.png
│       └── top_risk_explanation_profiles.csv
│
├── scripts/
│   └── load_train_to_postgres.py
│
├── sql/
│   ├── 01_data_validation.sql
│   ├── 02_customer_churn_overview.sql
│   ├── 03_customer_profile_analysis.sql
│   ├── 04_usage_engagement_analysis.sql
│   ├── 05_recharge_revenue_analysis.sql
│   ├── 06_service_behavior_analysis.sql
│   ├── 07_temporal_behavior_analysis.sql
│   ├── 08_revenue_exposure_analysis.sql
│   ├── 09_customer_segmentation_analysis.sql
│   └── 10_risk_priority_analysis.sql
│
├── src/
│   ├── build_final_model.py
│   ├── churniq_customer_value.py
│   ├── churniq_explainability.py
│   ├── churniq_inference.py
│   ├── cross_validation_stability.py
│   ├── customer_risk_intelligence.py
│   ├── customer_value_revenue_risk.py
│   ├── dummy_baseline.py
│   ├── feature_engineering.py
│   ├── final_model_selection.py
│   ├── heldout_validation.py
│   ├── hyperparameter_tuning.py
│   ├── logistic_regression_baseline.py
│   ├── model_development.py
│   ├── probability_calibration.py
│   ├── retention_prioritization_roi.py
│   ├── shap_customer_explanations.py
│   ├── shap_direction_analysis.py
│   ├── shap_explainability.py
│   ├── shap_top_risk_profiles.py
│   ├── threshold_optimization.py
│   ├── tree_based_baseline.py
│   └── xgboost_baseline.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** Raw and processed datasets are intentionally excluded from version control because of their size. The repository contains the compact deployment artifacts required by the Streamlit application.

---

## 📚 Documentation

ChurnIQ includes dedicated documentation covering model governance, limitations, monitoring, and final business interpretation.

| Document | Purpose |
|---|---|
| `docs/model_card.md` | Model purpose, methodology, evaluation, calibration, and governance |
| `docs/limitations.md` | Data, modeling, business, and interpretation limitations |
| `docs/monitoring_plan.md` | Data, model, calibration, and business monitoring framework |
| `docs/final_business_insights.md` | Evidence-based findings and business recommendations |

The `reports/` directory contains the supporting analysis, evaluation results, and quality-gate outputs produced throughout the project.

---

## 🔐 Data, Governance & Responsible Use

ChurnIQ is designed as an **analytical decision-support framework**, not an automated customer-decision system.

Key considerations:

- Churn probabilities are estimates, not guaranteed outcomes.
- Revenue exposure is scenario-based and uses August ARPU as a customer-value proxy.
- Retention economics depends on explicit assumptions such as intervention cost and success rate.
- September outcome information is excluded from prediction features.
- Model outputs should be evaluated alongside operational and business context.
- The system does not automatically determine which customers should receive an intervention.
- Real-world deployment would require monitoring for data drift, model performance changes, calibration changes, and business-outcome changes.
- Modeled ROI and revenue-exposure figures should not be interpreted as guaranteed financial results.

---

## 🚀 Running ChurnIQ Locally

### 1. Clone the repository

```bash
git clone https://github.com/shrutikadu23/churniq-customer-churn-revenue-risk.git
cd churniq-customer-churn-revenue-risk
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

The application provides customer-level churn risk, SHAP explanations, customer-value analysis, revenue-risk context, and business interpretation.

> **Deployment note:** The repository does not require the original large raw/processed datasets for the Streamlit inference workflow. The application uses the compact deployment artifacts included in `deployment_data/`.

---

## 🧪 Reproducibility

The final inference workflow uses:

- Frozen **169-feature schema**
- Final tuned XGBoost model
- Saved sigmoid probability calibrator
- Compact customer feature artifact
- Deployment imputation medians
- Customer-value deployment artifact

The application does **not retrain or recalibrate the model during inference**.

This separation keeps the deployed application focused on reproducible scoring, explainability, customer-value analysis, and business decision support.

---

## 📌 Key Takeaways

ChurnIQ demonstrates an end-to-end approach to customer retention analytics by connecting machine learning outputs with business-value analysis.

### What the Project Demonstrates

- **Business problem framing** before model development
- **Data validation and quality controls** throughout the analytical workflow
- **SQL-based business analysis** across customer, usage, recharge, revenue, service, and risk dimensions
- **Temporal feature engineering** that respects the defined prediction point
- **Model comparison and validation** rather than relying on a single algorithm
- **Probability calibration** to make churn probabilities more useful for decision support
- **SHAP explainability** for global and customer-level prediction interpretation
- **Customer risk intelligence** at the individual-customer level
- **Customer-value and revenue-exposure analysis**
- **Retention targeting and cohort analysis**
- **Scenario-based retention economics and sensitivity analysis**
- **Streamlit customer-level AI application**
- **Power BI executive communication layer**
- **Model governance, limitations, and monitoring considerations**

### Core Analytical Principle

The project deliberately keeps the major decision dimensions separate:

```text
Risk
  ↓
Customer Value
  ↓
Revenue Exposure
  ↓
Targeting Cohort
  ↓
Retention Economics
  ↓
Business Decision Support
```

This avoids reducing customer risk, customer value, and intervention economics into a single arbitrary score.

### Final Perspective

ChurnIQ demonstrates how analytical, machine-learning, and business-intelligence techniques can be combined to move from raw customer data toward structured, evidence-based decision support.

The project brings together:

**SQL + Python + Machine Learning + Explainable AI + Power BI + Streamlit**

with a strong focus on connecting technical outputs to business questions.

---

## ⚠️ Limitations & Responsible Use

ChurnIQ has several important limitations that should be considered before applying the framework to a real production environment.

### Data Limitations

- The project uses a historical telecom churn case-study dataset.
- Available variables do not represent every real-world factor that can influence churn.
- The dataset does not contain direct fields for some business concepts such as contracts, payment methods, customer satisfaction, or retention interventions.
- Customer-value analysis uses **August ARPU as a customer-value proxy**, not accounting revenue or customer lifetime value.

### Modeling Limitations

- Model performance is based on the available historical dataset and defined validation framework.
- Calibrated probabilities are estimates and may change when the underlying customer population changes.
- A model that performs well on historical data may not maintain the same performance after deployment.
- Changes in feature availability or data quality can affect predictions.

### Business Limitations

- Retention success rates and intervention costs used in the ROI analysis are **analyst-defined scenarios**, not observed intervention results.
- Modeled ROI does not establish that a retention campaign will generate the same return in practice.
- Revenue exposure represents potential scenario exposure, not guaranteed revenue loss.
- Operational capacity and campaign effectiveness would need to be validated with real business data.

### Responsible Use

ChurnIQ should be used as **decision support**, not as an automatic decision-maker.

Before real-world deployment, the framework should be evaluated using:

- New production outcomes
- Model performance monitoring
- Probability calibration monitoring
- Data-drift monitoring
- Business KPI monitoring
- Actual retention intervention results
- Appropriate human review and governance

---

## 🔗 Project Resources

- **GitHub Repository:**  
  https://github.com/shrutikadu23/churniq-customer-churn-revenue-risk

- **Streamlit Application:**  
  Run locally using the instructions above.

- **Power BI Dashboard:**  
  `powerbi/ChurnIQ_Executive_Dashboard.pbix`

- **Project Documentation:**  
  `docs/`

---

## 👩‍💻 Author

**Shruti Kadu**

Computer Science Student | Business Analytics & AI/ML

Interested in:

- Business Analysis
- Data Analytics
- Customer & Product Analytics
- AI/ML Applications
- Decision-Support Systems

**GitHub:**  
https://github.com/shrutikadu23

**LinkedIn:**  
https://www.linkedin.com/in/shruti-kadu-101b05371/

---

## ⭐ Explore the Project

Feel free to explore the repository and review the analytical workflow, model development, explainability, business analysis, retention economics, and Streamlit application.

---