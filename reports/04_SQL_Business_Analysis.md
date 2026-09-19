# ChurnIQ — SQL Business Analysis & Analytical Query Development

## 1. Phase Objective

The objective of the SQL analysis phase is to transform the validated telecom churn dataset into business-oriented analytical insights using PostgreSQL.

SQL will serve as a structured analytical layer for understanding:

- Customer churn behavior
- Customer characteristics associated with churn
- Usage and engagement patterns
- Recharge and revenue behavior
- Customer value
- Revenue exposure
- High-risk and high-value customer groups
- Retention prioritization opportunities

The purpose is not to demonstrate SQL syntax alone. Each analysis must answer a meaningful business question and contribute to the overall ChurnIQ decision framework:

**Customer → Churn → Revenue → Risk → Priority → Action**

---

## 2. Business Role of SQL in ChurnIQ

SQL will complement the Python-based machine-learning workflow rather than replace it.

The project will use:

**SQL**
→ Data validation, business analysis, aggregation, segmentation, analytical views, and reusable business metrics

**Python**
→ Data preprocessing, feature engineering, machine learning, model evaluation, calibration, threshold optimization, and explainability

**Power BI**
→ Executive reporting, KPI monitoring, segmentation, and business visualization

**Streamlit**
→ Interactive customer-level churn prediction and risk interpretation

This separation keeps each technology focused on the part of the analytical workflow where it provides the most value.

---

## 3. SQL Analytical Architecture

The SQL layer will follow a structured analytical flow:

```text
Raw CSV Files
     ↓
PostgreSQL Staging
     ↓
Structural & Data Validation
     ↓
Validated Analytical Dataset
     ↓
Reusable Analytical Views
     ↓
Business Analysis Queries
     ↓
Business Findings & Metrics
     ↓
Python / ML Feature Investigation
     ↓
Power BI / Streamlit Outputs

The raw dataset will remain unchanged.

Transformations required for SQL analysis will be performed within the PostgreSQL analytical layer so that the original source remains reproducible and traceable.

4. SQL Data Model

The primary analytical entity is the customer.

The initial customer-level analytical dataset will contain:

Customer identifier
Customer characteristics
Tenure information
Monthly revenue and ARPU measures
Recharge behavior
Voice usage
Data usage
Service adoption indicators
Relevant date information
Churn target for training analysis
Identifier

id will be retained as the customer identifier for:

Customer-level analysis
Analytical joins
Risk outputs
Tracking
Final customer-level reporting

id will not be used as a predictive feature.

Target

churn_probability is the available training target:

0 → Retained
1 → Churned

The target will be used for supervised business analysis on the training dataset.

5. Training and Test Data Boundary

The SQL architecture will maintain a strict separation between training and test data.

TRAIN
 ├── Data validation
 ├── Business analysis
 ├── Exploratory SQL
 ├── Segment analysis
 └── Model-development support

TEST
 └── Reserved for final competition-compatible prediction

The test dataset must not be used to:

Select predictive features
Tune models
Optimize thresholds
Calibrate probabilities
Make model-development decisions
Estimate model performance during development

This boundary will be maintained throughout the project.

6. Primary Business Stakeholder
Primary Stakeholder

Customer Retention / Business Manager

The stakeholder needs to understand:

Which customers are churning?
Which customer groups have elevated churn?
What behaviors are associated with churn?
Which customers represent greater revenue exposure?
Where should retention resources be focused?
Which customers may warrant different retention approaches?
7. Core SQL Business Questions

The SQL analysis will answer the following business questions.

7.1 Customer & Churn Overview
How many customers are in the dataset?
What is the overall churn rate?
How many customers are retained versus churned?
What proportion of the customer base is represented by churned customers?
7.2 Customer Characteristics
How does churn vary with customer tenure?
Which tenure groups have higher churn rates?
Are newer or longer-tenured customers more exposed to churn?
How does churn vary across relevant customer characteristics available in the dataset?
7.3 Revenue & ARPU Analysis
How does ARPU differ between retained and churned customers?
Which ARPU ranges show higher churn?
How does recharge value differ between retained and churned customers?
Are customers with lower recent revenue behavior more likely to churn?
Which customer groups combine higher revenue contribution with elevated churn?
7.4 Usage & Engagement Analysis
How does voice usage differ between retained and churned customers?
How does incoming and outgoing usage relate to churn patterns?
How does data usage differ between retained and churned customers?
Do customers with lower recent engagement show higher churn?
Which usage indicators show the strongest differences between retained and churned customers?
7.5 Recharge & Service Behavior
How does recharge frequency differ between retained and churned customers?
How does recharge amount differ across churn groups?
How does recent recharge behavior vary by churn status?
Does service adoption differ between retained and churned customers?
Does missing data-service information show meaningful differences across churn groups?

Missingness-based findings will be treated as associations rather than causal explanations.

7.6 Behavioral Change
How do customer metrics change from June to July to August?
Which customers show declining revenue behavior?
Which customers show declining usage?
Which customers show declining recharge activity?
Are negative behavioral trends associated with higher observed churn?

Temporal analyses will only be used for prediction-related conclusions after prediction-point and leakage validation.

7.7 Revenue Exposure
How much customer revenue is associated with churned customers?
Which customer groups contribute the greatest revenue exposure?
Which high-value customer groups also show elevated churn?
How does revenue exposure differ across risk-relevant customer segments?

Revenue exposure will be clearly distinguished from guaranteed future revenue loss.

7.8 Customer Prioritization
Which customer groups combine higher churn risk with higher customer value?
How can customers be segmented using churn risk and customer value?
Which groups should receive greater retention attention?

The final prioritization framework will use:

Risk × Customer Value

rather than treating churn risk alone as the definition of business priority.

Model-derived risk will only be introduced after the machine-learning workflow has been developed and validated.

8. Business Metric Definitions

To maintain consistency across SQL, Python, Power BI, and Streamlit, important metrics must have documented definitions.

Churn Rate
Churn Rate =
Churned Customers / Total Customers
Retention Rate
Retention Rate =
Retained Customers / Total Customers
Average ARPU

The average value of the selected ARPU measure across the relevant customer population.

The specific month or aggregation period must always be stated.

Recharge Value

The selected recharge amount measure for the relevant analysis period.

Customer Value

Customer value will be defined using revenue-related measures available in the dataset.

The final customer-value definition will be established after validating the temporal structure and business meaning of the revenue variables.

Revenue Exposure

Revenue exposure represents revenue associated with customers who are churned or identified as being at elevated risk, depending on the analysis stage.

It is an exposure measure, not a guaranteed future revenue loss.

Risk-Weighted Revenue

Where appropriate, model-derived churn probability may later be combined with customer revenue to estimate risk-weighted revenue exposure.

This metric will only be introduced after model probabilities have been validated and calibrated.

9. SQL Analytical Layers

The SQL work will be organized into progressive analytical layers.

Layer 1 — Data Validation

Validate:

Row counts
Column availability
Unique customer IDs
Target values
Missingness
Basic numerical ranges
Train/test separation
Layer 2 — Descriptive Churn Analysis

Analyze:

Customer counts
Churn counts
Churn rate
Retention rate
Churn distribution
Layer 3 — Customer Behavior Analysis

Analyze:

Tenure
ARPU
Recharge behavior
Voice usage
Data usage
Service adoption
Monthly behavior
Layer 4 — Segment Analysis

Analyze churn across meaningful customer segments created from available variables.

Examples may include:

Tenure bands
ARPU bands
Recharge behavior groups
Usage groups
Engagement groups

Segments will only be created when they have a clear business interpretation.

Layer 5 — Revenue Analysis

Analyze:

Customer revenue
Revenue contribution
Revenue associated with churn
High-value customer groups
Revenue exposure
Layer 6 — Risk & Priority Analysis

Combine:

Churn outcome during descriptive analysis
Model-derived churn risk after ML validation
Customer value
Revenue exposure

to support retention prioritization.

10. Planned SQL Deliverables

The SQL phase will produce reusable scripts rather than one large SQL file.

Planned structure:

sql/
├── 01_data_validation.sql
├── 02_customer_churn_overview.sql
├── 03_customer_profile_analysis.sql
├── 04_usage_engagement_analysis.sql
├── 05_recharge_revenue_analysis.sql
├── 06_service_behavior_analysis.sql
├── 07_temporal_behavior_analysis.sql
├── 08_revenue_exposure_analysis.sql
├── 09_customer_segmentation.sql
└── 10_risk_priority_analysis.sql

The exact scripts may be adjusted if the validated data structure makes a different organization more appropriate.

11. Planned Analytical Views

Where repeated calculations or customer-level summaries are required, reusable PostgreSQL views may be created.

Potential views include:

vw_customer_churn_summary
vw_customer_revenue_summary
vw_churn_segment_analysis
vw_revenue_exposure

Views will be created only where they improve:

Reusability
Readability
Consistency
Query maintainability
Downstream analysis

They will not be created merely to increase the apparent technical complexity of the project.

12. SQL Techniques to Demonstrate

The analysis will intentionally demonstrate practical analytical SQL skills.

Core SQL
SELECT
WHERE
GROUP BY
ORDER BY
CASE
HAVING
DISTINCT
Aggregation
COUNT
SUM
AVG
MIN
MAX
Conditional aggregation
Analytical SQL
Common Table Expressions (WITH)
Subqueries
Window functions
Ranking
Percentages
Comparative calculations where business-relevant
Data Handling
NULL handling
Date conversion
Conditional logic
Numeric bucketing
Data validation queries
PostgreSQL

Where useful, PostgreSQL-specific functionality may be used when it improves readability or analytical capability.

SQL complexity will not be added merely to make the project appear advanced.

13. SQL-to-Business Mapping

Every important query should have a clear connection between the technical calculation and the business question.

SQL Analysis	Business Meaning
Churn count	Size of churned customer population
Churn rate	Overall customer retention health
Average ARPU by churn status	Revenue-value difference between groups
Recharge frequency by churn status	Engagement / payment behavior
Usage by churn status	Customer activity differences
Tenure-band churn rate	Customer lifecycle risk pattern
Revenue associated with churn	Potential revenue exposure
High-value churned customers	Retention priority candidates
Risk × value segmentation	Business prioritization

This ensures that SQL outputs can later feed the broader ChurnIQ decision framework.

14. SQL-to-Python Handoff

SQL findings will be used to inform, not automatically determine, the machine-learning workflow.

The handoff will follow:

SQL Business Pattern
        ↓
Business Interpretation
        ↓
Candidate Feature / Segment
        ↓
Leakage & Prediction-Time Validation
        ↓
Python Feature Engineering
        ↓
Predictive Evaluation
        ↓
Final Feature Decision

A variable showing a strong churn association in SQL is not automatically a good predictive feature.

Python-based modeling will independently evaluate:

Predictive usefulness
Leakage safety
Stability
Redundancy
Interpretability
Performance on unseen validation data

This distinction prevents descriptive relationships from being mistaken for predictive evidence.

15. Query Validation & Reconciliation

Important SQL outputs will be validated before being used in downstream analysis.

Validation may include:

Row-count reconciliation
Customer-count reconciliation
Churn-count reconciliation
Null handling checks
Duplicate checks
Expected range checks
Cross-checking selected results against Python/EDA calculations
Verification of metric definitions

For example, the SQL churn count and churn rate should reconcile with the validated dataset totals.

This creates a controlled analytical chain:

Source → SQL Calculation → Validation → Business Insight

16. SQL Analysis Output Standard

Each major SQL analysis should produce:

1. Business Question

What decision or question is being addressed?

2. SQL Query

The reproducible SQL implementation.

3. Result

The important analytical output.

4. Interpretation

What does the result mean from a business perspective?

5. Modeling Relevance

Does the finding suggest:

A potential feature?
A potential segmentation?
A validation requirement?
A possible leakage concern?
A business KPI?
6. Limitation

What should not be concluded from the result?

This structure prevents SQL analysis from becoming a collection of disconnected queries.

17. Important Analytical Boundaries
17.1 Association Is Not Causation

SQL comparisons can identify patterns and associations.

For example:

Customers with lower recent ARPU may show higher observed churn.

This does not establish that low ARPU causes churn.

17.2 Revenue Exposure Is Not Guaranteed Loss

Revenue associated with churned customers can be used as an exposure measure.

It should not automatically be presented as guaranteed future revenue loss.

Future revenue protection will require explicit assumptions and scenario analysis.

17.3 Prediction Is Not Causation

Later model explanations may identify variables associated with predicted churn risk.

Neither SQL relationships nor SHAP importance should be interpreted as proof that changing a variable will prevent churn.

17.4 Temporal Analysis Requires Validation

The dataset contains monthly observations.

However, SQL analysis will not automatically assume that every monthly variable is valid at the prediction point.

Before using a variable for predictive modeling, ChurnIQ must establish:

Prediction Point → Available Information → Prediction Horizon → Future Outcome

17.5 Test Data Must Remain Isolated

The competition test dataset must not influence model-development decisions.

It remains reserved for the appropriate final prediction stage.

18. SQL Quality Principles

All SQL scripts should follow these principles:

Clear and readable naming
Logical query structure
Reusable CTEs where appropriate
Consistent calculations
Explicit handling of missing values
No unnecessary complexity
Business-oriented comments
Reproducible outputs
No silent assumptions
No target leakage
No test-set contamination
Consistent metric definitions
Clear analytical purpose
19. Connection to the ChurnIQ Pipeline

The SQL phase fits into the larger analytical workflow:

Raw Dataset
     ↓
Data Quality & Validation
     ↓
SQL Business Analysis
     ↓
EDA
     ↓
Feature Engineering
     ↓
Baseline Model
     ↓
Model Development
     ↓
Model Evaluation
     ↓
Calibration & Threshold Optimization
     ↓
SHAP Explainability
     ↓
Customer Risk Intelligence
     ↓
Revenue Exposure
     ↓
Retention Prioritization
     ↓
Power BI + Streamlit

SQL therefore acts as an important bridge between data understanding and predictive decision intelligence.

20. Definition of Done

The SQL analysis phase will be considered complete when:

 PostgreSQL environment and data structure are established
 Training data is loaded into the analytical environment
 Training/test boundaries are enforced
 Data validation queries are created
 Core business metrics are consistently defined
 Overall churn analysis is completed
 Customer characteristic analysis is completed
 Usage and engagement analysis is completed
 Recharge and revenue analysis is completed
 Service behavior analysis is completed
 Relevant temporal analysis is completed
 Revenue exposure analysis is completed
 Meaningful customer segments are identified
 Reusable analytical views are created where justified
 Key SQL findings are documented
 Business interpretation is documented
 Modeling-relevant findings are identified
 Leakage-sensitive findings are flagged
 Important outputs are validated and reconciled
 SQL scripts are organized and reproducible
 No test-set contamination occurs
 SQL findings are ready to inform EDA and feature engineering
21. Phase 4 Completion Statement

The SQL phase is designed to establish a strong business understanding of customer churn before machine-learning development begins.

The final objective is not simply to answer:

"Who churned?"

It is to understand:

"Who churned, what behavioral and revenue patterns distinguish them, where is the greatest revenue exposure, and how can these insights support better churn-risk decisions?"

## Final Status

SQL Business Analysis: COMPLETE

PostgreSQL analytical environment: CONFIGURED

Raw training data import: COMPLETE

Data validation queries: COMPLETE

Customer churn analysis: COMPLETE

Customer profile and engagement analysis: COMPLETE

Recharge and revenue analysis: COMPLETE

Temporal behavior analysis: COMPLETE

Revenue exposure analysis: COMPLETE

Customer segmentation analysis: COMPLETE

Risk and priority analysis: COMPLETE

SQL-to-Python analytical handoff: COMPLETE

Validation/test data separation: PASS

SQL analysis is complete and approved as the supporting business-analysis layer for ChurnIQ.