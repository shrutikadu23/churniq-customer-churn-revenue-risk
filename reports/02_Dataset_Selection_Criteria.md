# ChurnIQ — Dataset Selection Criteria

## 1. Purpose

The purpose of this document is to define the requirements and evaluation criteria that will be used to select the dataset for ChurnIQ.

The selected dataset must support not only customer churn prediction, but the broader business objective of:

**Predict → Explain → Quantify Risk → Prioritize → Recommend Action**

Therefore, dataset selection will be based on:

* Business relevance
* Analytical quality
* Modeling suitability
* Revenue-risk suitability
* Temporal realism
* Data provenance
* Portfolio value

Dataset popularity alone will not be considered a sufficient reason for selection.

---

## 2. Core Dataset Requirements

The selected dataset should satisfy the following requirements wherever realistically available.

### 2.1 Customer-Level Granularity

The dataset should support analysis at the individual customer level.

This is required to:

* Calculate customer-level churn risk
* Generate individual churn probabilities
* Explain individual predictions
* Assess customer value
* Quantify customer-level revenue exposure
* Prioritize individual customers

---

### 2.2 Clearly Defined Churn Target

The dataset must contain a sufficiently clear churn outcome that can be used as the machine learning target.

The target should allow customers to be meaningfully classified as:

* Churned
* Retained / Not Churned

The target definition must be documented and checked for:

* Ambiguity
* Inconsistent labeling
* Hidden leakage
* Post-outcome information

---

### 2.3 Revenue / Customer Value Information

The dataset should contain a meaningful customer revenue or value measure.

Preferred fields may include:

* Monthly recurring revenue
* Monthly charges
* Subscription revenue
* Customer revenue
* Annualized recurring revenue
* Customer lifetime value

Revenue information is important for:

* Revenue exposure
* Risk-weighted revenue exposure
* Customer value
* Retention prioritization
* Scenario-based ROI analysis

If the selected dataset does not contain a true recurring-revenue measure, this limitation will be explicitly documented.

**Artificially creating an MRR or revenue measure from unrelated fields will not be presented as actual business revenue.**

---

### 2.4 Customer Tenure

The dataset should contain customer tenure or sufficient information to derive it.

Tenure can support:

* Customer lifecycle analysis
* Early-life versus long-tenure risk analysis
* Feature engineering
* Customer segmentation
* Business interpretation

---

### 2.5 Customer & Service Attributes

The dataset should contain meaningful explanatory variables that allow churn risk to be analyzed across relevant business dimensions.

Potential categories include:

* Customer characteristics
* Contract information
* Service subscriptions
* Product or plan information
* Payment method
* Billing characteristics
* Usage or engagement indicators
* Support or interaction indicators
* Customer tenure
* Revenue-related attributes

The exact feature set will depend on the selected dataset.

---

## 3. Temporal Structure & Prediction Realism

Temporal structure is an important selection criterion because ChurnIQ aims to identify churn risk **before the outcome occurs**.

The dataset will therefore be classified as either:

### A. Temporally Structured Dataset

Contains sufficient dates, historical observations, snapshots, or event records to establish:

**Prediction Point → Future Prediction Horizon → Churn Outcome**

This may support:

* Time-based feature engineering
* Realistic prediction windows
* Time-aware train/test validation
* More defensible future-risk framing

### B. Cross-Sectional / Snapshot Dataset

Contains one customer-level record representing a particular observation period without sufficient historical timestamps.

Such a dataset may still support:

* Churn classification
* Customer segmentation
* Risk modeling
* Revenue-risk analysis
* Customer prioritization

However, it will **not** be presented as a genuine time-series forecasting problem.

If the final dataset is cross-sectional, the project will explicitly document this limitation and define the prediction framing accordingly.

---

## 4. Prediction-Time Feature Availability

Every candidate dataset will be evaluated based on whether its features would realistically be available at the prediction point.

A feature may be statistically useful but still unsuitable if it is only known after the customer has already churned.

Potentially problematic fields include:

* Cancellation information
* Post-churn status
* Future charges
* Future service status
* Final account information
* Variables generated after the outcome

The final modeling dataset will contain only features that are legitimately available at or before the defined prediction point.

This principle takes priority over maximizing model performance.

---

## 5. Data Leakage Safety

Potential leakage sources will be investigated, including:

* Direct churn indicators
* Post-outcome variables
* Future information
* Derived fields containing target information
* Features calculated using the complete customer lifecycle
* Variables created after cancellation

Leakage checks will be documented before model development.

---

## 6. Dataset Size & Statistical Suitability

The dataset should contain enough observations to support:

* Exploratory analysis
* Train/test evaluation
* Cross-validation where appropriate
* Model comparison
* Customer-level risk analysis
* Subgroup analysis where feasible

Dataset size will be evaluated relative to:

* Number of features
* Number of churn events
* Target balance
* Modeling complexity

No arbitrary minimum row count will be used.

---

## 7. Target Class Balance

The distribution of churned and retained customers will be measured.

If meaningful class imbalance exists, the project will account for it through appropriate methods such as:

* Precision
* Recall
* F1-score
* PR-AUC
* Stratified validation where appropriate
* Threshold analysis
* Lift and gains analysis
* Other justified techniques

Accuracy alone will not be treated as sufficient evidence of model quality.

---

## 8. Business Analysis Suitability

The dataset should support meaningful questions such as:

* Which customer groups have higher churn rates?
* Which contracts or services are associated with higher churn?
* How does churn vary with tenure?
* Which customers represent greater revenue exposure?
* Which high-value customers are also at elevated risk?
* How concentrated is churn within the highest-risk customers?
* Which customer groups should receive greater retention attention?

The dataset should therefore support both:

**Customer-level analysis + Segment-level analysis**

---

## 9. SQL Suitability

SQL should perform genuine business analysis rather than exist only as a technical demonstration.

Where appropriate, SQL should support:

* Customer-level analysis
* Churn-rate calculations
* Segment comparisons
* Revenue analysis
* Customer-value analysis
* Aggregations
* Ranking
* Filtering
* Risk-group analysis
* Business-oriented segmentation

If multiple related tables are available, their relationships should be sufficiently clear to support meaningful joins.

---

## 10. Machine Learning Suitability

The dataset should support:

* Binary classification
* Dummy Classifier baseline
* Multiple candidate models
* Cross-validation where appropriate
* Controlled hyperparameter tuning
* Probability prediction
* Probability calibration
* Threshold optimization
* Model comparison
* Lift and gains analysis

Model selection will consider:

**Predictive performance + probability quality + interpretability + business usefulness**

rather than a single metric such as ROC-AUC.

---

## 11. Explainability Suitability

The dataset should contain features that can be interpreted by business stakeholders.

The final solution should support:

### Global Explainability

Understanding which features are most influential in model predictions.

### Individual Explainability

Understanding why a particular customer received a particular churn-risk estimate.

SHAP or another appropriate explainability method may be used.

However:

**Feature importance or SHAP values will be interpreted as model associations, not proof of causation.**

---

## 12. Revenue-Risk Suitability

The dataset should ideally support a defensible connection between churn risk and customer value.

This enables analysis of:

### Revenue Exposure

Revenue associated with customers identified as being at risk.

### Risk-Weighted Revenue Exposure

An estimated risk measure such as:

**Predicted Churn Probability × Customer Recurring Revenue**

This represents potential exposure under the model assumptions.

It will not be presented as guaranteed future revenue loss.

---

## 13. Retention Prioritization Suitability

The dataset should support a decision framework combining:

**Churn Risk + Customer Value + Revenue Exposure**

This allows customers to be grouped into categories such as:

| Risk  | Customer Value | Business Interpretation         |
| ----- | -------------- | ------------------------------- |
| High  | High           | Highest retention priority      |
| High  | Lower          | Monitor / targeted intervention |
| Lower | High           | Protect valuable relationship   |
| Lower | Lower          | Lower immediate priority        |

The purpose is to avoid treating every predicted churner as equally important.

---

## 14. Retention ROI Suitability

Where the data supports it, the project may estimate hypothetical retention outcomes using clearly stated assumptions.

Potential scenarios:

* Conservative
* Expected
* Optimistic

Scenario calculations may consider:

* Number of customers targeted
* Estimated intervention cost
* Assumed retention success rate
* Revenue associated with targeted customers
* Potential revenue protected

These will be explicitly labeled as **scenario estimates**, not historical facts or guaranteed outcomes.

---

## 15. Power BI Suitability

The dataset should support meaningful executive-level analysis such as:

* Customer KPIs
* Churn KPIs
* Churn by segment
* Churn by contract/service
* Revenue exposure
* Risk-weighted revenue
* Customer risk distribution
* Risk × Value analysis
* Priority customer groups
* Churn capture
* Revenue exposure captured

Time-based visuals will only be included if the dataset supports valid temporal analysis.

---

## 16. Streamlit Suitability

The dataset should contain a realistic set of prediction-time features that can be supplied to the Streamlit application.

The application must:

* Use the same preprocessing logic as model training
* Accept only prediction-time features
* Apply the selected model consistently
* Return a churn probability
* Assign an appropriate risk category
* Provide relevant explanations where supported
* Avoid exposing post-outcome information

---

## 17. Data Quality Criteria

Candidate datasets will be evaluated for:

* Missing values
* Duplicate records
* Invalid values
* Inconsistent categories
* Incorrect data types
* Outliers
* Target quality
* Cardinality
* Logical inconsistencies
* Temporal inconsistencies
* Potential leakage
* Sampling limitations

Data-quality problems will not automatically disqualify a dataset.

The key question will be:

**Can the issue be identified, handled responsibly, and documented without compromising analytical validity?**

---

## 18. Data Provenance & Licensing

Each candidate dataset must be evaluated for:

* Original publisher
* Source platform
* Dataset documentation
* Dataset version
* Access date
* License or usage terms
* Redistribution restrictions
* Citation requirements
* Suitability for portfolio/GitHub use

A dataset with unclear provenance or questionable redistribution rights may be rejected even if its technical characteristics are strong.

---

## 19. Portfolio Quality Criteria

The selected dataset should allow ChurnIQ to demonstrate a coherent combination of:

* Business analysis
* Data cleaning
* SQL
* Python
* Exploratory data analysis
* Feature engineering
* Machine learning
* Model evaluation
* Probability calibration
* Threshold optimization
* Explainable AI
* Revenue-risk analysis
* Customer prioritization
* Power BI
* Streamlit
* Business communication

Technology will not be added merely to make the project appear more advanced.

---

## 20. Dataset Modification Policy

The project will follow these rules:

1. Real-world fields will not be artificially invented simply to satisfy project requirements.
2. Derived features will be clearly documented.
3. Synthetic or assumed values, if ever required, will be explicitly labeled.
4. Assumptions will never be presented as observed business facts.
5. Any dataset augmentation will be justified and documented.
6. Model performance will not be artificially improved through leakage or unrealistic preprocessing.

**Portfolio credibility takes priority over artificially impressive results.**

---

## 21. Dataset Selection Principles

The final selection will follow these principles:

1. **Business relevance over popularity**
2. **Data quality over dataset size alone**
3. **Realistic prediction over artificial performance**
4. **Meaningful revenue analysis over assumed financial metrics**
5. **Temporal realism over forced forecasting claims**
6. **Prediction-time availability over predictive convenience**
7. **Interpretability over unnecessary model complexity**
8. **Analytical depth over feature quantity**
9. **Reproducibility over one-off results**
10. **Business usefulness over technology collecting**

---

## 22. Dataset Evaluation Scorecard

Candidate datasets will be evaluated using the following framework:

| Criterion                    | Weight | Evaluation Focus                                |
| ---------------------------- | -----: | ----------------------------------------------- |
| Business Relevance           |    14% | Realistic subscription/churn business scenario  |
| Churn Target Quality         |    14% | Clear and reliable churn definition             |
| Revenue / Customer Value     |    13% | Ability to quantify customer value and exposure |
| Temporal Structure           |    13% | Prediction point and horizon feasibility        |
| Prediction-Time Availability |    10% | Features realistically available before outcome |
| Feature Richness             |     8% | Meaningful explanatory variables                |
| Data Quality                 |     8% | Completeness, consistency, usability            |
| Data Provenance & Licensing  |     6% | Source credibility and portfolio usage          |
| ML Suitability               |     5% | Classification and evaluation suitability       |
| SQL Suitability              |     3% | Meaningful business analysis                    |
| Explainability               |     2% | Business-interpretable features                 |
| BI / Application Suitability |     2% | Power BI and Streamlit usefulness               |
| Portfolio Value              |     2% | Professional demonstration                      |

**Total: 100%**

The weighting intentionally gives greater importance to business realism and analytical validity than to technical complexity.

---

## 23. Selection Decision Rule

Candidate datasets will be scored after reviewing:

* Dataset structure
* Documentation
* Data dictionary
* Target definition
* Feature availability
* Temporal structure
* Revenue semantics
* Data quality
* Provenance
* Licensing
* Modeling suitability

The highest-scoring dataset will normally be selected.

However:

**A dataset with a slightly lower numerical score may be selected if it provides substantially stronger temporal realism, revenue analysis, data quality, provenance, or business relevance that the numerical score does not fully capture.**

The final decision and reasoning will be documented before modeling begins.

---

## 24. Dataset Selection Outcome

Once selected, the dataset documentation will record:

* Dataset name
* Original source
* Publisher
* Dataset version
* Access date
* License / usage terms
* Number of records
* Number of features
* Target definition
* Customer-level granularity
* Revenue fields
* Time-related fields
* Key feature categories
* Data-quality considerations
* Leakage considerations
* Prediction-point feasibility
* Prediction-horizon feasibility
* Final selection score
* Known limitations
* Reason for final selection

---

## 25. Phase 2.1 Definition of Done

Phase 2.1 will be considered complete when:

* Dataset requirements are documented
* Business requirements are clearly separated from technical requirements
* Temporal realism has been explicitly defined
* Prediction-time feature availability is considered
* Revenue assumptions are controlled
* Data provenance and licensing are included
* Dataset modification rules are established
* Candidate datasets can be evaluated consistently
* The scoring framework is ready for use

The next step is **candidate dataset research and comparison**.

No dataset will be downloaded or selected before this evaluation is completed.
