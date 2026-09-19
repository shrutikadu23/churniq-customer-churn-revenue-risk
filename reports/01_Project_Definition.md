# ChurnIQ — Customer Churn Prediction & Revenue Risk Intelligence

## 1. Business Scenario

ChurnIQ is designed for a subscription-based company that generates recurring revenue from a diverse customer base.

The company collects customer, service, contract, tenure, payment, and revenue-related information, but primarily uses historical reporting to understand churn after it has occurred.

As customer acquisition becomes more expensive, retaining existing customers is increasingly important. However, the business currently lacks a proactive intelligence system that can identify customers who are likely to churn, explain the factors associated with their risk, quantify the recurring revenue exposed to potential churn, and help retention teams prioritize their efforts.

ChurnIQ addresses this gap by combining business analytics, SQL, machine learning, explainable AI, and business intelligence into an end-to-end customer retention decision-support system.

The system is designed to move the organization from reactive churn reporting toward proactive, risk-based retention planning.

## 2. Business Problem

The company currently has limited ability to identify customer churn risk before a customer leaves.

Historical churn analysis can show which customers have already churned and reveal broad patterns across customer groups, services, contracts, and revenue. However, descriptive analysis alone does not provide an individual-level estimate of future churn risk or a systematic way to prioritize retention efforts.

This creates several business challenges:

- Retention teams may identify customers only after churn has occurred.
- High-risk customers may not be distinguishable from lower-risk customers using simple reporting.
- The reasons associated with an individual customer's predicted risk may not be clear.
- Churn risk is not directly connected to customer value or recurring revenue exposure.
- Retention resources are limited, making it important to prioritize customers rather than treat every at-risk customer equally.
- The business lacks a structured framework for evaluating potential retention outcomes under different cost and success assumptions.

Therefore, the core business problem is:

> How can the company use customer data and predictive analytics to identify customers at risk of churn early, understand the factors associated with their risk, quantify potential revenue exposure, and prioritize retention efforts based on both customer risk and business value?

## 3. Project Objective

The primary objective of ChurnIQ is to develop an end-to-end customer churn intelligence system that can:

1. Estimate the probability of churn for individual customers.
2. Identify customer segments and characteristics associated with elevated churn risk.
3. Explain the key factors contributing to predicted churn risk.
4. Quantify recurring revenue associated with customers at risk.
5. Combine churn risk with customer value to identify retention priorities.
6. Evaluate potential retention outcomes using clearly defined hypothetical cost, success-rate, and revenue assumptions.
7. Present actionable insights through an executive Power BI dashboard.
8. Provide an interactive Streamlit interface for customer-level churn risk prediction and explanation.

The project is therefore designed not merely as a churn prediction model, but as a decision-support system that connects predictive analytics with customer retention and revenue-risk management.

## 4. Primary Stakeholder

**Primary stakeholder:** Customer Retention / Business Manager

The primary stakeholder is responsible for understanding customer retention performance and allocating limited retention resources.

The stakeholder needs answers to three core questions:

- Which customers require attention?
- Why are these customers considered at risk?
- Which customers should be prioritized based on their potential business value?

ChurnIQ is designed to translate machine learning outputs into these business-oriented decisions.

## 5. Prediction Point

ChurnIQ will be designed around a clearly defined prediction point: the point in the customer lifecycle at which the business would realistically want to estimate churn risk.

Only information that would reasonably be available at or before this prediction point should be used as predictive input.

This principle will guide:

- Feature selection
- Data leakage prevention
- Train/test methodology
- Model development
- Probability estimation
- Customer risk scoring
- Revenue-risk calculations
- Streamlit prediction inputs

Post-churn information or variables that would only become known after the churn event will not be used as predictive features.

The prediction framework will also define a prediction horizon: the future period over which churn risk is evaluated.

The prediction point and prediction horizon will be determined based on the temporal structure and business meaning of the selected dataset. This will ensure that the modeling framework reflects a realistic retention scenario rather than relying on information that would not be available at prediction time.

## 6. Key Business Questions

ChurnIQ is designed to answer the following business questions:

### Customer & Churn Understanding
- What proportion of customers churn?
- Which customer segments have the highest churn rates?
- How does churn vary by contract, service, payment method, tenure, and customer characteristics?
- Which customer behaviors or attributes are associated with elevated churn risk?

### Predictive Risk
- Which individual customers are most likely to churn?
- How accurately can churn probability be estimated?
- How early can meaningful churn risk be identified using information available at the prediction point?

### Churn Risk Drivers
- What factors are most strongly associated with predicted churn risk?
- Why is a particular customer considered high risk?
- Are the model's predictions sufficiently explainable for business use?

### Revenue Risk
- How much recurring revenue is associated with customers identified as high risk?
- Which high-risk customers also represent high customer value?
- How much potential revenue exposure is concentrated among the highest-priority customers?

### Retention Prioritization
- Which customers should retention teams prioritize?
- How does prioritization change when churn probability is considered together with customer value?
- What retention action category could be considered for different customer risk profiles?

### Business Impact
- How effectively can the model identify churners within the highest-risk customer groups?
- What proportion of total churn and revenue exposure can be captured by targeting the highest-priority customers?
- Under clearly stated hypothetical assumptions, what potential revenue protection and retention ROI could be achieved?

The overall decision framework is:

**Churn Probability → Customer Value → Revenue Exposure → Retention Priority → Recommended Action**

## 7. Machine Learning Objective

The machine learning objective is to develop a classification model that estimates the probability that an individual customer will churn.

The model should:

- Produce a reliable churn probability for each customer.
- Distinguish higher-risk customers from lower-risk customers.
- Identify as many genuine churners as practical while maintaining useful precision.
- Produce sufficiently calibrated probabilities for downstream revenue-risk analysis.
- Remain interpretable enough for business stakeholders to understand the key factors associated with predicted risk.
- Generalize reliably to unseen data without relying on data leakage or information that would only become available after the prediction point.

Model performance will be evaluated using multiple complementary measures, including:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Probability Calibration
- Lift and Gains
- Churn Capture Rate at selected targeting thresholds

Because churn datasets may contain an imbalance between churned and retained customers, evaluation will not rely on accuracy alone.

Model selection will consider:

- Predictive performance
- Ability to identify genuine churners
- Probability calibration
- Interpretability
- Generalization to unseen data
- Business usefulness

The evaluation will also consider the business implications of prediction errors. Missing a genuinely high-risk customer may represent a greater business cost than contacting a customer who ultimately would not churn. Therefore, threshold selection will consider the relative consequences of false negatives and false positives rather than relying only on the default classification threshold.

The final model will be selected based on a balance between predictive performance, probability quality, interpretability, and practical business value.

---

## 8. Business Impact Objective

The business objective is to convert churn predictions into a practical customer retention prioritization framework.

The system should enable the business to:

1. Identify customers with elevated predicted churn risk.
2. Understand the factors associated with each customer's predicted risk.
3. Combine churn probability with customer value and recurring revenue.
4. Quantify potential revenue exposure among customers identified as at risk.
5. Prioritize customers when retention resources are limited.
6. Evaluate how effectively different targeting thresholds capture churn and revenue exposure.
7. Estimate potential revenue protection and retention ROI under clearly defined hypothetical assumptions.

The business framework will distinguish between the following concepts:

**Model Risk**  
The estimated probability that a customer will churn.

**Revenue Exposure**  
The recurring revenue associated with customers identified as being at risk.

**Risk-Weighted Revenue Exposure**  
An estimated revenue-risk measure calculated by combining customer revenue with predicted churn probability.

**Retention Priority**  
A business prioritization measure that considers churn risk together with customer value and revenue exposure.

**Retention ROI**  
A scenario-based estimate of potential financial benefit after considering assumed retention success rates, intervention costs, and potentially protected revenue.

The effectiveness of the prioritization framework will also be evaluated using targeting-efficiency measures such as:

- Lift
- Gains
- Churn capture rate
- Revenue exposure captured within the highest-risk customer groups
- Performance at selected targeting levels such as the top 10%, 20%, or other relevant thresholds

These measures will help answer practical business questions such as:

- If the retention team can contact only a limited percentage of customers, how many potential churners could be identified?
- How much potential revenue exposure is concentrated within the highest-priority customers?
- Does risk-based prioritization perform better than treating all customers equally?

Any revenue protection or ROI estimates will be explicitly presented as scenario-based estimates using stated assumptions. They will not be interpreted as guaranteed financial outcomes or evidence that a particular retention action causes a customer to remain.

ChurnIQ therefore aims to optimize not simply for predictive accuracy, but for the broader business decision:

**Which customers should the business prioritize, why are they considered at risk, how much potential revenue is exposed, and what retention opportunity could be considered?**

## 9. KPIs

ChurnIQ will track three categories of KPIs: business performance, model performance, and decision effectiveness.

### 9.1 Business KPIs

| KPI | Definition | Business Purpose |
|---|---|---|
| Churn Rate | Percentage of customers who churn | Measures overall customer retention performance |
| Total Customers | Number of customers in the analysis population | Defines the customer base |
| Churned Customers | Number of customers classified as churned in historical data | Measures the scale of observed churn |
| Monthly Recurring Revenue | Recurring monthly revenue generated by customers | Measures the revenue base exposed to churn |
| Revenue Exposure | Recurring revenue associated with customers identified as at risk | Quantifies potential revenue exposure |
| Risk-Weighted Revenue Exposure | Customer recurring revenue multiplied by predicted churn probability | Provides a probability-weighted estimate of potential revenue risk |
| High-Risk Customers | Number of customers above the selected risk threshold | Measures the size of the actionable risk population |
| High-Value High-Risk Customers | Customers with both elevated churn risk and high customer value | Identifies strategically important retention opportunities |

Revenue exposure will be treated as potential exposure rather than guaranteed revenue loss.

### 9.2 Machine Learning KPIs

The model will be evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Brier Score or another appropriate probability-calibration metric
- Calibration Curve
- Lift
- Gains
- Churn Capture Rate at selected targeting thresholds

Accuracy may be reported for completeness but will not be treated as the primary model-selection metric, particularly if the target variable is imbalanced.

### 9.3 Decision Effectiveness KPIs

The practical usefulness of the model will be evaluated through:

- Churn Capture Rate at Top X% of customers
- Revenue Exposure Captured at Top X%
- Risk-Weighted Revenue Captured at Top X%
- Lift at selected targeting levels
- Gains at selected targeting levels
- High-Value Customer Capture Rate
- Estimated cost per captured churn-risk customer under defined targeting scenarios
- Estimated potential revenue protection under defined scenarios
- Scenario-based Retention ROI

These measures will determine whether the model can concentrate meaningful churn and revenue risk within a manageable customer population.

Model performance will also be evaluated at multiple probability thresholds to understand the trade-off between identifying more potential churners and limiting unnecessary retention outreach.

---

## 10. Success Criteria

ChurnIQ will be considered successful when it demonstrates value across three dimensions: predictive quality, business usefulness, and decision support.

### 10.1 Predictive Success

The final model should:

- Demonstrate meaningful predictive performance against a simple Dummy Classifier baseline.
- Identify churners substantially better than random targeting.
- Provide useful precision and recall at a business-relevant threshold.
- Produce sufficiently reliable churn probabilities for risk quantification.
- Generalize to unseen data without evidence of data leakage or severe overfitting.
- Provide interpretable risk drivers through appropriate explainability techniques.

No arbitrary universal performance threshold will be defined before the dataset and baseline results are evaluated. Model performance targets will be established after understanding the dataset, class balance, baseline performance, and business costs of prediction errors.

The model will also be evaluated across multiple probability thresholds rather than relying automatically on the default classification threshold. Threshold selection will consider the trade-off between churn capture, false-positive volume, and the practical cost of retention outreach.

### 10.2 Business Success

The project should demonstrate that:

- High-risk customers can be identified systematically.
- Customer risk can be connected to customer value and recurring revenue.
- A meaningful proportion of churn or revenue exposure can be concentrated within a smaller prioritized customer group.
- Risk-based targeting can provide greater efficiency than treating all customers equally.
- Retention opportunities can be ranked according to business value rather than churn probability alone.

### 10.3 Decision-Support Success

The final solution should allow a business stakeholder to answer:

- Who is most at risk?
- Why are they considered at risk?
- How much recurring revenue is potentially exposed?
- Which customers should be prioritized?
- How much churn or revenue exposure could potentially be captured by targeting the highest-risk customers?
- What retention action category could be considered?
- What could the potential financial impact look like under different assumptions?
- What is the estimated cost of targeting a selected customer group?

The final Power BI dashboard and Streamlit application should translate analytical and machine learning outputs into understandable, actionable decision-support information.

### 10.4 Responsible Use

ChurnIQ predictions will be treated as decision-support signals rather than guaranteed outcomes.

The project will clearly distinguish:

- Prediction from causation
- Risk from certainty
- Revenue exposure from realized revenue loss
- Scenario-based ROI from actual financial results
- Model explanations from proven causal drivers

Retention decisions should therefore incorporate appropriate business judgment rather than relying exclusively on model predictions.

### 10.5 Overall Success Definition

ChurnIQ will be considered successful when it demonstrates that predictive churn modeling can improve the identification and prioritization of potential churn risk compared with non-predictive or random targeting, while providing sufficiently reliable and interpretable outputs for business decision-making.

## 11. Project Scope & Boundaries

### 11.1 In Scope

ChurnIQ will cover the following analytical and business areas:

#### Customer Churn Analysis
- Historical customer churn analysis
- Churn patterns across customer, service, contract, payment, tenure, and revenue-related characteristics
- Customer segmentation and risk profiling
- Identification of customer groups associated with elevated churn risk

#### SQL Business Analysis
- Customer-level and segment-level churn analysis
- Churn-rate analysis across relevant business dimensions
- Revenue and recurring-revenue analysis
- Customer-value analysis
- High-risk and high-value customer analysis
- Revenue exposure and risk-weighted revenue analysis
- Customer prioritization analysis

#### Exploratory Data Analysis
- Distribution and relationship analysis of relevant variables
- Churn pattern identification
- Customer and revenue behavior analysis
- Identification of data-quality issues
- Identification of potential modeling risks and anomalies

#### Feature Engineering
- Creation of prediction-ready features using information available at the defined prediction point
- Business-relevant customer value and revenue features where appropriate
- Categorical encoding and numerical transformations where required
- Feature selection based on business relevance, data quality, and modeling requirements
- Prevention of target leakage and post-churn information leakage

#### Machine Learning
- Development of a simple Dummy Classifier baseline
- Development and comparison of suitable classification models
- Cross-validation where appropriate
- Controlled hyperparameter tuning
- Evaluation using predictive and business-oriented metrics
- Probability calibration
- Classification-threshold optimization
- Model selection based on performance, calibration, interpretability, generalization, and business usefulness

#### Explainable AI
- Global model interpretation
- Identification of important factors associated with churn predictions
- Individual customer-level prediction explanations
- Clear distinction between model explanation and causal inference

#### Customer Risk Intelligence
- Customer-level churn probability
- Risk categorization
- Customer value assessment
- Revenue exposure assessment
- Risk-weighted revenue assessment
- Risk × Customer Value prioritization

#### Retention Prioritization
- Identification of high-risk/high-value customers
- Top-X% targeting analysis
- Churn capture analysis
- Revenue exposure capture analysis
- Lift and gains analysis
- Targeting-efficiency analysis
- Scenario-based retention opportunity analysis
- Scenario-based retention ROI estimation

#### Business Intelligence
- Executive Power BI dashboard
- Interactive filtering and segmentation
- Customer risk and revenue-risk views
- Priority customer analysis
- Business insights and recommendations

#### Interactive Application
- Streamlit-based customer churn prediction interface
- Realistic prediction-time feature inputs
- Churn probability and risk classification
- Customer-level explanation of important risk factors
- Business-oriented interpretation of predicted risk

#### Documentation & Responsible Use
- Data dictionary
- Project methodology
- Model comparison and evaluation
- Model Card
- Assumptions and limitations
- Responsible-use considerations
- Reproducibility documentation
- Conceptual model monitoring and data-drift considerations

---

### 11.2 Out of Scope

The following areas are intentionally outside the scope of ChurnIQ:

- Real-time production deployment
- Full-scale MLOps infrastructure
- Automated model retraining pipelines
- Cloud infrastructure deployment
- Deep learning unless demonstrated to provide meaningful additional business value
- Real-time customer intervention systems
- Automated retention campaigns
- Direct CRM or marketing-platform integration
- Actual customer communication
- Guaranteed prediction of individual customer behavior
- Guaranteed prediction of future revenue loss
- Causal inference about why customers churn
- Claiming that a specific retention action will prevent churn without appropriate intervention data
- Measuring actual retention campaign effectiveness when historical intervention outcomes are unavailable

These exclusions are intentional. The project prioritizes analytical depth, predictive modeling, explainability, and business decision support over unnecessary technical complexity.

---

### 11.3 Data Availability & Temporal Boundary

The scope of prediction and temporal validation will depend on the structure of the selected dataset.

If the dataset contains appropriate time-related information, ChurnIQ will define a realistic prediction point and prediction horizon and may incorporate time-aware validation.

If the dataset does not contain sufficient temporal information, the project will clearly document this limitation and will not present a cross-sectional model as a fully time-based forecasting system.

Only information that would realistically be available at or before the defined prediction point will be eligible for predictive modeling.

Variables that directly reveal churn or become available only after the prediction point will be excluded to prevent data leakage.

---

### 11.4 Business Decision Boundary

ChurnIQ is a decision-support system rather than an autonomous decision-making system.

The model will identify and prioritize potential churn risk, but final retention decisions remain with business stakeholders.

Recommended actions will represent business-oriented action categories or considerations rather than guaranteed interventions.

Depending on the available data and identified risk patterns, the system may suggest consideration of categories such as:

- Contract or pricing review
- Service or support review
- Engagement follow-up
- Value-based retention consideration
- General retention outreach

These recommendations will not be interpreted as proven causal interventions.

The project will not claim that a particular action will prevent churn unless appropriate intervention or experimental data is available.

---

### 11.5 Analytical Boundary

ChurnIQ will distinguish between four analytical layers:

**Descriptive Analysis**  
What happened historically?

**Diagnostic / Risk Driver Analysis**  
Which characteristics and factors are associated with observed or predicted churn risk?

**Predictive Analysis**  
What is the estimated probability that a customer will churn?

**Prescriptive Decision Support**  
Which customers may warrant greater retention attention based on predicted risk, customer value, and revenue exposure?

The project will not interpret predictive relationships as causal relationships without appropriate evidence.

---

### 11.6 Model & Data Risk Boundary

The project will explicitly consider important analytical risks, including:

- Target leakage
- Post-churn information leakage
- Overfitting
- Class imbalance
- Poor probability calibration
- Unstable predictions
- Dataset limitations
- Sampling limitations
- Potential subgroup performance differences

Where the dataset contains relevant demographic or customer-group attributes, appropriate subgroup performance checks may be performed to identify material differences in model behavior.

These checks will support responsible interpretation but will not be presented as a complete fairness or regulatory audit.

---

### 11.7 Scope Principle

ChurnIQ will follow a business-value-first principle:

**Every analytical method, feature, model, visualization, and application component must have a clear business purpose.**

Technology will be added only when it contributes meaningful analytical or decision-support value.

The project will prioritize:

**Business Relevance → Analytical Rigor → Model Quality → Explainability → Decision Usefulness**

rather than technical complexity for its own sake.

## 12. Assumptions

The following assumptions will guide the development of ChurnIQ. These assumptions will be validated or revised after the dataset is selected and analyzed.

### 12.1 Business Assumptions

- The company operates a subscription-based business model with recurring customer relationships.
- Historical customer and service information can provide useful signals for identifying churn risk.
- Customer retention resources are limited, making prioritization important.
- Recurring revenue is an appropriate measure for assessing potential revenue exposure.
- Retention decisions should consider both customer churn risk and business value.
- Retention interventions involve costs and have uncertain success rates; therefore, ROI analysis will use explicit scenario assumptions rather than guaranteed outcomes.

### 12.2 Data Assumptions

- The selected dataset contains sufficient customer-level information to support churn analysis.
- The churn target is sufficiently defined to distinguish churned and retained customers.
- Relevant customer, service, contract, tenure, payment, and revenue-related variables are available where applicable.
- Data quality issues can be identified and appropriately handled.
- Variables used for prediction are available at or before the defined prediction point.
- Information that directly reveals churn or becomes available only after the prediction point will not be used as a predictive feature.
- The dataset may contain limitations that affect temporal modeling, generalization, or business interpretation. Such limitations will be documented rather than hidden.

### 12.3 Modeling Assumptions

- Churn is treated as a classification problem.
- Predicted probabilities represent estimated risk rather than certainty.
- Model performance should be evaluated on unseen data.
- Class imbalance, if present, will be explicitly considered during model evaluation.
- Probability calibration is important because predicted probabilities will be used in downstream revenue-risk calculations.
- Model explanations describe factors associated with predictions and should not automatically be interpreted as causal drivers.
- The final model and preprocessing pipeline should be reproducible using documented inputs, parameters, and procedures.

---

## 13. Project Risks & Mitigation

| Risk | Potential Impact | Mitigation |
|---|---|---|
| Data Leakage | Artificially inflated model performance | Define the prediction point and exclude post-prediction information |
| Poor Data Quality | Unreliable analysis and predictions | Perform systematic data validation and document data-quality issues |
| Class Imbalance | Poor identification of the minority churn class | Use appropriate metrics, stratification, and evaluation techniques |
| Overfitting | Poor performance on unseen data | Use cross-validation, controlled tuning, and holdout evaluation |
| Poor Probability Calibration | Misleading revenue-risk estimates | Evaluate and improve probability calibration |
| Dataset Limitations | Reduced generalizability to real-world customers | Clearly document dataset characteristics and limitations |
| Sampling Bias | Model may not represent the broader customer population | Evaluate customer distributions and document limitations |
| Unstable Model Performance | Inconsistent predictions across datasets or segments | Compare models and evaluate performance stability |
| Subgroup Performance Differences | Uneven model behavior across customer groups | Perform appropriate subgroup performance checks where data permits |
| Threshold Selection Risk | Too many false positives or missed churners | Evaluate multiple thresholds and consider business costs |
| Revenue Assumption Risk | Misleading ROI or revenue-protection estimates | Use clearly stated scenario assumptions and sensitivity analysis |
| Interpretation Risk | Business users may mistake association for causation | Clearly distinguish prediction, explanation, and causal inference |
| Recommendation Risk | Suggested actions may be interpreted as guaranteed solutions | Present actions as decision-support categories rather than guaranteed interventions |
| Data Drift | Model performance may decline as customer behavior changes | Document conceptual monitoring and relevant drift indicators |
| Experiment / Configuration Risk | Different preprocessing, splits, parameters, or random seeds may produce inconsistent results | Maintain consistent experiment configuration, record parameters and metrics, and use reproducible random seeds |
| Reproducibility Risk | Results may be difficult to reproduce | Maintain requirements, document preprocessing and modeling steps, and save the selected model and pipeline |

---

## 14. Expected Deliverables

ChurnIQ will produce the following deliverables across the project lifecycle.

### 14.1 Project Documentation

- Project Definition Document
- Dataset Documentation
- Data Dictionary
- Methodology Documentation
- Model Comparison Documentation
- Model Card
- Assumptions and Limitations
- Final Business Insights and Recommendations

### 14.2 Data & Analysis

- Raw dataset
- Processed dataset
- Data-validation report documenting schema, missing values, duplicates, invalid values, target quality, and key data-quality findings
- SQL business-analysis queries
- Exploratory Data Analysis notebook(s)
- Feature-engineering pipeline
- Customer-level analytical dataset

### 14.3 Machine Learning

- Dummy Classifier baseline
- Candidate machine learning models
- Model comparison results
- Cross-validation results
- Hyperparameter tuning results
- Final selected model
- Probability calibration results
- Threshold optimization results
- Model evaluation metrics
- Lift and gains analysis
- Saved preprocessing and modeling pipeline

### 14.4 Explainability

- Global feature-importance analysis
- SHAP-based global explanations
- Customer-level SHAP explanations
- Documentation of model interpretation limitations

### 14.5 Customer Risk Intelligence

- Customer-level churn probabilities
- Risk categories
- Customer-value classification
- Revenue exposure analysis
- Risk-weighted revenue analysis
- Risk × Customer Value matrix
- Retention-priority dataset
- Top-X% targeting analysis
- Churn and revenue exposure capture analysis

### 14.6 Business Intelligence

- Executive Power BI dashboard
- KPI summary
- Churn-risk analysis
- Revenue-risk analysis
- Customer prioritization views
- Business insights and recommendations

### 14.7 Interactive Application

- Streamlit prediction application
- Prediction-time input interface
- Churn probability output
- Risk classification
- Customer-level explanation
- Business-oriented risk interpretation

### 14.8 Portfolio & Reproducibility

- Clean GitHub repository
- Structured project folders
- Requirements file
- Reproducible analysis workflow
- README with project overview, methodology, results, and business impact
- Clearly documented limitations and responsible-use considerations

---

## 15. Phase 1 Completion Criteria

Phase 1 will be considered complete when:

- The business scenario is clearly defined.
- The business problem and objectives are documented.
- The prediction point and prediction horizon framework are established.
- Key business questions are defined.
- Machine learning and business objectives are separated.
- Business, ML, and decision-effectiveness KPIs are defined.
- Project scope and boundaries are documented.
- Assumptions and major analytical risks are identified.
- Expected project deliverables are defined.
- The project definition is sufficiently stable to guide dataset selection and subsequent analysis.

### 15.1 Definition of Done

Phase 1 is complete when the project definition provides a stable framework for dataset selection, analysis, modeling, and business reporting without requiring major structural changes.

Minor adjustments may be made later if the selected dataset introduces constraints or opportunities that could not reasonably be determined during the initial planning stage.
