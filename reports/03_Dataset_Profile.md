# ChurnIQ — Dataset Profile

## 1. Dataset Overview

ChurnIQ uses a telecom customer churn case-study dataset containing customer-level behavioral, usage, recharge, service, and revenue-related information across multiple monthly periods.

The dataset was selected because its structure supports the complete ChurnIQ analytical workflow:

**Predict → Explain → Quantify Risk → Prioritize → Recommend Action**

Unlike a purely static customer snapshot, the dataset contains repeated monthly observations across key customer activity and revenue measures. This creates opportunities for temporal feature engineering and early-risk analysis, subject to validation of the dataset's prediction point and churn horizon.

The dataset supports the following analytical layers:

- Customer and churn analysis
- SQL-based business analysis
- Exploratory data analysis
- Temporal behavioral analysis
- Feature engineering
- Supervised machine learning
- Model evaluation and calibration
- Explainable AI
- Customer risk scoring
- Customer-value analysis
- Revenue-risk analysis
- Retention prioritization
- Power BI reporting
- Streamlit-based prediction interface

---

## 2. Dataset Source

**Dataset:** Telecom Churn Case Study / Hackathon Dataset

**Source:** Kaggle competition dataset

The dataset was accessed through the Kaggle competition environment and downloaded for analytical and modeling purposes.

### Source Handling

The original downloaded files are treated as source data and will be preserved locally under:

```text
data/raw/

Raw competition data will not be committed to the public GitHub repository unless the applicable competition and dataset terms explicitly permit redistribution.

The GitHub repository will document the dataset source, expected file structure, and reproduction requirements instead.

3. Raw Dataset Files

The downloaded dataset contains four files:

File	Purpose
train.csv	Training dataset containing customer features and the churn target
test.csv	Test dataset containing customer features without the churn target
sample.csv	Sample submission/reference structure
data_dictionary.csv	Description of dataset variables
Raw Data Principle

The files stored in data/raw/ will remain unchanged.

No cleaning, imputation, feature engineering, or manual modification will be performed directly on the raw files.

All transformations will be performed through reproducible processing steps and saved separately under:

data/processed/

This preserves data lineage, supports reproducibility, and maintains a clear separation between source data and analytical outputs.

4. Dataset Size
Training Dataset
Rows: 69,999
Columns: 172
Test Dataset
Rows: 30,000
Columns: 171

The one-column difference is explained by the presence of the target variable in the training dataset.

Train/Test Column Integrity
Train-only column: churn_probability
Test-only columns: None
Duplicate IDs in training data: None identified
Duplicate IDs in test data: None identified
Train/Test ID overlap: None identified

Therefore, the train and test datasets have aligned predictor structures with the churn target available only in the training data.

5. Target Variable

The supervised learning target is:

churn_probability

The target is binary:

Value	Meaning
0	Customer retained
1	Customer churned

The training dataset contains 69,999 records, including approximately 10.19% churned records and 89.81% retained records.

This represents a moderately imbalanced classification problem. Therefore, model evaluation will not rely on accuracy alone and will include metrics such as Precision, Recall, F1-score, ROC-AUC, PR-AUC, calibration, and business-oriented lift and capture measures.

6. Data Structure

The dataset follows a customer-level tabular structure, where each row represents an individual customer and the columns contain customer attributes, behavioral measures, usage patterns, recharge activity, service-related indicators, and revenue-related variables.

The dataset contains both customer-level attributes and repeated monthly observations, allowing ChurnIQ to evaluate customer behavior across multiple periods.

Core Data Components
Component	Description
Customer Identifier	Unique identifier used for record-level tracking
Customer Attributes	Customer-level characteristics and account information
Tenure	Customer age-on-network information represented by aon
Revenue	Monthly ARPU and related revenue measures
Usage	Outgoing, incoming, local, STD, voice, and data usage measures
Recharge	Recharge frequency, amount, and related recharge behavior
Data Services	2G/3G usage, recharge, and service indicators
Service Indicators	Variables representing subscription or service usage behavior
Temporal Variables	Monthly observations and recharge dates
Target	Binary churn outcome represented by churn_probability
7. Monthly Data Structure

A major characteristic of the dataset is the presence of monthly variables for June, July, and August, represented through suffixes such as:

_6 = June
_7 = July
_8 = August

Examples include:

arpu_6
arpu_7
arpu_8

total_rech_amt_6
total_rech_amt_7
total_rech_amt_8

total_og_mou_6
total_og_mou_7
total_og_mou_8

This structure creates opportunities to analyze:

Month-to-month changes in customer activity
Usage increases or declines
Recharge behavior changes
Revenue movement
Customer engagement trends
Potential early-warning signals

However, monthly variables will only be used for prediction after their availability at the defined prediction point has been validated. This is necessary to prevent target leakage and ensure that the model uses only information that would realistically be available when a prediction is made.

8. Major Variable Groups

The dataset contains a large number of variables that can be organized into meaningful analytical groups.

8.1 Customer Tenure

Key variable:

aon

This represents the customer's age on the network and can support analysis of churn behavior across customer lifecycle stages.

8.2 Revenue & ARPU

Examples include:

arpu_6
arpu_7
arpu_8

arpu_3g_6
arpu_3g_7
arpu_3g_8

arpu_2g_6
arpu_2g_7
arpu_2g_8

These variables support customer-value analysis, revenue-risk estimation, and behavioral trend analysis.

8.3 Recharge Behavior

Examples include:

total_rech_num_6
total_rech_num_7
total_rech_num_8

total_rech_amt_6
total_rech_amt_7
total_rech_amt_8

max_rech_amt_6
max_rech_amt_7
max_rech_amt_8

Recharge variables can help identify changes in customer engagement and spending behavior.

8.4 Voice Usage

Examples include:

total_og_mou_6
total_og_mou_7
total_og_mou_8

total_ic_mou_6
total_ic_mou_7
total_ic_mou_8

These variables represent outgoing and incoming usage and can contribute to behavioral churn analysis.

8.5 Data Usage

Examples include:

vol_2g_mb_6
vol_2g_mb_7
vol_2g_mb_8

vol_3g_mb_6
vol_3g_mb_7
vol_3g_mb_8

These variables provide information about mobile data consumption and customer engagement.

8.6 Data Recharge & Service Adoption

Examples include:

total_rech_data_6
total_rech_data_7
total_rech_data_8

count_rech_2g_6
count_rech_2g_7
count_rech_2g_8

count_rech_3g_6
count_rech_3g_7
count_rech_3g_8

These variables can support analysis of data-service adoption and changes in customer behavior.

9. Date Variables

The dataset contains monthly reference dates and recharge-related dates, including:

last_date_of_month_6
last_date_of_month_7
last_date_of_month_8

date_of_last_rech_6
date_of_last_rech_7
date_of_last_rech_8

date_of_last_rech_data_6
date_of_last_rech_data_7
date_of_last_rech_data_8

These variables will be evaluated for:

Recency calculations
Activity gaps
Month-level behavioral features
Temporal consistency
Potential leakage

Date fields that provide no analytical variation or business value will be excluded from modeling after validation.

10. Identifier & Constant-Variable Considerations

The customer identifier will be retained for record-level tracking and final customer-level outputs but will not be used as a predictive feature.

Initial profiling also identified the following constant variables:

circle_id
last_date_of_month_6

Because constant variables contain no predictive variation within the training data, they are candidates for exclusion from the modeling feature set.

Their treatment will be formally documented during preprocessing rather than modifying the raw dataset.

11. Missingness Considerations

The dataset contains missing values across multiple variable groups, with particularly high missingness in several data-recharge and data-service variables.

Missing values will therefore be treated as a data-quality and business-meaning issue rather than automatically being replaced with zero.

The project will investigate:

Missingness by variable
Missingness by month
Missingness by churn outcome
Whether missingness represents non-usage or unavailable information
Whether missingness itself contains predictive information
Appropriate imputation strategies
Potential leakage introduced through preprocessing

Any final treatment will be applied through the reproducible preprocessing pipeline.

12. Date & Temporal Considerations

Date variables are initially stored as text and will be converted to appropriate date formats when required.

Temporal analysis will consider:

Chronological ordering
Recency
Month-over-month changes
Customer activity gaps
Prediction-point availability
Potential target leakage

The exact prediction point and prediction horizon will be finalized after validating the dataset's temporal structure and target construction.

13. Feature Selection Principle

ChurnIQ will not automatically use every available variable simply because it exists in the dataset.

Feature selection will consider:

Business relevance
Data quality
Missingness
Variance
Redundancy
Leakage risk
Temporal availability
Predictive usefulness
Interpretability
Stability across validation samples

The objective is to build a business-relevant feature set, not a feature-maximization model.

14. Feature Engineering Direction

Feature engineering will focus on converting raw monthly observations into interpretable customer-behavior signals.

Potential feature families include:

Feature Family	Example Business Signal
Recency	Time since last recharge/activity
Revenue Trend	Increase or decline in ARPU
Recharge Trend	Change in recharge frequency or value
Usage Trend	Increase or decline in voice/data usage
Engagement	Consistency of customer activity
Service Adoption	Usage of available data/service offerings
Customer Value	Revenue-related customer value indicators

These features will only be created when they are supported by the available data and valid at the defined prediction point.

15. Data Quality Flags for Further Investigation

Initial profiling has identified several areas requiring deeper validation:

Approximately 10.19% of training records belong to the churn class.
Several variables contain substantial missingness.
Some data-service variables have missingness above 70%.
circle_id and last_date_of_month_6 are constant.
Several ARPU variables contain negative values that require business/data validation.
Date fields require appropriate parsing and temporal validation.
Monthly variables require leakage assessment before modeling.

These findings will be investigated during the formal data-quality and preprocessing stages.

16. Dataset Suitability for ChurnIQ

The dataset provides the key characteristics required for ChurnIQ's business-first framework:

Requirement	Dataset Support
Customer-level analysis	Available
Binary churn target	Available
Revenue-related variables	Available
Recharge behavior	Available
Usage behavior	Available
Multiple monthly periods	Available
Feature engineering opportunities	Strong
Churn prediction	Suitable
Customer risk scoring	Suitable
Revenue-risk analysis	Suitable
SQL business analysis	Suitable
Explainable ML	Suitable
Power BI reporting	Suitable
Streamlit prediction workflow	Suitable

The dataset therefore provides a strong foundation for connecting predictive churn risk with customer value and potential revenue exposure rather than treating churn prediction as an isolated machine learning exercise.

17. Data Structure Summary

The dataset provides a customer-level analytical structure combining:

Customer → Tenure → Revenue → Recharge → Usage → Services → Monthly Behavior → Churn

This structure enables ChurnIQ to move beyond simply predicting churn and instead connect:

Churn Risk → Customer Value → Revenue Exposure → Retention Priority

The detailed variable-level profiling, missing-value analysis, data-quality assessment, temporal validation, and leakage analysis will be documented in the subsequent dataset-audit stage.

# ChurnIQ — Initial Dataset Audit

## 18. Audit Objective

The initial dataset audit establishes the structural, quality, and modeling characteristics of the raw training and test datasets before any cleaning, transformation, or feature engineering is performed.

The objective is to identify potential data-quality, modeling, and leakage risks early and determine which areas require deeper validation before the analytical pipeline is finalized.

The audit covers:

- Dataset dimensions
- Column structure
- Data types
- Missing values
- Duplicate records
- Identifier integrity
- Target distribution
- Constant variables
- Date variables
- Key numerical distributions
- Initial churn-related patterns
- Potential data leakage
- Data-quality risks

---

## 19. Dataset Dimensions

The raw datasets contain:

| Dataset | Rows | Columns |
|---|---:|---:|
| Training | 69,999 | 172 |
| Test | 30,000 | 171 |

The training dataset contains one additional column because it includes:

```text
churn_probability

The test dataset does not contain the target variable.

20. Identifier Integrity

The customer identifier is:

id

Initial validation identified:

Check	Result	Status
Training duplicate IDs	0	Validated
Test duplicate IDs	0	Validated
Train/Test ID overlap	0	Validated
Unique training IDs	69,999	Validated
Unique test IDs	30,000	Validated

The absence of duplicate IDs and train/test ID overlap provides a clean starting point for record-level analysis and model evaluation.

The id field will be retained for customer-level tracking and final risk outputs but excluded from predictive modeling.

21. Target Distribution

The target variable is:

churn_probability

Initial profiling shows the following distribution:

Target	Customer Count	Share
0 — Retained	62,867	89.81%
1 — Churned	7,132	10.19%
Total	69,999	100.00%

The target therefore represents an imbalanced binary classification problem.

This has important modeling implications:

Accuracy alone will not be sufficient.
Recall will be important because missed churners may represent lost revenue opportunities.
Precision will matter because unnecessary retention interventions consume resources.
F1-score will provide a balance between Precision and Recall.
PR-AUC will provide additional insight into minority-class performance.
Probability calibration will be evaluated because ChurnIQ uses predicted probabilities for risk prioritization.
Thresholds will be evaluated according to business trade-offs.
Lift, gains, and churn-capture analysis will be used to assess targeting effectiveness.

The exact meaning and construction of the target will also be validated against the dataset documentation before the final prediction framework is established.

22. Missing Value Assessment

The initial audit identified missing values across multiple variable groups.

Examples include:

Variable Group	Initial Observation
Monthly usage variables	Several thousand missing records in multiple fields
Last recharge dates	Approximately 1,100–2,500 missing records depending on month
Data recharge variables	Approximately 74% missing in several fields
Data-service variables	Frequently above 70% missing

The high missingness observed in data-service variables requires particular attention.

Missing values will not automatically be treated as zero because a missing value may represent:

No usage
No recharge
Non-subscription
Unavailable information
A structural property of the dataset

The final treatment will therefore depend on the business meaning of each variable and the results of further validation.

23. Missingness and Churn Relationship

Initial profiling indicates that missingness in several data-service variables differs substantially between retained and churned customers.

For example, selected August data-service variables show substantially higher missingness among churned customers than retained customers.

This suggests that missingness may contain useful behavioral information.

However, this observation is associational, not causal.

The project will investigate:

Whether missingness is systematically related to customer behavior
Whether missingness indicators improve predictive performance
Whether the pattern remains stable across validation samples
Whether the information would legitimately be available at the prediction point
Whether any missingness pattern introduces leakage

No missingness-based feature will be retained solely because it improves performance on one validation split.

24. Constant Variables

The initial audit identified two constant variables:

circle_id
last_date_of_month_6

Because constant variables contain no variation within the training data, they provide no useful predictive information in their current form.

Treatment Status
Variable	Initial Finding	Modeling Status
circle_id	Constant	Candidate for exclusion
last_date_of_month_6	Constant	Candidate for exclusion

The raw dataset will remain unchanged.

Final exclusion decisions will be implemented through the reproducible preprocessing pipeline and documented in the feature-selection stage.

25. Date Variables

The dataset contains the following date-related variables:

last_date_of_month_6
last_date_of_month_7
last_date_of_month_8

date_of_last_rech_6
date_of_last_rech_7
date_of_last_rech_8

date_of_last_rech_data_6
date_of_last_rech_data_7
date_of_last_rech_data_8

Initial observations indicate:

Monthly reference dates have limited unique values.
Recharge dates contain multiple dates across each month.
Some recharge dates contain missing values.
Data-recharge dates have particularly high missingness.

These variables may support features such as:

Recency
Activity gaps
Recharge recency
Monthly behavioral timing

However, such features will only be created after confirming that the underlying information was available at the defined prediction point.

26. Key Numerical Variable Profile

Selected variables show substantial variation across customers.

Customer Tenure
aon
Statistic	Value
Mean	1,220.64
Minimum	180
Maximum	4,337
August ARPU
arpu_8
Statistic	Retained	Churned
Mean	297.54	114.23
Median	210.03	9.97
Q1	103.42	0.00
Q3	388.17	137.01
August Recharge Amount
total_rech_amt_8
Statistic	Value
Mean	323.85
Minimum	0
Maximum	45,320

These observations demonstrate substantial customer-level variation and support further investigation of revenue, engagement, and behavioral features.

They should not yet be interpreted as evidence of causal drivers of churn.

27. Negative Value Investigation

Some ARPU-related variables contain negative values.

Examples include:

arpu_6
arpu_7
arpu_8

Negative values will not automatically be classified as errors.

Possible explanations may include:

Adjustments
Credits
Reversals
Accounting treatments
Dataset-specific business rules

The data dictionary and distribution of these observations will be reviewed before deciding whether such values should be retained, transformed, or excluded.

This prevents unsupported assumptions from being introduced during preprocessing.

28. Initial Churn-Related Patterns

Initial descriptive analysis shows differences between retained and churned customers across several observed August measures.

Variable	Retained Mean	Churned Mean
aon	1,264.0	838.4
arpu_8	297.54	114.23
total_rech_amt_8	346.88	120.82
total_og_mou_8	328.55	92.62
total_ic_mou_8	216.70	37.92
vol_2g_mb_8	54.56	11.06
vol_3g_mb_8	146.93	34.60

In the observed data, churned customers have lower average values across these selected measures.

These are descriptive associations only and should not be interpreted as causal drivers.

Further analysis will determine whether these relationships remain meaningful after considering other variables, temporal behavior, and potential confounding factors.

29. Data Leakage — Gating Condition

Data leakage is one of the highest-priority risks in ChurnIQ because the dataset contains multiple monthly observations and a churn target.

No feature engineering or final model development will proceed until the prediction point, prediction horizon, and information-availability rules have been formally validated.

The leakage assessment will investigate:

Whether a feature was recorded after the prediction point
Whether target-derived information is present
Whether future customer behavior is indirectly represented
Whether variables were calculated using future information
Whether preprocessing uses validation or test information
Whether feature engineering crosses temporal boundaries
Whether train/test separation is maintained throughout preprocessing

A feature will be considered eligible for modeling only when its information would realistically have been available at the prediction point.

30. Initial Data Quality Status
Area	Initial Finding	Status
Dataset dimensions	69,999 train / 30,000 test	Validated
Target balance	10.19% churn	Validated
Duplicate IDs	None identified	Validated
Train/Test ID overlap	None identified	Validated
Missing values	High in several variable groups	Requires investigation
Data-service missingness	Frequently above 70%	Requires investigation
Constant variables	2 identified	Requires investigation
Negative ARPU values	Present	Requires investigation
Date fields	Stored as text	Requires preprocessing
Temporal structure	Multiple monthly periods	Requires validation
Prediction point	Not yet finalized	Requires validation
Leakage risk	Potentially significant	Gating condition
31. Initial Audit Conclusion

The initial audit indicates that the dataset is structurally suitable for the ChurnIQ workflow, while several areas require formal validation before modeling.

The most important findings are:

The target is imbalanced, requiring evaluation beyond accuracy.
Missingness is substantial in several data-service variables and may contain behavioral information.
Constant variables exist and should not enter the predictive feature set.
Customer IDs are structurally clean but will not be used as predictive features.
Negative ARPU values require investigation rather than automatic removal.
Monthly and date variables create valuable temporal opportunities, but their prediction-time availability must be validated.
Potential leakage is a gating issue and must be resolved before final feature engineering.
Observed differences between retained and churned customers are associations, not causal conclusions.

The next stage will establish the formal data-quality validation and preprocessing rules required to create a reliable modeling dataset.

# ChurnIQ — Data Quality & Validation Framework

## 32. Validation Objective

Before cleaning, feature engineering, or machine learning, ChurnIQ will establish a reproducible data-quality validation process.

The objective is to ensure that the analytical dataset is:

- Structurally consistent
- Sufficiently complete for the intended analysis
- Free from unintended duplicates
- Correctly typed
- Correctly separated into predictors and target
- Temporally valid
- Protected against data leakage
- Suitable for downstream SQL, analytics, and machine learning

The validation process will identify and document issues before applying transformations rather than silently modifying the data.

---

## 33. Validation Principles

The following principles will guide the data-quality process:

1. **Raw data will never be modified directly.**
2. **Every transformation will be reproducible.**
3. **Validation will occur before feature engineering.**
4. **Data-quality issues will be documented rather than hidden.**
5. **Missing values will be interpreted according to business meaning.**
6. **Potential leakage will be treated as a critical modeling risk.**
7. **Train and test data will remain strictly separated during modeling.**
8. **Validation rules will be applied consistently across datasets.**
9. **Features will be evaluated for both analytical usefulness and predictive eligibility.**
10. **Major exclusion and transformation decisions will have documented reasons.**

---

## 34. Validation Categories

The validation framework is divided into the following categories:

| Category | Purpose |
|---|---|
| Structural Validation | Confirm rows, columns, and expected schema |
| Identifier Validation | Detect duplicates and unexpected ID overlap |
| Data Type Validation | Confirm appropriate types for each variable |
| Missingness Validation | Measure and investigate missing values |
| Target Validation | Confirm target values and class distribution |
| Range Validation | Identify unusual or potentially invalid values |
| Constant Variable Validation | Identify variables with no useful variation |
| Date Validation | Validate date formats and chronological consistency |
| Temporal Validation | Establish valid information availability |
| Leakage Validation | Prevent future or target-derived information |
| Train/Test Validation | Ensure predictor consistency and separation |
| Business Logic Validation | Check whether values make sense in context |

---

## 35. Structural Validation

The first validation layer will confirm the basic structure of the datasets.

### Checks

- Expected files exist
- Training dataset loads successfully
- Test dataset loads successfully
- Data dictionary is available
- Expected row counts are recorded
- Expected column counts are recorded
- Column names are unique
- Train and test predictor columns are aligned
- Target exists only in the training dataset

### Expected Baseline

| Dataset | Expected Rows | Expected Columns |
|---|---:|---:|
| Training | 69,999 | 172 |
| Test | 30,000 | 171 |

Any unexpected structural change will be flagged before further processing.

---

## 36. Identifier Validation

The customer identifier will be validated to ensure that each record can be uniquely tracked.

### Checks

- Missing customer IDs
- Duplicate customer IDs
- Duplicate IDs within training data
- Duplicate IDs within test data
- Train/test ID overlap
- Unexpected ID type or formatting

### Modeling Rule

The `id` variable will not be used as a predictive feature because an identifier represents record identity rather than customer behavior.

It will remain available for:

- Customer-level risk outputs
- Prediction result tracking
- Business reporting
- Approved analytical joins

---

## 37. Data Type Validation

Each variable will be reviewed against its expected analytical role.

### Expected Categories

- Identifier
- Numeric measure
- Count
- Binary indicator
- Date
- Target variable

Date fields currently stored as text will be parsed during processing when required.

Data types will be checked for:

- Numeric values stored as text
- Invalid date values
- Unexpected categorical values
- Target type inconsistencies
- Unexpected object/string values

The final processed dataset will use appropriate data types for analysis and modeling.

---

## 38. Missing Value Validation

Missingness will be assessed at multiple levels.

### Variable-Level Analysis

For each variable, the validation process will calculate:

- Missing count
- Missing percentage
- Non-missing count

### Pattern-Level Analysis

Missingness will also be examined:

- By month
- By variable group
- By churn outcome
- Across related variables
- Across relevant customer groups

### Treatment Principle

Missing values will not automatically be replaced with:

```text
0

A missing value will only be converted or imputed after determining what the missingness represents.

Where appropriate, missingness indicators may be created if they provide legitimate, stable, and leakage-safe predictive information.

39. Target Validation

The target variable will undergo dedicated validation.

churn_probability
Checks
Target exists in training data
Target does not exist in test data
No unexpected target values
No missing target values
Target is binary
Class distribution is documented
Target construction is reviewed against the source documentation
Expected Values
0 = Retained
1 = Churned

The project will not assume a specific future churn horizon until the source methodology and temporal structure have been validated.

40. Range & Outlier Validation

Numerical variables will be screened for unusual values.

Examples include:

Negative ARPU
Extremely high recharge amounts
Extremely high usage values
Zero-inflated variables
Unusually high tenure values

Outlier detection will not automatically result in removing records.

Each unusual value will be classified as:

Valid business value
Potential data-quality issue
Extreme but plausible observation
Requires source clarification

This prevents legitimate high-value or unusual customer observations from being incorrectly removed.

41. Constant & Low-Variation Variables

Variables with no variation will be identified before modeling.

Initial constant variables include:

circle_id
last_date_of_month_6

Additional low-variation variables will be assessed during preprocessing.

The decision to remove a variable will consider:

Variance
Business meaning
Predictive usefulness
Potential future variation
Interpretability

Constant variables in the current dataset will not be used for predictive modeling.

42. Date & Temporal Validation

Temporal validation is a critical component of ChurnIQ.

The project will establish:

Prediction Point → Available Information → Future Outcome

Before creating temporal features, the following will be determined:

Which months represent the observation period
Which information is available at prediction time
When the churn outcome is measured
Whether any variables contain future information
The appropriate prediction horizon
Whether the dataset supports time-aware validation

Potential temporal features such as:

Recharge recency
Usage decline
Revenue change
Activity gaps

will only be created from information available at the prediction point.

43. Data Leakage Validation

Data leakage will be treated as a gating condition for the modeling workflow.

A feature will not be considered eligible simply because it improves model performance.

Potential leakage sources include:

Future-month information
Target-derived variables
Post-churn activity
Future revenue information
Aggregations calculated using future records
Preprocessing performed before train/validation separation
Imputation or scaling fitted using validation or test data
Feature selection performed using validation or test information
Leakage Prevention Rule

All learned preprocessing steps will be fitted using training data only and then applied to validation and test data.

The same principle will apply to:

Imputation
Scaling
Encoding
Feature selection
Probability calibration
Threshold optimization

Where cross-validation is used, these operations will be performed within the appropriate training folds.

44. Train/Test Integrity

The training and test datasets will remain separated throughout the modeling workflow.

Rules
The test dataset will not be used for model selection.
The test dataset will not be used for feature selection.
The test dataset will not be used to fit preprocessing transformations.
The test dataset will not influence threshold selection.
The test dataset will only be used for final evaluation or competition-compatible prediction where appropriate.

Cross-validation will be performed using the training data only.

Where temporal structure supports it, time-aware validation will be preferred over a purely random split.

45. Business Logic Validation

Statistical validity alone is not sufficient for ChurnIQ.

Selected variables will also be reviewed for business plausibility.

Examples:

Recharge amount should be interpreted alongside recharge frequency.
Usage values should be evaluated with their units and definitions.
Missing data-service activity should not automatically mean zero usage.
Negative ARPU should be investigated using the data dictionary.
Customer tenure should be consistent with the dataset's measurement period.
Revenue-risk calculations should use clearly defined revenue measures.

Business logic checks help ensure that technically valid transformations do not produce misleading business conclusions.

46. Validation Status Framework

Each validation issue will receive a clear status.

Status	Meaning
Validated	Evidence supports the current treatment
Requires Investigation	Additional analysis or source clarification is needed
Requires Preprocessing	Valid issue with a defined transformation requirement
Gating Condition	Must be resolved before modeling proceeds
Monitor	Not currently blocking but should be tracked

This framework prevents unresolved assumptions from being silently carried into later project stages.

47. Validation Output

The validation process will produce a reproducible audit report containing:

Dataset dimensions
Column inventory
Data types
Missing-value summary
Duplicate checks
Identifier checks
Target validation
Class distribution
Constant-variable report
Range and outlier checks
Date validation
Train/test comparison
Leakage assessment
Validation status
Recommended preprocessing actions

The output will be stored separately from the raw data.

48. Data Quality Decision Framework

The overall decision process will follow:

Validate → Understand → Document → Transform → Revalidate

The project will avoid:

Blindly dropping rows
Blindly replacing missing values with zero
Automatically removing outliers
Using every available feature
Using future information because it improves model performance
Making unsupported business assumptions
Fitting preprocessing steps on validation or test data

Every major preprocessing decision should be traceable to either a validated data-quality issue or a clearly stated business requirement.

49. Modeling Readiness Gate

Before feature engineering and model development begin, ChurnIQ must satisfy the following conditions:

Requirement	Status
Dataset structure validated	Required
Identifier integrity validated	Required
Target structure validated	Required
Missingness profiled	Required
Constant variables identified	Required
Numerical anomalies investigated	Required
Date fields assessed	Required
Prediction point established	Gating condition
Prediction horizon established	Gating condition
Major leakage risks assessed	Gating condition
Train/test separation documented	Required
Preprocessing actions defined	Required

This gate ensures that model performance is not optimized on top of an invalid or leakage-prone analytical design.

50. Definition of Done

The data-quality validation stage will be considered complete when:

Dataset structure has been validated.
Identifier integrity has been confirmed.
Target structure has been validated.
Missingness has been profiled and classified.
Constant and low-variation variables have been identified.
Numerical anomalies have been investigated.
Date fields have been assessed.
Prediction-point availability has been established.
Prediction horizon has been established or its limitation has been explicitly documented.
Major leakage risks have been assessed.
Train/test separation rules are documented.
Required preprocessing actions are defined.
Validation findings are reproducible.
No unresolved gating condition remains before feature engineering.

Only after these conditions are satisfied will ChurnIQ proceed to the final processed modeling dataset and feature-engineering stage.

# ChurnIQ — Dataset Audit Results

## 51. Audit Outcome Overview

The initial audit establishes the current condition of the ChurnIQ dataset before any irreversible transformation.

Overall, the dataset is **structurally sound and suitable for further analysis**, but several characteristics require controlled investigation before feature engineering and modeling.

| Area | Finding | Status |
|---|---|---|
| Dataset structure | Train/test dimensions and schema validated | Validated |
| Customer identifiers | No duplicate IDs or train/test overlap | Validated |
| Target variable | Binary target with 10.19% churn | Validated |
| Missing values | Significant missingness in selected feature groups | Requires Investigation |
| Constant variables | 2 constant variables identified | Requires Preprocessing |
| Date fields | Stored as date-like text | Requires Preprocessing |
| Negative ARPU | Negative observations identified | Requires Investigation |
| Temporal structure | June–August monthly observations available | Requires Validation |
| Class imbalance | Churn represents 10.19% of training records | Requires Modeling Strategy |
| Data leakage | Prediction-time availability not yet fully established | **Gating Condition** |

---

## 52. Structural Integrity Results

The raw datasets passed the primary structural integrity checks.

### Training Dataset

- **Records:** 69,999
- **Columns:** 172
- **Unique customer IDs:** 69,999
- **Duplicate IDs:** 0

### Test Dataset

- **Records:** 30,000
- **Columns:** 171
- **Unique customer IDs:** 30,000
- **Duplicate IDs:** 0

### Train/Test Separation

- **Customer IDs appearing in both datasets:** 0
- **Train-only column:** `churn_probability`
- **Test-only columns:** None

This confirms clean customer-level separation between training and test data.

The `id` field will remain available for customer tracking and final risk outputs but will **not** be used as a predictive feature.

---

## 53. Target Variable Assessment

The target variable is:

```text
churn_probability

with:

0 = Retained
1 = Churned
Target Distribution
Outcome	Customers	Share
Retained	62,867	89.81%
Churned	7,132	10.19%
Total	69,999	100.00%
Modeling Implication

The target is moderately imbalanced.

Therefore, overall accuracy will not be treated as the primary model-selection metric.

ChurnIQ will evaluate:

Precision
Recall
F1-score
ROC-AUC
PR-AUC
Confusion Matrix
Brier Score
Calibration
Lift
Gains
Churn Capture Rate
Revenue Exposure Capture

This ensures that model evaluation reflects the business objective of identifying customers who are genuinely at risk.

54. Missingness Assessment

Missing values are concentrated in several behavioral and data-service variables.

The most significant pattern occurs in data-recharge and related service variables, where missingness is frequently above 70%.

Examples include:

total_rech_data_6
total_rech_data_7
total_rech_data_8

date_of_last_rech_data_6
date_of_last_rech_data_7
date_of_last_rech_data_8

arpu_3g_6
arpu_3g_7
arpu_3g_8
Treatment Principle

Missing values will not automatically be replaced with zero.

Each feature or feature group will be evaluated according to its business meaning.

Possible interpretations include:

No service activity
No recharge
No usage
Not applicable
Information unavailable

Where appropriate, missingness indicators may be created if they are:

Predictively useful
Business interpretable
Available at prediction time
Stable across validation data
Free from leakage
55. Missingness and Churn Relationship

Initial analysis shows that missingness in selected data-service variables differs substantially between retained and churned customers.

For selected August data-service variables:

Churn Status	Missing
Retained	~71.69%
Churned	~91.31%

This indicates that missingness itself may contain predictive information.

However, ChurnIQ will treat this as an association rather than causation.

The final pipeline will test whether such missingness patterns provide stable and useful predictive information without introducing leakage.

56. Constant Variable Assessment

Two constant variables were identified:

circle_id
last_date_of_month_6

Because they contain no meaningful variation within the training dataset, they provide no useful predictive separation in their current form.

Planned Treatment
Variable	Decision
circle_id	Exclude from modeling
last_date_of_month_6	Exclude from modeling

The raw variables will remain unchanged in data/raw/.

Any exclusions will occur within the reproducible preprocessing pipeline.

57. Numerical Data Findings

Several important behavioral and financial variables show meaningful differences between retained and churned customers.

Customer Tenure — aon
Mean: 1,220.64
Minimum: 180
Maximum: 4,337
August ARPU — arpu_8
Metric	Retained	Churned
Mean	297.54	114.23
Median	210.03	9.97
Q1	103.42	0.00
Q3	388.17	137.01
August Recharge Amount — total_rech_amt_8
Mean: 323.85
Minimum: 0
Maximum: 45,320

These differences indicate that customer tenure, revenue, recharge behavior, and engagement are promising areas for deeper analysis.

They are not yet considered confirmed churn drivers.

58. Negative ARPU Investigation

Negative values were identified in ARPU-related variables.

Examples include:

arpu_6
arpu_7
arpu_8

Negative values will not be automatically removed or converted.

The preprocessing stage will determine whether these observations represent:

Valid financial adjustments
Credits or reversals
Data-recording behavior
Genuine anomalies
Extreme but legitimate observations

The decision will be based on the dataset documentation, distributions, and relationships with related financial variables.

59. Temporal Structure Findings

The dataset contains monthly observations for:

June → July → August

This creates an opportunity to analyze customer behavior over time rather than relying only on static values.

Potential feature families include:

Revenue trends
Recharge trends
Usage trends
Engagement decline
Activity recency
Service adoption changes
Month-over-month behavioral changes

However, these features will only be created after confirming that the underlying information would have been available at the defined prediction point.

60. Prediction-Time Data Validation

The most important unresolved analytical question is:

What information would realistically have been available when the churn prediction was made?

ChurnIQ will establish:

Prediction Point → Available Customer Information → Prediction Horizon → Churn Outcome

This determines which variables are eligible for modeling.

A feature will not be considered valid simply because it improves validation performance.

It must also satisfy:

Available at prediction time + Leakage-safe + Business meaningful

61. Data Leakage as a Modeling Gate

Data leakage is treated as a hard modeling gate.

The following will be checked before final feature engineering:

Post-prediction customer behavior
Future usage or recharge information
Target-derived variables
Post-churn information
Future revenue values
Preprocessing fitted outside training folds
Feature selection using validation/test information
Calibration using data not reserved for calibration
Threshold optimization using the final test set

All learned preprocessing steps will be fitted only on appropriate training data.

This includes:

Imputation
Encoding
Scaling
Feature selection
Model tuning
Calibration
Threshold optimization
62. Initial Preprocessing Decisions

The audit produces the following preliminary decisions:

Issue	Preliminary Decision
Raw files	Preserve unchanged
Customer ID	Retain for tracking, exclude from modeling
Constant variables	Exclude from modeling
Date fields	Parse and validate
Missing values	Investigate before treatment
High missingness	Evaluate by feature and business meaning
Negative ARPU	Investigate before transformation
Class imbalance	Address through evaluation and decision strategy
Temporal features	Create only after prediction-time validation
Test dataset	Keep isolated until final prediction/evaluation stage
Leakage	Resolve before final modeling

These decisions are preliminary and may be updated if further validation reveals new evidence.

63. Audit-to-Modeling Decision Framework

ChurnIQ will follow a controlled transformation process:

Raw Data
↓
Audit
↓
Investigate
↓
Document Decision
↓
Transform
↓
Revalidate
↓
Feature Engineering
↓
Modeling

This prevents undocumented cleaning decisions from influencing model performance or business conclusions.

64. Current Dataset Readiness

The dataset is classified as:

Structurally Ready — Analytically Not Yet Modeling-Ready

The dataset has passed the major structural and integrity checks required to proceed.

However, modeling must wait until the following gating decisions are finalized:

Prediction point
Prediction horizon
Feature availability at prediction time
Leakage assessment
Missing-value treatment
Negative-value treatment
Final preprocessing rules

This distinction is important because a dataset can be technically valid while still being unsuitable for predictive modeling until its temporal and business logic are established.

65. Audit Conclusion

The audit confirms that the ChurnIQ dataset has the required structure and behavioral richness to support:

Customer churn prediction
Churn-driver analysis
Customer risk scoring
Revenue exposure analysis
Risk × Customer Value prioritization
Retention decision support

The main analytical challenge is not simply cleaning the data. It is ensuring that every feature represents information that could legitimately have been known before the churn outcome occurred.

Therefore, the next stage will focus on establishing the validated preprocessing and feature-readiness framework while maintaining complete data lineage from the untouched raw dataset.

Audit Status: COMPLETE

Modeling Status: BLOCKED pending temporal and leakage validation

# ChurnIQ — Final Dataset Readiness & Preprocessing Roadmap

## 66. Dataset Readiness Objective

The objective of this stage is to establish a controlled path from the untouched raw dataset to a **validated, reproducible, modeling-ready dataset**.

Preprocessing will not be treated as routine data cleaning. Every transformation must preserve information, respect prediction-time availability, prevent leakage, and have a documented business or analytical justification.

---

## 67. Raw-to-Model Data Architecture

ChurnIQ will maintain clear separation between source data, processed data, engineered features, and model inputs.

```text
data/raw/
    ↓
Raw Source Data
    ↓
Initial Audit
    ↓
Data Quality & Temporal Validation
    ↓
Preprocessing
    ↓
data/processed/
    ↓
Feature Engineering
    ↓
Model-Ready Features
    ↓
Model Training & Evaluation

Data Handling Rules
Raw files remain unchanged.
Transformations are reproducible.
Processed data is separated from source data.
Feature engineering occurs only after temporal and leakage validation.
Test data remains isolated from model-development decisions.
Major transformations are documented with their rationale.
The same preprocessing logic must be reusable during future prediction.
68. Preprocessing Principles

ChurnIQ will follow six core principles:

1. Preserve Information

Unusual values will not be removed simply because they appear inconvenient.

2. Understand Before Transforming

Missing values, outliers, negative values, and unusual distributions will be investigated before treatment.

3. Prevent Leakage

Only information legitimately available at the prediction point may be used.

4. Prioritize Business Meaning

Statistical treatment will be aligned with the operational meaning of each variable wherever possible.

5. Maintain Reproducibility

Preprocessing must be implemented consistently across training, validation, and future prediction data.

6. Revalidate After Transformation

Every major transformation must be followed by data-quality and integrity checks.

69. Preprocessing Roadmap

The preprocessing workflow will follow these stages.

Stage 1 — Data Loading
Load the raw training and test datasets.
Confirm expected files and schemas.
Record dataset dimensions.
Preserve the original source files.
Stage 2 — Data Type Standardization
Convert date-like fields to appropriate date formats.
Validate numerical variables.
Validate binary variables.
Validate the target datatype.
Keep id as an identifier rather than a predictive feature.
Stage 3 — Structural Cleaning
Exclude constant variables from the modeling feature set.
Check duplicate records.
Check duplicate customer IDs.
Confirm train/test schema alignment.
Verify that the target exists only in training data.
Stage 4 — Missing-Value Treatment

For each relevant variable or feature group:

Measure missingness.
Determine its business meaning.
Distinguish between "no activity" and "unknown/unavailable".
Evaluate whether missingness itself is informative.
Select an appropriate treatment.
Revalidate the resulting feature.

No blanket rule such as replacing every missing value with zero will be applied.

Stage 5 — Numerical Validation

Investigate:

Negative values
Extreme values
Zero-heavy distributions
Highly skewed variables
Implausible ranges
Redundant numerical measures

Transformations will only be applied when justified by the data and business context.

Stage 6 — Temporal Validation

Before creating temporal features:

Establish the prediction point.
Establish the prediction horizon.
Identify information available at prediction time.
Identify information representing future behavior.
Define the valid observation window.

Only leakage-safe temporal features will proceed to modeling.

Stage 7 — Train/Validation Preparation

The validation strategy will reflect the dataset's temporal structure.

Where appropriate, time-aware validation will be preferred over a purely random split.

Any learned preprocessing operation will be fitted only on the relevant training data.

Stage 8 — Final Revalidation

The processed dataset will be checked for:

Missing values
Invalid values
Unexpected categories
Duplicate records
Leakage
Schema consistency
Target integrity
Distribution changes caused by preprocessing
70. Feature Eligibility Framework

A candidate variable can become a model feature only if it satisfies the following criteria:

Criterion	Requirement
Business relevance	Represents meaningful customer behavior
Prediction availability	Available at the prediction point
Leakage safety	Contains no future information
Data quality	Meets defined quality standards
Variation	Provides meaningful information
Stability	Performs consistently across validation data
Interpretability	Understandable enough for business use where possible
Reproducibility	Can be generated consistently at prediction time

Predictive performance alone is not sufficient for feature approval.

71. Planned Feature Engineering Families

After preprocessing and temporal validation, the following feature families will be evaluated.

Customer Tenure
Tenure in days
Customer lifecycle stage
Tenure bands where business interpretation is useful
Revenue & Customer Value
Recent ARPU
Revenue trend
Revenue change
Revenue volatility
Recent revenue level
Recharge Behavior
Recharge frequency
Recharge amount
Average recharge value
Recharge trend
Recharge recency
Usage & Engagement
Outgoing usage
Incoming usage
Voice usage trend
Data usage trend
Usage decline
Engagement intensity
Service Adoption
Data-service activity
2G/3G usage
Service adoption indicators
Changes in service usage
Recency

Where valid at the prediction point:

Days since last recharge
Days since recent activity
Activity gaps
Behavioral Change

Potential month-over-month measures include:

June → July
July → August

Examples:

ARPU change
Recharge change
Usage change
Engagement change

The final feature set will be determined by prediction-time availability, leakage checks, business relevance, redundancy, and validation performance.

72. Feature Redundancy & Dimensionality Control

The dataset contains many related monthly variables. Retaining every available variable would increase redundancy and reduce interpretability.

Feature selection will therefore consider:

Highly correlated variables
Duplicate information
Near-constant variables
Repeated monthly measures
Business interpretability
Model stability
Predictive contribution

Where multiple variables communicate substantially similar information, the most useful and interpretable representation will be preferred.

The objective is informative dimensionality, not maximum feature count.

73. Data Lineage & Feature Traceability

Every model feature should be traceable back to its source.

Example:

arpu_8
   ↓
August Revenue Level
   ↓
Feature Engineering
   ↓
recent_arpu
   ↓
Churn Model

For engineered features, the project will document:

Source variable(s)
Calculation logic
Transformation
Prediction-time availability
Business interpretation

This creates a transparent connection between raw customer behavior and model predictions.

74. Processed Dataset Deliverable

The preprocessing stage will produce validated data under:

data/processed/

The processed dataset will contain:

Validated customer records
Approved modeling variables
Correct data types
Documented missing-value treatment
Approved numerical treatments
Leakage-safe temporal information
Valid target labels
Customer identifier for traceability

The raw source files will remain untouched.

75. Preprocessing Validation Checklist

Before the dataset is considered modeling-ready:

 Raw files remain unchanged
 Train/test structure validated
 Duplicate records checked
 Duplicate IDs checked
 Train/test ID overlap checked
 Target validated
 Constant variables handled
 Date fields parsed where required
 Missingness investigated
 Missing-value treatment documented
 Negative values investigated
 Outliers assessed
 Prediction point established
 Prediction horizon established
 Leakage assessment completed
 Feature availability validated
 Validation strategy defined
 Processed data revalidated
 Feature lineage documented
76. Modeling Readiness Gate

ChurnIQ will be classified as Modeling Ready only after all critical gates have passed.

Gate 1 — Data Integrity

Dataset structure, records, identifiers, and schema are validated.

Gate 2 — Target Integrity

Target definition, values, and class distribution are validated.

Gate 3 — Prediction Definition

Prediction point and prediction horizon are explicitly established.

Gate 4 — Leakage Prevention

All candidate features are confirmed to use only permitted information.

Gate 5 — Preprocessing

Required treatments for missing values, dates, constants, numerical anomalies, and other data-quality issues are documented.

Gate 6 — Validation Strategy

A defensible train/validation strategy has been established.

Gate 7 — Reproducibility

The complete preprocessing process can be consistently applied to future prediction data.

Only after all seven gates pass will final model development begin.

77. Phase 2 Completion Criteria

The dataset strategy and profiling phase will be considered complete when:

Dataset source and access conditions are documented.
Raw files are preserved.
Dataset structure is understood.
Major variable groups are documented.
Data-quality issues are identified.
Missingness has been investigated.
Target distribution has been validated.
Temporal structure has been assessed.
Prediction point and horizon are defined.
Leakage risks have been evaluated.
Preprocessing decisions are documented.
Feature eligibility rules are established.
A reproducible processed-data workflow is defined.
78. Final Dataset Readiness Statement

The ChurnIQ dataset contains sufficient customer, behavioral, revenue, recharge, usage, and temporal information to support the project's complete decision framework:

Predict → Explain → Quantify Risk → Prioritize → Recommend Action

The project will therefore proceed to preprocessing and feature-readiness work while maintaining strict controls over:

Data lineage
Prediction-time availability
Leakage prevention
Reproducibility
Business interpretability

The objective is not simply to produce a high-performing churn model.

The objective is to establish a reliable analytical foundation for:

Churn probability estimation
Customer-level risk scoring
Churn-driver interpretation
Revenue exposure analysis
Risk × Customer Value prioritization
Retention decision support

## Final Status

Dataset Profile & Audit Documentation: COMPLETE

Dataset Modeling Readiness: COMPLETE

Prediction point: End of August 2014

Prediction horizon: September 2014 / subsequent churn outcome

Temporal leakage review: PASS

Feature eligibility review: PASS

Modeling readiness: APPROVED for downstream feature engineering and model development.