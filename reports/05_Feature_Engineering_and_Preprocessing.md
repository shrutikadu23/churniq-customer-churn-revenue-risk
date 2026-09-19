# ChurnIQ — Feature Engineering & Preprocessing Framework

## 1. Purpose

This document defines the preprocessing and feature engineering framework for ChurnIQ — Customer Churn Prediction & Revenue Risk Intelligence.

The objective is to transform the validated raw telecom customer dataset into a modeling-ready analytical dataset while preserving business meaning, preventing data leakage, controlling redundancy, and creating features that support both churn prediction and business decision-making.

Feature engineering will follow a business-first principle:

> Every engineered feature must represent a meaningful customer behavior, value signal, engagement pattern, or risk indicator that could reasonably support churn prediction or retention prioritization.

Feature creation will not be performed solely to increase the number of model inputs.

---

## 2. Prediction Framework

The modeling framework is based on the temporal structure established during dataset validation and the case-study methodology reviewed for the dataset.

### Prediction Point

The working prediction point is the end of August 2014.

### Prediction Horizon

The target represents the subsequent churn outcome associated with the case-study churn phase, framed as September 2014.

### Available Predictive Information

The model may use customer information from:

- June 2014
- July 2014
- August 2014

### Restricted Information

Information representing the future churn outcome or any information unavailable at the prediction point must not be used as a predictive feature.

The target variable is:

`churn_probability`

where:

- `0` = retained customer
- `1` = churned customer

The target is used only for supervised learning and evaluation.

The supplied training dataset contains June–August predictor variables and the churn target. September raw predictor columns are not present in the supplied training data. Therefore, September is treated as the outcome horizon rather than as a feature period.

---

## 3. Core Preprocessing Principles

Preprocessing will follow these principles:

1. Raw source data will remain unchanged.
2. Transformations will be reproducible through documented scripts or notebooks.
3. Training-derived preprocessing parameters will be learned only from the training data.
4. Validation data will not influence preprocessing fitting, feature selection, or model tuning.
5. The competition test dataset will remain isolated from development decisions.
6. Missing values will not automatically be replaced with zero.
7. Structurally meaningful missingness will be investigated before treatment.
8. Constant and non-informative variables will be excluded from predictive modeling where justified.
9. Extreme values will be investigated before clipping, transformation, or removal.
10. Date information will be transformed into meaningful temporal features where appropriate.
11. Highly redundant features will be controlled to improve model stability and interpretability.
12. Features unavailable at the prediction point will be excluded.
13. Feature engineering decisions will be documented with their business rationale.
14. Model-generated outputs such as churn probabilities will not be used as inputs to the same model.
15. Any feature-selection decision must be based only on information available within the development process.

---

## 4. Data Lineage

The feature engineering pipeline will follow this sequence:

Raw Dataset  
→ Data Quality Validation  
→ Validated Analytical Dataset  
→ Development Split  
→ Preprocessing  
→ Feature Engineering  
→ Feature Quality Checks  
→ Feature Selection  
→ Modeling Dataset  
→ Model Development  
→ Final Locked Test Prediction

The raw files under `data/raw/` will not be overwritten.

Processed datasets will be stored under:

`data/processed/`

Feature definitions and supporting documentation will be maintained under:

`data/data_dictionary/`

Scripts responsible for repeatable transformations will be maintained under:

`src/` or `scripts/`, depending on implementation purpose.

---

## 5. Identifier Handling

The `id` column uniquely identifies customers and is retained for:

- Customer-level reporting
- Risk output tables
- Business prioritization
- Prediction tracking
- Joining predictions with customer-level information

However, `id` will not be used as a predictive model feature.

The identifier does not represent customer behavior, engagement, or customer value and could introduce meaningless patterns into the model.

---

## 6. Constant and Non-Informative Variables

The initial dataset audit identified:

- `circle_id`
- `last_date_of_month_6`

as constant variables in the training dataset.

These variables contain no variation across customers and therefore provide no predictive information in their current form.

They will be excluded from the modeling feature set unless later investigation identifies a legitimate transformation or business use.

The exclusion will be recorded in the feature register and preprocessing audit trail.

---

## 7. Feature Eligibility Framework

A variable may enter the modeling dataset only when it satisfies the required criteria below.

### 7.1 Prediction-Time Availability

The information could realistically be known at the prediction point.

### 7.2 Business Relevance

The variable represents customer behavior, engagement, revenue, service usage, tenure, or another meaningful business characteristic.

### 7.3 Data Quality

The variable has acceptable data quality after documented preprocessing.

### 7.4 Predictive Utility

The variable provides potentially useful information without directly encoding the target or future outcome.

### 7.5 Leakage Safety

The variable does not contain direct or indirect information about the future churn outcome.

### 7.6 Stability

The feature should be sufficiently stable and interpretable to support reliable predictions and business use.

### 7.7 Redundancy Control

Highly duplicated or strongly overlapping information should be reviewed so that the final feature set remains efficient and interpretable.

---

## 8. Feature Status Framework

Each candidate feature will be assigned one of the following statuses:

| Status | Meaning |
|---|---|
| **Approved** | Passed business, temporal, quality, leakage, and redundancy checks |
| **Candidate** | Potentially useful but requires further validation |
| **Investigate** | Data meaning, quality, or availability requires additional investigation |
| **Excluded** | Fails a required criterion or provides no meaningful modeling value |

A feature will not be considered final merely because it improves model performance in one experiment.

Final feature decisions will consider predictive performance, stability, interpretability, business usefulness, and leakage safety.

---

## 9. Feature Families

Feature engineering will focus on the following business-driven families:

1. Customer tenure
2. Revenue and ARPU
3. Recharge behavior
4. Voice usage
5. Data usage
6. Customer engagement
7. Service adoption
8. Recency
9. Behavioral change
10. Customer value
11. Missingness-based behavioral signals

These families will be evaluated before final inclusion.

The purpose is to capture different dimensions of churn risk rather than generate large numbers of overlapping variables.

---

## 10. Tenure Features

The `aon` variable represents the number of days the customer has been using the operator network.

Potential features include:

- Tenure in days
- Tenure in months
- Tenure bands

Business purpose:

Tenure can help distinguish newer customers from long-standing customers and identify whether churn risk varies across customer lifecycle stages.

The original `aon` value will be retained where appropriate, while derived bands will be considered only if they provide additional business interpretability.

---

## 11. Revenue and ARPU Features

Revenue-related features will represent customer monetary value and recent changes in monetization.

Candidate features include:

- `arpu_6`
- `arpu_7`
- `arpu_8`
- ARPU change from July to August
- ARPU percentage change where mathematically appropriate
- Average ARPU across available months
- Recent ARPU trend

Example:

`arpu_8 - arpu_7`

Business interpretation:

A negative value represents a decline in recent customer revenue contribution.

Revenue features will later support both churn prediction and revenue-risk analysis.

Negative ARPU values identified during the data-quality audit will not be automatically removed or transformed. Their source/business meaning must be considered before final preprocessing.

---

## 12. Recharge Behavior Features

Recharge variables will be used to represent customer payment and engagement behavior.

Candidate features include:

- Monthly recharge amount
- Monthly recharge frequency
- Maximum recharge amount
- Recent recharge change
- Recharge frequency change
- Average recharge amount
- Recharge trend

Examples:

`total_rech_amt_8 - total_rech_amt_7`

`total_rech_num_8 - total_rech_num_7`

Business interpretation:

Declining recharge activity may indicate weakening customer engagement or reduced likelihood of continued service usage.

The previously identified difference between recharge counts and recharge amounts will be investigated before assuming that every apparent inconsistency represents a data error.

---

## 13. Voice Usage Features

Voice usage features will capture customer communication activity.

Candidate variables include:

- Total outgoing usage
- Total incoming usage
- Local usage
- STD usage
- On-network usage
- Off-network usage
- Recent changes in usage

Example:

`total_og_mou_8 - total_og_mou_7`

Business interpretation:

A significant decline in customer usage may indicate weakening engagement before churn.

---

## 14. Data Usage Features

Data usage will represent customer mobile internet engagement.

Potential features include:

- 2G data volume
- 3G data volume
- Total data usage
- Data recharge behavior
- Data service adoption
- Recent change in data usage

Because several data-service variables contain substantial missingness, these features require additional treatment and validation before inclusion.

Missingness will not automatically be interpreted as zero.

Where appropriate, missingness indicators may be evaluated as separate behavioral signals.

---

## 15. Service Adoption Features

Variables representing service adoption will be evaluated for their ability to describe customer product usage and engagement.

Potential areas include:

- Data services
- 2G/3G services
- Facebook/social networking service
- Monthly packs
- Sachet packs
- Night packs

Business purpose:

Additional services may indicate deeper customer engagement and therefore potentially different churn behavior.

The relationship between service adoption and churn will be treated as associative rather than causal.

---

## 16. Recency Features

Recharge-date fields may be transformed into recency measures where their availability and semantics support reliable calculation.

Potential examples include:

- Days since last recharge
- Days since last data recharge
- Recent recharge recency
- Change in recharge recency across months

Business purpose:

Recency features can identify customers whose activity has weakened close to the prediction point.

Date calculations will use only information available at or before the prediction point.

---

## 17. Behavioral Change Features

A major component of ChurnIQ will be identifying deterioration in customer behavior rather than relying only on static monthly values.

Candidate change features include:

- ARPU change
- Recharge amount change
- Recharge frequency change
- Outgoing usage change
- Incoming usage change
- Data usage change
- Service engagement change

Examples:

`arpu_8 - arpu_7`

`total_rech_amt_8 - total_rech_amt_7`

`total_og_mou_8 - total_og_mou_7`

Where appropriate, relative or percentage changes may also be considered.

Behavioral-change features are intended to capture deterioration or improvement approaching the prediction point.

---

## 18. Customer Value Features

Because ChurnIQ is designed for revenue-risk intelligence rather than churn prediction alone, customer value will be considered as a separate analytical dimension.

Potential customer-value signals include:

- Latest ARPU
- Recent recharge value
- Average revenue across available months
- Recent revenue trend
- Recent engagement level

These features will support the later framework:

**Churn Risk + Customer Value → Revenue Risk → Retention Priority**

Customer value features must remain independent of the target and future churn outcome.

---

## 19. Missingness-Based Behavioral Signals

Missingness will be treated as a potential source of business information rather than automatically as a data-quality defect.

Candidate indicators may include:

- Data recharge information available/not available
- Data service information available/not available
- Recharge-date information available/not available

For example:

`is_data_recharge_available_8`

may capture whether a customer had recorded data-recharge activity.

These indicators will be evaluated for:

- Business meaning
- Relationship with churn
- Stability
- Leakage risk
- Redundancy with the underlying variable

Missingness indicators will only be retained when they provide defensible business or predictive value.

---

## 20. Temporal Feature Design

Temporal features will focus on customer behavior leading up to the prediction point.

The primary temporal sequence is:

**June → July → August → Future Churn Outcome**

Candidate temporal transformations include:

- Month-over-month change
- Recent trend
- Recent activity level
- Recency
- Acceleration/deceleration where meaningful
- Multi-month averages

August features may be used because August represents the final available predictive period in the established prediction framework.

No future outcome information will be incorporated into these features.

---

## 21. Leakage Taxonomy

ChurnIQ will explicitly guard against the following forms of leakage.

### Direct Target Leakage

A feature directly contains or is derived from `churn_probability`.

**Decision:** Exclude.

### Temporal Leakage

A feature contains information occurring after the prediction point.

**Decision:** Exclude.

### Aggregation Leakage

An aggregate includes observations from the future outcome period.

**Decision:** Exclude.

### Preprocessing Leakage

Imputation, scaling, encoding, or other transformations are fitted using validation or test information.

**Decision:** Exclude.

### Feature-Selection Leakage

Validation/test performance is used improperly to select features before final evaluation.

**Decision:** Exclude.

### Customer-Level Leakage

The same customer's information is improperly distributed across development boundaries in a way that allows information to cross the modeling boundary.

**Decision:** Investigate and prevent where applicable.

---

## 22. Train, Validation, and Test Separation

The development process will maintain strict separation between datasets.

### Training Data

Used to:

- Fit preprocessing
- Engineer features
- Train models
- Perform cross-validation
- Tune model parameters

### Validation Data

Used to:

- Compare models
- Evaluate thresholds
- Assess probability quality
- Support model selection

Validation information must not be used to fit preprocessing transformations.

### Competition Test Data

`test.csv` will remain isolated from feature selection, model tuning, threshold selection, calibration decisions, and development performance evaluation.

It will be used only at the appropriate final stage for generating predictions.

---

## 23. Feature Redundancy Control

The dataset contains many related monthly variables and overlapping usage measures.

Redundancy will therefore be evaluated using:

- Correlation analysis
- Feature-family review
- Domain/business logic
- Model behavior
- Feature importance stability
- Interpretability

Highly correlated features will not automatically be removed.

Removal will depend on whether the information is genuinely redundant and whether retaining it improves model performance, stability, or interpretability.

---

## 24. Feature Register

A feature register will be maintained during development.

The register will document:

| Field | Purpose |
|---|---|
| Feature Name | Final technical name |
| Feature Family | Business category |
| Source Variables | Original variables used |
| Business Meaning | What the feature represents |
| Time Window | Period represented |
| Prediction-Time Availability | Whether available at prediction point |
| Leakage Check | Leakage assessment |
| Missingness Treatment | Applied handling |
| Transformation | Mathematical/logic transformation |
| Status | Approved / Candidate / Investigate / Excluded |
| Final Decision | Reason for inclusion or exclusion |

Example:

| Feature | Family | Business Meaning | Time Window | Status |
|---|---|---|---|---|
| `arpu_8` | Revenue | Latest customer revenue | August | Candidate |
| `arpu_change_8_7` | Behavioral Change | Recent revenue deterioration | July→August | Candidate |
| `rech_amt_change_8_7` | Recharge | Recent recharge deterioration | July→August | Candidate |
| `days_since_recharge_8` | Recency | Recent recharge inactivity | August | Candidate |

The feature register will become the audit trail for the final modeling dataset.

---

## 25. Feature Selection Philosophy

Final feature selection will not be based solely on statistical significance or model importance.

Features will be assessed across five dimensions:

1. **Business relevance**
2. **Prediction-time validity**
3. **Data quality**
4. **Predictive usefulness**
5. **Interpretability and stability**

A feature that appears predictive but cannot be justified as available at prediction time will be excluded.

Similarly, a feature that is statistically useful but has unclear business meaning may require additional investigation before final inclusion.

---

## 26. Business Decision Architecture

The feature engineering framework supports the broader ChurnIQ decision architecture:

Customer Behavior  
↓  
Churn Risk  
↓  
Calibrated Churn Probability  
↓  
Customer Value  
↓  
Revenue Exposure  
↓  
Risk × Value Prioritization  
↓  
Retention Action

This ensures that the modeling layer remains connected to the final business decision rather than becoming an isolated machine-learning exercise.

---

## 27. Feature Engineering Decision Rule

Every candidate feature will pass through the following sequence:

**Business Meaning  
→ Prediction-Time Availability  
→ Data Quality  
→ Leakage Check  
→ Missingness Assessment  
→ Redundancy Review  
→ Predictive Utility  
→ Stability  
→ Interpretability  
→ Final Status**

Only features that pass the required checks will be included in the final modeling dataset.

---

## 28. Definition of Done

Feature engineering and preprocessing will be considered complete when:

- Prediction point and horizon are documented.
- Feature eligibility rules are documented.
- Leakage rules are documented.
- Raw data remains unchanged.
- Train/validation/test separation is enforced.
- Missing-value treatment is justified.
- Outlier treatment is justified.
- Constant/non-informative features are handled.
- Temporal features are validated.
- Business-driven feature families are evaluated.
- Redundant features are reviewed.
- A feature register is maintained.
- Final feature decisions are reproducible.
- The processed dataset passes quality checks.
- No feature uses future outcome information.
- The final modeling dataset is ready for baseline and model development.

## 29. Current Status

**Feature Engineering Framework: Defined**

**Prediction-Time Framework: Defined**

**Leakage Controls: Defined**

**Feature Construction: Pending Implementation**

**Preprocessing Pipeline: Pending Implementation**

**Final Modeling Dataset: Pending**

# ChurnIQ — Feature Engineering & Preprocessing Framework
## Part 2: Missing Values, Outliers, Temporal Features & Feature Register

## 30. Preprocessing and Feature Engineering Objective

The objective of this stage is to transform the validated ChurnIQ dataset into a reliable, reproducible, modeling-ready feature dataset.

The process must preserve:

- Customer-level identity
- Business meaning
- Temporal ordering
- Prediction-time availability
- Data quality
- Reproducibility
- Model interpretability

The process will not focus on maximizing the number of features.

Instead:

> ChurnIQ will create the smallest practical set of business-relevant features that captures customer behavior, customer value, behavioral deterioration, and churn risk without introducing leakage.

---

## 31. Processing Sequence

The preprocessing and feature engineering workflow will follow a controlled sequence:

**Raw Training Data**

→ Structural Validation

→ Development Split

→ Missingness Assessment

→ Data-Type Conversion

→ Missing-Value Treatment

→ Outlier Assessment

→ Temporal Feature Engineering

→ Behavioral Feature Engineering

→ Recency Feature Engineering

→ Customer-Value Feature Engineering

→ Feature Quality Checks

→ Redundancy Review

→ Feature Register Update

→ Leakage Validation

→ Final Feature Freeze

→ Modeling

The exact implementation will be maintained in reproducible Python code rather than relying on undocumented manual transformations.

---

## 32. Development Data Separation

Feature engineering decisions must be made within the development dataset without using the locked competition test data.

The development workflow will maintain the following conceptual separation:

```text
Training Data
    ↓
Development Split
    ├── Training Portion
    └── Validation Portion

Locked Competition Test
    ↓
Reserved for Final Prediction

Any transformation that learns parameters from data must be fitted using the training portion only.

This includes:

Imputation values
Scaling parameters
Transformation parameters
Outlier thresholds
Encoding rules
Feature-selection thresholds

Validation data will only be transformed using parameters learned from the training portion.

The competition test dataset will remain isolated until the final prediction stage.

33. Temporal Validation Principle

Because ChurnIQ is based on customer behavior over time, temporal validity will take priority over randomly mixing future and past observations where the modeling design requires temporal separation.

The conceptual timeline is:

June → July → August → Future Churn Outcome

The model predicts the subsequent churn outcome using information available through August.

Where the available dataset structure permits meaningful temporal validation, the validation strategy will respect this ordering.

A random split may be used only when justified by the final modeling design and documented appropriately.

The final validation strategy will be selected based on:

Dataset structure
Prediction point
Prediction horizon
Customer-level independence
Availability of historical periods
Leakage risk
34. Missing Value Strategy

Missing values will be treated according to their business and data-generating meaning.

The core rule is:

Missing does not automatically mean zero.

The initial audit identified substantial missingness in several data-service and recharge-related variables.

Therefore, each important missingness pattern will be investigated before treatment.

35. Missingness Categories

Missing values will be classified into four broad categories:

Category	Meaning	Initial Treatment
Structural Missingness	Feature is not applicable to the customer/service	Preserve meaning
Conditional Missingness	Value exists only when an activity/service occurs	Investigate semantics
Data-Quality Missingness	Expected information is unexpectedly absent	Investigate source/data quality
Unknown Missingness	Cause cannot be confidently established	Conservative treatment + monitoring

This classification prevents a single imputation rule from being applied across unrelated feature families.

36. Missingness Assessment

Missingness will be assessed using:

Overall missing percentage
Missingness by month
Missingness by feature family
Missingness by churn status
Related-variable missingness patterns
Business interpretation
Predictive usefulness
Stability

The observed difference in missingness between retained and churned customers suggests that some missingness patterns may carry behavioral information.

However:

Missingness associated with churn does not establish that missingness causes churn.

Therefore, missingness indicators will be treated as candidate predictive signals and validated independently.

37. Numeric Missing Value Treatment

Numeric imputation will only be applied where appropriate.

Potential strategies include:

Median imputation
Business-rule-based treatment
Explicit missing indicators
Native model handling of missing values where appropriate

For median or other learned imputation:

The imputation value must be learned from the training data only.

It must not be calculated using:

Validation data
Competition test data
Future outcome information

For skewed variables, median-based treatment may be preferred over mean-based treatment.

38. Structural Missingness Treatment

Variables related to data services may have high missingness because the underlying service or activity may not apply to every customer.

Examples include:

Data recharge variables
Data recharge dates
2G/3G-related variables
Service adoption variables

These variables will not automatically be converted to zero.

Before treatment, the relationship between:

Missing → No activity → Zero → Not applicable

must be understood.

If business semantics support it, a combination of:

Value treatment
Missingness indicator

may be used.

39. Missingness Indicators

Where justified, binary indicators may be created.

Example:

data_recharge_missing_8

with:

1 = original value was missing
0 = original value was available

Potential indicators include:

data_recharge_missing_8
last_data_recharge_missing_8
arpu_3g_missing_8
fb_user_missing_8

Each indicator must pass:

Business interpretation
Leakage check
Predictive usefulness
Stability assessment
Redundancy review
40. Missing Date Treatment

Missing dates will not be replaced with artificial dates.

For example, a missing last-recharge date must not automatically be assigned:

Month start
Month end
Zero
A random date

Instead, the feature design will distinguish between:

Valid activity date
Missing activity date

Where useful, the missingness state may become a separate indicator.

41. Date Conversion

Relevant date fields will be converted into appropriate date/datetime representations before temporal feature creation.

Relevant fields include:

last_date_of_month_6
last_date_of_month_7
last_date_of_month_8
date_of_last_rech_6
date_of_last_rech_7
date_of_last_rech_8
date_of_last_rech_data_6
date_of_last_rech_data_7
date_of_last_rech_data_8

The constant month-end fields will not be treated as customer-level predictive variables.

42. Outlier Philosophy

The dataset contains highly skewed revenue, recharge, and usage variables.

Extreme values will not automatically be considered errors.

A high-value customer may legitimately have:

High recharge amounts
High usage
High ARPU
High service consumption

Therefore:

Outlier detection is an investigation step, not an automatic deletion rule.

43. Outlier Detection

Potential techniques include:

Minimum/maximum inspection
Percentile analysis
IQR analysis
Distribution analysis
Business-rule validation
Log-scale inspection
Model sensitivity analysis

Outliers will be classified as:

Classification	Treatment
Valid Extreme	Retain
Potential Data Error	Investigate
Valid but Highly Skewed	Consider transformation
Clearly Invalid	Correct/remove only with evidence

No clipping or winsorization will be performed simply because a value is statistically extreme.

44. Negative ARPU Treatment

The initial audit identified negative ARPU values.

These values will not automatically be:

Deleted
Replaced with zero
Clipped
Winsorized

Their meaning must first be understood using the data dictionary and source context.

Until then:

Negative ARPU Treatment: Investigate

If the values are valid business records, they should not be removed simply because they appear unusual.

If they are determined to be invalid, the treatment will be documented and revalidated.

45. Recharge Amount and Frequency Anomaly

The audit identified customers where:

Recharge frequency > 0
Recharge amount = 0

This pattern occurs frequently enough that it should not automatically be treated as an error.

Possible explanations must be investigated through source semantics before creating derived inconsistency features.

Therefore:

Recharge Count/Amount Relationship: Investigate

No business conclusion will be drawn from this pattern until its meaning is established.

46. Temporal Feature Principle

Temporal feature engineering will use only:

June → July → August

with August representing the latest available predictive period.

The objective is to capture:

Current level
Direction of change
Deterioration
Improvement
Stability
Recent inactivity
Customer trajectory

No future churn-period information may enter the feature set.

47. Month-over-Month Change Features

For suitable variables, absolute changes may be calculated.

Examples:

arpu_change_7_6 = arpu_7 - arpu_6

arpu_change_8_7 = arpu_8 - arpu_7

recharge_amt_change_8_7 =
total_rech_amt_8 - total_rech_amt_7

outgoing_usage_change_8_7 =
total_og_mou_8 - total_og_mou_7

These features represent recent behavioral direction.

A negative value may indicate deterioration, depending on the underlying metric.

48. Percentage Change Features

Percentage change may provide additional context where the denominator is meaningful.

Conceptually:

Percentage Change =
(Current Value - Previous Value) / Absolute(Previous Value)

Percentage-change features must include safeguards for:

Zero denominators
Very small denominators
Missing values
Infinite values
Extremely large ratios

If the calculation produces unstable values, the feature may be excluded or replaced with a more robust representation such as absolute change.

49. Ratio Feature Controls

Derived ratios will only be created when their denominator has a meaningful business interpretation.

Before final inclusion, ratio features must be checked for:

Zero denominator
Near-zero denominator
Infinite values
Extreme skew
Meaningless mathematical combinations
Leakage

Example:

recharge_amount / recharge_count

is potentially meaningful as average recharge value per event.

However, the feature must handle customers with zero recharge events appropriately.

No ratio will be created merely because two variables can mathematically be divided.

50. Multi-Month Summary Features

Customer behavior may also be summarized across June–August.

Potential features include:

Three-month average
Three-month total
Minimum
Maximum
Standard deviation
Range
Recent value versus historical average

Examples:

avg_arpu_6_8

avg_recharge_amt_6_8

avg_outgoing_usage_6_8

These features capture customer level and consistency across the observed period.

51. Recent-vs-Baseline Features

A key ChurnIQ feature concept is comparison of the latest behavior with the customer's own historical baseline.

For example:

arpu_8 - average(arpu_6, arpu_7)

This measures how August performance differs from the customer's earlier behavior.

This approach can identify:

Stable customers
Improving customers
Deteriorating customers
Sudden declines

The feature is based on customer-specific behavior rather than only population-level comparison.

52. Trend Features

Where sufficient information exists, trend features may represent:

Direction
Magnitude
Consistency
Volatility
Recent acceleration/deceleration

For example:

High → Medium → Low

represents a different behavioral trajectory from:

Low → Low → Low

even if both customers have similar August values.

Therefore, ChurnIQ will evaluate both:

Behavior Level

and

Behavior Trajectory

53. Zero-Activity Features

For features where zero has a valid business interpretation, zero-activity indicators may be evaluated.

Examples:

No recharge in August
No outgoing usage in August
No incoming usage in August
No data usage in August

The distinction between:

Zero activity

and

Missing information

must remain explicit.

These indicators will only be created where the source semantics support the distinction.

54. Recency Features

Recency features will measure how recently a customer performed an activity before the prediction point.

Conceptually:

Recency =
Prediction Point - Last Activity Date

Potential features include:

Days since last recharge
Days since last data recharge

Recency features must:

Use only available information
Handle missing dates correctly
Produce valid non-negative values
Avoid artificial dates
Be checked for extreme values
55. Customer Value Features

Customer value will be treated as a distinct analytical dimension.

Potential value signals include:

Latest ARPU
Average ARPU
Recent recharge value
Recharge frequency
Recent revenue trend
Recent engagement

These features support the broader ChurnIQ architecture:

Customer Behavior → Churn Risk

and separately:

Customer Value → Revenue Exposure

which later combine into:

Churn Risk + Customer Value → Retention Priority

Customer value features must not use:

Churn target
Future outcome information
Model-generated churn probability
56. Risk and Revenue Boundary

ChurnIQ will maintain a strict distinction between:

Predictive Layer

Customer information available at prediction time is used to estimate:

Probability of Churn

Business Intelligence Layer

The resulting churn probability may later be combined with customer value to estimate:

Revenue exposure
Risk-weighted revenue
High-value/high-risk customers
Retention priority

Therefore:

Model predictions are downstream outputs and must not be fed back into the same model as predictive inputs.

57. Feature Interaction Policy

Feature interactions and ratios will be created selectively.

A candidate interaction must have:

Clear business meaning
Stable mathematical behavior
Prediction-time availability
Low leakage risk
Potential predictive value

The project will avoid generating large numbers of arbitrary interaction terms simply to increase model complexity.

Where tree-based models can naturally capture relationships, manually engineered interactions will only be added when they provide clear business or analytical value.

58. Feature Redundancy Control

The dataset contains many related monthly and usage variables.

Redundancy will be evaluated using:

Correlation analysis
Feature-family review
Domain knowledge
Model behavior
Feature importance stability
Interpretability

Highly correlated variables will not automatically be removed.

The final decision will consider whether the feature contributes meaningful information beyond another variable.

59. Feature Naming Convention

Feature names will follow a consistent naming system.

Examples:

arpu_change_8_7
avg_arpu_6_8
recharge_amt_change_8_7
days_since_recharge_8
data_recharge_missing_8

Names should communicate:

Metric
Transformation
Time period

This improves:

Reproducibility
Debugging
Model interpretation
SHAP analysis
SQL/Python consistency
Dashboard documentation
60. Feature Provenance

Every engineered feature must be traceable to its original source variables.

For example:

Feature:
arpu_change_8_7

Source:
arpu_8
arpu_7

Transformation:
arpu_8 - arpu_7

Business Meaning:
Recent change in customer ARPU

Prediction Availability:
Available at end of August

Leakage Status:
Pass

Feature provenance will make the modeling dataset auditable and easier to explain to business stakeholders.

61. Leakage Taxonomy

ChurnIQ will explicitly test for the following leakage types.

61.1 Direct Target Leakage

A feature directly contains or is derived from:

churn_probability

Decision: Exclude

61.2 Temporal Leakage

A feature uses information occurring after the prediction point.

Decision: Exclude

61.3 Outcome Leakage

A feature indirectly reflects the future churn outcome.

Decision: Exclude

61.4 Aggregation Leakage

An aggregate includes future-period observations.

Decision: Exclude

61.5 Preprocessing Leakage

A preprocessing parameter is learned using validation or test information.

Decision: Exclude

61.6 Feature-Selection Leakage

Validation/test information improperly determines the final feature set.

Decision: Exclude

61.7 Customer-Level Leakage

Information from the same customer crosses a development boundary in a way that exposes future or validation information.

Decision: Prevent and validate

62. Automated Feature Quality Checks

The final feature dataset will be checked programmatically for:

Structural Checks
Expected row count
Expected customer count
Unique customer IDs
Expected feature columns
Numerical Checks
Missing values
Infinite values
Unexpected negative values
Unexpected zero values
Extreme derived ratios
Invalid calculations
Temporal Checks
No future information
Valid date arithmetic
Correct month ordering
Valid recency values
Business Checks
Feature definitions match their intended meaning
Units are consistent
Direction of change is interpretable
Derived values are mathematically valid
63. Feature Leakage Tests

Before modeling, explicit leakage tests will be performed.

Examples include checking for:

Target-derived columns
September/future-period variables
Duplicate target information
Post-prediction dates
Suspiciously perfect relationships with the target
Features created using full-dataset preprocessing
Validation/test contamination

The objective is not simply to avoid obvious target columns, but to detect indirect leakage introduced during preprocessing or feature construction.

64. Feature Risk Classification

Each feature will receive a risk classification.

Risk	Meaning
Low	Clearly available, stable, well-defined
Medium	Requires transformation or additional interpretation
High	High missingness, unusual semantics, complex transformation, or leakage concern
Excluded	Fails required criteria

High-risk features require additional validation before final inclusion.

65. Feature Status Framework

Each feature will receive one of four statuses:

Status	Meaning
Approved	Passed required business, temporal, quality, leakage, and redundancy checks
Candidate	Potentially useful but requires modeling validation
Investigate	Meaning or quality requires further investigation
Excluded	Fails required criteria or provides insufficient value

A feature will not become approved simply because it improves one model metric.

66. Feature Register

The final feature register will contain:

Field	Purpose
Feature Name	Final technical name
Feature Family	Business category
Source Variables	Original variables
Transformation	Calculation/logic
Business Meaning	What it represents
Time Window	Period represented
Prediction-Time Availability	Available at prediction point
Missingness Treatment	Handling method
Leakage Check	Leakage result
Risk Level	Modeling risk
Status	Approved/Candidate/Investigate/Excluded
Final Decision	Reason for final inclusion/exclusion

Example:

Feature	Family	Source	Business Meaning	Time	Leakage	Risk	Status
arpu_8	Revenue	arpu_8	Latest customer revenue	Aug	Pass	Low	Candidate
arpu_change_8_7	Behavioral Change	arpu_8, arpu_7	Recent revenue change	Jul→Aug	Pass	Low	Candidate
rech_amt_change_8_7	Recharge	total_rech_amt_8, total_rech_amt_7	Recent recharge change	Jul→Aug	Pass	Low	Candidate
days_since_recharge_8	Recency	date_of_last_rech_8	Recharge inactivity	Aug	Pending	Medium	Candidate
data_recharge_missing_8	Missingness	total_rech_data_8	Data-recharge availability	Aug	Pass	Medium	Candidate
67. Revalidation After Transformation

The original dataset audit is not considered sufficient after preprocessing.

The processed feature dataset must undergo a second validation cycle:

Validate → Transform → Revalidate

This must confirm that preprocessing has not introduced:

Duplicate customers
Unexpected missing values
Infinite values
Invalid calculations
Impossible dates
Temporal leakage
Unexpected distributions
68. Reproducibility Requirements

The feature engineering pipeline must be reproducible.

The implementation will maintain:

Fixed random seeds where randomness is used
Documented transformation logic
Consistent feature names
Recorded feature definitions
Reproducible train/validation split logic
Versioned Python dependencies
Saved preprocessing/model pipeline where appropriate

A future run of the project should produce the same feature definitions under the same input and configuration.

69. Class Imbalance Boundary

The initial dataset contains approximately 10.19% churned customers.

Class imbalance will be addressed during model development rather than by altering the raw or processed customer population during feature engineering.

Potential modeling approaches may later include:

Class weighting
Threshold optimization
Appropriate sampling strategies where justified

Any sampling technique will be applied only within the training process and never to validation/test evaluation data.

The original customer distribution will remain available for business reporting.

70. Feature Freeze

Before baseline model development, the project will establish a formal Feature Freeze.

At feature freeze:

Final feature definitions are documented.
Feature provenance is documented.
Leakage checks are passed.
Missing-value treatment is finalized.
Outlier decisions are documented.
Temporal features are finalized.
Redundant features are reviewed.
Validation checks pass.
The final feature register is updated.

After the feature freeze, changes to the feature set will be versioned and documented rather than silently replacing the original design.

71. Final Feature Engineering Workflow

The implementation workflow will be:

1. Load validated training data

↓

2. Establish development split

↓

3. Assess missingness

↓

4. Investigate ambiguous missingness

↓

5. Convert data types

↓

6. Evaluate outliers

↓

7. Investigate negative ARPU

↓

8. Investigate recharge semantics

↓

9. Create approved temporal features

↓

10. Create behavioral-change features

↓

11. Create recency features

↓

12. Create customer-value features

↓

13. Evaluate missingness indicators

↓

14. Validate derived ratios and transformations

↓

15. Review feature redundancy

↓

16. Update feature register

↓

17. Run automated quality checks

↓

18. Run leakage checks

↓

19. Revalidate processed dataset

↓

20. Freeze final feature definition

↓

21. Begin baseline modeling

72. Definition of Done — Part 2

Part 2 will be considered complete when:

Missingness categories are defined.
Missing-value treatment rules are documented.
High-missingness service variables are investigated.
Missingness indicators are evaluated appropriately.
Negative ARPU treatment is documented.
Recharge amount/frequency behavior is investigated.
Outlier treatment principles are defined.
Date conversion rules are defined.
Recency features are defined.
Month-over-month features are defined.
Percentage-change safeguards are defined.
Ratio-feature safeguards are defined.
Multi-month summary features are defined.
Recent-vs-baseline features are defined.
Temporal trend features are defined.
Leakage taxonomy is documented.
Temporal validation principles are documented.
Train/validation/test separation is enforced.
Feature provenance is documented.
Feature naming standards are established.
Feature risk classification is established.
Feature register is established.
Automated quality checks are defined.
Revalidation requirements are defined.
Reproducibility requirements are defined.
Class-imbalance handling is separated from preprocessing.
Feature freeze requirements are defined.
73. Current Status

Preprocessing Framework: Defined

Missing Value Strategy: Defined

Outlier Strategy: Defined

Temporal Feature Strategy: Defined

Recency Strategy: Defined

Customer Value Feature Strategy: Defined

Leakage Controls: Defined

Feature Provenance: Defined

Feature Register: Defined

Quality & Revalidation Framework: Defined

Reproducibility Framework: Defined

Feature Implementation: Pending

Processed Modeling Dataset: Pending

Feature Freeze: Pending

Baseline Modeling: Pending

## 74. Missingness Treatment Policy

### 74.1 Purpose

The missingness investigation identified distinct patterns across customer
usage, recharge, data-service, service-status, and date-related variables.

Missing values will therefore not be handled using a single global imputation
rule.

The objective of the missingness treatment strategy is to:

- preserve meaningful customer-behavior information;
- distinguish structural missingness from ordinary missingness;
- avoid converting unknown values into artificial behavioral observations;
- preserve potentially informative missingness signals;
- prevent validation or test information from influencing preprocessing;
- maintain consistent prediction-time behavior;
- make every treatment decision traceable and reproducible.

No final missing-value transformation will be applied until the relevant
feature semantics have been reviewed and the treatment decision has been
validated.

---

### 74.2 Core Treatment Principles

ChurnIQ follows the following principles for missing-value handling:

1. Missing values are not automatically treated as data-quality errors.
2. Missingness may represent customer behavior, service non-adoption,
   unavailable information, or data-recording conditions.
3. Missingness must be interpreted using both feature semantics and observed
   value patterns.
4. High missingness alone is not sufficient reason to remove a feature.
5. Missingness may itself contain predictive information and should not be
   automatically eliminated through imputation.
6. Zero will only be used when zero has a valid business interpretation.
7. Dates will not be replaced with arbitrary or statistical dates.
8. Learned preprocessing parameters will be fitted using development data
   only.
9. Validation data will not influence preprocessing parameters or treatment
   decisions.
10. Competition test data will remain isolated until the development workflow
    is frozen.
11. Target-wise missingness analysis may identify potentially informative
    patterns, but the churn target must not be used as the sole basis for
    assigning treatment values.
12. Every feature-level treatment decision will be recorded in the feature
    treatment matrix.
13. Post-treatment validation is mandatory before the modeling dataset is
    considered ready.

---

### 74.3 Missingness Mechanism Classification

Each feature with missing values will be assigned to one of four analytical
categories.

#### Category A — Structural / Service Non-Use

Missingness may indicate that a customer did not use, subscribe to, or activate
the corresponding service.

Examples include certain data-service and service-status variables.

Treatment may involve:

- a valid zero transformation where the data semantics confirm non-use;
- a service-availability indicator;
- a missingness indicator where it provides additional information.

Exact treatment remains feature-dependent.

---

#### Category B — Potentially Informative Missingness

Missingness may contain predictive information about customer behavior.

This category is particularly relevant where missingness differs materially
between retained and churned customers.

Treatment will aim to preserve the distinction between:

`observed value`

and

`missing value`

where that distinction has analytical value.

Missingness indicators may therefore be retained.

Target-wise analysis is used for investigation only and does not establish
causality.

---

#### Category C — Ordinary / Limited Missingness

Missingness appears relatively limited and does not currently show evidence of
structural service behavior.

If a complete numeric modeling matrix requires treatment, controlled
statistical imputation may be considered.

The default candidate for skewed numeric telecom variables is development-set
median imputation.

This is a candidate treatment rather than an automatic rule.

---

#### Category D — Unknown / Requires Investigation

Where the meaning of missingness cannot be confidently established from the
data dictionary and observed value patterns, the feature will remain under
investigation.

No zero conversion or arbitrary imputation will be applied until the semantic
meaning is sufficiently established.

---

### 74.4 Structural Data-Service Missingness

A major synchronized missingness block was identified across data-service
variables, including:

- `total_rech_data_*`
- `count_rech_2g_*`
- `count_rech_3g_*`
- `av_rech_amt_data_*`
- `max_rech_data_*`
- `arpu_2g_*`
- `arpu_3g_*`
- `fb_user_*`
- `night_pck_user_*`

These variables exhibit highly synchronized missingness patterns, suggesting
that the missingness is structured rather than independently random.

The investigation also found strong co-occurrence among multiple
data-service missingness fields.

However, synchronized missingness does not by itself prove that every missing
value represents zero usage.

Therefore, treatment will be determined at the feature or feature-family
level.

Where the data semantics confirm:

`missing = service not used / service not applicable`

then an appropriate zero or non-user transformation may be applied.

Where missingness instead represents:

`unknown / unavailable / not recorded`

the missing state will be preserved or handled through controlled imputation.

---

### 74.5 Data-Service Missingness as a Behavioral Signal

Data-service missingness, particularly in the later observation period, shows
meaningful differences across churn outcomes.

This makes missingness potentially useful as a predictive signal.

However:

- predictive association does not imply causation;
- target-wise differences will not determine the treatment value;
- missingness indicators will be created only where they provide information
  that would otherwise be lost;
- treatment will be consistent across development, validation, and prediction
  data.

The final model will therefore be allowed to learn from meaningful
missingness without treating missingness as proof of future churn.

---

### 74.6 August Voice-Usage Missingness

Several August voice-usage variables show substantially different missingness
rates between retained and churned customers.

This pattern makes August voice-usage missingness a high-priority area for
feature investigation.

The project will not automatically interpret these missing values as zero.

Before transformation, each relevant feature will be checked against its
documented business definition.

Where the semantics support:

`missing = no recorded usage`

the value may be converted to zero while preserving a missingness indicator
when useful.

Where the semantics indicate unavailable or unknown information, the missing
state will be retained or handled through controlled imputation.

---

### 74.7 Date Missingness

Date variables require separate treatment because statistical date imputation
can create artificial customer activity.

Relevant examples include:

- `date_of_last_rech_6`
- `date_of_last_rech_7`
- `date_of_last_rech_8`
- `date_of_last_rech_data_6`
- `date_of_last_rech_data_7`
- `date_of_last_rech_data_8`

Missing dates will not be replaced with:

- the median date;
- the mean date;
- the minimum date;
- the maximum date;
- an arbitrary fixed date.

Instead, dates will be used to derive business-relevant features such as:

- recharge recency;
- data-recharge recency;
- recharge activity indicators;
- data-service activity indicators.

The original date fields will not automatically be passed directly into the
final modeling matrix.

---

### 74.8 Low-Missingness Numeric Features

Numeric features with limited missingness will first be evaluated for whether
imputation is actually necessary.

If imputation is required:

- the imputation parameter will be learned from development data only;
- median imputation will be preferred for skewed telecom usage and monetary
  variables;
- the same fitted parameter will be applied to validation and future
  prediction data;
- a missingness indicator will be considered where the missing state may
  contain additional information.

No feature will receive median imputation merely because it contains missing
values.

---

### 74.9 Zero-Imputation Boundary

Zero will only be used when zero has a valid business interpretation.

Potential examples include:

- no recharge activity;
- no recorded service usage;
- no applicable service activity;
- confirmed non-adoption of a service.

Zero will not be used merely because:

- a feature has high missingness;
- a feature is numeric;
- the modeling algorithm requires a complete matrix;
- zero is computationally convenient;
- the feature belongs to a service-related family.

This boundary prevents unknown values from being converted into false
behavioral observations.

---

### 74.10 Missingness Indicators

Missingness indicators will be considered when:

- missingness has a meaningful behavioral interpretation;
- missingness differs materially across customer outcomes;
- structural missingness exists;
- imputation could otherwise remove useful information.

Naming convention:

`<original_feature>_missing`

Example:

`onnet_mou_8_missing`

Encoding:

- `1` = original value was missing;
- `0` = original value was observed.

Indicators will be generated before transformations that could otherwise
remove the distinction between missing and observed values.

---

### 74.11 Target Independence in Treatment Decisions

The churn target is available for training analysis and can be used to
investigate whether missingness patterns differ across outcomes.

However, the target will not be used to calculate:

- imputation values;
- feature-level replacement values;
- missingness encodings;
- service-status mappings.

Target-wise missingness analysis is therefore treated as an analytical
diagnostic rather than a preprocessing input.

This prevents treatment rules from becoming dependent on the observed training
target distribution.

---

### 74.12 Development-Validation Boundary

All learned preprocessing parameters will follow the development/validation
boundary.

The development dataset will be used to learn:

- imputation parameters;
- scaling parameters where required;
- encoding mappings where required;
- transformation parameters.

Validation data will only receive transformations learned from development
data.

Validation statistics will not be used to fit preprocessing operations.

This rule will also apply to any preprocessing introduced during later model
experimentation.

---

### 74.13 Test Isolation

The competition test dataset will remain completely isolated during
development.

`test.csv` will not be used to:

- determine missingness treatment rules;
- calculate imputation parameters;
- select features;
- tune preprocessing;
- select model thresholds;
- evaluate model performance.

The final fitted preprocessing pipeline will only be applied to the
competition test dataset after the development workflow has been frozen.

---

### 74.14 Feature-Level Treatment Matrix

The final treatment decision will be maintained at feature or feature-family
level rather than applying one global rule to all missing variables.

Each relevant feature will receive:

- missingness mechanism;
- business interpretation;
- proposed treatment;
- missingness-indicator decision;
- preprocessing method;
- validation result;
- final status.

The treatment matrix will serve as the authoritative record of preprocessing
decisions.

Example structure:

| Feature / Family | Missingness Mechanism | Business Interpretation | Treatment | Indicator | Status |
|---|---|---|---|---|---|
| Low-missing numeric | Ordinary / limited | Unknown value | Development median if required | Consider | Pending |
| Data-service usage | Structural candidate | Non-use / non-applicability candidate | Semantic validation before zero treatment | Yes if informative | Pending |
| Service-status flags | Structural candidate | Service availability candidate | Semantic mapping | Consider | Pending |
| Recharge dates | Structural / activity | No observed recharge date | Recency/activity features | Yes | Approved |
| Data-recharge dates | Structural / activity | No observed data recharge date | Recency/activity features | Yes | Approved |
| August voice usage | Potentially informative | Requires semantic validation | Preserve signal; treatment after validation | Yes if justified | Pending |

The matrix will be updated after implementation and post-treatment validation.

---

### 74.15 Zero-Inflation and Missingness Distinction

Zero and missing values will be treated as analytically distinct unless the
feature semantics explicitly establish that they represent the same business
state.

For applicable features:

- zero = observed absence of activity;
- missing = no recorded / unavailable / structurally non-applicable value.

The distinction will be preserved where it provides meaningful information.

This prevents preprocessing from collapsing two potentially different
customer states into a single value.

---

### 74.16 Outlier and Missingness Interaction

Missing-value treatment will not be used as a substitute for outlier
treatment.

Extreme values will be investigated separately using feature semantics and
distributional diagnostics.

In particular:

- negative ARPU values will not be automatically removed;
- recharge anomalies will not be automatically corrected;
- extreme usage values will not be winsorized solely to improve model
  performance.

Any outlier transformation must receive a separate documented justification.

---

### 74.17 Post-Treatment Validation

After missingness treatment is implemented, the pipeline will validate:

#### Structural integrity
- row count;
- column count;
- customer IDs;
- target values;
- development/validation separation.

#### Missingness integrity
- remaining missing-value counts;
- expected missingness indicators;
- unexpected newly introduced missing values.

#### Numeric integrity
- data types;
- infinite values;
- unexpected negative values;
- minimum and maximum values;
- zero rates before and after transformation.

#### Transformation integrity
- feature naming;
- feature lineage;
- feature count;
- transformation consistency between development and validation data.

The transformed dataset must reconcile with the original development and
validation row counts.

---

### 74.18 Before-vs-After Treatment Audit

A treatment audit will compare important feature properties before and after
preprocessing.

The audit will include:

| Metric | Before | After |
|---|---:|---:|
| Row count | Recorded | Must reconcile |
| Feature count | Recorded | Documented |
| Missing count | Recorded | Expected reduction |
| Zero count | Recorded | Expected change |
| Minimum | Recorded | Investigated if changed |
| Maximum | Recorded | Investigated if changed |
| Data type | Recorded | Validated |
| Infinite values | Recorded | Must be zero |

Unexpected changes will trigger investigation before the modeling dataset is
approved.

---

### 74.19 Missingness Treatment Decision Status

| Treatment Area | Policy | Status |
|---|---|---|
| 0% missing features | No treatment | Approved |
| Low-missingness numeric features | Assess necessity before imputation | Approved |
| Structural data-service missingness | Confirm semantics before zero treatment | Pending Validation |
| Data-service missingness indicators | Preserve where analytically justified | Approved |
| Date missingness | Derive recency/activity features; no fake dates | Approved |
| August voice missingness | Preserve signal; validate semantic meaning | Pending Validation |
| High missingness alone | Do not automatically drop | Approved |
| Arbitrary date imputation | Prohibited | Approved |
| Blanket zero imputation | Prohibited | Approved |
| Blanket median imputation | Prohibited | Approved |
| Target-driven treatment decisions | Prohibited | Approved |
| Test-based preprocessing decisions | Prohibited | Approved |

---

### 74.20 Revalidation After Transformation

After implementation, all treatment decisions will be revalidated against:

- feature semantics;
- missingness distributions;
- development/validation stability;
- feature distributions;
- model input requirements;
- prediction-time availability;
- leakage controls.

A feature treatment may be revised if post-treatment validation identifies
unexpected behavior.

The final treatment matrix will be updated to reflect the implemented and
validated state.

---

### 74.21 Reproducibility

All preprocessing decisions must be reproducible.

The implementation will record:

- source columns;
- derived columns;
- treatment rules;
- imputation strategy;
- fitted parameters;
- missingness indicators;
- transformation order;
- development/validation split;
- random seed where applicable.

The same fitted preprocessing logic must produce consistent transformations
when applied to future prediction data.

---

### 74.22 Current Status

Missingness profiling and investigation are complete.

The evidence supports a differentiated treatment strategy rather than blanket
imputation or feature removal.

The following policies are now locked:

- no blanket zero imputation;
- no blanket median imputation;
- no arbitrary date imputation;
- no automatic removal based on missingness percentage alone;
- preserve informative missingness where justified;
- validate structural/service semantics before zero conversion;
- keep preprocessing within the development-data boundary;
- keep the competition test set isolated;
- perform before-vs-after validation after treatment.

The exact feature-level treatment remains pending for features where business
semantics require confirmation.

## Final Status

Feature Engineering & Preprocessing: COMPLETE

Prediction-point framework: COMPLETE

Temporal feature eligibility review: PASS

Structural missingness treatment: COMPLETE

Behavioral missingness treatment: COMPLETE

Development-only imputation fitting: PASS

Temporal change features: COMPLETE

Percentage-change features: COMPLETE

Multi-month summary features: COMPLETE

Recent-vs-baseline features: COMPLETE

Recency and activity signals: COMPLETE

Customer-value features: COMPLETE

Feature redundancy review: COMPLETE

Predictive screening: COMPLETE

Feature stability review: COMPLETE

Permutation-importance review: COMPLETE

Final model feature set: 169 features

Feature freeze: ACTIVE

Development / validation separation: PASS

Test data usage during feature engineering: NONE

Target leakage review: PASS

Feature Engineering & Preprocessing is complete and the frozen 169-feature dataset is approved for model development and downstream risk intelligence.