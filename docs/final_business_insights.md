# ChurnIQ — Final Business Insights & Recommendations

## 1. Executive Summary

ChurnIQ is an end-to-end customer retention intelligence project that connects predictive analytics with customer-value and business-impact analysis.

The objective is to identify customers with elevated potential churn risk, understand the behavioral signals associated with those predictions, quantify potential customer-value exposure, and support evidence-based retention prioritization.

The overall decision framework is:

**Predict → Explain → Quantify Risk → Prioritize → Recommend Action**

---

## 1.1 Business Objective

The business problem addressed by ChurnIQ is not simply identifying who may churn.

The broader objective is to help a retention team answer four connected questions:

1. **Which customers show elevated potential churn risk?**
2. **What behavioral signals are associated with that risk?**
3. **Where is the greatest potential customer-value exposure?**
4. **Which customer groups can be considered for retention attention under different business assumptions?**

The project therefore combines predictive modeling, explainable AI, customer-value analysis, revenue-exposure analysis, and retention economics into one decision-support framework.

---

## 1.2 Overall Analytical Findings

### Observed Evidence

The historical customer dataset contains:

- **69,999 customers**
- **10.19% historical churn rate**

August ARPU is used throughout the business-impact analysis as a **customer-value proxy**, not as booked revenue.

Under the project's exposure definition, the historical customer universe represents approximately:

- **₹776,208.24 potential monthly revenue exposure**
- **₹9,314,498.90 annualized scenario exposure**

These figures represent modeled exposure under the project's analytical definition and should not be interpreted as guaranteed future revenue loss.

### Model-Derived Findings

The final tuned XGBoost model uses **169 frozen features**.

On the held-out validation set:

- **PR-AUC:** 0.7614
- **ROC-AUC:** 0.9502
- **Precision:** 0.7686
- **Recall:** 0.6220
- **F1-score:** 0.6876

When customers are ranked by predicted churn risk, the **top 10% of customers capture 71.04% of historical churn**.

This demonstrates the usefulness of the model's ranking for concentrating analytical attention, while not implying that every high-risk customer will churn.

---

## 1.3 Customer Risk Intelligence

The calibrated scoring framework produces a probability-based risk view across the full historical customer universe.

At the primary analytical threshold of **0.10**:

- **12,492 customers** are above the threshold
- **6,152 customers** are classified as Very High Risk
- **6,340 customers** are classified as High Risk
- **57,507 customers** remain Below Primary Threshold

The threshold is an analytical operating point used for risk classification and should not be interpreted as a guaranteed intervention rule.

The risk score is therefore most useful as a **prioritization signal**, rather than as a statement that a customer will definitely churn.

---

## 1.4 Explainability and Behavioral Signals

ChurnIQ uses SHAP-based explanations to connect individual predictions with the model's learned behavioral signals.

The strongest global signals in the final explanation analysis include measures related to:

- overall customer activity
- voice recharge recency
- recent recharge behavior
- incoming voice usage
- roaming usage
- recent activity deterioration
- multi-month usage patterns
- customer tenure

These signals describe relationships learned by the model. They should **not be interpreted as causal drivers** of churn without additional causal analysis or controlled business experimentation.

This distinction is important because ChurnIQ is designed to explain **why the model produced a prediction**, not to prove why a customer churns.

---

## 1.5 Customer Value and Revenue Exposure

Customer risk alone does not describe the potential business impact of churn.

ChurnIQ therefore combines calibrated churn probability with August ARPU as a customer-value proxy to estimate potential revenue exposure.

Exposure is concentrated among higher-ranked customers:

| Risk / Exposure Cohort | Share of Potential Monthly Exposure |
|---|---:|
| Top 10% risk-ranked customers | **77.61%** |
| Top 20% risk-ranked customers | **88.19%** |
| High + Very High Risk customers | **71.04%** |

This concentration indicates that customer prioritization can be informed by both **predicted risk** and **customer-value exposure**, rather than relying on risk probability alone.

The exposure figures remain analytical scenario estimates and do not represent realized revenue loss.

---

## 1.6 Retention Targeting Insights

The retention analysis evaluates several targeting cohorts:

- Top 1% Exposure
- Top 5% Exposure
- Top 10% Exposure
- Top 20% Exposure
- High + Very High Risk

These cohorts provide different ways of concentrating retention attention.

The analysis shows that narrower targeting cohorts can concentrate a substantial share of potential revenue exposure while involving fewer customers.

For example:

- The **Top 1% Exposure** cohort contains approximately **700 customers** and captures **32.02%** of potential monthly exposure.
- The **Top 5% Exposure** cohort contains approximately **3,500 customers** and captures **63.84%**.
- The **Top 10% Exposure** cohort contains approximately **7,000 customers** and captures **77.61%**.
- The **Top 20% Exposure** cohort contains approximately **14,000 customers** and captures **88.19%**.

These figures describe exposure concentration. They do not establish that a particular cohort will produce a specific retention outcome.

---

## 1.7 Retention Economics

Retention economics were evaluated across:

- five targeting cohorts
- three intervention scenarios
- multiple retention success assumptions
- multiple intervention-cost assumptions
- 135 sensitivity scenarios

Across the tested sensitivity space:

- **94 of 135 scenarios were economically viable**
- Tested break-even success rates ranged from **2.35% to 94.39%**

The economics are therefore highly dependent on the assumptions used for retention success and intervention cost.

The analysis is intended as **decision-support scenario modeling**, not as a forecast of realized ROI.

Historical churn is used descriptively in the targeting analysis and is not treated as realized retention ROI.

---

## 1.8 What the Analysis Means for the Business

Taken together, the analysis supports a layered decision process:

### Layer 1 — Risk

Use calibrated churn probability to identify customers with elevated potential churn risk.

### Layer 2 — Explanation

Use SHAP explanations to understand the behavioral signals associated with an individual prediction.

### Layer 3 — Value

Use customer-value information to understand the potential financial exposure associated with a customer's predicted risk.

### Layer 4 — Prioritization

Use exposure-ranked and risk-based cohorts to concentrate analytical attention according to available customer-retention capacity.

### Layer 5 — Economics

Evaluate whether a proposed retention approach remains economically feasible under explicit success-rate and intervention-cost assumptions.

This prevents churn prediction from being treated as the final business decision.

---

## 1.9 Executive Decision Framework

A practical retention workflow supported by ChurnIQ is:

**Identify Risk**

↓  

**Understand the Prediction**

↓  

**Assess Customer Value**

↓  

**Estimate Potential Exposure**

↓  

**Define a Targeting Cohort**

↓  

**Evaluate Retention Economics**

↓  

**Apply Operational and Business Constraints**

↓  

**Measure Actual Outcomes**

The final step is essential. Real retention decisions should ultimately be evaluated against observed customer outcomes after interventions are deployed.

---

## 1.10 Final Executive Takeaway

ChurnIQ demonstrates how predictive modeling can be connected to business analysis rather than being treated as an isolated machine-learning exercise.

The project moves from:

**Customer behavior → Churn probability → Explainability → Customer value → Revenue exposure → Retention prioritization → Economic scenarios**

The central business insight is that **risk identification, customer value, and retention economics answer different questions and should be interpreted together**.

ChurnIQ therefore serves as a decision-support framework for prioritizing analytical attention, while final retention actions remain dependent on operational capacity, customer context, intervention strategy, and validation through real-world outcomes.

---

## 1.11 Interpretation Boundaries

The following principles apply to all findings in this document:

- Predicted churn probability is **not guaranteed churn**.
- Historical churn is evidence about the past, not a guaranteed future outcome.
- SHAP explanations describe model behavior and **do not establish causality**.
- August ARPU is a **customer-value proxy**, not booked revenue.
- Revenue exposure is a **scenario estimate**, not guaranteed revenue loss.
- Retention ROI is based on explicit assumptions and is **not realized business ROI**.
- Targeting cohorts may overlap and should not be interpreted as mutually exclusive customer populations.
- The **0.10 primary threshold** is an analytical operating point, not a universal intervention rule.
- Business decisions should incorporate operational constraints and subsequently measured outcomes.

# 2. Customer Risk Insights

## 2.1 Risk Distribution Across the Customer Universe

ChurnIQ scores the complete historical customer universe of **69,999 customers** using the final tuned XGBoost model with sigmoid probability calibration.

The resulting analytical risk framework is:

- **Below Primary Threshold:** calibrated churn probability below 0.10
- **High Risk:** calibrated churn probability from 0.10 to below 0.50
- **Very High Risk:** calibrated churn probability of 0.50 or higher

The resulting distribution is:

| Risk Level | Customers | Share of Customer Universe |
|---|---:|---:|
| Below Primary Threshold | 57,507 | 82.15% |
| High Risk | 6,340 | 9.06% |
| Very High Risk | 6,152 | 8.79% |
| **Total** | **69,999** | **100.00%** |

Therefore, **12,492 customers**, or approximately **17.85% of the customer universe**, are above the primary 0.10 analytical threshold.

---

## 2.2 What the Risk Probability Represents

The calibrated churn probability is a **model-estimated likelihood**, not a guaranteed outcome.

A higher probability indicates higher modeled churn risk relative to customers with lower predicted probabilities.

The probability is therefore useful for:

- ranking customers by modeled risk
- identifying elevated-risk populations
- supporting customer prioritization
- providing a consistent analytical input for downstream business analysis

It should not be interpreted as a deterministic statement that an individual customer will churn.

---

## 2.3 Primary Threshold Analysis

The primary analytical threshold is **0.10**.

Using this operating point:

- **12,492 customers** are above the threshold.
- These customers represent **17.85%** of the historical customer universe.
- The above-threshold population contains **88.04% of historical churned customers** in the analyzed dataset.

The final figure is a **descriptive historical measurement** of the selected risk threshold. It does not imply that the threshold guarantees equivalent capture of future churn.

The threshold was established during the project's threshold-optimization analysis and should be treated as an analytical operating point rather than a universal intervention rule.

An operational threshold could differ depending on:

- available retention capacity
- intervention cost
- required precision
- acceptable customer-contact volume
- retention strategy
- observed post-intervention outcomes

---

## 2.4 Risk Ranking Performance

Risk ranking provides an additional way to concentrate analytical attention.

When customers are ordered from highest to lowest predicted churn probability, the **top 10% of risk-ranked customers capture 71.04% of historical churn**.

This means the model's ranking places a substantial proportion of historically churned customers within a relatively small portion of the customer universe.

This is a **historical model-performance measurement**, not a forecast that the same proportion of future churn will occur within the top 10%.

The ranking should therefore be interpreted as evidence that the model provides useful separation of historical risk, while future performance would need to be monitored after deployment.

---

## 2.5 Very High Risk Population

The Very High Risk population contains **6,152 customers**, representing approximately **8.79%** of the customer universe.

These customers have calibrated churn probabilities of **0.50 or higher**.

The Very High Risk category provides a more concentrated view of customers with the highest modeled risk.

However, even within this group, predicted risk remains probabilistic. Very High Risk should therefore be used as a prioritization signal rather than as confirmation that a customer will churn.

---

## 2.6 Risk Concentration Supports Prioritization

The risk analysis demonstrates that the customer base can be organized into analytically meaningful risk populations rather than being treated as one homogeneous group.

This creates two useful levels of prioritization:

### Broad risk prioritization

The **12,492 customers above the 0.10 threshold** provide a broader population for retention analysis.

### Concentrated risk prioritization

The **6,152 Very High Risk customers** provide a more concentrated population for situations where retention capacity is limited.

These groups serve different analytical purposes and should not automatically be treated as equivalent intervention populations.

---

## 2.7 Risk Alone Does Not Represent Business Impact

Churn probability answers:

> **“Which customers show elevated modeled churn risk?”**

It does not answer:

> **“Which customers represent the greatest potential financial exposure?”**

A high-risk customer may have relatively low customer-value exposure, while another customer with lower modeled risk may have substantially higher customer-value exposure.

This is why ChurnIQ separates the **risk layer** from the **customer-value and revenue-exposure layer**.

The two are brought together in the subsequent business-impact analysis.

---

## 2.8 Business Interpretation

The risk analysis changes the retention question from a single aggregate measure:

> **“What is the overall churn rate?”**

to a more actionable analytical sequence:

> **“Which customers show elevated modeled risk, how concentrated is that risk, and how does that risk intersect with customer value?”**

This provides the foundation for moving from **risk identification** toward **business-impact assessment and retention prioritization**.

---

## 2.9 Key Risk Insights

The main conclusions from the risk layer are:

1. The model produces a complete probability-based risk view across **69,999 customers**.
2. **17.85%** of customers are above the primary 0.10 analytical threshold.
3. The top 10% of risk-ranked customers contain **71.04% of historical churn**.
4. The Very High Risk population contains **6,152 customers**.
5. Risk probability provides a prioritization signal but does not guarantee future churn.
6. Risk should be combined with customer value and revenue exposure before making business-prioritization decisions.

---

## 2.10 Risk Interpretation Boundaries

The following principles apply to all ChurnIQ risk outputs:

- Predicted churn probability is a **model estimate**, not certainty.
- Risk classification does not establish causality.
- Historical churn-capture measurements describe performance against historical outcomes.
- The 0.10 threshold is an **analytical operating point**, not a universal intervention rule.
- Model ranking performance may change when applied to future populations.
- Customer risk should be interpreted alongside customer value and operational constraints.
- Actual retention effectiveness must ultimately be evaluated using observed post-intervention outcomes.

# 3. Customer Value & Revenue Exposure Insights

## 3.1 Customer Value as a Business Context Layer

ChurnIQ adds customer value as a second dimension to the modeled churn-risk analysis.

For this project, **August ARPU is used as a customer-value proxy**.

This proxy is not intended to represent booked revenue, customer lifetime value, or total future customer revenue. Its purpose is to provide a consistent value measure that can be combined with calibrated churn probability for analytical exposure estimation.

The resulting framework separates three questions:

- **Risk:** Which customers have elevated modeled churn probability?
- **Value:** Which customers have greater value according to the selected proxy?
- **Exposure:** Where do modeled churn risk and customer value combine into greater potential financial exposure?

---

## 3.2 Customer Value Profile

Across the 69,999-customer historical universe:

- **Average August ARPU:** ₹278.86
- **Median August ARPU:** ₹192.23
- **Customers with negative August ARPU:** 352

The underlying August ARPU values are retained for analysis.

For exposure calculations, the exposure-eligible ARPU is constrained to a minimum of zero. This prevents negative ARPU observations from producing negative revenue exposure while preserving the original customer-value observation.

This is an analytical treatment for the exposure calculation and does not imply that negative ARPU observations represent zero actual revenue.

---

## 3.3 Potential Revenue Exposure Methodology

ChurnIQ estimates potential monthly revenue exposure using:

**Calibrated churn probability × exposure-eligible August ARPU**

The resulting historical-universe exposure is:

- **Potential monthly revenue exposure:** ₹776,208.24
- **Annualized scenario exposure:** ₹9,314,498.90

The annualized figure extends the monthly exposure calculation across 12 months.

These values are analytical scenario estimates based on:

1. the model's calibrated churn probabilities, and
2. August ARPU as the selected customer-value proxy.

They should not be interpreted as:

- guaranteed revenue loss
- realized churn-related revenue loss
- booked revenue
- customer lifetime value
- a financial forecast

---

## 3.4 Exposure Concentration

Potential monthly revenue exposure is concentrated among customers ranked by exposure.

The targeting analysis shows:

| Exposure-Ranked Cohort | Customers | Potential Monthly Exposure Captured |
|---|---:|---:|
| Top 1% Exposure | 700 | **32.02%** |
| Top 5% Exposure | 3,500 | **63.84%** |
| Top 10% Exposure | 7,000 | **77.61%** |
| Top 20% Exposure | 14,000 | **88.19%** |

These results demonstrate the concentration of modeled potential exposure within progressively larger targeting populations.

The relationship can be viewed as a targeting trade-off:

**Customers considered ↑ → Exposure covered ↑**

The rate at which exposure coverage increases provides context for evaluating potential targeting capacity.

---

## 3.5 Risk-Based Exposure Context

A separate risk-based population is represented by the **High + Very High Risk** cohort.

This population contains:

- **12,492 customers**
- approximately **17.85%** of the customer universe
- **71.04% of potential monthly revenue exposure**

This demonstrates why risk and value should be examined together.

The High + Very High Risk population is defined by modeled risk classification, while the Top 1%, Top 5%, Top 10%, and Top 20% cohorts are exposure-ranked populations.

They therefore answer different business questions and should not be treated as interchangeable cohorts.

---

## 3.6 Risk, Value, and Exposure Are Different Measures

The three measures have distinct meanings:

### Risk

**Calibrated churn probability**

Measures the model's estimated likelihood of churn.

### Value

**August ARPU**

Provides a customer-value proxy.

### Exposure

**Calibrated churn probability × exposure-eligible August ARPU**

Provides an analytical estimate of potential monthly revenue exposure.

This distinction prevents a high-risk customer from automatically being interpreted as a high-value customer, and prevents a high-value customer from automatically being interpreted as a high-risk customer.

---

## 3.7 Targeting Capacity and Exposure Coverage

The exposure-ranked cohorts provide a useful way to examine different operational capacities.

| Cohort | Customers Considered | Exposure Coverage |
|---|---:|---:|
| Top 1% | 700 | 32.02% |
| Top 5% | 3,500 | 63.84% |
| Top 10% | 7,000 | 77.61% |
| Top 20% | 14,000 | 88.19% |

For example, moving from the Top 5% to the Top 10% doubles the number of customers considered while increasing exposure coverage from **63.84% to 77.61%**.

Moving from the Top 10% to the Top 20% doubles the targeting population again while increasing exposure coverage from **77.61% to 88.19%**.

These comparisons illustrate the operational trade-off between targeting breadth and exposure coverage.

They do not establish which cohort should be selected in practice.

---

## 3.8 High-Risk and High-Value Intersection

The combined risk-and-value framework provides a basis for identifying customers where elevated modeled risk coincides with greater potential customer-value exposure.

Conceptually:

**Higher Risk + Higher Value → Greater modeled exposure**

However, the exposure measure should remain an analytical prioritization signal rather than a prediction of realized financial loss.

The framework is therefore intended to help the business identify where further investigation or retention planning may be warranted.

---

## 3.9 Business Interpretation

The customer-value layer changes the analytical question from:

> **“Who is most likely to churn?”**

to:

> **“Where is potential customer-value exposure concentrated among customers with modeled churn risk?”**

This creates a bridge between predictive modeling and business prioritization.

The model identifies risk.

The customer-value proxy provides financial context.

The exposure calculation combines the two.

The targeting analysis then shows how much of that modeled exposure is concentrated within different customer populations.

Only after these steps does ChurnIQ evaluate retention economics.

---

## 3.10 Key Customer-Value Insights

The main findings from this layer are:

1. August ARPU provides a consistent customer-value proxy for the business-impact analysis.
2. The historical customer universe produces **₹776,208.24** in potential monthly exposure under the project's exposure definition.
3. The corresponding annualized scenario exposure is **₹9,314,498.90**.
4. The Top 1% exposure-ranked cohort represents **32.02%** of potential monthly exposure.
5. The Top 5% represents **63.84%**.
6. The Top 10% represents **77.61%**.
7. The Top 20% represents **88.19%**.
8. The High + Very High Risk population represents **71.04%** of potential monthly exposure.
9. Exposure-ranked cohorts and risk-based cohorts represent different analytical populations and should not be treated as interchangeable.
10. Exposure concentration provides business context for evaluating targeting capacity before considering intervention economics.

---

## 3.11 Interpretation Boundaries

The following boundaries apply to the customer-value and exposure analysis:

- August ARPU is a **customer-value proxy**, not booked revenue.
- Potential monthly exposure is based on calibrated churn probability and exposure-eligible August ARPU.
- Annualized exposure extends the monthly scenario across 12 months.
- Exposure does not represent realized or guaranteed revenue loss.
- The analysis does not estimate customer lifetime value.
- Exposure concentration does not prove that retention intervention will prevent churn.
- Exposure-ranked cohorts and risk-classification cohorts have different definitions.
- Targeting cohorts may overlap with other analytical populations.
- Negative ARPU observations are retained in the underlying data but are excluded from creating negative exposure.
- Actual financial impact would require observation of customer outcomes after any real-world retention intervention.

# 4. Retention Prioritization & Economics Insights

## 4.1 Purpose of Retention Prioritization

ChurnIQ moves from identifying potential churn risk toward evaluating how a business could prioritize customers for retention attention.

The objective is not to prescribe a single intervention strategy.

Instead, the analysis evaluates how different targeting populations behave under different assumptions about:

- retention success rate
- intervention cost
- potential exposure captured
- net value
- ROI
- break-even success rate

This creates an assumption-driven decision-support layer between predictive risk analysis and potential operational action.

---

## 4.2 Targeting Cohorts Evaluated

Five targeting cohorts were evaluated:

1. **Top 1% Exposure**
2. **Top 5% Exposure**
3. **Top 10% Exposure**
4. **Top 20% Exposure**
5. **High + Very High Risk**

The exposure-ranked cohorts are based on customer exposure ranking, while High + Very High Risk is based on the model's risk classification.

These cohorts represent different analytical approaches to concentrating retention attention.

They are not mutually exclusive customer populations.

---

## 4.3 Retention Economics Framework

The retention economics analysis evaluates the potential economics of a retention intervention under explicit assumptions.

The analysis considers:

**Potential Exposure → Assumed Retention Success → Expected Value Retained → Intervention Cost → Net Value → ROI**

The economics therefore depend directly on the assumptions selected for:

- retention success
- intervention cost
- targeted population
- potential exposure

The results should consequently be interpreted as **scenario analysis**, not realized business performance.

---

## 4.4 Scenario Results

Across the five targeting cohorts and three intervention scenarios, the analysis evaluated **15 cohort-scenario combinations**.

The resulting ROI values demonstrate that economic outcomes vary substantially across targeting populations and intervention assumptions.

Selected results include:

| Targeting Cohort | Low-Cost Scenario ROI | Targeted Scenario ROI | High-Cost Scenario ROI |
|---|---:|---:|---:|
| Top 1% Exposure | 752.19% | 496.53% | 326.09% |
| Top 5% Exposure | 239.78% | 137.85% | 69.89% |
| Top 10% Exposure | 106.54% | 44.58% | 3.27% |
| Top 20% Exposure | 17.35% | -17.85% | -41.32% |
| High + Very High Risk | 5.94% | -25.84% | -47.03% |

These values represent the project's defined scenarios and assumptions.

They do not represent actual historical ROI or guaranteed future ROI.

---

## 4.5 Economics Change With Targeting Breadth

The scenario results illustrate an important trade-off.

Narrower targeting cohorts can concentrate more potential exposure per customer, while broader cohorts involve more customers and therefore incur greater total intervention cost under the same per-customer cost assumption.

For example:

- The Top 1% Exposure cohort has positive modeled ROI across all three tested scenarios.
- The Top 5% Exposure cohort also has positive modeled ROI across all three scenarios.
- The Top 10% Exposure cohort remains positive across the tested scenarios, although the high-cost scenario produces a much smaller modeled return.
- The Top 20% Exposure and High + Very High Risk cohorts produce negative modeled ROI under some tested scenarios.

This demonstrates that expanding the targeting population does not automatically improve economic efficiency.

The result depends on the relationship between:

**exposure captured ↔ assumed retention success ↔ intervention cost**

---

## 4.6 Break-Even Analysis

The analysis also calculates the retention success rate required for an intervention to break even under each tested cohort and cost assumption.

Across the sensitivity analysis:

- Minimum tested break-even success rate: **2.35%**
- Maximum tested break-even success rate: **94.39%**

A lower break-even requirement means that the modeled economics require a smaller assumed retention-success rate to cover intervention costs.

A higher break-even requirement indicates greater sensitivity to the assumed retention-success rate.

Break-even therefore provides a useful way to examine economic feasibility without treating any particular success rate as guaranteed.

---

## 4.7 Sensitivity and Robustness

ChurnIQ evaluates **135 sensitivity scenarios** across:

- 5 targeting cohorts
- 9 retention success rates
- 3 intervention costs

The tested retention success rates range from:

**10%, 20%, 30%, 35%, 40%, 50%, 60%, 70%, and 80%**

The tested intervention costs are:

- ₹100
- ₹250
- ₹500

Across all scenarios:

- **94 of 135 scenarios were economically viable**
- **41 of 135 scenarios were not economically viable**
- Break-even success rates ranged from **2.35% to 94.39%**

The sensitivity analysis demonstrates that the economics are not fixed.

They change materially when the underlying assumptions change.

---

## 4.8 Robustness Interpretation

The sensitivity framework categorizes scenarios according to the required retention-success assumption:

- **Robust:** break-even requirement at or below 10%
- **Conditional:** above 10% and up to 50%
- **Weak:** above 50% and up to 80%
- **Not viable:** above 80% or infeasible within the tested range

These categories are **analytical labels defined by the project methodology**.

They should not be interpreted as universal business standards.

The purpose is to distinguish scenarios that require relatively low assumed success from scenarios that depend on increasingly demanding assumptions.

---

## 4.9 What the ROI Results Actually Mean

The ROI calculations answer a specific scenario question:

> **“If the assumed retention success rate and intervention cost occur as defined, what would the modeled economic outcome look like for this targeting cohort?”**

They do not answer:

> **“How much money will the business actually recover?”**

Actual realized ROI would depend on factors that are outside the historical analysis, including:

- actual intervention effectiveness
- customer response
- intervention delivery cost
- operational execution
- changes in customer behavior
- future customer value
- measurement of retained revenue after intervention

Therefore, the scenario ROI should be used to evaluate **economic conditions and sensitivity**, not as a guaranteed financial forecast.

---

## 4.10 Targeting and Economics Should Be Considered Together

The exposure analysis and retention economics provide complementary information.

### Exposure analysis asks:

> **How much potential exposure is concentrated within a targeting cohort?**

### Economics analysis asks:

> **Under stated assumptions, does the potential retained value justify the intervention cost?**

A cohort can capture substantial potential exposure while still producing unfavorable modeled economics if:

- intervention cost is high, or
- the assumed retention success rate is insufficient.

Conversely, a smaller cohort may produce favorable modeled economics because it concentrates higher exposure per targeted customer.

This is why exposure concentration alone should not be treated as an intervention decision.

---

## 4.11 Decision-Support Framework

The retention prioritization framework can be expressed as:

**1. Identify potential churn risk**

↓

**2. Estimate customer-value exposure**

↓

**3. Define a targeting cohort**

↓

**4. Specify intervention assumptions**

↓

**5. Calculate modeled economic outcomes**

↓

**6. Test sensitivity to assumptions**

↓

**7. Apply operational constraints**

↓

**8. Measure actual outcomes**

This sequence keeps predictive analytics, financial modeling, and real-world business outcomes conceptually separate.

---

## 4.12 Business Interpretation

The retention economics analysis demonstrates that there is no single economic result independent of assumptions.

Instead, the business can evaluate:

- how much exposure a cohort contains
- how many customers would be targeted
- what success rate would be required to break even
- how intervention cost changes the economics
- how sensitive the result is to different assumptions

This turns retention prioritization into a transparent **scenario-based decision framework** rather than an unsupported claim about expected financial returns.

---

## 4.13 Key Retention Economics Insights

The main findings from this layer are:

1. Five targeting cohorts were evaluated using exposure and risk-based definitions.
2. Fifteen cohort-scenario combinations were evaluated in the primary economics analysis.
3. The modeled ROI varies substantially across targeting cohorts and intervention-cost assumptions.
4. **94 of 135 sensitivity scenarios were economically viable**.
5. Tested break-even success rates ranged from **2.35% to 94.39%**.
6. Narrower exposure-ranked cohorts generally concentrate more potential exposure per targeted customer.
7. Broader targeting increases the number of customers requiring intervention and can materially change modeled economics.
8. High modeled exposure does not automatically imply favorable retention economics.
9. ROI results are assumption-based and should not be interpreted as realized business ROI.
10. Actual retention effectiveness must be established through observed post-intervention outcomes.

---

## 4.14 Interpretation Boundaries

The following boundaries apply to the retention prioritization and economics analysis:

- Retention success rates are **analytical assumptions**, not observed intervention results.
- Intervention costs are **scenario assumptions**, not necessarily actual operational costs.
- ROI is **modeled scenario ROI**, not realized business ROI.
- Break-even rates depend on the selected exposure and cost assumptions.
- Exposure is based on calibrated churn probability and August ARPU as a customer-value proxy.
- Historical churn is used descriptively and is not treated as realized retention ROI.
- Targeting cohorts may overlap.
- Sensitivity results describe the tested scenario space and do not cover every possible business condition.
- The robustness categories are project-defined analytical classifications, not universal standards.
- Actual business outcomes may differ materially from scenario results.

# 5. Final Business Recommendations

The recommendations below translate the ChurnIQ findings into a practical decision-support framework for customer-retention analysis.

They are based on observed historical evidence, model-derived outputs, customer-value analysis, and assumption-based retention economics.

They are not guarantees of future churn reduction, revenue recovery, or realized ROI.

---

## 5.1 Recommendation 1 — Use Calibrated Risk Ranking to Structure Retention Analysis

ChurnIQ should use calibrated churn probability as the starting point for identifying customers with elevated modeled risk.

The complete customer universe contains **69,999 customers**, of which **12,492 customers** are above the 0.10 primary analytical threshold.

The model's ranking also places **71.04% of historical churn within the top 10% of risk-ranked customers**.

### Business application

A retention team can use the risk ranking to:

- identify customers requiring further analysis
- organize the customer population by modeled risk
- define candidate targeting populations
- support capacity-based prioritization

The risk output should remain a prioritization signal rather than a deterministic churn decision.

---

## 5.2 Recommendation 2 — Combine Risk With Customer Value

Retention analysis should consider modeled churn risk together with customer-value information.

Churn probability and customer value answer different questions:

- **Risk:** likelihood of churn according to the model
- **Value:** customer value represented by the August ARPU proxy
- **Exposure:** modeled churn probability combined with exposure-eligible customer value

The exposure analysis shows that:

- Top 10% exposure-ranked customers represent **77.61%** of potential monthly exposure.
- Top 20% represent **88.19%**.
- High + Very High Risk customers represent **71.04%**.

### Business application

Use the sequence:

**Risk → Value → Potential Exposure**

rather than treating churn probability as the only prioritization variable.

This provides financial context around the modeled risk population.

---

## 5.3 Recommendation 3 — Match Targeting Population to Operational Capacity

Different targeting populations provide different levels of exposure coverage.

| Targeting Cohort | Customers | Exposure Coverage |
|---|---:|---:|
| Top 1% Exposure | 700 | 32.02% |
| Top 5% Exposure | 3,500 | 63.84% |
| Top 10% Exposure | 7,000 | 77.61% |
| Top 20% Exposure | 14,000 | 88.19% |

The results show a clear trade-off between:

**targeting volume ↔ potential exposure coverage**

### Business application

The business can evaluate the targeting population against its actual retention capacity.

A constrained retention operation may evaluate narrower cohorts, while greater operational capacity may allow broader cohorts to be considered.

The analysis does not establish a universally correct targeting population.

---

## 5.4 Recommendation 4 — Evaluate Intervention Economics Before Scaling

Retention interventions should be evaluated against explicit assumptions before being considered for broader operational use.

ChurnIQ evaluates:

- targeting cohort
- retention success assumption
- intervention cost
- potential exposure
- modeled retained value
- net value
- ROI
- break-even success rate

The sensitivity analysis tested **135 scenarios**, of which **94 were economically viable under the project's assumptions**.

Break-even success requirements ranged from **2.35% to 94.39%** across the tested scenarios.

### Business application

Before scaling an intervention, define:

1. expected intervention cost
2. measurable retention-success assumption
3. required break-even success rate
4. targeting population
5. sensitivity to changes in assumptions

Then compare the scenario results with actual outcomes after implementation.

---

## 5.5 Recommendation 5 — Validate Predictions and Economics With Real Outcomes

ChurnIQ should be treated as a decision-support prototype until its predictions and economic assumptions are validated using real intervention outcomes.

A future operational process could record:

- customers identified as elevated risk
- customers selected for intervention
- intervention type
- intervention cost
- customer response
- subsequent churn outcome
- retained customer value
- realized financial impact

These observations can then be compared with the original predictions and scenario assumptions.

The resulting feedback loop is:

**Predict → Intervene → Observe → Measure → Validate → Improve**

This is necessary because historical model performance and scenario economics do not establish future intervention effectiveness.

---

## 5.6 Recommendation 6 — Monitor Data, Model, and Business Outcomes Separately

ChurnIQ should be monitored at three distinct levels.

### Data layer

Monitor:

- schema consistency
- missingness
- feature distributions
- identifier integrity
- data-quality issues

### Model layer

Monitor:

- churn probability distribution
- risk-level distribution
- probability calibration
- prediction stability
- predictive performance once outcomes become available

### Business layer

Monitor:

- targeting volume
- customer-value exposure
- intervention cost
- retention success
- realized customer outcomes
- realized financial impact

Separating these layers helps determine whether a change originates from the **data**, the **model**, or the **business process**.

---

## 5.7 Recommended Decision Process

The ChurnIQ framework can be applied through the following sequence:

### Step 1 — Identify Risk

Use calibrated churn probability to identify customers with elevated modeled risk.

### Step 2 — Explain

Use SHAP explanations to understand the behavioral signals associated with individual predictions.

### Step 3 — Quantify Potential Exposure

Combine modeled risk with customer-value information to estimate potential exposure.

### Step 4 — Define a Targeting Population

Evaluate risk-based and exposure-ranked cohorts according to available operational capacity.

### Step 5 — Test Economics

Evaluate intervention scenarios using explicit retention-success and cost assumptions.

### Step 6 — Apply Business Constraints

Consider customer context, intervention feasibility, operational capacity, and other business requirements.

### Step 7 — Measure Outcomes

Track actual retention, churn, customer value, intervention cost, and financial outcomes.

### Step 8 — Reassess

Use observed outcomes and monitoring evidence to reassess the model, targeting approach, and economic assumptions.

---

## 5.8 Practices to Avoid

The ChurnIQ framework should not be used to:

- treat predicted churn probability as certainty
- treat SHAP explanations as causal evidence
- treat August ARPU as booked revenue
- treat modeled exposure as guaranteed revenue loss
- treat scenario ROI as realized ROI
- assume the 0.10 threshold is appropriate for every operational environment
- expand targeting without considering intervention capacity and cost
- evaluate the model only through predictive metrics
- assume historical performance will automatically continue in future populations

---

## 5.9 Recommendation Summary

| Recommendation | Evidence Base | Intended Use |
|---|---|---|
| Use calibrated risk ranking | Churn probability and historical ranking performance | Structure risk analysis |
| Combine risk with value | Customer-value and exposure analysis | Add financial context |
| Match targeting to capacity | Exposure concentration analysis | Evaluate targeting breadth |
| Test economics before scaling | ROI and sensitivity analysis | Evaluate scenario feasibility |
| Validate with real outcomes | Outcome and monitoring framework | Establish actual effectiveness |
| Monitor data, model, and business separately | Monitoring plan | Identify sources of performance change |

---

## 5.10 Final Recommendation Statement

ChurnIQ should be used as a **decision-support framework rather than an automated decision maker**.

Its analytical workflow connects:

**Risk Prediction → Explainability → Customer Value → Revenue Exposure → Targeting → Retention Economics → Outcome Measurement**

Each layer answers a different business question.

The framework can therefore support a structured process for identifying potential risk, understanding model signals, assessing potential customer-value exposure, evaluating targeting populations, and testing retention economics under explicit assumptions.

Final retention decisions should remain dependent on customer context, operational capacity, intervention feasibility, cost, and validated real-world outcomes.

# 6. Final Decision Framework & Executive Takeaway

## 6.1 End-to-End Decision Framework

ChurnIQ connects predictive modeling with customer-value analysis and retention economics through a structured decision-support workflow.

The framework is:

**1. Define the Business Problem**

Establish the retention objective, prediction point, outcome horizon, and analytical scope.

↓

**2. Predict Churn Risk**

Use the final tuned XGBoost model and sigmoid-calibrated probabilities to identify customers with elevated modeled churn risk.

↓

**3. Explain the Prediction**

Use SHAP to identify the behavioral signals associated with individual model predictions.

↓

**4. Assess Customer Value**

Use August ARPU as the project's customer-value proxy.

↓

**5. Quantify Potential Exposure**

Combine calibrated churn probability with exposure-eligible customer value to estimate potential monthly exposure.

↓

**6. Evaluate Targeting Populations**

Compare risk-based and exposure-ranked cohorts according to exposure coverage and operational capacity.

↓

**7. Evaluate Retention Economics**

Test retention-success and intervention-cost assumptions using net value, ROI, break-even analysis, and sensitivity analysis.

↓

**8. Apply Business Constraints**

Consider operational capacity, intervention feasibility, customer context, and other business requirements.

↓

**9. Measure Outcomes**

Track actual intervention results, subsequent churn outcomes, retained customer value, intervention costs, and financial impact.

↓

**10. Reassess**

Use observed outcomes and monitoring evidence to evaluate whether the model, targeting approach, and economic assumptions remain appropriate.

---

## 6.2 What Each Analytical Layer Contributes

Each component of ChurnIQ answers a different business question.

| Analytical Layer | Business Question | Primary Output |
|---|---|---|
| Risk Prediction | Which customers show elevated modeled churn risk? | Calibrated churn probability |
| Explainability | Which behavioral signals are associated with the prediction? | SHAP explanations |
| Customer Value | What customer-value proxy is represented? | August ARPU |
| Revenue Exposure | Where is potential exposure concentrated? | Modeled exposure |
| Targeting | Which customer populations can be evaluated? | Targeting cohorts |
| Economics | How do retention assumptions affect financial feasibility? | ROI, net value, break-even, sensitivity |
| Monitoring | Is the analytical system changing over time? | Data and model monitoring evidence |
| Outcome Measurement | What actually happened after intervention? | Observed business outcomes |

This separation helps prevent one analytical output from being interpreted as evidence for a different business question.

For example:

- churn probability does not represent customer value
- SHAP does not establish causality
- potential exposure does not represent guaranteed revenue loss
- scenario ROI does not represent realized business ROI

---

## 6.3 Evidence Hierarchy

ChurnIQ distinguishes between four levels of evidence.

### Level 1 — Observed Evidence

Directly measured from the historical dataset.

Examples:

- customer counts
- historical churn outcomes
- historical churn rate
- August ARPU

### Level 2 — Model-Derived Evidence

Produced by the predictive analytics pipeline.

Examples:

- calibrated churn probability
- risk classification
- risk ranking
- SHAP explanations

### Level 3 — Scenario-Based Analysis

Dependent on explicit analytical assumptions.

Examples:

- potential revenue exposure
- retention-success assumptions
- intervention costs
- modeled ROI
- break-even requirements
- sensitivity results

### Level 4 — Business Decision Support

Uses the evidence above to structure potential business decisions while considering:

- operational capacity
- customer context
- intervention feasibility
- actual costs
- validated outcomes

This hierarchy provides a clear distinction between:

**What was observed → What the model estimated → What was assumed → What the business may consider**

---

## 6.4 Decision-Support Logic

The central analytical logic of ChurnIQ can be summarized as:

**Risk → Explanation → Value → Exposure → Prioritization → Economics → Validation**

Each stage adds business context to the previous stage.

A customer may have elevated modeled churn risk, but that alone does not establish the customer's business importance or the economic value of an intervention.

Similarly, a high potential exposure estimate does not establish that the corresponding revenue will actually be lost or recovered.

The framework therefore treats retention analysis as a sequence of evidence-building steps rather than a single automated decision.

---

## 6.5 Final Executive Takeaway

ChurnIQ demonstrates how machine learning can be connected to practical business analysis.

The project combines:

- predictive churn modeling
- probability calibration
- customer-level explainability
- customer-value analysis
- potential revenue-exposure analysis
- targeting-cohort analysis
- retention economics
- sensitivity analysis
- monitoring and governance

The result is an end-to-end framework for moving from **customer behavior to structured retention decision support**.

The model provides the risk signal, while the surrounding analytical layers provide explanation, financial context, prioritization logic, and economic evaluation.

---

## 6.6 Production and Decision Boundary

The current ChurnIQ implementation should be considered an **analytical prototype and decision-support system**, not a fully validated production retention engine.

Before operational deployment, the following areas would require validation:

- production data pipelines
- future-period prediction performance
- probability calibration on new populations
- intervention workflows
- actual intervention costs
- observed retention effectiveness
- realized customer-value outcomes
- realized financial impact
- production monitoring
- retraining and model-governance procedures

The existing model card, limitations document, and monitoring plan define the current methodological and governance boundaries.

---

## 6.7 Final Statement

ChurnIQ's central deliverable is not a single churn probability, risk category, targeting cohort, or ROI estimate.

It is a **connected analytical framework** that integrates:

**Prediction → Explanation → Value → Exposure → Prioritization → Economics → Validation**

while maintaining a clear distinction between historical evidence, model-derived outputs, analytical assumptions, and observed business outcomes.

This structure provides a foundation for informed customer-retention analysis while keeping final business decisions dependent on validated evidence, operational context, and real-world results.