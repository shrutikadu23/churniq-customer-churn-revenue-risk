
# Phase 12 — Customer Value & Revenue Risk

## Status

**FAIL**

## Purpose

Phase 12 connects the locked Phase 11 calibrated churn probabilities
with August customer-value economics to quantify scenario-based
potential revenue exposure.

The purpose is to understand **where customer-value exposure is
concentrated**, not to prescribe retention actions.

---

## Customer Value

August ARPU (`arpu_8`) is used as the customer-value proxy.

- Customer universe: **69,999**
- Average August ARPU: **₹278.86**
- Median August ARPU: **₹192.23**
- Minimum August ARPU: **₹-945.81**
- Maximum August ARPU: **₹33,543.62**
- Negative August ARPU customers: **352**

Raw ARPU is preserved.

For revenue exposure calculations only, negative ARPU values are
clipped to zero.

---

## Revenue Exposure

### Monthly Formula

`Potential Monthly Revenue Exposure =
Calibrated Churn Probability × max(August ARPU, 0)`

### Annualized Formula

`Annualized Scenario Exposure =
Potential Monthly Revenue Exposure × 12`

### Overall Results

- Total potential monthly exposure: **₹776,208.24**
- Annualized scenario exposure: **₹9,314,498.90**
- High + Very High monthly exposure: **₹551,432.56**
- Very High monthly exposure: **₹264,492.92**

Very High-risk customers represent **8.79%**
of the customer universe and account for **34.07%**
of potential monthly exposure.

This comparison describes exposure concentration; it does not establish
that the customers will churn or that the exposure will become realized
revenue loss.

---

## Revenue Risk by Risk Level

| risk_level              |   customers |   customer_share_pct |   avg_churn_probability |   avg_arpu |   total_exposure_eligible_arpu |   potential_monthly_revenue_exposure |   annualized_scenario_exposure |   exposure_share_pct |   exposure_minus_customer_share_pp |
|:------------------------|------------:|---------------------:|------------------------:|-----------:|-------------------------------:|-------------------------------------:|-------------------------------:|---------------------:|-----------------------------------:|
| Below Primary Threshold |       57507 |             82.154   |               0.0147234 |   310.875  |                    1.78795e+07 |                               224776 |                    2.69731e+06 |              28.9582 |                           -53.1959 |
| High                    |        6340 |              9.05727 |               0.237609  |   199.183  |                    1.26427e+06 |                               286940 |                    3.44328e+06 |              36.9668 |                            27.9096 |
| Very High               |        6152 |              8.7887  |               0.770131  |    61.6913 |               380457           |                               264493 |                    3.17392e+06 |              34.075  |                            25.2863 |

---

## Revenue Risk by Customer Value Tier

| customer_value_tier   |   customers |   avg_arpu |   avg_churn_probability |   potential_monthly_revenue_exposure |   annualized_scenario_exposure |   customer_share_pct |   exposure_share_pct |
|:----------------------|------------:|-----------:|------------------------:|-------------------------------------:|-------------------------------:|---------------------:|---------------------:|
| Lower Value           |       17499 |    32.9071 |               0.276532  |                              53256.5 |               639078           |              24.9989 |              6.86111 |
| Mid-Lower Value       |       17500 |   136.274  |               0.053444  |                             124017   |                    1.4882e+06  |              25.0004 |             15.9772  |
| Mid-Upper Value       |       17500 |   269.75   |               0.0430579 |                             198700   |                    2.3844e+06  |              25.0004 |             25.5989  |
| Higher Value          |       17500 |   676.49   |               0.0321805 |                             400235   |                    4.80282e+06 |              25.0004 |             51.5628  |

---

## Risk × Value Matrix

| risk_level              | customer_value_tier   |   customers |   avg_churn_probability |   avg_arpu |   potential_monthly_revenue_exposure |   annualized_scenario_exposure |   exposure_share_pct |
|:------------------------|:----------------------|------------:|------------------------:|-----------:|-------------------------------------:|-------------------------------:|---------------------:|
| Below Primary Threshold | Lower Value           |        9801 |               0.0231299 |   47.6481  |                              9310.88 |               111731           |              1.19953 |
| Below Primary Threshold | Mid-Lower Value       |       15545 |               0.014976  |  136.667   |                             31368.5  |               376422           |              4.04125 |
| Below Primary Threshold | Mid-Upper Value       |       15849 |               0.0126575 |  270.325   |                             53580.4  |               642965           |              6.90284 |
| Below Primary Threshold | Higher Value          |       16312 |               0.0114388 |  674.449   |                            130516    |                    1.56619e+06 |             16.8146  |
| High                    | Lower Value           |        2708 |               0.250893  |   25.9384  |                             17158.2  |               205898           |              2.21051 |
| High                    | Mid-Lower Value       |        1409 |               0.226143  |  134.032   |                             42453.9  |               509446           |              5.46939 |
| High                    | Mid-Upper Value       |        1282 |               0.235028  |  265.015   |                             79426.1  |               953113           |             10.2326  |
| High                    | Higher Value          |         941 |               0.220063  |  705.609   |                            147902    |                    1.77482e+06 |             19.0544  |
| Very High               | Lower Value           |        4990 |               0.788161  |    7.73562 |                             26787.5  |               321450           |              3.45107 |
| Very High               | Mid-Lower Value       |         546 |               0.702991  |  130.858   |                             50194.2  |               602330           |              6.46659 |
| Very High               | Mid-Upper Value       |         369 |               0.681838  |  261.501   |                             65694    |               788327           |              8.46344 |
| Very High               | Higher Value          |         247 |               0.686189  |  700.332   |                            121817    |                    1.46181e+06 |             15.6939  |

---

## Exposure Concentration

|   top_customer_percent |   customer_count |   monthly_exposure |   exposure_share_pct |
|-----------------------:|-----------------:|-------------------:|---------------------:|
|                      1 |              700 |             248555 |              32.0217 |
|                      5 |             3500 |             495515 |              63.8379 |
|                     10 |             7000 |             602398 |              77.6078 |
|                     20 |            14000 |             684559 |              88.1927 |
|                     50 |            35000 |             754123 |              97.1547 |

The top 10% of customers by potential revenue exposure account for
approximately **77.61%** of total potential monthly exposure.

---

## Governance

- Phase 11 calibrated churn probabilities used as locked inputs.
- Primary threshold retained at **0.10**.
- August ARPU used as the customer-value proxy.
- Raw negative ARPU values preserved.
- Negative ARPU clipped to zero only for exposure calculations.
- No September predictors used.
- No model retraining.
- No model tuning.
- No recalibration.
- No feature selection.
- No threshold modification.
- No retention actions.
- No retention-cost assumptions.
- No ROI calculations.

---

## Interpretation Boundary

Potential revenue exposure is a **scenario-based financial exposure
measure** calculated from model-estimated churn probability and August
customer-value proxy.

It is **not**:

- guaranteed revenue loss,
- a forecast of booked revenue,
- guaranteed future churn,
- guaranteed revenue that can be saved,
- or a retention recommendation.

---

## Phase 13 Boundary

Retention prioritization, intervention costs, expected retention success,
potential revenue saved, net benefit, ROI, and sensitivity scenarios belong
to **Phase 13 — Retention Prioritization & ROI**.
