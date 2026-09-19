# ChurnIQ — Step 7.12
## Threshold Optimization & Business Operating Points

### Objective

Determine practical churn-risk operating points using sigmoid-calibrated probabilities from the finalized tuned XGBoost model.

### Data Boundary

- Development OOF rows: 55,999
- Held-out validation rows: 14,000
- Frozen features: 169
- Test data: not loaded
- Threshold selection: development OOF only

### Calibration

The exact Step 7.11 sigmoid calibrator was applied to the raw tuned-XGBoost OOF probabilities.

Calibration method: **sigmoid**

No additional calibration was fitted.

### Primary Operating Point

| Metric | Value |
|---|---:|
| Threshold | 0.10 |
| Targeting rate | 17.93% |
| Precision | 50.14% |
| Churn capture | 88.21% |
| Lift | 4.92x |

This operating point is an analytical recommendation. Actual deployment should consider intervention capacity, retention strategy, contact cost, and business value.

### Capacity-Constrained Operating Point

A separate 10% intervention-capacity scenario was evaluated.

| Metric | Development OOF |
|---|---:|
| Capacity | 10.00% |
| Threshold | 0.402017 |
| Churn capture | 71.24% |
| Precision | 72.60% |

The capacity point is intentionally separated from the probability threshold selected from the broader trade-off.

### Risk Bands

Analytical churn-probability bands:

- Very Low: 0–20%
- Low: 20–40%
- Medium: 40–60%
- High: 60–80%
- Very High: 80%+

These are prioritization bands and do not guarantee individual churn outcomes.

### Hypothetical Cost Scenarios

Three scenario assumptions were evaluated:

1. Conservative: false positive cost = 1, false negative cost = 1
2. Retention-focused: false positive cost = 1, false negative cost = 3
3. High missed-churn cost: false positive cost = 1, false negative cost = 5

These are illustrative assumptions rather than measured financial costs.

### Held-Out Validation Confirmation

The selected operating point was evaluated on the untouched validation set only after threshold selection.

| Metric | Validation |
|---|---:|
| Targeting rate | 17.53% |
| Precision | 50.77% |
| Recall / churn capture | 87.38% |
| F1 | 0.6423 |
| Lift | 4.98x |

### Governance

- Exact Step 7.11 sigmoid calibrator used: YES
- Threshold selection from development OOF: YES
- Validation used for threshold selection: NO
- Test data loaded: NO
- Additional training: NO
- Additional tuning: NO
- Additional calibration: NO
- Feature selection: NO
- Feature freeze: ACTIVE

### Quality Gate

**PASS**

Step 7.12 is complete.
