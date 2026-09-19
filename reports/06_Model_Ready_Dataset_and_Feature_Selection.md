# ChurnIQ — Model-Ready Dataset & Feature Selection

## 1. Purpose

This stage converts the engineered feature dataset into a controlled, evidence-supported modeling feature set.

The objectives are to:

- remove technically unusable features
- control excessive feature redundancy
- evaluate predictive usefulness
- assess predictive stability
- preserve meaningful behavioral signals
- maintain feature-family coverage
- prevent target and future-information leakage
- establish a reproducible modeling feature set
- freeze the approved feature set before baseline model development

Feature selection is performed using development data only.

The validation dataset is not used to select features, and the competition test dataset remains untouched.

---

## 2. Prediction Framework

The feature-selection process follows the established ChurnIQ prediction framework.

### Prediction Point

End of August 2014.

### Available Information

Customer information and behavioral data available through June, July, and August 2014.

### Future Outcome

The subsequent churn outcome associated with the case-study framework.

### Target

`churn_probability`

Although the source column is named `churn_probability`, it functions as the binary churn target in this dataset.

It is not treated as an existing probability score.

Target encoding:

- `0` = retained customer
- `1` = churned customer

Future outcome information must not be used as a predictive feature.

---

## 3. Input Dataset

The feature-selection stage uses the output of the feature-engineering stage.

Source dataset:

`data/processed/feature_engineered_train.csv`

Source dimensions:

- Rows: 69,999
- Columns: 429

The source contains:

- `id`
- `churn_probability`
- 427 candidate predictive features

The customer ID is retained for traceability but is not used as a predictive feature.

The target variable is protected and excluded from all feature-selection calculations.

---

## 4. Candidate Feature Accounting

The feature-selection process begins with:

**427 candidate predictive features**

The following columns are outside the candidate feature pool:

- `id`
- `churn_probability`

The feature-selection process resulted in:

- Candidate predictors: 427
- Retained predictors: 169
- Excluded predictors: 258

Therefore:

**427 candidate predictors → 169 retained predictors**

The final count of 169 was not predetermined.

It resulted from sequential technical screening, redundancy control, predictive screening, stability analysis, model-based evidence, and final feature-family review.

---

## 5. Development and Validation Separation

The existing stratified 80/20 split is reconstructed using:

- Random seed: `42`
- Development rows: 55,999
- Validation rows: 14,000

The development dataset is used for:

- redundancy analysis
- missingness review
- mutual-information screening
- predictive stability analysis
- model-based feature screening
- permutation-importance evaluation
- final feature selection

The validation dataset is reserved for later model evaluation.

The competition test dataset is not loaded during feature selection.

This separation prevents validation performance from becoming an indirect feature-selection criterion.

---

## 6. Feature Selection Decision Philosophy

Feature selection follows a staged evidence-based framework:

```text
Technical validity
        ↓
Missingness review
        ↓
Redundancy control
        ↓
Predictive screening
        ↓
Predictive stability
        ↓
Model-based evidence
        ↓
Feature-family coverage
        ↓
Leakage review
        ↓
Final feature freeze

No single statistical measure is used as the sole basis for final feature selection.

A feature may be retained because it provides useful predictive evidence even when its individual relationship with churn is not obvious.

Similarly, a feature is not retained solely because it appears intuitively important.

Selection decisions are based on measurable evidence, modeling usefulness, feature redundancy, stability, and business relevance.

7. Stage 1 — Constant Feature Screening

Features with no variation provide no useful predictive information.

The first screening stage identifies constant features within the development dataset.

Results:

Constant features excluded: 21
Near-constant features excluded automatically: 0

Constant features are documented in the feature-selection register.

No business interpretation is assigned to constant features because they contain no variation within the modeling data.

8. Stage 2 — Missingness Screening

Feature missingness is reviewed before predictive screening.

The purpose is not to automatically delete highly incomplete variables.

High missingness may represent:

unavailable service
non-adoption
behavioral inactivity
structural absence
collection limitations

Therefore, high-missingness features are reviewed rather than automatically removed.

Result:

Features with more than 90% development-set missingness: 2

These features were flagged for review rather than automatically deleted solely because of missingness.

9. Stage 3 — Redundancy Analysis

Telecom datasets contain many closely related variables.

Examples include:

monthly usage measures
recharge amounts
recharge frequencies
voice activity measures
data usage measures
derived temporal variables

Highly redundant features can:

increase model complexity
make interpretation more difficult
create unstable feature-importance rankings
provide limited incremental information

Spearman correlation was therefore used as a redundancy screen.

Redundancy Threshold

|Spearman correlation| >= 0.95

The correlation review was performed using a development-data sample.

Development sample size:

15,000 rows.

Results:

Numeric features entering redundancy review: 397
Redundancy groups identified: 34
Features involved in highly correlated groups: 125
Features excluded through redundancy decisions: 91

The redundancy process does not imply that excluded correlated features are incorrect or meaningless.

It identifies cases where retaining every highly similar variable provides limited additional information.

10. Stage 4 — Missingness Signal Redundancy

Missingness indicators can themselves contain overlapping information.

Therefore, missingness patterns were also reviewed.

Result:

Highly similar missingness signals identified: 435

These signals were evaluated as part of the overall feature-selection process.

The objective was to avoid retaining large numbers of indicators carrying essentially the same missingness information.

11. Stage 5 — Predictive Candidate Screening

After technical and redundancy screening, the remaining candidate features were evaluated for predictive usefulness.

Results:

Predictive candidates: 315
Numeric candidates entering predictive screening: 308

These candidates were evaluated using development data only.

The purpose of this stage was to identify features with measurable predictive information before more computationally intensive model-based evaluation.

12. Stage 6 — Mutual Information Screening

Mutual information was used to identify features containing potentially useful information about the churn target.

Mutual information can identify nonlinear dependencies that may not be captured by simple linear association measures.

The observed development-data distribution was used to establish the screening cutoff.

Result:

Mutual Information cutoff = 0.012391

Features meeting the screening requirement progressed to subsequent predictive stability analysis.

Mutual information was used as a screening mechanism rather than as the sole basis for final feature selection.

13. Stage 7 — Predictive Stability

A feature may appear predictive in one sample but fail to provide stable evidence across development folds.

Candidate features were therefore evaluated using cross-validated univariate predictive stability.

Results:

Stability candidates: 292
Stable features: 171

Only development data was used for this stage.

The validation dataset was not used to determine feature stability.

14. Stage 8 — Model-Based Predictive Evidence

A HistGradientBoosting-based screening model was used to evaluate predictive usefulness in a nonlinear modeling context.

Permutation importance was then used as additional evidence for feature contribution.

Results:

Model-based candidates: 169
Permutation-importance evaluation: completed

Model-based importance was treated as supporting evidence rather than as the sole selection criterion.

15. Final Feature Selection

The final selection decision combined evidence from:

technical validity
missingness behavior
redundancy analysis
mutual information
predictive stability
nonlinear model-based evidence
permutation importance
feature-family coverage
leakage controls

Final selected features:

169

These 169 features constitute the approved modeling feature set.

16. Why 169 Features?

The final feature count was not chosen in advance.

The 169 retained features survived the established sequence of technical, statistical, predictive, and modeling-oriented screening stages.

The purpose of selection was not to minimize the number of features at any cost.

Instead, the objective was to achieve a controlled feature set that:

removes clearly unusable predictors
reduces excessive redundancy
retains predictive behavioral signals
maintains relevant feature families
supports model interpretability
avoids unnecessary dimensionality

Therefore, 169 should be interpreted as the outcome of the selection methodology rather than as a predetermined target.

17. Feature-Family Composition

The final feature set contains the following major groups:

Feature Family	Count
Voice / Call Behavior	106
Recharge Activity	31
Customer Value	13
Data Engagement	7
Temporal Change	4
Activity Signal	3
Original / Other	2
Behavioral Trend	2
Customer Tenure	1
Total	169

The relatively large Voice / Call Behavior component reflects the structure of the source telecom dataset and the predictive evidence observed during selection.

Family balance was reviewed to ensure that feature selection did not unintentionally eliminate entire behavioral dimensions.

Feature-family balance was not artificially forced.

18. Final Model-Ready Feature Set

The final selected feature set contains:

Development rows: 55,999
Validation rows: 14,000
Selected features: 169
Numeric features: 169
Categorical features: 0

The final selected feature matrices used for baseline modeling contain no missing values after development-fitted preprocessing.

Preprocessing parameters are learned using development data only.

The validation dataset receives transformations learned from the development dataset.

19. Preprocessing Boundary

The feature-selection stage does not modify the raw source data.

The workflow preserves a clear distinction between:

Raw source data
Engineered features
Selected features
Development-fitted preprocessing
Model-ready matrices

Imputation parameters are learned from development data only.

No test information is used to learn preprocessing parameters.

20. Leakage Controls

The following controls were applied:

Target excluded from predictive feature matrices.
Customer ID excluded from predictive modeling.
Future-period information excluded.
Validation outcomes not used for feature selection.
Test dataset not loaded.
Development-only preprocessing maintained.
Feature-selection calculations performed using development data.
Feature freeze established before model development.

Leakage checks passed.

21. Feature Selection Register

A complete feature-selection register was generated.

Register:

reports/03D_feature_selection_register.csv

The register contains all 427 candidate features and records their final status.

Results:

Total candidate features: 427
Retained: 169
Excluded: 258

The register provides feature-level traceability for selection decisions.

22. Selection Decision Artifacts

The following artifacts were generated:

reports/03D_feature_selection_register.csv

reports/03D_correlation_decisions.csv

reports/03D_missingness_review.csv

reports/03D_mutual_information_screening.csv

reports/03D_predictive_stability.csv

reports/03D_permutation_importance.csv

reports/03D_feature_family_summary.csv

These artifacts provide an auditable record of the selection process.

23. Final Modeling Artifacts

The model-ready datasets were generated as:

data/processed/final_model_features.csv

data/processed/X_dev.csv

data/processed/X_validation.csv

data/processed/y_dev.csv

data/processed/y_validation.csv

These artifacts form the controlled input to the baseline modeling stage.

24. Feature Freeze

The final feature set is now frozen at:

169 features

No new feature should be added during baseline model comparison based on validation performance.

If new features are later considered, they must be treated as a new feature-selection iteration and must undergo the appropriate:

leakage review
development-only screening
redundancy analysis
predictive evaluation
validation controls

This prevents iterative feature engineering from contaminating model evaluation.

25. Validation Discipline After Feature Freeze

Validation performance will be used only after the feature freeze to assess model generalization and compare candidate models.

If substantial validation degradation is observed, the project will investigate potential causes such as:

overfitting
preprocessing differences
distribution differences
model instability
feature behavior
data-quality issues

The feature set will not be repeatedly modified using validation performance simply to improve the validation score.

Any required feature changes will constitute a new controlled modeling iteration.

26. Feature Selection Limitations

Feature-selection results depend on:

the available dataset
the development sample
screening thresholds
correlation thresholds
predictive stability methodology
model-based screening methodology

Therefore, the retained feature set should be considered a controlled modeling configuration rather than proof that excluded features have no relationship with churn.

An excluded feature may still contain information that is redundant with retained features or may become useful under a different modeling methodology.

Feature selection also does not establish causality.

A feature being predictive or associated with churn does not mean that changing the feature will necessarily prevent churn.

27. Interpretation Boundary

Feature selection identifies variables with predictive usefulness.

It does not establish causal relationships.

Business recommendations will therefore be developed only after combining:

model evidence
SHAP explanations
business context
revenue exposure
customer prioritization
appropriate assumptions
documented limitations

The model will be used as a decision-support mechanism rather than as proof of individual customer behavior.

28. Reproducibility

The feature-selection workflow records:

Python environment
random seed
development/validation split
feature counts
missingness criteria
correlation threshold
mutual-information cutoff
predictive stability methodology
model-based screening methodology
final retained features
feature-selection statuses
generated artifacts

The objective is to make the feature-selection process reproducible and auditable.

29. Quality Gate

The Model-Ready Dataset & Feature Selection stage passes when:

Candidate features are documented.
Candidate accounting is reconciled.
Constant features are controlled.
Missingness is reviewed.
Redundant features are screened.
Predictive evidence is evaluated.
Predictive stability is evaluated.
Model-based evidence is evaluated.
Feature-family coverage is reviewed.
Development/validation separation is preserved.
Target leakage is checked.
Future leakage is checked.
Validation data is not used for feature selection.
Test data remains untouched.
Final feature set is reproducible.
Feature-selection artifacts are saved.
Final feature set is frozen.
Quality Gate Result

PASS

30. Final Outcome

The feature-engineering and feature-selection pipeline transformed:

427 candidate predictive features

into:

169 evidence-supported modeling features

through a controlled sequence of technical screening, redundancy analysis, predictive screening, stability evaluation, model-based evidence, and leakage review.

The resulting 169-feature set is frozen and ready for baseline machine-learning development.

The feature-selection process maintains the established prediction-time boundary and prevents validation/test contamination.