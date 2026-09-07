# Semiconductor Manufacturing Yield & Process Excursion Analysis

## Project Objective

This project uses the UCI SECOM dataset to build a reproducible and interpretable workflow for semiconductor manufacturing pass/fail analysis.

The project focuses on:

- Inspecting manufacturing measurement data quality
- Building leakage-safe baselines for imbalanced pass/fail classification
- Evaluating failure detection with appropriate metrics
- Identifying anonymous candidate signals associated with failure outcomes
- Exploring process behavior over time without claiming physical root causes from anonymized data

## Dataset

The UCI SECOM dataset contains:

- 1,567 production samples
- 590 anonymized process measurement features
- Pass/fail labels
- Timestamps associated with each sample

The original label encoding is:

- `-1`: Pass
- `1`: Fail

## Dataset Inspection

Initial inspection produced the following results:

| Item | Result |
|---|---:|
| Samples | 1,567 |
| Anonymous measurement features | 590 |
| Pass samples (`-1`) | 1,463 |
| Fail samples (`1`) | 104 |
| Observed pass yield | 93.36% |
| Observed failure rate | 6.64% |
| Missing measurement cells | 41,951 / 924,530 |
| Overall missing rate | 4.54% |
| Constant features | 116 |
| Near-constant candidate features | 10 |
| Earliest timestamp | 2008-07-19 11:55:00 |
| Latest timestamp | 2008-10-17 06:07:00 |
| Chronologically ordered | Yes |
| Duplicate timestamp occurrences | 33 |

The dataset is strongly class-imbalanced: only 104 of 1,567 samples are failures. Therefore, accuracy alone is not an appropriate measure of failure-detection performance. An always-pass prediction would already achieve approximately 93.36% accuracy while detecting none of the failures.

Missing values are also unevenly distributed across features. Although the overall missing rate is approximately 4.54%, several individual features have missing rates above 85%, including some above 91%.

Near-constant candidates were defined during exploratory inspection as non-constant features whose most frequent observed value accounts for at least 98% of their non-missing observations. This threshold is an exploratory screening rule based on the observed feature distribution, not a physical process specification.

## Milestone 2 — Leakage-Safe Data Preprocessing

A leakage-safe preprocessing workflow was developed for the imbalanced SECOM pass/fail classification problem.

The dataset was split into a stratified 80/20 holdout:

- Training set: 1,253 samples
  - 1,170 Pass
  - 83 Fail
- Test set: 314 samples
  - 293 Pass
  - 21 Fail

The test set is reserved for final unseen evaluation and is not used for preprocessing or model-development decisions.

Model development uses 5-fold stratified cross-validation within the training set.

The preprocessing workflow applies:

1. Removal of features with more than 80% missing values, determined from the data used to fit each preprocessing step
2. Removal of features with no usable observed variation
3. Median imputation for remaining missing values
4. Standard scaling

The greater-than-80% missingness threshold is an MVP analysis policy rather than a universal statistical rule or semiconductor process specification. Median imputation was selected as a simple robust approach after training-only analysis showed substantial skewness among the retained measurement features.

The preprocessing steps are implemented in a scikit-learn `Pipeline` using training-fitted feature filters, `SimpleImputer`, and `StandardScaler`. This allows preprocessing parameters and feature-selection decisions to be learned independently within each cross-validation training fold rather than globally before validation.

Pipeline verification on the holdout training set produced:

- Input shape: 1,253 × 590
- Output shape: 1,253 × 466
- Missing values after preprocessing: 0
- Average scaled feature mean: approximately 0
- Average scaled feature standard deviation: 1.0

Fold-by-fold verification confirmed that preprocessing was independently fitted within each cross-validation training fold. One fold retained 460 features while the other four retained 466, demonstrating that feature filtering can adapt to the observations available in each training fold while applying the same fitted feature set to its corresponding validation fold.

No predictive model performance is reported yet. Modeling and failure-detection evaluation begin after completion of the preprocessing milestone.