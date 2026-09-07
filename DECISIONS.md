# Engineering Decision Log

This file records important analysis and modeling decisions, including the reasoning, alternatives considered, and limitations.

## Decision 1 — Preserve the original SECOM label definition

**Decision:**
Use the original SECOM encoding during dataset inspection:

- `-1` = Pass
- `1` = Fail

Treat the failure class (`1`) as the positive class for failure-detection evaluation.

**Reason:**
The label meaning was verified from the original `secom.names` documentation rather than inferred from the numerical values. Failure is the event of interest for recall, precision, F1, and related evaluation metrics.

**Alternative considered:**
Re-encode Pass/Fail as `0/1` immediately.

**Why not yet:**
Re-encoding is not necessary for dataset inspection. If later modeling code benefits from `0/1` encoding, the transformation will be explicit and documented.

---

## Decision 2 — Remove strictly constant features during preprocessing

**Decision:**
Remove features with zero usable variation during model preprocessing.

**Observed during inspection:**
116 of the 590 measurement features have at most one observed non-missing value.

**Reason:**
A strictly constant feature cannot distinguish between pass and fail samples because it provides no variation across observations.

**Important implementation constraint:**
Feature filtering used for modeling must be implemented in a leakage-safe way with respect to the training data rather than using validation outcomes to guide feature selection.

---

## Decision 3 — Inspect but do not automatically remove near-constant features

**Decision:**
Define near-constant candidates during exploratory inspection as non-constant features with a dominant observed-value frequency of at least 98%. Do not automatically remove them from the MVP solely because they meet this definition.

**Observed during inspection:**
10 features meet this exploratory criterion.

**Reason:**
The observed dominant-frequency distribution showed a small group around 98.6%–99.9%, followed by a large drop to approximately 58%. The 98% threshold therefore provides a useful screening rule for this dataset.

A rare value may still contain predictive information, particularly because the SECOM features are anonymized. Low variation alone is not sufficient evidence that a non-constant feature is useless.

**Alternative considered:**
Apply an arbitrary raw-variance threshold or automatically remove all features above the dominant-frequency threshold.

**Why not:**
Raw variance depends on feature scale, and the physical units of the anonymous SECOM measurements are unknown. Automatic removal could also discard rare but potentially informative signals.

**Limitation:**
The 98% threshold is an exploratory analysis choice, not a semiconductor process specification or universal statistical rule.
## Decision 4 — Keep downloaded raw data outside version control

**Decision:**
Do not commit the downloaded UCI SECOM raw data files or generated processed datasets to the Git repository.

The following directories are excluded through `.gitignore`:

- `data/raw/`
- `data/processed/`

**Reason:**
The SECOM dataset is an external data source rather than project-generated source code. Keeping downloaded raw data outside version control makes data provenance clearer and keeps the repository focused on reproducible code, documentation, and analysis decisions.

The project README should provide instructions for obtaining the dataset from UCI and placing the required files under `data/raw/`.

**Reproducibility implication:**
A user who clones the repository cannot run the analysis immediately without first obtaining the SECOM dataset. Reproducibility therefore depends on documenting the external data source and expected local file structure.

**Alternative considered:**
Commit the raw SECOM files directly into the repository.

**Why not:**
The project can remain reproducible by documenting the external data source and download procedure without duplicating externally maintained raw data in the Git repository.

## Decision 5 — Remove extremely high-missing features during preprocessing

**Decision:**
During modeling preprocessing, remove features whose missing rate is greater than 80% in the data used to fit the preprocessing step.

**Observed during exploratory analysis:**
8 of the 590 features have more than 80% missing values. In the full-dataset exploration, these features form two extreme-missingness groups at approximately 85.58% and 91.19% missingness.

A training-only robustness check using the current stratified holdout split also identified 8 features above the 80% threshold, with the same feature indices as the full-dataset exploratory result.

**Reason:**
Features with more than 80% missingness contain observed measurements for fewer than 20% of samples and would therefore depend heavily on imputation. For the MVP, the 80% threshold provides a conservative policy that reduces dependence on extensively imputed features while avoiding more aggressive removal of anonymous features that may still contain predictive information.

**Important implementation constraint:**
The specific features to remove must be determined from the training data used to fit each preprocessing step. Feature indices identified from the full dataset must not be hard-coded into cross-validation preprocessing.

**Limitation:**
The 80% threshold is an analysis policy for this MVP, not a universal statistical rule or semiconductor process specification. Features with very high missingness may still contain predictive information in their observed values, so removing them involves a potential information-loss trade-off.

## Decision 6 — Use median imputation for retained measurement features

**Decision:**
Use median imputation for missing values in measurement features that remain after high-missing and constant-feature filtering.

**Observed during training-only analysis:**
Complete-case row deletion is not viable for the SECOM measurement matrix because zero samples contain complete observations across all 590 measurement features.

After applying the current training-only exploratory filtering steps:

- 8 features with more than 80% missing values were excluded.
- 116 constant features were excluded.
- 466 features remained.

The retained training features showed substantial distributional skewness:

- Median feature skewness: approximately 2.04
- 75th percentile of feature skewness: approximately 9.45
- Observed skewness range: approximately -21.90 to 35.38

**Reason:**
Median imputation is less sensitive than mean imputation to skewed distributions and extreme values. Given the substantial skewness observed across the retained training features, median imputation provides a simple and robust approach for the MVP.

**Important implementation constraint:**
Imputation values must be learned only from the data used to fit the preprocessing step. During cross-validation, the imputer must be fitted separately on each set of training folds and then applied to the corresponding validation fold without refitting.

**Limitation:**
Median imputation does not establish or correct the underlying missing-data mechanism. The anonymized SECOM data does not provide enough information to determine whether missingness is MCAR, MAR, or MNAR. Median imputation is an MVP preprocessing choice, not a claim that it is universally optimal.

## Decision 7 — Use class-weighted Logistic Regression as the interpretable baseline model

**Decision:**
Use Logistic Regression with `class_weight="balanced"` as the primary interpretable modeling baseline for Milestone 3.

The classifier is combined with the existing leakage-safe preprocessing workflow inside a single scikit-learn `Pipeline`.

**Reason:**
The SECOM training data is strongly class-imbalanced, so a majority-class classifier can achieve high accuracy while detecting no failures. Class weighting increases the training importance of the minority failure class without creating synthetic samples or changing the observed class distribution.

Logistic Regression was selected as the primary baseline because it provides a relatively simple and interpretable starting point for failure detection and remains compatible with later candidate-signal analysis.

**Evaluation design:**
Model development used 5-fold stratified cross-validation within the 1,253-sample training set. The final 314-sample holdout test set remained isolated until the model and evaluation procedure were fixed.

The same cross-validation framework was used to compare the Logistic Regression model against a `DummyClassifier(strategy="most_frequent")` baseline.

**Observed cross-validation results:**

| Metric | DummyClassifier mean | Logistic Regression mean | Logistic Regression std |
|---|---:|---:|---:|
| Accuracy | 0.9338 | 0.8499 | 0.0170 |
| Balanced accuracy | 0.5000 | 0.5726 | 0.0638 |
| Failure precision | 0.0000 | 0.1403 | 0.0589 |
| Failure recall | 0.0000 | 0.2529 | 0.1315 |
| Failure F1 | 0.0000 | 0.1786 | 0.0796 |
| Average precision | 0.0662 | 0.1600 | 0.0814 |
| ROC-AUC | 0.5000 | 0.6027 | 0.0799 |

The Logistic Regression baseline sacrificed overall accuracy relative to the always-Pass dummy classifier but provided non-zero failure detection and improved balanced accuracy, failure recall, precision, F1, average precision, and ROC-AUC.

Cross-validation also showed meaningful fold-to-fold variation, particularly in failure recall and ranking metrics. This variability is an important limitation given the small number of failures available in each validation fold.

**Final holdout evaluation:**
After model-development decisions were complete, the modeling pipeline was fitted using the full 1,253-sample training set and evaluated once on the untouched 314-sample holdout test set.

The final confusion matrix was:

```text
[[259  34]
 [ 17   4]]
```

with Fail (1) treated as the positive class.
This corresponds to:
- True negatives: 259
- False positives: 34
- False negatives: 17
- True positives: 4
Final holdout metrics:
- Accuracy: 0.8376
- Balanced accuracy: 0.5372
- Failure precision: 0.1053
- Failure recall: 0.1905
- Failure F1: 0.1356
- Average precision: 0.1192
- ROC-AUC: 0.6239
Interpretation:
The class-weighted Logistic Regression showed modest predictive value beyond the trivial majority-class baseline, but failure detection remained limited and cross-validation performance varied meaningfully across folds.
The model should therefore be treated as an interpretable engineering baseline rather than a production-ready failure detector.
Important constraint:
The final holdout results must not now be used to tune the classification threshold, preprocessing choices, class weights, hyperparameters, or model selection. Doing so would allow the final test set to influence model development.
Limitation:
Predictive performance does not establish physical causation. Any later analysis of influential anonymous SECOM features must describe them as candidate signals associated with failure outcomes rather than identified physical root causes.