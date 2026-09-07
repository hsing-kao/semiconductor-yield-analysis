# Learning Notes

## 1. Yield and Failure Rate

Manufacturing yield is the proportion of observed units that pass:

\[
\text{Yield} =
\frac{\text{Pass samples}}{\text{Total samples}}
\]

For the inspected SECOM dataset:

- Pass: 1,463
- Fail: 104
- Total: 1,567
- Observed pass yield: approximately 93.36%
- Observed failure rate: approximately 6.64%

These values describe the samples in this dataset and should not automatically be interpreted as the long-term production yield of an actual fabrication line.

## 2. Pass/Fail Classification

SECOM defines:

- `-1` = Pass
- `1` = Fail

For failure-detection evaluation, Fail (`1`) is treated as the positive class.

"Positive" does not mean that failure is desirable. It means that failure is the event the classifier is intended to detect.

## 3. Class Imbalance

The dataset is imbalanced because there are many more Pass samples than Fail samples:

- Pass: 1,463
- Fail: 104

This creates a problem for accuracy.

A classifier that always predicts Pass would achieve approximately 93.36% accuracy while detecting zero failures.

Therefore, accuracy alone is not sufficient for evaluating failure detection.

## 4. Confusion Matrix

With Fail defined as the positive class:

- True Positive (TP): actual Fail, predicted Fail
- True Negative (TN): actual Pass, predicted Pass
- False Positive (FP): actual Pass, predicted Fail
- False Negative (FN): actual Fail, predicted Pass

A false negative is especially important for failure detection because it represents a missed failure.

## 5. Recall, Precision, and F1

### Failure Recall

\[
\text{Recall} =
\frac{TP}{TP+FN}
\]

Question answered:

> Of all actual failures, how many did the model detect?

High failure recall means fewer failures are missed.

### Failure Precision

\[
\text{Precision} =
\frac{TP}{TP+FP}
\]

Question answered:

> Of all samples predicted as failures, how many were actually failures?

Low precision means the model produces many false alarms.

### F1 Score

\[
F1 =
2\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}
\]

F1 combines precision and recall using their harmonic mean. It becomes low when either precision or recall is poor.
## 6. Missing Data Inspection

The SECOM measurement matrix contains:

- 924,530 total measurement cells
- 41,951 missing cells
- Overall missing rate: approximately 4.54%

The overall missing rate does not describe how missing values are distributed across features. Some individual features have more than 85% missing values, and several exceed 91%.

Therefore, missingness should be inspected both globally and feature by feature before deciding how to handle missing data.

### Missing-data mechanisms

Three common conceptual mechanisms are:

- **MCAR (Missing Completely At Random):** missingness is unrelated to observed or unobserved values.
- **MAR (Missing At Random):** missingness may depend on other observed information.
- **MNAR (Missing Not At Random):** missingness may depend on the missing value itself or other unobserved factors.

The anonymized SECOM data does not provide enough information to determine which mechanism generated the missing values. The analysis should therefore describe missingness patterns without claiming a physical missing-data mechanism.


## 7. Constant and Near-Constant Features

A **constant feature** has no usable variation across its observed values.

During inspection:

- 116 of 590 features were constant.
- Strictly constant features provide no information for distinguishing Pass from Fail and are candidates for removal during preprocessing.

A **near-constant feature** still has variation, but one value dominates most observations.

Near-constant candidates were screened using:

\[
\text{Dominant Frequency}
=
\frac{\text{Count of most common observed value}}
{\text{Number of non-missing observations}}
\]

For exploratory inspection, a non-constant feature was considered a near-constant candidate when its dominant frequency was at least 98%.

This produced 10 candidates.

Near-constant features were not automatically removed because rare values may still contain predictive information. A feature with only two values could also represent a meaningful binary or two-state signal.

The 98% threshold is an exploratory screening choice based on the observed SECOM distribution, not a universal statistical rule or semiconductor process specification.


## 8. Association Versus Causation

A feature associated with failure is not automatically a physical root cause.

Even if an unusual feature value occurs more frequently among failures, possible explanations include:

- the feature contributes to the failure,
- another process condition influences both the feature and the failure,
- the measurement is a consequence rather than a cause,
- or the observed relationship occurs by chance.

Because SECOM features are anonymized and the dataset is observational, important features should be described as **candidate signals associated with failure outcomes**, not identified physical root causes.


## 9. Timestamp Inspection

SECOM timestamps were converted from strings to a datetime representation using the explicit format:

`%d/%m/%Y %H:%M:%S`

Observed dataset range:

- Earliest: 2008-07-19 11:55:00
- Latest: 2008-10-17 06:07:00
- Chronologically ordered: Yes
- Duplicate timestamp occurrences: 33

Chronological ordering only means that observations are arranged in non-decreasing timestamp order.

It does **not** prove:

- the manufacturing process was stable,
- the process was under statistical control,
- there was no process drift,
- or there were no yield excursions.

Duplicate timestamps also mean that timestamp should not be treated as a unique sample identifier.


## 10. Reproducibility and File Paths

A Python DataFrame such as `X` or `labels` exists in memory while the Python process is running. It does not need to be permanently stored in memory.

A reproducible analysis preserves:

- the raw input data,
- the analysis code,
- the required software environment,
- and the documented analysis decisions.

The DataFrames can then be recreated whenever the analysis is executed.

`pathlib.Path` is used in the formal inspection script to locate the project root relative to the script location rather than hard-coding a Windows-specific absolute path.

This makes the source code more portable.


## 11. Git Basics

Git is a version-control system. GitHub is a service that can host Git repositories remotely.

The basic Git workflow is:

`Working Directory -> Staging Area -> Commit History`

- `git add` moves selected changes into the staging area.
- `git commit` records the staged state as a version-control checkpoint.
- `.gitignore` tells Git which local files should not be tracked.

Files such as `.venv/`, raw downloaded data, temporary verification output, and superseded exploratory scripts are excluded from the repository.

The Milestone 1 checkpoint was committed on the `main` branch as:

`Complete Milestone 1 dataset inspection`

## 12. Train/Test Split and Unseen Evaluation

A train/test split separates model development from final evaluation.

- The training set is used for learning model parameters and developing preprocessing/modeling choices.
- The test set is reserved for final evaluation on unseen data.
- Repeatedly changing the analysis based on test-set performance would cause the test set to influence model development.

For the SECOM dataset, a stratified 80/20 holdout split with `random_state=42` produced:

- Training samples: 1,253
  - Pass: 1,170
  - Fail: 83
  - Failure rate: approximately 6.62%
- Test samples: 314
  - Pass: 293
  - Fail: 21
  - Failure rate: approximately 6.69%

Stratification preserves approximately the same Pass/Fail class proportions in the training and test sets. It does not balance the classes to 50/50.

The test set is kept separate from preprocessing and cross-validation development.


## 13. Stratified Cross-Validation

Five-fold stratified cross-validation is used within the training set.

In each cross-validation round:

- Four folds are used for training.
- One fold is used for validation.
- Each fold serves as the validation fold once.
- The held-out final test set does not participate in cross-validation.

`StratifiedKFold` helps preserve the imbalanced Pass/Fail distribution across validation folds.

Observed validation-fold failure counts were:

- Fold 1: 17 failures
- Fold 2: 17 failures
- Fold 3: 17 failures
- Fold 4: 16 failures
- Fold 5: 16 failures

These sum to the 83 failures in the holdout training set.

Cross-validation performance should be interpreted across all folds rather than reporting only the best-performing fold.


## 14. Data Leakage and Preprocessing Leakage

Data leakage occurs when model development uses information that would not legitimately be available when predicting unseen data.

Leakage can occur even without using test labels.

For example, calculating an imputation value using both training and validation/test measurements allows information from unseen data to influence training preprocessing.

The leakage-safe pattern is:

`fit on training -> transform training/validation/test using training-fitted parameters`

During cross-validation, preprocessing must be fitted separately inside each set of training folds.

Examples of preprocessing quantities that must be learned only from training data include:

- Missing-rate-based feature selection
- Constant-feature selection
- Imputation statistics
- Scaling means and standard deviations

The final test set must not be used to choose preprocessing methods or parameters.


## 15. High-Missing Feature Policy

Exploratory missingness analysis found:

- 32 features with more than 20% missing values
- 28 features with more than 50% missing values
- 8 features with more than 80% missing values
- 4 features with more than 90% missing values

The extremely high-missing features formed groups at approximately 85.58% and 91.19% missingness.

For the MVP, features with more than 80% missingness are removed during preprocessing.

This is a conservative analysis policy intended to reduce dependence on extensively imputed features while retaining less-sparse anonymous features that may still contain predictive information.

The 80% threshold is not a universal statistical rule or semiconductor process specification.

The specific feature columns removed during modeling must be determined from the training data used to fit each preprocessing step rather than hard-coded from full-dataset inspection.


## 16. Median Imputation

Complete-case row deletion is not viable for the raw SECOM measurement matrix:

- Complete rows across all 590 measurement features: 0

Therefore, missing measurements require another handling strategy.

After training-only high-missing and constant-feature filtering:

- Original features: 590
- High-missing features removed: 8
- Constant features removed: 116
- Remaining features: 466

The retained training features showed substantial skewness:

- Median feature skewness: approximately 2.04
- 75th percentile of feature skewness: approximately 9.45
- Observed range: approximately -21.90 to 35.38

Median imputation was selected for the MVP because the median is less sensitive than the mean to skewed distributions and extreme values.

Median imputation does not determine or correct the underlying missing-data mechanism.

For leakage-safe use, the median must be learned from training data and then applied without refitting to validation or test data.


## 17. Fit, Transform, and Fit-Transform

In scikit-learn preprocessing:

- `fit()` learns parameters from data.
- `transform()` applies already learned parameters.
- `fit_transform()` learns parameters and immediately transforms the same training data.

For example, a median imputer fitted on training values `10`, `20`, and `NaN` learns a median of `15`.

If validation values are `100`, `200`, and `NaN`, leakage-safe transformation still replaces the validation `NaN` with the training-derived value `15`, rather than recalculating a validation median of `150`.

During cross-validation:

- Training folds may use `fit_transform()`.
- Validation folds should use `transform()` with preprocessing fitted on the corresponding training folds.


## 18. Feature Scaling

`StandardScaler` standardizes each retained feature using statistics learned from training data.

Conceptually, standardization expresses a measurement relative to its training mean and standard deviation.

Scaling is important for the planned regularized Logistic Regression baseline because feature scale affects optimization and regularization.

Like imputation, scaling must be leakage-safe:

- Fit the scaler on training data.
- Apply the fitted scaler to validation/test data without refitting.

Training-only exploratory verification after filtering and median imputation produced:

- Shape after imputation: 1,253 x 466
- Missing values after imputation: 0
- Shape after scaling: 1,253 x 466
- Average scaled feature mean: approximately 0
- Average scaled feature standard deviation: 1.0


## 19. Leakage-Safe Preprocessing Pipeline

The formal preprocessing workflow contains four ordered steps:

1. Remove features with training-data missing rate greater than 80%.
2. Remove features with at most one observed non-missing unique value in the training data.
3. Apply median imputation.
4. Apply standard scaling.

The feature filters are implemented as scikit-learn-compatible transformers so their feature-selection decisions are learned during `fit()` and reused during `transform()`.

The formal preprocessing pipeline transformed the full holdout training set from:

- Input: 1,253 x 590
- Output: 1,253 x 466
- Missing values after preprocessing: 0
- Average scaled feature mean: approximately 0
- Average scaled feature standard deviation: 1.0

Fold-by-fold preprocessing verification produced:

- Fold 1: training 1,002 x 460; validation 251 x 460
- Fold 2: training 1,002 x 466; validation 251 x 466
- Fold 3: training 1,002 x 466; validation 251 x 466
- Fold 4: training 1,003 x 466; validation 250 x 466
- Fold 5: training 1,003 x 466; validation 250 x 466

Fold 1 retained fewer features because feature filtering was fitted independently on that fold's training subset. The corresponding validation data was transformed using the same training-fitted feature selection.

This fold-specific behavior is important evidence that preprocessing decisions are being learned inside the cross-validation workflow rather than globally before validation.