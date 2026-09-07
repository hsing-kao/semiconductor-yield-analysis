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

好，DECISIONS.md 的 Decision 7 完成。下一步更新：
Step 10 — LEARNING_NOTES.md
這份文件和 DECISIONS.md 功能不同：
- DECISIONS.md：為什麼我們做這些工程決策
- LEARNING_NOTES.md：你之後複習、準備 technical interview 時，需要真正理解的概念
目前 LEARNING_NOTES.md 已經到 Section 19 — Leakage-Safe Preprocessing Pipeline。LEARNING_NOTES.mdMD
所以我們從 Section 20 接續。
請在 LEARNING_NOTES.md 最下面加入以下內容。
## 20. Dummy Classifier Baseline
A baseline provides a simple reference point for determining whether a predictive model learns useful information beyond a trivial strategy.
For the imbalanced SECOM classification problem, the baseline uses:
DummyClassifier(strategy="most_frequent")
This classifier always predicts the majority class, which is Pass (-1).
Five-fold stratified cross-validation produced:
- Accuracy: 0.9338
- Balanced accuracy: 0.5000
- Failure precision: 0.0000
- Failure recall: 0.0000
- Failure F1: 0.0000
- Average precision: 0.0662
- ROC-AUC: 0.5000
The high accuracy is misleading because the classifier detects no failures.
This provides a useful minimum reference for evaluating a real failure-detection model.
## 21. Class-Weighted Logistic Regression
The primary Milestone 3 model is Logistic Regression with:
class_weight="balanced"
Class weighting gives greater training importance to the minority failure class and lower relative importance to the majority Pass class.
It does not:
- create synthetic samples,
- duplicate failure samples,
- change the observed class distribution,
- or perform oversampling.
The Logistic Regression classifier is placed after the existing preprocessing workflow inside one scikit-learn Pipeline.
Conceptually:
raw data -> leakage-safe preprocessing -> Logistic Regression
During each cross-validation round, both preprocessing and model fitting use only that round's training folds.
This prevents validation observations from determining feature filtering, imputation statistics, scaling parameters, or fitted model coefficients.
## 22. Balanced Accuracy
Ordinary accuracy can be misleading when one class is much more common than the other.
Balanced accuracy gives equal importance to classification performance on the two classes by averaging their recalls.
For the always-Pass DummyClassifier:
- Pass recall = 1
- Failure recall = 0
- Balanced accuracy = 0.5
Therefore, the DummyClassifier can have approximately 93% ordinary accuracy while having only 0.5 balanced accuracy.
The Logistic Regression cross-validation mean balanced accuracy was:
0.5726
The final holdout balanced accuracy was:
0.5372
This indicates some improvement beyond majority-class prediction, although the improvement is limited.
## 23. ROC-AUC and Ranking Ability
Logistic Regression can produce a score or probability representing how strongly each sample is associated with the failure class.
ROC-AUC evaluates how well the model ranks actual Fail samples above actual Pass samples across possible classification thresholds.
Conceptually:
- ROC-AUC = 1.0 represents perfect ranking.
- ROC-AUC = 0.5 represents no useful ranking ability beyond random ordering.
The Logistic Regression results were:
- Cross-validation mean ROC-AUC: 0.6027
- Cross-validation standard deviation: 0.0799
- Final holdout ROC-AUC: 0.6239
These results indicate modest ranking ability beyond random ordering.
ROC-AUC does not mean that 62.39% of all samples were classified correctly. It describes ranking/discrimination ability rather than ordinary classification accuracy.
A model can rank a failure above many Pass samples while still classify that failure as Pass at a particular decision threshold.
Therefore, ROC-AUC and failure recall can behave differently.
## 24. Precision-Recall and Average Precision
For a strongly imbalanced failure-detection problem, precision-recall behavior is especially useful because it focuses directly on performance for the positive failure class.
Recall asks:
Of the actual failures, how many were detected?

Precision asks:
Of the samples predicted as failures, how many were actually failures?

Average precision summarizes precision-recall performance across model score thresholds.
For the DummyClassifier:
- Cross-validation mean average precision: 0.0662
For Logistic Regression:
- Cross-validation mean average precision: 0.1600
- Cross-validation standard deviation: 0.0814
- Final holdout average precision: 0.1192
The final holdout failure prevalence was approximately:
21 / 314 = 0.0669
Therefore, the final average precision of 0.1192 was above the approximate failure prevalence baseline of 0.0669.
This supports the conclusion that the model contains some predictive ranking information for failures, but the absolute performance remains limited.
Average precision should not be interpreted as physical process understanding or causal evidence.
## 25. Classification Threshold
A classification model can first produce a continuous failure score or probability.
A classification threshold converts that score into a final Pass/Fail prediction.
For example, conceptually:
failure probability >= threshold -> predict Fail
failure probability < threshold -> predict Pass
Changing the threshold creates a trade-off between missed failures and false alarms.
A lower threshold can generally increase failure recall but may also increase false positives and reduce precision.
The Milestone 3 final holdout test must not be used to choose a better threshold after observing its results.
Any future threshold-selection analysis would need to make that decision using training/cross-validation data rather than repeatedly evaluating candidate thresholds on the final test set.
## 26. Fold-to-Fold Stability
Five-fold cross-validation should be examined across individual folds rather than using only the mean score.
The Logistic Regression failure recall results were:
- Fold 1: 0.1765
- Fold 2: 0.3529
- Fold 3: 0.2353
- Fold 4: 0.0625
- Fold 5: 0.4375
- Mean: 0.2529
- Standard deviation: 0.1315
ROC-AUC ranged from:
- 0.4618 in Fold 4
- to 0.6958 in Fold 2
Average precision also varied substantially across folds.
Each validation fold contained only 16 or 17 failures, so a small change in the number of correctly detected failures can produce a substantial change in failure recall.
The baseline should therefore not be described as highly stable.
A defensible interpretation is that the model shows predictive signal but meaningful fold-to-fold variability.
## 27. Final Holdout Evaluation
After preprocessing choices, model choice, and evaluation procedures were fixed using the training/CV workflow, the complete modeling pipeline was fitted on all 1,253 training samples.
It was then evaluated on the untouched 314-sample final holdout test set.
The confusion matrix was:
[[259  34]
 [ 17   4]]
with Fail (1) as the positive class.
Therefore:
- TN = 259
- FP = 34
- FN = 17
- TP = 4
Of the 21 actual failures:
- 4 were detected.
- 17 were missed.
Final metrics were:
- Accuracy: 0.8376
- Balanced accuracy: 0.5372
- Failure precision: 0.1053
- Failure recall: 0.1905
- Failure F1: 0.1356
- Average precision: 0.1192
- ROC-AUC: 0.6239
The model predicted 38 samples as failures:
34 false positives + 4 true positives = 38 predicted failures.
Therefore, the failure precision was:
4 / 38 = 0.1053
The failure recall was:
4 / 21 = 0.1905
The model showed modest predictive value beyond the trivial majority-class baseline, but failure detection remained limited.
It should be treated as an interpretable engineering baseline rather than a production-ready failure detector.
## 28. Why the Final Test Set Must Now Remain Closed
The final holdout test has now been used for its intended purpose: one final unseen evaluation of the fixed Milestone 3 modeling workflow.
Its results must not now be used to choose:
- a different classification threshold,
- different class weights,
- different preprocessing settings,
- different hyperparameters,
- or a different model.
Doing so would convert the final test set into development feedback and weaken the validity of its role as unseen evaluation data.
Future Milestone 4 candidate-signal analysis should preserve the leakage-safe evaluation principles established in Milestones 2 and 3.