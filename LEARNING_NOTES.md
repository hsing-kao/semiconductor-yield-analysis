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

## 29. Logistic Regression Coefficients for Candidate-Signal Analysis
After Milestone 3 established Logistic Regression as the interpretable baseline, Milestone 4 uses its coefficients as one source of feature-association evidence.
Because the retained features are standardized before Logistic Regression, coefficient magnitudes are more comparable than they would be on the original heterogeneous measurement scales.
With Fail (1) as the positive class:
- A positive coefficient means that a higher standardized feature value is associated with the model's Fail direction, conditional on the other features in the fitted model.
- A negative coefficient means that a higher standardized feature value is associated with the model's Pass direction, conditional on the other features.
- A larger absolute coefficient indicates a stronger contribution within that fitted Logistic Regression model.
A coefficient does not establish physical causation.
It also should not be interpreted in isolation because correlated features can influence how Logistic Regression distributes coefficients among measurements.
The full holdout-training preprocessing retained 466 features and produced 466 Logistic Regression coefficients.
The feature indices had to be recovered from the fitted preprocessing filters because preprocessing removes some original columns. The coefficient position alone is not the original SECOM feature index.
For example, the first retained feature indices were:
0, 1, 2, 3, 4, 6, 7, 8, 9, 10
so transformed-column position and original anonymous feature index are not interchangeable.

## 30. Cross-Validation Coefficient Stability
A large coefficient from one fit may not represent a stable association.
Milestone 4 therefore refitted the complete preprocessing and Logistic Regression workflow independently in each of the five cross-validation training folds.
The retained feature counts were:
- Fold 1: 460
- Fold 2: 466
- Fold 3: 466
- Fold 4: 466
- Fold 5: 466
This reproduced the fold-specific preprocessing behavior observed in Milestone 2.
For each feature, coefficient stability was examined using:
- number of folds in which the feature was retained,
- coefficient direction,
- mean coefficient,
- mean absolute coefficient,
- minimum and maximum coefficient across folds.
Among the 460 features present in all five folds, 210 had the same coefficient direction in all five folds.
Therefore:
5/5 direction consistency alone is not a sufficiently selective definition of a strong candidate signal.

Feature 59 was particularly prominent in the coefficient analysis:
- Present in 5/5 folds
- Positive coefficient in 5/5 folds
- Mean coefficient: approximately 1.6681
- Minimum coefficient: approximately 1.0482
- Maximum coefficient: approximately 2.1715
This makes Feature 59 a strong coefficient-based candidate, but coefficient evidence alone is not sufficient for final candidate screening.

## 31. Mann–Whitney U and Rank-Biserial Effect Size
Milestone 4 uses a training-only univariate analysis as a complementary perspective to the multivariable Logistic Regression model.
The Mann–Whitney U test compares the relative ranks of observed feature values between the Fail and Pass groups.
It was selected because the retained SECOM features showed substantial skewness and there is no basis for assuming that every anonymous measurement feature follows a normal distribution.
The analysis uses observed values rather than median-imputed values so that large groups of artificially identical imputed values do not determine the rank comparison.
The Mann–Whitney U test should not automatically be interpreted as a test of medians alone. It provides evidence about differences in the relative distributions/ranks between groups.
Rank-biserial correlation is used as an effect-size measure.
In this project:
- Positive rank-biserial values indicate that Fail samples tend to have higher observed values than Pass samples.
- Negative values indicate that Fail samples tend to have lower observed values than Pass samples.
- The absolute value describes the strength of the rank-based separation.
A p-value and an effect size answer different questions.
A small p-value does not automatically mean that an association is large or engineering-significant.
For example, Feature 103 had the largest observed absolute rank-biserial effect among the initial univariate results:
- Rank-biserial: approximately 0.3628
- Raw p-value: approximately 3.21e-08
However, Feature 103 was not among the strongest coefficient-stability candidates.
This demonstrates why Milestone 4 does not rely on a single feature-ranking method.

## 32. Multiple Testing and False Discovery Rate
Testing hundreds of features creates a multiple-testing problem.
Milestone 4 performed 466 univariate feature tests.
Using unadjusted p-values:
- 79 features had raw p < 0.05.
However, when many tests are performed, some small p-values can occur by chance.
The Benjamini–Hochberg false-discovery-rate procedure was therefore applied.
After correction:
- 15 features had FDR-adjusted q < 0.05.
This reduction from 79 raw-significant results to 15 FDR-significant results demonstrates why unadjusted p-values should not be used alone for high-dimensional feature screening.
The FDR procedure controls the expected proportion of false discoveries among the set of declared discoveries under its assumptions.
An FDR-significant association still does not establish physical causation.

## 33. Permutation Importance
Permutation importance provides a validation-based predictive perspective.
For each cross-validation round:
1. The complete preprocessing and Logistic Regression pipeline is fitted on the CV training folds.
2. Baseline performance is evaluated on the corresponding validation fold.
3. One raw input feature is randomly shuffled across validation samples.
4. Validation performance is evaluated again.
5. The performance change measures how much the fitted model depended on the original feature-to-sample relationship.
Milestone 4 uses average precision as the permutation-importance scoring metric because failure is the minority class and average precision focuses on precision-recall performance.
A positive permutation importance means that disrupting the feature tended to reduce validation average precision.
A value near zero means that shuffling the feature had little average effect on validation performance.
Negative importance can occur when shuffling happens to improve validation performance.
Permutation importance was repeated 10 times per feature within each validation fold.
The analysis showed meaningful fold-to-fold variability.
For example:
- Feature 75 had the largest mean permutation importance among the displayed results, approximately 0.0208, but was positive in only 3/5 folds.
- Feature 14 and Feature 64 had positive permutation importance in 5/5 folds.
- Feature 59 had mean permutation importance approximately 0.00716 and positive importance in 4/5 folds.
Therefore, permutation-importance magnitude and cross-fold consistency should both be considered.
Permutation importance does not establish physical causation and can also be affected by correlated or redundant features.

## 34. Cross-Method Candidate-Signal Screening
Milestone 4 combines complementary evidence rather than manually selecting features that look interesting.
The MVP candidate-signal rule requires all three criteria:
1. The feature is retained in all five CV training folds and its Logistic Regression coefficient has the same direction in all five folds.
2. The training-only univariate association has Benjamini–Hochberg FDR q < 0.05.
3. Validation-fold permutation importance for average precision is positive in at least 4/5 folds.
The criteria are explicit and reproducible, but they are analysis policies for this project rather than universal statistical thresholds.
Five anonymous features satisfied all three criteria:
- Feature 59
- Feature 129
- Feature 21
- Feature 477
- Feature 341
All five had:
- positive Logistic Regression coefficients in 5/5 folds,
- positive rank-biserial associations,
- agreement between coefficient and univariate direction,
- FDR q < 0.05,
- positive permutation importance in 4/5 folds.
The selected features are therefore described as:
candidate signals associated with failure outcomes
They are not described as identified root causes.
The five candidates are also not treated as a definitive ordered ranking of physical importance because coefficient magnitude, rank-biserial effect size, and permutation importance measure different quantities on different scales.

## 35. Candidate Robustness and Missingness
Candidate-signal evidence should also be interpreted in the context of measurement availability.
The five selected features had low training missingness:
- Feature 59: approximately 0.40%
- Feature 129: approximately 0.72%
- Feature 21: approximately 0.08%
- Feature 477: approximately 0.32%
- Feature 341: approximately 0.32%
All 83 training failures had observed values for all five candidate features.
Therefore, the candidate results are not based on a small subset of observed failure measurements.
Observed training-group medians were:
Feature	Pass median	Fail median
59	0.8018	4.9745
129	-0.1419	0.0000
21	-5526.50	-5426.25
477	5.19305	6.4994
341	2.37485	2.7761


For all five candidates, the Fail-group median was higher than the Pass-group median, consistent with their positive rank-biserial direction.
These values describe associations in the SECOM training data. They do not establish that increasing a feature physically causes failure.

## 36. Irregular Time Sampling

Chronological ordering does not mean that observations are equally spaced in time.

Milestone 5 examined the consecutive timestamp gaps across the SECOM dataset.

Observed results:

- Consecutive timestamp gaps: 1,566
- Median gap: 37 minutes
- Mean gap: approximately 82.5 minutes
- Minimum gap: 0 minutes
- Zero-length gaps: 33
- Maximum gap: 2 days and 38 minutes

The observation intervals therefore vary substantially.

The 33 zero-length gaps are consistent with the duplicate timestamps identified during Milestone 1. Timestamp should therefore not be treated as a unique sample identifier.

Irregular sampling affects visualization and interpretation.

For example, connecting consecutive observations with a continuous line can visually imply knowledge of the measurement trajectory between observations even when no measurements were recorded during that interval.

Milestone 5 therefore uses scatter-based timestamp visualization for the raw candidate measurements.

Irregular time spacing does not by itself prove process instability, drift, or an excursion. It describes the structure of the available observations.


## 37. Temporal Distribution Variation

A measurement distribution can change over the observed timeline.

This can involve changes in:

- location, such as the median,
- dispersion, such as the interquartile range (IQR),
- or both.

Milestone 5 divided the complete chronological observation sequence into four approximately equal-count blocks as a reproducible descriptive summary.

These blocks are based on observation order rather than manually selected visual boundaries.

For Feature 59, the formal chronological-block summary showed:

- Block 1 median: approximately 9.62
- Block 1 IQR: approximately 19.17
- Block 2 median: approximately 1.03
- Block 2 IQR: approximately 4.05
- Block 3 median: approximately -0.25
- Block 3 IQR: approximately 4.69
- Block 4 median: approximately -0.55
- Block 4 IQR: approximately 4.75

Feature 59 therefore had substantially higher measurements and greater dispersion in the earliest chronological block than in the later three blocks.

The other frozen candidate signals also showed temporal variation, but the patterns were not identical.

For example:

- Feature 129 showed a distinct lower-valued and more dispersed distribution in Block 3.
- Feature 21 showed substantially larger IQRs in Blocks 1–2 than in Blocks 3–4.
- Features 477 and 341 showed more moderate temporal differences in measurement level and dispersion.

Because the patterns differ across features, the analysis does not support describing all five candidates as undergoing one common process shift.

Temporal distribution variation in an anonymous measurement does not identify the physical reason for the change.


## 38. Separating Time Effects from Outcome Composition

A temporal difference in a candidate signal could partly reflect changes in the proportion of Pass and Fail observations over time.

This matters because the five Milestone 4 candidates were already selected for positive association with failure outcomes.

Conceptually:

more Fail observations in a time period
-> candidate values may appear higher
-> even if the measurement distribution within Pass observations did not change

Milestone 5 therefore performed a supporting exploratory check by examining chronological-block summaries separately within Pass and Fail observations.

Temporal differences remained visible within Pass observations for multiple candidates.

For example, Feature 59 Pass-only measurements showed:

- Block 1 median: approximately 8.94
- Block 2 median: approximately 0.97
- Block 3 median: approximately -0.44
- Block 4 median: approximately -0.55

Its Pass-only IQR also decreased substantially after Block 1.

Therefore, changing Pass/Fail composition alone does not explain all of the observed Feature 59 temporal variation.

This does not identify the physical cause of the temporal change. It only shows that the pattern is not solely a consequence of different Pass/Fail proportions.


## 39. Association Can Vary Across Time

An overall association does not guarantee that the same descriptive separation appears in every chronological period.

For Feature 59, the within-block Fail median minus Pass median was:

- Block 1: approximately +5.13
- Block 2: approximately +4.31
- Block 3: approximately +3.67
- Block 4: approximately -0.18

The first three blocks showed higher Fail-group medians, while the final block did not show the same median separation.

Other candidates also showed different degrees of within-block stability.

This means that the Milestone 4 candidate signals should not automatically be interpreted as temporally invariant standalone failure indicators.

The chronological-block comparison is descriptive. It does not establish a statistically significant feature-by-time interaction.

Small Fail subgroup sizes are also an important limitation:

- Block 1: 53 Fail observations
- Block 2: 14
- Block 3: 13
- Block 4: 24

Therefore, within-block Fail medians and IQRs, especially in Blocks 2 and 3, should be interpreted cautiously.


## 40. Chronological Blocks Are Not Change Points

Milestone 5 uses four approximately equal-count chronological observation blocks.

The purpose is to create a simple and reproducible way to summarize measurement distributions over the observed timeline.

The block boundaries are not:

- detected process change points,
- known manufacturing interventions,
- process-state boundaries,
- excursion start or end times,
- or physical operating regimes.

For example, a boundary generated by dividing observations into equal-count blocks should not be interpreted as evidence that the manufacturing process physically changed at that timestamp.

A formal change-point analysis would answer a different question and would require additional assumptions and interpretation.


## 41. Why Formal SPC Was Not Used

Statistical process control should not be added simply because the dataset comes from semiconductor manufacturing.

A defensible control chart requires enough process and sampling context to define what the chart represents and how its control limits should be interpreted.

The anonymized SECOM dataset does not provide important context such as:

- process subgroup definitions,
- tool or chamber identity,
- recipe or product context,
- sampling policy,
- engineering specification limits,
- or a known stable baseline operating period.

The timestamps are also irregularly spaced and contain duplicates.

Milestone 5 therefore uses descriptive time-oriented visualization rather than presenting formal SPC control limits that cannot be adequately justified from the available information.

Not using SPC is not evidence that the process was stable or unstable.

It means that the available dataset does not provide enough context for a defensible SPC interpretation in this MVP.


## 42. Display Zoom Versus Data Removal

The formal Milestone 5 figures use the combined observed 1st–99th percentile range for each candidate signal as a display-only zoom.

This improves readability because a small number of extreme observations can otherwise compress the central distribution into a narrow region of the figure.

The zoom does not:

- delete observations,
- modify the raw dataset,
- change candidate selection,
- change numerical summaries,
- or redefine an outlier-removal policy.

Measurements outside the displayed range remain part of the dataset and numerical analysis.

The same percentile rule is applied consistently rather than manually choosing a different visual range to make a particular candidate look more strongly associated with failure.