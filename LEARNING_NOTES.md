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