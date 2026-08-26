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