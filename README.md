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