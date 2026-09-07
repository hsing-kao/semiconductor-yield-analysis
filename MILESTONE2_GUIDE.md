---
title: Semiconductor Manufacturing Yield & Process Excursion Analysis — Milestone 2
---

# Semiconductor Manufacturing Yield & Process Excursion Analysis

## Milestone 2 — Leakage-Safe Data Preprocessing

## Milestone goal

Milestone 2 builds a reproducible, leakage-safe preprocessing workflow for the UCI SECOM pass/fail classification problem.

The goal is not to train a predictive model yet. The goal is to make sure that later model evaluation is trustworthy by preventing validation or test information from influencing preprocessing decisions.

The formal preprocessing workflow developed in this milestone is:

```text
Raw measurement features
        ↓
High-missing feature filtering
        ↓
Constant-feature filtering
        ↓
Median imputation
        ↓
Standard scaling
        ↓
Model-ready features
```

The most important principle is:

> Any preprocessing step that learns something from data must be fitted only on the training data available at that stage.

During cross-validation, this means preprocessing must be fitted separately inside each set of cross-validation training folds and then applied to the corresponding validation fold without refitting.

---

# Learning Log

## 1. Environment update — installing scikit-learn

Milestone 1 ended with:

- Python 3.14.7
- pandas 3.0.5
- numpy 2.5.2
- scikit-learn not yet installed

I first checked whether scikit-learn was already installed:

```powershell
python -m pip show scikit-learn
```

Output:

```text
WARNING: Package(s) not found: scikit-learn
```

I then installed it inside the project virtual environment:

```powershell
python -m pip install scikit-learn
```

The installation completed successfully with:

```text
scikit-learn-1.9.0
```

I verified that Python could import the package and checked the installed version:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

Output:

```text
1.9.0
```

The project `requirements.txt` was updated to include:

```text
numpy==2.5.2
pandas==3.0.5
scikit-learn==1.9.0
```

### Key lesson

Using:

```powershell
python -m pip
```

helps ensure that `pip` belongs to the Python interpreter currently being used.

---

## 2. Virtual-environment issue encountered later

After returning to the project on a later day, running the learning script produced:

```text
ModuleNotFoundError: No module named 'sklearn'
```

I checked which Python executable was actually running:

```powershell
python -c "import sys; print(sys.executable)"
```

Output:

```text
C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe
```

This was the system Python, not the project virtual environment.

The intended interpreter is:

```text
C:\Users\User\Downloads\secom\.venv\Scripts\python.exe
```

PowerShell initially blocked virtual-environment activation because script execution was disabled. I used a process-scoped execution-policy change:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activated the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt then showed:

```text
(.venv) PS C:\Users\User\Downloads\secom>
```

A second interpreter check confirmed:

```text
C:\Users\User\Downloads\secom\.venv\Scripts\python.exe
```

### Key lesson

A package can be correctly installed but still appear missing if the script is executed with a different Python interpreter.

The `-Scope Process` execution-policy change applies only to the current PowerShell process and disappears when that terminal session is closed.

---

## 3. Learning code versus formal project code

At first, Milestone 2 experiments were temporarily added to:

```text
src/inspect_data.py
```

This was corrected because `src/inspect_data.py` is the formal Milestone 1 reproducible inspection script and should remain a stable Milestone 1 artifact.

Git was used to inspect the accidental changes:

```powershell
git diff -- src/inspect_data.py
```

The file was restored to the committed Milestone 1 version:

```powershell
git restore src/inspect_data.py
```

After restoration, `git status` showed only the intended `requirements.txt` modification.

A separate learning script was created:

```text
milestone2_learning.py
```

This file was used for incremental experiments, verification prints, and small teaching examples.

The formal Milestone 2 implementation was later placed in:

```text
src/preprocessing.py
```

`milestone2_learning.py` was added to `.gitignore` so it remains available locally for study but is not part of the formal GitHub portfolio code.

### Key lesson

The project separates:

```text
learning / experimentation code
            ↓
understand and verify decisions
            ↓
formal reproducible code in src/
```

This keeps the portfolio implementation concise without losing the local learning history.

---

## 4. Recreating X and y for Milestone 2

The learning script loads the raw SECOM measurement and label files.

The measurement matrix is:

```text
X
```

and the target is:

```python
y = labels["label"]
```

The target retains the original SECOM encoding:

```text
-1 = Pass
 1 = Fail
```

Verification output:

```text
X shape: (1567, 590)
y shape: (1567,)
```

Label counts:

```text
-1    1463
 1     104
```

### Interpretation

- `X` contains 1,567 samples and 590 anonymous measurement features.
- `y` contains one Pass/Fail label for each sample.
- The failure class remains the positive class.

The timestamp is not included in `X` for this baseline preprocessing workflow.

---

# Train/Test Split

## 5. Why unseen test data is required

If the same data is used both to train a model and to evaluate it, the evaluation does not tell us how well the model generalizes to unseen samples.

The basic roles are:

```text
Training data
→ learn model/preprocessing parameters

Validation data
→ compare methods during development

Test data
→ final unseen evaluation
```

A test set should not repeatedly guide preprocessing or model choices. Even if test labels are not directly used in `fit()`, repeatedly changing the analysis based on test performance allows the test set to influence development.

---

## 6. Stratified 80/20 holdout split

The holdout split uses:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
```

### Parameter meanings

`test_size=0.20`

> Approximately 20% of samples are reserved for the final test set.

`stratify=y`

> Preserve approximately the same Pass/Fail proportions in training and test sets.

`random_state=42`

> Make the pseudo-random split reproducible. The value `42` is not statistically optimal; it is simply a fixed seed.

### Actual split results

```text
X_train shape: (1253, 590)
X_test shape:  (314, 590)
y_train shape: (1253,)
y_test shape:  (314,)
```

Training labels:

```text
Pass (-1): 1170
Fail (1):     83
```

Test labels:

```text
Pass (-1): 293
Fail (1):    21
```

Observed failure rates:

```text
Original dataset ≈ 6.64%
Training set     ≈ 6.62%
Test set         ≈ 6.69%
```

### Interpretation

The class proportions remain very similar after the split, showing that stratification worked as intended.

Stratification does not change the dataset to 50% Pass / 50% Fail. It preserves the original class imbalance approximately.

The 314-sample test set is reserved for final evaluation and does not participate in cross-validation development.

---

# Cross-Validation

## 7. Why cross-validation is used

The holdout training set contains only 83 failures.

If a permanent validation set were removed from the training set, even fewer failures would remain for model fitting.

Five-fold cross-validation allows the training data to be used more efficiently:

```text
5 folds total

Each round:
4 folds → training
1 fold  → validation
```

Each fold serves as validation exactly once.

The final 314-sample test set does not participate.

---

## 8. StratifiedKFold configuration

The cross-validation splitter is:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

### Meanings

`n_splits=5`

> Create five folds and therefore five validation rounds.

`shuffle=True`

> Shuffle samples before assigning stratified folds for this baseline evaluation design.

`random_state=42`

> Make the shuffled fold assignment reproducible.

SECOM is chronologically ordered, so shuffled CV should not be confused with time-aware validation. A future supplemental time-aware analysis would answer a different question, such as training on earlier observations and evaluating later observations.

---

## 9. Actual fold sizes and failure counts

The first fold contained:

```text
Fold 1 training samples: 1002
Fold 1 validation samples: 251
```

The Fold 1 validation labels were:

```text
Pass: 234
Fail:  17
```

Failure rate:

```text
17 / 251 ≈ 6.77%
```

This is close to the holdout training failure rate of approximately 6.62%.

All five validation folds were then checked:

```text
Fold 1: training = 1002 validation = 251 failures = 17
Fold 2: training = 1002 validation = 251 failures = 17
Fold 3: training = 1002 validation = 251 failures = 17
Fold 4: training = 1003 validation = 250 failures = 16
Fold 5: training = 1003 validation = 250 failures = 16
```

The validation failure counts sum to:

```text
17 + 17 + 17 + 16 + 16 = 83
```

which equals the total number of failures in the holdout training set.

### Key lesson

Cross-validation results should be interpreted across all folds. Reporting only the best-performing fold would cherry-pick an optimistic result and would not represent overall validation performance.

---

# Data Leakage

## 10. What data leakage means

Data leakage occurs when model development uses information that would not legitimately be available when predicting unseen data.

A very obvious example is allowing the model to see `y_test` while making test predictions.

However, leakage can also happen during preprocessing without using test labels.

---

## 11. Imputation leakage example

Suppose a feature contains:

```text
Training observed values:
10, 20

Test observed values:
100, 200
```

If training and test are combined before mean imputation:

```text
(10 + 20 + 100 + 200) / 4 = 82.5
```

The training missing value would then be influenced by test measurements.

This is preprocessing leakage.

The leakage-safe training mean is:

```text
(10 + 20) / 2 = 15
```

The correct pattern is:

```text
fit preprocessing on training
        ↓
learn parameter = 15
        ↓
transform training
transform validation/test using the same 15
```

---

## 12. Leakage inside cross-validation

The same rule applies inside each CV round.

For example, if Fold 3 is validation:

```text
Fold 1 + Fold 2 + Fold 4 + Fold 5
→ CV training

Fold 3
→ CV validation
```

The imputation statistics for that round must be calculated only from Folds 1, 2, 4, and 5.

It would still be leakage to calculate preprocessing statistics from all 1,253 holdout training samples before starting CV, because the current validation fold would already have influenced preprocessing.

Therefore:

```text
CV Round 1
→ fit preprocessing on Round 1 training folds

CV Round 2
→ fit preprocessing again on Round 2 training folds

...
```

Each round must learn its own preprocessing parameters.

---

## 13. fit(), transform(), and fit_transform()

In scikit-learn preprocessing:

### `fit()`

Learns parameters from data.

For a median imputer:

```text
Training: 10, 20, NaN
        ↓ fit
learn median = 15
```

### `transform()`

Uses already learned parameters without relearning them.

### `fit_transform()`

Fits parameters and immediately transforms the same training data.

A leakage-safe CV pattern is:

```text
CV training
→ fit_transform()

CV validation
→ transform()
```

Validation should not call `fit_transform()` because that would learn preprocessing parameters from validation data.

---

# High-Missing Feature Policy

## 14. Missingness distribution exploration

Milestone 1 found an overall missing rate of approximately 4.54%, but missingness was highly uneven across features.

Milestone 2 examined the number of features above several missingness thresholds:

```text
> 20% missing: 32 features
> 50% missing: 28 features
> 80% missing:  8 features
> 90% missing:  4 features
```

All features with more than 20% missingness were:

```text
157    0.911934
292    0.911934
293    0.911934
158    0.911934
220    0.855775
85     0.855775
492    0.855775
358    0.855775
245    0.649649
244    0.649649
246    0.649649
111    0.649649
382    0.649649
383    0.649649
110    0.649649
109    0.649649
384    0.649649
518    0.649649
517    0.649649
516    0.649649
580    0.605616
579    0.605616
581    0.605616
578    0.605616
346    0.506701
345    0.506701
72     0.506701
73     0.506701
247    0.456286
112    0.456286
385    0.456286
519    0.456286
```

This distribution forms several clear groups:

```text
91.19% → 4 features
85.58% → 4 features
64.96% → 12 features
60.56% → 4 features
50.67% → 4 features
45.63% → 4 features
```

There is also a large gap between the >20% group and the remaining features.

---

## 15. Why high missingness does not automatically mean useless

A feature with 90% missing values could still contain useful predictive information in its 10% observed values.

Therefore:

```text
high missingness ≠ no predictive value
```

However, an extremely sparse feature also has a major drawback: most of its values would need to be imputed, making its model representation highly dependent on the imputation strategy.

The trade-off is:

```text
Keep sparse feature
→ may retain predictive signal
→ but relies heavily on imputation

Remove sparse feature
→ reduces dependence on imputation
→ but may discard predictive signal
```

---

## 16. Decision — remove features with >80% training missingness

For the MVP, the selected policy is:

```text
Remove a feature when its missing rate is greater than 80%
in the data used to fit the preprocessing step.
```

This is a conservative policy compared with removing all features above 50% missingness.

The reasoning is:

- Features above 80% missingness contain observed measurements for fewer than 20% of samples.
- Keeping them would make their representation heavily dependent on imputation.
- A more aggressive threshold would remove substantially more anonymous features that may still contain predictive information.

This threshold is:

- an MVP analysis policy,
- not a universal statistical rule,
- not a semiconductor process specification.

---

## 17. Training-only robustness check

Full-dataset exploration found 8 features above 80% missingness:

```text
[85, 157, 158, 220, 292, 293, 358, 492]
```

The holdout training set independently produced:

```text
8 features
```

with the same indices:

```text
[85, 157, 158, 220, 292, 293, 358, 492]
```

This supports the stability of the extreme-missing group for the current holdout split.

However, the formal preprocessing implementation does not hard-code these indices.

Instead, each training fold calculates its own missing rates and applies the >80% policy.

---

# Constant-Feature Filtering

## 18. Why constant filtering must also be fitted on training data

Suppose a feature is:

```text
CV training:
5, 5, 5, 5

CV validation:
5, 5, 8
```

Within the CV training data, the feature is constant.

If validation is inspected before deciding whether to remove it, the value `8` would cause the feature to appear non-constant.

That would allow validation information to influence feature selection.

Therefore, constant-feature status must also be learned only from training data.

The operational rule remains:

```text
at most one observed non-missing unique value
→ constant
→ remove
```

---

## 19. Training-only filtering results

After first removing the 8 features above 80% training missingness:

```text
590 - 8 = 582 features
```

The training-only constant-feature check found:

```text
116 constant features
```

After removing them:

```text
582 - 116 = 466 features
```

Actual output:

```text
Constant features after high-missing filtering: 116
Features remaining after both filters: 466
```

Although this training-only count matched the Milestone 1 full-dataset count of 116, the formal code does not assume every CV fold will always identify exactly 116 constant features.

---

# Choosing an Imputation Method

## 20. Why complete-case row deletion is impossible here

A simple missing-data strategy would be to remove every sample containing any missing measurement.

This was tested with:

```python
complete_rows = X.dropna().shape[0]
```

Actual result:

```text
Complete rows with no missing measurements:
0
```

Therefore:

```text
Original samples: 1567
Complete rows:       0
```

Every sample contains at least one missing value somewhere across the 590 measurement features.

Complete-case deletion would remove 100% of the dataset and is therefore not viable.

This does not contradict the overall missing-cell rate of approximately 4.54%. With 590 columns, missing cells can be distributed such that every row contains at least one missing value.

---

## 21. Mean versus median imputation

Mean imputation is sensitive to extreme values.

Example:

```text
Observed values:
10, 11, 12, 13, 100
```

Mean:

```text
29.2
```

Median:

```text
12
```

The extreme value `100` pulls the mean upward much more strongly than the median.

Important causal direction:

> The mean does not cause skewness. A skewed distribution or extreme observations can pull the mean away from the center of the main data mass.

Median imputation is therefore generally more robust to skewness and extreme values than mean imputation.

---

## 22. Training-only skewness exploration

To avoid using the final test set to choose an imputation method, skewness was examined using training data only.

Before filtering, the 590 training features had:

```text
count    590.000000
mean       4.929776
std        8.863797
min      -21.900518
25%        0.000000
50%        0.932882
75%        6.295575
max       35.380830
```

However, constant and extremely high-missing features are not part of the intended final imputation population.

After training-only >80% missing filtering and constant filtering, 466 features remained.

Their skewness summary was:

```text
count    466.000000
mean       6.220590
std        9.566388
min      -21.900518
25%        0.340885
50%        2.044866
75%        9.451553
max       35.380830
```

### Interpretation

The retained training features show substantial distributional asymmetry.

In particular:

```text
Median feature skewness ≈ 2.04
75th percentile         ≈ 9.45
```

This supports using median imputation as a simple, robust MVP choice.

It does not prove that median imputation is universally optimal.

---

## 23. Decision — median imputation

The selected MVP policy is:

```text
After high-missing and constant-feature filtering,
use median imputation for remaining missing measurements.
```

Reason:

> The retained training features showed substantial skewness, and median imputation is less sensitive than mean imputation to skewed distributions and extreme values.

Limitation:

The SECOM dataset does not provide enough information to determine whether the actual missing-data mechanism is MCAR, MAR, or MNAR.

Median imputation does not establish or correct the underlying physical missingness mechanism.

---

# SimpleImputer Experiment

## 24. What SimpleImputer learns during fit()

A small example was used before applying imputation to SECOM:

```python
imputer_example = pd.DataFrame({
    "feature_A": [10.0, 20.0, None],
})

example_imputer = SimpleImputer(strategy="median")
example_imputer.fit(imputer_example)
```

The learned statistic was printed with:

```python
print(example_imputer.statistics_)
```

Output:

```text
[15.]
```

This confirms that `fit()` learned:

```text
median = 15
```

from the observed training values 10 and 20.

---

## 25. transform() experiment

The same training example was transformed:

```text
Before:
10
20
NaN
```

Output:

```text
[[10.]
 [20.]
 [15.]]
```

The missing value was replaced with the fitted median of 15.

---

## 26. Validation transformation without refitting

A separate validation example was created:

```text
100
200
NaN
```

Its own observed median would be 150.

However, the already-fitted training imputer was used only with:

```python
example_imputer.transform(validation_example)
```

Output:

```text
[[100.]
 [200.]
 [ 15.]]
```

The validation missing value became `15`, not `150`.

### Key lesson

This directly demonstrates leakage-safe preprocessing:

```text
Training
→ fit median = 15

Validation
→ transform using 15
→ do not refit
```

---

# Feature Scaling

## 27. Why scaling is needed

Anonymous SECOM features may have very different numerical scales.

For example:

```text
Feature A ≈ 0 to 1
Feature B ≈ 0 to 100000
```

The planned Logistic Regression baseline uses regularization, and feature scale affects optimization and regularization behavior.

Standardization puts features onto more comparable numerical scales.

The conceptual StandardScaler transformation is:

```text
standardized value
= (original value - training feature mean)
  / training feature standard deviation
```

After scaling, each training feature is expected to have approximately:

```text
mean ≈ 0
standard deviation ≈ 1
```

Scaling does not reveal physical units, determine feature importance, or convert anonymous measurements into known process parameters.

---

## 28. Scaling leakage

StandardScaler must learn:

```text
training feature means
training feature standard deviations
```

Therefore, scaling follows the same leakage rule as imputation:

```text
CV training folds
→ fit scaler

CV validation fold
→ transform using training-fitted scaler
```

Validation data must not participate in calculating the scaler parameters.

---

## 29. Exploratory preprocessing verification

After training-only high-missing filtering and constant filtering, the 466 retained training features were processed with:

```text
Median imputation
        ↓
StandardScaler
```

Actual output:

```text
Imputed shape: (1253, 466)
Missing values after imputation: 0
Scaled shape: (1253, 466)
Average scaled feature mean: 4.125199033573938e-17
Average scaled feature std: 1.0
```

The mean value:

```text
4.125199033573938e-17
```

is effectively zero within floating-point precision.

### Interpretation

The exploratory preprocessing behaved as intended:

- no samples were lost,
- 466 features remained,
- no missing values remained after imputation,
- scaled feature means were approximately zero,
- average scaled feature standard deviation was 1.0.

---

# Formal Leakage-Safe Transformers

## 30. Why custom transformers are needed

`SimpleImputer` and `StandardScaler` are already scikit-learn transformers.

However, this project also needs two custom data-dependent operations:

1. Remove features with training missing rate >80%.
2. Remove features with at most one observed non-missing unique value.

If these operations were performed globally before cross-validation, validation folds could influence feature-selection decisions.

Therefore, both operations were implemented as scikit-learn-compatible transformers with separate `fit()` and `transform()` behavior.

---

## 31. HighMissingFeatureFilter

The formal transformer is defined in:

```text
src/preprocessing.py
```

Its behavior is conceptually:

```text
fit(training)
→ calculate training missing rates
→ remember features with missing rate <= 80%

transform(validation/test)
→ use the same remembered feature list
→ do not recalculate missing rates
```

Formal verification produced:

```text
Learned features to keep: 582
Training shape after filter: (1253, 582)
Test shape after filter: (314, 582)
```

This matches:

```text
590 - 8 = 582
```

The test set was transformed using the feature list learned from training; it was not fitted independently.

---

## 32. ConstantFeatureFilter

The second custom transformer behaves conceptually as:

```text
fit(training)
→ count observed unique values per feature
→ remember features with >1 observed unique value

transform(validation/test)
→ use the same training-fitted feature list
```

Formal verification after high-missing filtering produced:

```text
Learned features to keep: 466
Training shape after both filters: (1253, 466)
Test shape after both filters: (314, 466)
```

This matched the earlier training-only exploratory result.

---

## 33. Implementation error encountered — missing class

When `ConstantFeatureFilter` was first imported, Python returned:

```text
ImportError: cannot import name 'ConstantFeatureFilter' from 'src.preprocessing'
```

Inspection showed that the class had not actually been added to `src/preprocessing.py`.

After adding it, another error appeared:

```text
IndentationError: expected an indented block after class definition
```

The formal file was rewritten with clean Python indentation.

### Key lesson

Python class methods must be indented inside their class definitions. A class definition at the left margin should contain methods indented beneath it.

The corrected module successfully imported and passed its formal verification.

---

# Formal Preprocessing Pipeline

## 34. Pipeline structure

The formal preprocessing pipeline is built by:

```python
build_preprocessing_pipeline()
```

Its ordered steps are:

```text
high_missing_filter
        ↓
constant_filter
        ↓
median_imputer
        ↓
scaler
```

A formal import/build smoke test produced:

```text
['high_missing_filter', 'constant_filter', 'median_imputer', 'scaler']
```

The pipeline intentionally does not contain Logistic Regression yet because Milestone 2 covers preprocessing only. Modeling begins in Milestone 3.

---

## 35. Full holdout-training Pipeline verification

The formal pipeline was fitted on the complete holdout training set:

```text
Input shape: (1253, 590)
Output shape: (1253, 466)
Missing values after pipeline: 0
Average scaled feature mean: 4.125199033573938e-17
Average scaled feature std: 1.0
```

### Interpretation

The complete formal Pipeline successfully performed:

```text
590 raw features
        ↓
>80% missing filtering
        ↓
constant filtering
        ↓
median imputation
        ↓
standard scaling
        ↓
466 processed features
```

with no missing values remaining.

---

# Cross-Validation Leakage-Safety Verification

## 36. Why the Pipeline must be fitted separately in each fold

It is not sufficient to fit preprocessing once on all 1,253 holdout training samples and then perform cross-validation.

That would allow each CV validation fold to influence:

- missing-rate feature selection,
- constant-feature selection,
- imputation medians,
- scaling parameters.

Instead, a new preprocessing Pipeline is created and fitted independently in each CV round.

Conceptually:

```text
Round 1
CV training 1 → fit Pipeline 1
CV validation 1 → transform with Pipeline 1

Round 2
CV training 2 → fit new Pipeline 2
CV validation 2 → transform with Pipeline 2

...
```

---

## 37. Actual fold-by-fold Pipeline results

The formal Pipeline was verified inside the 5-fold split.

Actual output:

```text
Fold 1: train=(1002, 460) validation=(251, 460)
Fold 2: train=(1002, 466) validation=(251, 466)
Fold 3: train=(1002, 466) validation=(251, 466)
Fold 4: train=(1003, 466) validation=(250, 466)
Fold 5: train=(1003, 466) validation=(250, 466)
```

### Important observation — Fold 1 retained only 460 features

The full holdout training set retained 466 features, but Fold 1 retained only 460.

This is not automatically an error.

A feature may have usable variation in the complete 1,253-sample training set but become constant within a smaller CV-training subset.

The important result is that Fold 1 produced:

```text
training   → 460 features
validation → 460 features
```

The validation fold used the feature selection learned from that fold's training data rather than independently deciding which features to keep.

The other folds produced matching training/validation dimensions of 466 features.

### Interpretation

This fold-specific behavior is useful evidence that data-dependent preprocessing is being fitted inside the CV workflow rather than globally before validation.

It should not be described as mathematical proof that every possible form of leakage is impossible, but it is consistent with the intended leakage-safe implementation and the transformer code structure.

---

# Final Formal Implementation

## 38. Formal Milestone 2 source file

The formal preprocessing implementation is:

```text
src/preprocessing.py
```

It contains:

```text
HighMissingFeatureFilter
ConstantFeatureFilter
build_preprocessing_pipeline()
```

The Pipeline combines:

```text
HighMissingFeatureFilter(threshold=0.80)
ConstantFeatureFilter()
SimpleImputer(strategy="median")
StandardScaler()
```

The learning experiments remain locally in:

```text
milestone2_learning.py
```

and this file is excluded from Git through `.gitignore`.

---

# Documentation Decisions

## 39. Decision 5 — extremely high-missing features

Formal decision:

> During modeling preprocessing, remove features whose missing rate is greater than 80% in the data used to fit the preprocessing step.

Important limitation:

> The threshold is an MVP analysis policy, not a universal statistical rule or semiconductor process specification. Sparse features may still contain predictive information.

---

## 40. Decision 6 — median imputation

Formal decision:

> Use median imputation for missing values in measurement features that remain after high-missing and constant-feature filtering.

Reason:

> The retained training features showed substantial skewness, and median imputation is less sensitive than mean imputation to skewed distributions and extreme values.

Important limitation:

> Median imputation does not establish or correct the unknown physical missing-data mechanism.

---

# Milestone 2 Final Status

## 41. Completed items

- [x] Install and verify scikit-learn 1.9.0
- [x] Update `requirements.txt`
- [x] Define `X` and `y`
- [x] Create stratified 80/20 holdout split
- [x] Reserve final test set for unseen evaluation
- [x] Learn 5-fold cross-validation
- [x] Implement and verify `StratifiedKFold`
- [x] Understand preprocessing leakage
- [x] Understand `fit()`, `transform()`, and `fit_transform()`
- [x] Explore feature missingness distribution
- [x] Adopt >80% training-missingness removal policy
- [x] Verify training-only high-missing group
- [x] Implement leakage-safe high-missing filtering
- [x] Implement leakage-safe constant filtering
- [x] Show complete-case deletion leaves zero samples
- [x] Examine training-only feature skewness
- [x] Select median imputation for the MVP
- [x] Verify `SimpleImputer` behavior with training/validation examples
- [x] Learn and verify standard scaling
- [x] Build formal scikit-learn preprocessing Pipeline
- [x] Verify full holdout-training Pipeline behavior
- [x] Verify fold-by-fold preprocessing inside 5-fold CV
- [x] Update `DECISIONS.md`
- [x] Update `LEARNING_NOTES.md`
- [x] Update `README.md`
- [x] Exclude `milestone2_learning.py` from Git
- [x] Run formal Pipeline smoke test
- [x] Commit Milestone 2

---

## 42. Git checkpoint

Formal Milestone 2 commit:

```text
5290797 — Complete Milestone 2 leakage-safe preprocessing
```

Files included in the commit:

```text
.gitignore
DECISIONS.md
LEARNING_NOTES.md
README.md
requirements.txt
src/preprocessing.py
```

Commit output:

```text
[main 5290797] Complete Milestone 2 leakage-safe preprocessing
6 files changed, 347 insertions(+), 5 deletions(-)
create mode 100644 src/preprocessing.py
```

Final repository status:

```text
On branch main
nothing to commit, working tree clean
```

Milestone 2 is formally complete.

---

# What I Should Be Able to Explain in an Interview

## 43. Short explanation of the preprocessing design

A concise explanation is:

> I reserved a stratified 20% holdout test set and used five-fold stratified cross-validation within the remaining training data. I implemented missing-rate and constant-feature filters as scikit-learn-compatible transformers, followed by median imputation and standard scaling. All data-dependent preprocessing is fitted only on the corresponding training data, so validation and test observations do not determine feature selection, imputation statistics, or scaling parameters.

---

## 44. Why not use accuracy alone?

The original SECOM dataset contains:

```text
1463 Pass
104 Fail
```

An always-Pass classifier would already achieve approximately 93.36% accuracy while detecting zero failures.

Therefore, Milestone 3 must evaluate failure detection with metrics appropriate for class imbalance rather than relying on accuracy alone.

---

## 45. Why use stratification?

Because failures are rare, an unlucky random split could produce validation or test sets with too few failures for meaningful evaluation.

Stratification preserves approximately the same Pass/Fail proportions across splits and folds.

Observed holdout failure rates were:

```text
Original ≈ 6.64%
Training ≈ 6.62%
Test     ≈ 6.69%
```

---

## 46. Why use an >80% missingness filter?

Features with more than 80% missingness have observed values for fewer than 20% of samples and would rely heavily on imputation.

The threshold is deliberately conservative so that less-sparse anonymous features are not aggressively removed before their predictive value is evaluated.

It is an MVP analysis policy, not a universal rule.

---

## 47. Why use median imputation?

After training-only high-missing and constant-feature filtering, 466 retained features showed substantial skewness:

```text
Median skewness ≈ 2.04
75th percentile ≈ 9.45
```

Median imputation was therefore selected as a simple robust method because it is less sensitive than mean imputation to skewed distributions and extreme values.

---

## 48. Why use StandardScaler?

The anonymous measurement features may exist on very different numerical scales.

The planned regularized Logistic Regression baseline is sensitive to feature scale through optimization and regularization.

StandardScaler therefore standardizes each feature using training-derived statistics.

---

## 49. How preprocessing leakage was prevented

The most important interview answer from this milestone is:

> Missing-rate filtering, constant-feature filtering, median imputation, and standard scaling are all inside a scikit-learn preprocessing Pipeline. Each step is fitted only on the training data available in that cross-validation fold, and the fitted transformations are then applied to the corresponding validation data without refitting.

This means validation/test observations do not determine:

- which features are removed for missingness,
- which features are removed for zero usable variation,
- imputation medians,
- scaling means,
- scaling standard deviations.

---

# Limitations

## 50. What Milestone 2 does not claim

Milestone 2 does not claim that:

- the >80% threshold is universally optimal,
- median imputation is universally optimal,
- SECOM missingness is MCAR, MAR, or MNAR,
- anonymous features correspond to known physical sensors or process parameters,
- preprocessing identifies physical root causes,
- model performance has been established.

No predictive model performance is reported in Milestone 2.

The purpose of this milestone is to establish a defensible, reproducible, leakage-safe preprocessing foundation before modeling.

---

# Next Milestone

## 51. Milestone 3 — Baseline & Evaluation

The next stage begins predictive modeling and evaluation.

Planned components include:

- `DummyClassifier`
- class-weighted Logistic Regression
- 5-fold stratified cross-validation
- confusion matrix
- failure recall
- precision
- F1 score
- balanced accuracy
- PR-AUC
- ROC-AUC comparison and interpretation
- explicit comparison against trivial baselines

The final holdout test set remains reserved for final unseen evaluation rather than routine model-development decisions.

No model-performance values should be documented until they are produced by actual execution.
