from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessing_pipeline

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------
# Load SECOM dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

X = pd.read_csv(
    RAW_DATA_DIR / "secom.data",
    sep=r"\s+",
    header=None,
    na_values="NaN",
)

labels = pd.read_csv(
    RAW_DATA_DIR / "secom_labels.data",
    sep=r"\s+",
    header=None,
    names=["label", "timestamp"],
)

y = labels["label"]

# --------------------------------------------------
# Holdout split (same as Milestone 2)
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# --------------------------------------------------
# Dummy baseline
# --------------------------------------------------

dummy_model = DummyClassifier(
    strategy="most_frequent",
    random_state=42,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(
        precision_score,
        pos_label=1,
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        pos_label=1,
    ),
    "f1": make_scorer(
        f1_score,
        pos_label=1,
    ),
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
}

dummy_scores = cross_validate(
    dummy_model,
    X_train,
    y_train,
    cv=cv,
    scoring=cv_scoring,
)

print("\n===== DummyClassifier CV Results =====\n")

for metric in dummy_scores:
    if metric.startswith("test_"):
        values = dummy_scores[metric]
        print(
            f"{metric:25s}"
            f"mean={values.mean():.4f}   "
            f"std={values.std():.4f}"
        )

# --------------------------------------------------
# Logistic Regression baseline
# --------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            build_preprocessing_pipeline(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            ),
        ),
    ]
)

print("\n===== Logistic Regression CV Results =====\n")

logistic_scores = cross_validate(
    logistic_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=cv_scoring,
)

for metric in logistic_scores:
    if metric.startswith("test_"):
        values = logistic_scores[metric]

        print(f"\n{metric}")

        for i, value in enumerate(values, start=1):
            print(f"  Fold {i}: {value:.4f}")

        print(f"  Mean : {values.mean():.4f}")
        print(f"  Std  : {values.std():.4f}")

# --------------------------------------------------
# Final holdout evaluation
# --------------------------------------------------

print("\n===== Final Holdout Test Results =====\n")

# Fit the complete leakage-safe modeling pipeline
# using only the full training set.
logistic_pipeline.fit(X_train, y_train)

# Class predictions for threshold-based metrics.
y_test_pred = logistic_pipeline.predict(X_test)

# Failure probabilities for threshold-independent metrics.
y_test_score = logistic_pipeline.predict_proba(X_test)[:, 1]

# Confusion matrix:
# rows = actual [-1 Pass, 1 Fail]
# columns = predicted [-1 Pass, 1 Fail]
cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[-1, 1],
)

accuracy = accuracy_score(y_test, y_test_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
precision = precision_score(
    y_test,
    y_test_pred,
    pos_label=1,
    zero_division=0,
)
recall = recall_score(
    y_test,
    y_test_pred,
    pos_label=1,
)
f1 = f1_score(
    y_test,
    y_test_pred,
    pos_label=1,
)
average_precision = average_precision_score(
    y_test,
    y_test_score,
    pos_label=1,
)
roc_auc = roc_auc_score(
    y_test,
    y_test_score,
)

print("Confusion matrix:")
print(cm)

print(f"\nAccuracy:           {accuracy:.4f}")
print(f"Balanced accuracy:  {balanced_accuracy:.4f}")
print(f"Failure precision:  {precision:.4f}")
print(f"Failure recall:     {recall:.4f}")
print(f"Failure F1:         {f1:.4f}")
print(f"Average precision:  {average_precision:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")