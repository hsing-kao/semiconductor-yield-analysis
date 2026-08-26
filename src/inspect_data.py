from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SECOM_DATA_PATH = RAW_DATA_DIR / "secom.data"
SECOM_LABELS_PATH = RAW_DATA_DIR / "secom_labels.data"

X = pd.read_csv(
    SECOM_DATA_PATH,
    sep=r"\s+",
    header=None,
    na_values="NaN",
)

labels = pd.read_csv(
    SECOM_LABELS_PATH,
    sep=r"\s+",
    header=None,
    names=["label", "timestamp"],
)

labels["timestamp"] = pd.to_datetime(
    labels["timestamp"],
    format="%d/%m/%Y %H:%M:%S",
)

assert len(X) == len(labels), "Measurement and label row counts do not match."

def summarize_labels(labels):
    pass_count = (labels["label"] == -1).sum()
    fail_count = (labels["label"] == 1).sum()
    total_count = len(labels)

    yield_rate = pass_count / total_count
    failure_rate = fail_count / total_count

    return pass_count, fail_count, yield_rate, failure_rate

def summarize_missingness(X):
    total_cells = X.size
    missing_cells = X.isna().sum().sum()
    missing_rate = missing_cells / total_cells
    missing_rate_by_feature = X.isna().mean()

    return total_cells, missing_cells, missing_rate, missing_rate_by_feature

def dominant_value_frequency(column):
    counts = column.value_counts(dropna=True)

    if counts.empty:
        return float("nan")

    return counts.iloc[0] / counts.sum()

def summarize_feature_variation(X, near_constant_threshold=0.98):
    unique_counts = X.nunique(dropna=True)

    constant_features = unique_counts[unique_counts <= 1]

    dominant_frequencies = X.apply(dominant_value_frequency)

    nonconstant_dominant_frequencies = dominant_frequencies[
        unique_counts > 1
    ]

    near_constant_features = nonconstant_dominant_frequencies[
        nonconstant_dominant_frequencies >= near_constant_threshold
    ]

    return constant_features, near_constant_features

def summarize_timestamps(labels):
    timestamps = labels["timestamp"]

    earliest_timestamp = timestamps.min()
    latest_timestamp = timestamps.max()
    chronologically_ordered = timestamps.is_monotonic_increasing
    duplicate_timestamps = timestamps.duplicated().sum()

    return (
        earliest_timestamp,
        latest_timestamp,
        chronologically_ordered,
        duplicate_timestamps,
    )

constant_features, near_constant_features = (
    summarize_feature_variation(X)
)

pass_count, fail_count, yield_rate, failure_rate = summarize_labels(labels)
total_cells, missing_cells, missing_rate, missing_rate_by_feature = (
    summarize_missingness(X)
)

(
    earliest_timestamp,
    latest_timestamp,
    chronologically_ordered,
    duplicate_timestamps,
) = summarize_timestamps(labels)

print("Measurement data path:", SECOM_DATA_PATH)
print("Label data path:", SECOM_LABELS_PATH)
print("Measurement shape:", X.shape)
print("Label shape:", labels.shape)
print("Label counts:")
print(labels["label"].value_counts().sort_index())
print("Observed yield:", f"{yield_rate:.2%}")
print("Observed failure rate:", f"{failure_rate:.2%}")
print("Total measurement cells:", total_cells)
print("Missing measurement cells:", missing_cells)
print("Overall missing rate:", f"{missing_rate:.2%}")
print("\nFeatures with highest missing rates:")
print(
    missing_rate_by_feature
    .sort_values(ascending=False)
    .head(10)
)
print("\nNumber of constant features:", len(constant_features))
print("Constant feature indices:")
print(constant_features.index.tolist())

print(
    "\nNumber of near-constant candidate features:",
    len(near_constant_features),
)
print("Near-constant candidate indices:")
print(near_constant_features.index.tolist())
print("\nEarliest timestamp:", earliest_timestamp)
print("Latest timestamp:", latest_timestamp)
print("Chronologically ordered:", chronologically_ordered)
print("Duplicate timestamp occurrences:", duplicate_timestamps)