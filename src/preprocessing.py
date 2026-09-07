from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HighMissingFeatureFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.80):
        self.threshold = threshold

    def fit(self, X, y=None):
        missing_rates = X.isna().mean()

        self.features_to_keep_ = missing_rates[
            missing_rates <= self.threshold
        ].index

        return self

    def transform(self, X):
        return X.loc[:, self.features_to_keep_]


class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        unique_counts = X.nunique(dropna=True)

        self.features_to_keep_ = unique_counts[
            unique_counts > 1
        ].index

        return self

    def transform(self, X):
        return X.loc[:, self.features_to_keep_]


def build_preprocessing_pipeline():
    return Pipeline(
        steps=[
            (
                "high_missing_filter",
                HighMissingFeatureFilter(threshold=0.80),
            ),
            (
                "constant_filter",
                ConstantFeatureFilter(),
            ),
            (
                "median_imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )