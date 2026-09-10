from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessing_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REPORTS_DIR = PROJECT_ROOT / "reports"
CANDIDATE_OUTPUT_PATH = (
    REPORTS_DIR / "candidate_signals.csv"
)

RANDOM_STATE = 42
N_SPLITS = 5
N_PERMUTATION_REPEATS = 10


def load_data():
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

    return X, y


def build_modeling_pipeline():
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessing_pipeline(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )


def calculate_coefficient_stability(X_train, y_train, cv):
    records = []

    for fold_number, (train_index, _) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]

        pipeline = build_modeling_pipeline()
        pipeline.fit(X_fold_train, y_fold_train)

        preprocessing = pipeline.named_steps["preprocessing"]

        constant_filter = preprocessing.named_steps[
            "constant_filter"
        ]

        feature_indices = constant_filter.features_to_keep_

        coefficients = pipeline.named_steps[
            "classifier"
        ].coef_[0]

        for feature_index, coefficient in zip(
            feature_indices,
            coefficients,
        ):
            records.append(
                {
                    "fold": fold_number,
                    "feature_index": feature_index,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                }
            )

    coefficient_df = pd.DataFrame(records)

    stability = (
        coefficient_df
        .groupby("feature_index")
        .agg(
            folds_present=("fold", "count"),
            mean_coefficient=("coefficient", "mean"),
            mean_absolute_coefficient=(
                "absolute_coefficient",
                "mean",
            ),
            positive_folds=(
                "coefficient",
                lambda values: (values > 0).sum(),
            ),
            negative_folds=(
                "coefficient",
                lambda values: (values < 0).sum(),
            ),
        )
        .reset_index()
    )

    stability["direction_consistency"] = (
        stability[
            ["positive_folds", "negative_folds"]
        ].max(axis=1)
        / stability["folds_present"]
    )

    return stability


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values)

    number_of_tests = len(p_values)

    sorted_order = np.argsort(p_values)
    sorted_p_values = p_values[sorted_order]

    ranks = np.arange(
        1,
        number_of_tests + 1,
    )

    adjusted_sorted = (
        sorted_p_values
        * number_of_tests
        / ranks
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(
        adjusted_sorted,
        0,
        1,
    )

    adjusted_p_values = np.empty(
        number_of_tests
    )

    adjusted_p_values[sorted_order] = (
        adjusted_sorted
    )

    return adjusted_p_values


def calculate_univariate_association(
    X_train,
    y_train,
    feature_indices,
):
    records = []

    for feature_index in feature_indices:
        pass_values = X_train.loc[
            y_train == -1,
            feature_index,
        ].dropna()

        fail_values = X_train.loc[
            y_train == 1,
            feature_index,
        ].dropna()

        if len(pass_values) == 0 or len(fail_values) == 0:
            continue

        u_statistic, p_value = mannwhitneyu(
            fail_values,
            pass_values,
            alternative="two-sided",
        )

        n_fail = len(fail_values)
        n_pass = len(pass_values)

        rank_biserial = (
            2 * u_statistic
            / (n_fail * n_pass)
            - 1
        )

        records.append(
            {
                "feature_index": feature_index,
                "pass_n": n_pass,
                "fail_n": n_fail,
                "pass_median": pass_values.median(),
                "fail_median": fail_values.median(),
                "p_value": p_value,
                "rank_biserial": rank_biserial,
                "absolute_effect": abs(rank_biserial),
            }
        )

    univariate = pd.DataFrame(records)

    univariate["fdr_q_value"] = (
        benjamini_hochberg(
            univariate["p_value"].to_numpy()
        )
    )

    return univariate


def calculate_permutation_stability(
    X_train,
    y_train,
    cv,
):
    records = []

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        X_fold_train = X_train.iloc[train_index]
        y_fold_train = y_train.iloc[train_index]

        X_fold_validation = X_train.iloc[
            validation_index
        ]
        y_fold_validation = y_train.iloc[
            validation_index
        ]

        pipeline = build_modeling_pipeline()

        pipeline.fit(
            X_fold_train,
            y_fold_train,
        )

        result = permutation_importance(
            pipeline,
            X_fold_validation,
            y_fold_validation,
            scoring="average_precision",
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        for feature_index, importance in zip(
            X_fold_validation.columns,
            result.importances_mean,
        ):
            records.append(
                {
                    "fold": fold_number,
                    "feature_index": feature_index,
                    "importance": importance,
                }
            )

    permutation_df = pd.DataFrame(records)

    stability = (
        permutation_df
        .groupby("feature_index")
        .agg(
            mean_permutation_importance=(
                "importance",
                "mean",
            ),
            std_permutation_importance=(
                "importance",
                "std",
            ),
            positive_importance_folds=(
                "importance",
                lambda values: (values > 0).sum(),
            ),
        )
        .reset_index()
    )

    return stability


def build_candidate_table(
    coefficient_stability,
    univariate,
    permutation_stability,
):
    evidence = coefficient_stability.merge(
        univariate,
        on="feature_index",
        how="left",
    )

    evidence = evidence.merge(
        permutation_stability,
        on="feature_index",
        how="left",
    )

    evidence["direction_consistent_5fold"] = (
        (evidence["folds_present"] == N_SPLITS)
        & (evidence["direction_consistency"] == 1.0)
    )

    evidence["fdr_significant"] = (
        evidence["fdr_q_value"] < 0.05
    )

    evidence["permutation_positive_4plus"] = (
        evidence["positive_importance_folds"] >= 4
    )

    evidence["evidence_count"] = (
        evidence[
            [
                "direction_consistent_5fold",
                "fdr_significant",
                "permutation_positive_4plus",
            ]
        ].sum(axis=1)
    )

    candidates = evidence[
        evidence["evidence_count"] == 3
    ].copy()

    candidates = candidates.sort_values(
        [
            "mean_absolute_coefficient",
            "absolute_effect",
            "mean_permutation_importance",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return candidates


def main():
    X, y = load_data()

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    coefficient_stability = (
        calculate_coefficient_stability(
            X_train,
            y_train,
            cv,
        )
    )

    full_training_pipeline = build_modeling_pipeline()

    full_training_pipeline.fit(
        X_train,
        y_train,
    )

    preprocessing = full_training_pipeline.named_steps[
        "preprocessing"
    ]

    retained_feature_indices = (
        preprocessing.named_steps[
            "constant_filter"
        ].features_to_keep_
    )

    univariate = calculate_univariate_association(
        X_train,
        y_train,
        retained_feature_indices,
    )

    permutation_stability = (
        calculate_permutation_stability(
            X_train,
            y_train,
            cv,
        )
    )

    candidates = build_candidate_table(
        coefficient_stability,
        univariate,
        permutation_stability,
    )

    missing_rates = X_train.isna().mean()

    candidates["training_missing_rate"] = (
        candidates["feature_index"].map(
            missing_rates
        )
    )
    output_columns = [
        "feature_index",
        "training_missing_rate",
        "pass_n",
        "fail_n",
        "pass_median",
        "fail_median",
        "mean_coefficient",
        "positive_folds",
        "negative_folds",
        "rank_biserial",
        "fdr_q_value",
        "mean_permutation_importance",
        "std_permutation_importance",
        "positive_importance_folds",
    ]

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates[
        output_columns
    ].to_csv(
        CANDIDATE_OUTPUT_PATH,
        index=False,
    )
    print(
        "\n===== Candidate Signals =====\n"
    )

    print(
        candidates[
            output_columns
        ].to_string(index=False)
    )

    print(
        "\nCandidate count:",
        len(candidates),
    )

    print(
        "Candidate table saved to:",
        CANDIDATE_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()