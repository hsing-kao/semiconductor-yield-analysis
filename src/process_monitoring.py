from pathlib import Path


import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

CANDIDATE_PATH = (
    REPORTS_DIR / "candidate_signals.csv"
)


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

    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"],
        format="%d/%m/%Y %H:%M:%S",
    )

    assert len(X) == len(labels), (
        "Measurement and label row counts do not match."
    )

    return X, labels


def load_candidates():
    candidates = pd.read_csv(
        CANDIDATE_PATH
    )

    candidate_indices = (
        candidates["feature_index"]
        .astype(int)
        .tolist()
    )

    return candidates, candidate_indices

def build_process_view(
    X,
    labels,
    candidate_indices,
):
    process_view = X[
        candidate_indices
    ].copy()

    process_view.insert(
        0,
        "label",
        labels["label"].to_numpy(),
    )

    process_view.insert(
        0,
        "timestamp",
        labels["timestamp"].to_numpy(),
    )

    return process_view

def plot_candidate_distributions(
    process_view,
    candidate_indices,
):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
    )

    axes = axes.flatten()

    for ax, feature_index in zip(
        axes,
        candidate_indices,
    ):
        pass_values = process_view.loc[
            process_view["label"] == -1,
            feature_index,
        ].dropna()

        fail_values = process_view.loc[
            process_view["label"] == 1,
            feature_index,
        ].dropna()

        combined_values = pd.concat(
            [pass_values, fail_values]
        )

        zoom_lower = combined_values.quantile(
            0.01
        )

        zoom_upper = combined_values.quantile(
            0.99
        )

        ax.boxplot(
            [pass_values, fail_values],
            tick_labels=["Pass", "Fail"],
            showfliers=False,
        )

        ax.set_ylim(
            zoom_lower,
            zoom_upper,
        )

        ax.set_title(
            f"Feature {feature_index}"
        )

        ax.set_ylabel(
            "Raw measurement"
        )

        ax.text(
            0.02,
            0.96,
            "Display: 1st–99th percentile",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
        )

    for ax in axes[
        len(candidate_indices):
    ]:
        ax.set_visible(False)

    fig.suptitle(
        "Frozen Candidate-Signal Distributions by Outcome"
    )

    fig.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / "candidate_distributions.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

def plot_candidate_timelines(
    process_view,
    candidate_indices,
):
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 14),
        sharex=True,
    )

    for ax, feature_index in zip(
        axes,
        candidate_indices,
    ):
        plot_data = process_view[
            ["timestamp", "label", feature_index]
        ].dropna()
        zoom_lower = plot_data[
            feature_index
        ].quantile(0.01)

        zoom_upper = plot_data[
            feature_index
        ].quantile(0.99)

        pass_data = plot_data[
            plot_data["label"] == -1
        ]

        fail_data = plot_data[
            plot_data["label"] == 1
        ]

        ax.scatter(
            pass_data["timestamp"],
            pass_data[feature_index],
            s=10,
            alpha=0.30,
            label="Pass",
        )

        ax.scatter(
            fail_data["timestamp"],
            fail_data[feature_index],
            s=28,
            marker="x",
            alpha=0.90,
            label="Fail",
        )

        ax.set_ylim(
            zoom_lower,
            zoom_upper,
        )

        ax.text(
            0.01,
            0.94,
            "Display: 1st–99th percentile",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
        )

        ax.set_ylabel(
            f"Feature {feature_index}"
        )

    axes[0].legend(
        loc="upper right"
    )

    axes[-1].set_xlabel(
        "Timestamp"
    )

    fig.suptitle(
        "Frozen Candidate-Signal Measurements Over Time"
    )

    fig.autofmt_xdate()
    fig.tight_layout()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / "candidate_timelines.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path

def build_temporal_summary(
    process_view,
    candidate_indices,
):
    chronological_view = (
        process_view
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    chronological_view["time_block"] = pd.qcut(
        chronological_view.index,
        q=4,
        labels=[
            "Block 1",
            "Block 2",
            "Block 3",
            "Block 4",
        ],
    )

    records = []

    for feature_index in candidate_indices:
        for block_name, block_data in (
            chronological_view.groupby(
                "time_block",
                observed=True,
            )
        ):
            observed_values = (
                block_data[feature_index]
                .dropna()
            )

            q1 = observed_values.quantile(0.25)
            q3 = observed_values.quantile(0.75)

            records.append(
                {
                    "feature_index": feature_index,
                    "time_block": block_name,
                    "start_time": block_data[
                        "timestamp"
                    ].min(),
                    "end_time": block_data[
                        "timestamp"
                    ].max(),
                    "n": len(observed_values),
                    "median": observed_values.median(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )

    temporal_summary = pd.DataFrame(
        records
    )

    return temporal_summary

def save_temporal_summary(
    temporal_summary,
):
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORTS_DIR
        / "candidate_temporal_summary.csv"
    )

    temporal_summary.to_csv(
        output_path,
        index=False,
    )

    return output_path

def main():
    X, labels = load_data()

    candidates, candidate_indices = (
        load_candidates()
    )

    process_view = build_process_view(
        X,
        labels,
        candidate_indices,
    )

    distribution_figure_path = (
        plot_candidate_distributions(
            process_view,
            candidate_indices,
        )
    )

    timeline_figure_path = (
        plot_candidate_timelines(
            process_view,
            candidate_indices,
        )
    )

    temporal_summary = build_temporal_summary(
        process_view,
        candidate_indices,
    )

    temporal_summary_path = (
        save_temporal_summary(
            temporal_summary
        )
    )

    print(
        "Measurement shape:",
        X.shape,
    )

    print(
        "Label shape:",
        labels.shape,
    )

    print(
        "Candidate table shape:",
        candidates.shape,
    )

    print(
        "Candidate feature indices:",
        candidate_indices,
    )

    print(
        "Process-view shape:",
        process_view.shape,
    )

    print(
        "Process-view columns:",
        process_view.columns.tolist(),
    )

    print(
        "Candidate missing counts:",
    )

    print(
        process_view[
            candidate_indices
        ].isna().sum()
    )
    print(
        "Distribution figure saved to:",
        distribution_figure_path,
    )
    print(
        "Timeline figure saved to:",
        timeline_figure_path,
    )
    print(
        "Temporal summary saved to:",
        temporal_summary_path,
    )

    print(
        "\nTemporal summary:"
    )

    print(
        temporal_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()