"""Create descriptive tables and figures for extracted dog characteristics."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd

CHARACTERISTICS = (
    "breed_mentioned",
    "kc_breed_group",
    "size_category",
    "life_stage",
)
REQUIRED_COLUMNS = {
    "stage",
    "gender",
    "age_group",
    "tenure",
    "platform",
    *CHARACTERISTICS,
}
GROUP_ORDERS = {
    "Gender (Stage 1)": [
        "female",
        "male",
        "non-binary",
        "term not listed",
        "prefer not to say",
    ],
    "Age group (Stage 1)": [
        "24 and under",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
        "prefer not to say",
    ],
    "Tenure (Stage 2)": [
        "new prospect",
        "infrequent low-value contributor",
        "seasonal (Christmas-only) supporter",
        "long-term high-value supporter",
    ],
    "Platform (complete corpus)": ["gpt", "gemini", "claude"],
}
BREED_GROUP_ORDER = [
    "Toy",
    "Terrier",
    "Gundog",
    "Hound",
    "Pastoral",
    "Utility",
    "Working",
    "unspecified/mixed",
    "not stated",
]
BREED_GROUP_COLOURS = {
    "Toy": "#E69F00",
    "Terrier": "#D55E00",
    "Gundog": "#56B4E9",
    "Hound": "#F0E442",
    "Pastoral": "#009E73",
    "Utility": "#0072B2",
    "Working": "#000000",
    "unspecified/mixed": "#CC79A7",
    "not stated": "#777777",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Summarise and plot extracted dog characteristics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "judged_outputs_final.csv",
        help="CSV containing generated outputs and extracted dog fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dog_characteristics_output",
        help="Directory for CSV summaries and figures.",
    )
    return parser.parse_args()


def validate_columns(data: pd.DataFrame) -> None:
    """Check that the required columns are present.

    Args:
        data: Input data to summarise.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def clean_category(series: pd.Series) -> pd.Series:
    """Trim labels and represent missing values consistently.

    Args:
        series: Values to clean.

    Returns:
        Cleaned category labels with missing values marked as not stated.
    """
    values = series.astype("string").str.strip()
    return values.mask(values.isna() | values.eq(""), "not stated")


def overall_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise dog characteristics across the complete dataset.

    Args:
        data: Input data to summarise.

    Returns:
        Counts and percentages for each dog characteristic.
    """
    summaries: list[pd.DataFrame] = []
    for characteristic in CHARACTERISTICS:
        counts = (
            clean_category(data[characteristic])
            .value_counts(dropna=False)
            .rename_axis("category")
            .reset_index(name="n")
        )
        counts["percentage"] = 100 * counts["n"] / counts["n"].sum()
        counts.insert(0, "characteristic", characteristic)
        summaries.append(counts)
    return pd.concat(summaries, ignore_index=True)


def composition(
    data: pd.DataFrame,
    grouping_variable: str,
    comparison: str,
) -> pd.DataFrame:
    """Calculate category shares within each comparison group.

    Args:
        data: Input data to summarise.
        grouping_variable: Column used to define comparison groups.
        comparison: Display label for the comparison.

    Returns:
        Breed-group counts and shares by comparison group.
    """
    prepared = pd.DataFrame(
        {
            "group": clean_category(data[grouping_variable]),
            "breed_group": clean_category(data["kc_breed_group"]),
        }
    )
    counts = prepared.groupby(["group", "breed_group"], observed=True).size().rename("n").reset_index()
    counts["group_total"] = counts.groupby("group", observed=True)["n"].transform("sum")
    counts["percentage"] = 100 * counts["n"] / counts["group_total"]

    baseline = prepared["breed_group"].value_counts(normalize=True).mul(100).rename("overall_percentage")
    counts = counts.join(baseline, on="breed_group")
    counts["percentage_point_difference"] = counts["percentage"] - counts["overall_percentage"]
    counts["comparison"] = comparison
    return counts


def build_compositions(data: pd.DataFrame) -> pd.DataFrame:
    """Build breed-group compositions for the planned comparisons.

    Args:
        data: Input data to summarise.

    Returns:
        Combined breed-group compositions for all comparisons.
    """
    stage1 = data.loc[data["stage"].eq("stage1")]
    stage2 = data.loc[data["stage"].eq("stage2")]
    return pd.concat(
        [
            composition(stage1, "gender", "Gender (Stage 1)"),
            composition(stage1, "age_group", "Age group (Stage 1)"),
            composition(stage2, "tenure", "Tenure (Stage 2)"),
            composition(data, "platform", "Platform (complete corpus)"),
        ],
        ignore_index=True,
    )


def characteristic_composition(
    data: pd.DataFrame,
    grouping_variable: str,
    comparison: str,
) -> pd.DataFrame:
    """Summarise every extracted dog characteristic within a comparison group.

    Args:
        data: Input data to summarise.
        grouping_variable: Column used to define comparison groups.
        comparison: Display label for the comparison.

    Returns:
        Characteristic counts and shares by comparison group.
    """
    summaries: list[pd.DataFrame] = []
    groups = clean_category(data[grouping_variable])
    for characteristic in CHARACTERISTICS:
        prepared = pd.DataFrame(
            {
                "group": groups,
                "category": clean_category(data[characteristic]),
            }
        )
        counts = prepared.groupby(["group", "category"], observed=True).size().rename("n").reset_index()
        counts["group_total"] = counts.groupby("group", observed=True)["n"].transform("sum")
        counts["percentage"] = 100 * counts["n"] / counts["group_total"]
        baseline = prepared["category"].value_counts(normalize=True).mul(100).rename("overall_percentage")
        counts = counts.join(baseline, on="category")
        counts["percentage_point_difference"] = counts["percentage"] - counts["overall_percentage"]
        counts.insert(0, "characteristic", characteristic)
        counts.insert(0, "comparison", comparison)
        summaries.append(counts)
    return pd.concat(summaries, ignore_index=True)


def build_characteristic_compositions(data: pd.DataFrame) -> pd.DataFrame:
    """Compare breed, size, and life-stage selections across the study factors.

    Args:
        data: Input data to summarise.

    Returns:
        Combined compositions for all dog characteristics.
    """
    stage1 = data.loc[data["stage"].eq("stage1")]
    stage2 = data.loc[data["stage"].eq("stage2")]
    return pd.concat(
        [
            characteristic_composition(stage1, "gender", "Gender (Stage 1)"),
            characteristic_composition(stage1, "age_group", "Age group (Stage 1)"),
            characteristic_composition(stage2, "tenure", "Tenure (Stage 2)"),
            characteristic_composition(data, "platform", "Platform (complete corpus)"),
        ],
        ignore_index=True,
    )


def ordered_categories(values: pd.Series) -> list[str]:
    """Return categories in the preferred display order.

    Args:
        values: Category values in their observed order.

    Returns:
        Unique category labels in display order.
    """
    return list(dict.fromkeys(values.astype(str)))


def order_table(table: pd.DataFrame, comparison: str) -> pd.DataFrame:
    """Order a composition table for plotting.

    Args:
        table: Composition table to reorder.
        comparison: Display label for the comparison.

    Returns:
        The table reordered for plotting.
    """
    group_order = [group for group in GROUP_ORDERS[comparison] if group in table.index]
    breed_order = [breed_group for breed_group in BREED_GROUP_ORDER if breed_group in table.columns]
    return table.reindex(index=group_order, columns=breed_order, fill_value=0)


def plot_stacked_bars(
    compositions: pd.DataFrame,
    comparison_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Plot category shares as stacked bars.

    Args:
        compositions: Category compositions to plot.
        comparison_names: Comparisons to include.
        title: Figure title.
        output_path: Destination for the figure.
    """
    observed_groups = set(compositions["breed_group"])
    breed_groups = [breed_group for breed_group in BREED_GROUP_ORDER if breed_group in observed_groups]
    colour_map = {group: BREED_GROUP_COLOURS[group] for group in breed_groups}

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharey=True, constrained_layout=True)
    for axis, comparison_name in zip(axes, comparison_names, strict=True):
        subset = compositions.loc[compositions["comparison"].eq(comparison_name)]
        table = subset.pivot(index="group", columns="breed_group", values="percentage").fillna(0)
        table = order_table(table, comparison_name)
        table.plot(
            kind="bar",
            stacked=True,
            ax=axis,
            color=[colour_map[column] for column in table.columns],
            width=0.8,
            legend=False,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set_title(comparison_name, fontsize=13)
        axis.set_xlabel("")
        axis.set_ylabel("Percentage of generated letters", fontsize=11)
        axis.set_ylim(0, 100)
        axis.set_xticklabels(
            [fill(label.get_text(), width=22) for label in axis.get_xticklabels()],
            rotation=0,
            fontsize=10,
        )
        axis.tick_params(axis="y", labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour_map[group]) for group in breed_groups]
    fig.legend(
        handles,
        breed_groups,
        title="Extracted breed group",
        loc="outside lower center",
        ncol=min(4, len(breed_groups)),
        fontsize=10,
        title_fontsize=11,
    )
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_difference_heatmap(
    compositions: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot deviations from overall breed-group shares.

    Args:
        compositions: Category compositions to plot.
        output_path: Destination for the figure.
    """
    comparisons = ordered_categories(compositions["comparison"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    limit = max(1.0, compositions["percentage_point_difference"].abs().max())

    image = None
    for axis, comparison_name in zip(axes.flat, comparisons, strict=True):
        subset = compositions.loc[compositions["comparison"].eq(comparison_name)]
        table = subset.pivot(
            index="group",
            columns="breed_group",
            values="percentage_point_difference",
        ).fillna(0)
        table = order_table(table, comparison_name)
        numeric_values = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        image = axis.imshow(
            numeric_values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(comparison_name)
        axis.set_xticks(
            range(len(table.columns)),
            table.columns,
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(table.index)), table.index)
        axis.set_xlabel("Extracted breed group")
        axis.set_ylabel("")

    if image is not None:
        colour_bar = fig.colorbar(image, ax=axes, shrink=0.85)
        colour_bar.set_label("Difference from comparison-wide share (percentage points)")
    fig.suptitle(
        "Breed-group shares relative to each comparison-wide distribution",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate the dog-characteristic summaries and figures."""
    args = parse_args()
    data = pd.read_csv(args.input)
    validate_columns(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = overall_summary(data)
    compositions = build_compositions(data)
    characteristic_compositions = build_characteristic_compositions(data)
    summary.to_csv(
        args.output_dir / "dog_characteristic_summary.csv",
        index=False,
    )
    compositions.to_csv(
        args.output_dir / "breed_group_composition.csv",
        index=False,
    )
    characteristic_compositions.to_csv(
        args.output_dir / "dog_characteristic_composition.csv",
        index=False,
    )
    plot_stacked_bars(
        compositions,
        ["Gender (Stage 1)", "Age group (Stage 1)"],
        "Dog breed-group distributions by demographic group",
        args.output_dir / "breed_group_demographic_distributions.png",
    )
    plot_stacked_bars(
        compositions,
        ["Tenure (Stage 2)", "Platform (complete corpus)"],
        "Dog breed-group distributions by tenure and platform",
        args.output_dir / "breed_group_tenure_platform_distributions.png",
    )
    plot_difference_heatmap(
        compositions,
        args.output_dir / "breed_group_difference_heatmap.png",
    )

    print(f"Saved descriptive outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
