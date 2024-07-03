from typing import Literal
from matplotlib.dates import MonthLocator
import seaborn as sns
import matplotlib.pyplot as plt
from .parameters import explain_vars


def get_ts_plot(df, y_axis="gwl_cm", group_by="region_id", group_name="region_id"):
    """Plot a time series of the data."""

    print(f"Plotting time series for {group_by}...")

    # Get the unique region IDs
    group_ids = sorted(df[group_by].unique())

    # Set the figure size
    fig, axs = plt.subplots(len(group_ids), 1, figsize=(9, 4 * len(group_ids)))

    # Iterate over the region IDs and create a separate plot for each region
    for i, group_id in enumerate(group_ids):
        ax = axs[i]

        data = df[df[group_by] == group_id]
        if not len(data):
            continue

        # Sort the data by date
        data = data.sort_values(by="date")

        # create a title for each plot

        sns.lineplot(x="date", y=y_axis, data=data, ax=ax)

        # get the name of the region
        region_name = data[group_name].iloc[0]
        extra_label = data["phu_id"].iloc[0] if group_name == "phu_name" else ""

        # Add number of points to the title
        ax.set_title(f"{region_name}, {extra_label}, (n={len(data)} points)")
        ax.set_xlabel("Date")
        ax.set_ylabel("GWL (cm)")

        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

        # Use MonthLocator for sparse labeling
        ax.xaxis.set_major_locator(MonthLocator())

    # Adjust the spacing between the subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


def get_precipitation_plot(
    df, group_by=Literal["station", "phu_id", "region_id"], value: str = None
):
    """Return a multi-line plot of the precipitation data, grouped by station, region, or PHU.

    Args:
        value (str): The value to filter the data by in the group_by column.

    """

    if group_by == "station":
        group_by = "id"

    if value:
        df = df[df[group_by] == value]

    cols = ["gwl_cm", group_by, "date"] + explain_vars

    # Group the DataFrame by the specified group_by column and date
    df = df[cols].groupby([group_by, "date"]).mean().reset_index().copy()
    df = df.sort_values(by="date")

    sns.set_theme(style="darkgrid")
    # Get the group names
    group_ids = sorted(df[group_by].unique())

    if len(group_ids) > 1:
        fig, axs = plt.subplots(len(group_ids), 1, figsize=(9, 4 * len(group_ids)))

    else:
        fig, axs = plt.subplots(1, 1, figsize=(13, 4))
        axs = [axs]

    # Iterate over the region IDs and create a separate plot for each region
    for i, group_id in enumerate(group_ids):
        ax = axs[i]

        data = df[df[group_by] == group_id]
        print("len(df)", len(data))
        if not len(data):
            continue

        # Sort the data by date
        data = data.sort_values(by="date")

        # create a title for each plot
        ax1 = sns.lineplot(
            x="date", y="gwl_cm", data=data, ax=ax, label="GWL (cm)", color="red"
        )
        ax1.set_ylabel("GWL (cm)", color="tab:blue")

        # Create the second y-axis for "precipitation" data
        ax2 = ax1.twinx()
        # Set the second y-axis label
        ax2.set_ylabel("Precipitation (mm)", color="tab:red")

        # Plot the data and add the handles and labels to the lists
        for y, color, label in zip(
            [
                "precipitation",
                "prec_3",
                "prec_7",
                "prec_30",
                "prec_3_sum",
                "prec_7_sum",
                "prec_30_sum",
            ],
            ["black", "orange", "blue", "purple", "orange", "blue", "purple"],
            [
                "Precipitation (mm)",
                "prec_3 (mm)",
                "prec_7 (mm)",
                "prec_30 (mm)",
                "prec_3_sum (mm)",
                "prec_7_sum (mm)",
                "prec_30_sum (mm)",
            ],
        ):
            sns.lineplot(
                x="date", y=y, data=data, color=color, label=label, legend=False
            )
        # get the name of the region
        region_name = data[group_by].iloc[0]

        # Add number of points to the title
        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

        # Use MonthLocator for sparse labeling
        # ax.xaxis.set_major_locator(MonthLocator())

        # Set the title and x-axis label
        plt.title(
            f"GWL and Precipitation Over Time. "
            + f"{group_by}:{region_name}, (n={len(data)} points)"
        )
        plt.xlabel("Date")
        # ax.set_ylabel("GWL (cm)")

        # Add legends for both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

    # Adjust the spacing between the subplots
    plt.tight_layout()
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
    plt.show()


def plot_observed_vs_predicted(y_test, y_pred_test, color=1):
    """Use seaborn to plot the observed vs predicted values"""

    # generate two random colors for the plot based on the color parameter

    colors = ["#297496", "#0C9B53", "#23A4D2"]

    color = colors[color]

    # Customizing with professional aesthetics
    sns.set_theme(style="whitegrid")

    # Plot the residuals after fitting a linear model
    plt.figure(figsize=(4, 3))
    sns.residplot(x=y_test, y=y_pred_test, color=color)
    plt.xlabel("Observed values")
    plt.ylabel("Residuals")
    plt.title("Residuals plot")
    plt.show()

    # Plot the observed vs predicted values
    plt.figure(figsize=(4, 3))
    sns.scatterplot(x=y_test, y=y_pred_test, color=color)
    # Add a line for perfect correlation
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="r",
        linestyle="--",
    )
    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Observed vs predicted values")
    plt.show()


def plot_ts(df, variable: str = "gwl_cm", title: str = "Groundwater level vs date"):

    assert df.date.dtype == "datetime64[ns]", "Date column must be in datetime format"

    # Sort the data by date
    df = df.sort_values(by="date")

    # Group the

    plt.figure(figsize=(15, 5))
    sns.lineplot(data=df, x="date", y=variable, hue="id")
    plt.title(f"Groundwater level vs date {title}")

    # remove the legend
    plt.legend().remove()

    plt.show()
