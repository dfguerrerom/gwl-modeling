from matplotlib.dates import MonthLocator
import seaborn as sns
import matplotlib.pyplot as plt


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

        sns.lineplot(x="date", y="gwl_cm", data=data, ax=ax)

        # get the name of the region
        region_name = data[group_name].iloc[0]

        # Add number of points to the title
        ax.set_title(f"{region_name}, (n={len(data)} points)")
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
