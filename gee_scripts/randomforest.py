import pylab
from typing import Literal, Union
from tqdm import tqdm

from IPython.display import display
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .parameters import explain_vars, temporal_expl
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (19, 19)


def get_regressor():
    """Get a random forest regressor."""

    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=1,
        random_state=42,
        oob_score=True,
        criterion="friedman_mse",
        n_jobs=-1,
    )


def run_randomforest(
    training_df,
    variable="gwl_cm",
    type_="allbutone",
):
    """Run a random forest model on the data."""

    print(f"total points: {len(training_df)}")
    print(f"total stations: {len(training_df.id.unique())}")
    print("Starting random forest model...")

    row = {}

    # All but one PHU for training
    for i, station_id in tqdm(enumerate(training_df.id.unique())):
        explans = []

        # define training subset
        train_df = training_df[training_df.id != station_id]

        # define test subset
        test_df = training_df[training_df.id == station_id]

        X_train, X_test = train_df[explain_vars], test_df[explain_vars]
        y_train, y_test = train_df[variable], test_df[variable]

        regr = get_regressor()

        regr.fit(X_train, y_train)
        y_pred_test = regr.predict(X_test)

        r, p = pearsonr(y_test, y_pred_test)
        explans.append(r)

        explans.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        # add correlation of explanatories
        for expl in temporal_expl:
            explans.append(test_df[variable].corr(test_df[expl]))

        row[station_id] = explans

    return pd.DataFrame.from_dict(row, orient="index")


def get_heatmap(stats_df, type_=Literal["r_local", "rmse_local"]):
    """Get a heatmap of the correlation of the explanatories."""
    stats_df.columns = ["r_local", "rmse_local"] + temporal_expl
    stats_df.loc["mean"] = stats_df.mean()

    if type_ == "r_local":
        # Set figure size
        plt.rcParams["figure.figsize"] = (19, len(stats_df) / 2)
        cmap = sns.color_palette("Spectral", len(stats_df))
        # invert the color palette
        display(
            sns.heatmap(stats_df[["r_local"] + temporal_expl], annot=True, cmap=cmap)
        )

    if type_ == "rmse_local":
        plt.rcParams["figure.figsize"] = (1, len(stats_df) / 2)
        # Change the font size
        plt.rcParams.update({"font.size": 10})
        # change the y-axis font size
        plt.yticks(fontsize=8)
        stats_df = stats_df.sort_values(by="rmse_local")
        stats_df = stats_df.drop("mean", axis=0)
        stats_df.loc["mean"] = stats_df.mean()

        # set figure title
        plt.title("RMSE of stations")

        cmap = sns.color_palette("Spectral", len(stats_df))
        # invert the color palette
        cmap = cmap[::-1]

        display(
            sns.heatmap(
                stats_df[["rmse_local"]],
                annot=True,
                cmap=cmap,
            )
        )


def bootstrap(
    df,
    variable="gwl_cm",
    iterations=100,
    train_size=0.8,
):
    """Run a random forest model on the data."""

    train_size = train_size
    bootstrap_stations = df.id.unique()
    size = int(train_size * len(bootstrap_stations))

    r_list, r2_list, rmse_list, samples_train, samples_test = [], [], [], [], []

    i = 0
    while i < iterations:
        train_list = np.random.choice(bootstrap_stations, size=size, replace=False)

        gdf_train = df[df.id.isin(train_list)].copy()
        gdf_test = df[~df.id.isin(gdf_train.id.unique())].copy()

        X_train, X_test = gdf_train[explain_vars], gdf_test[explain_vars]
        y_train, y_test = gdf_train[variable], gdf_test[variable]

        regr = get_regressor()
        regr.fit(X_train, y_train)
        y_pred_test = regr.predict(X_test)

        samples_train.append(len(gdf_train))
        samples_test.append(len(gdf_test))
        r, p = pearsonr(y_test, y_pred_test)
        r_list.append(r)
        r2_list.append(r2_score(y_test, y_pred_test))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        i += 1

    return pd.DataFrame(
        {
            "r": get_stats(r_list),
            "r2": get_stats(r2_list),
            "rmse": get_stats(rmse_list),
            "samples_train": get_stats(samples_train, is_sample=True),
            "samples_test": get_stats(samples_test, is_sample=True),
        }
    ).T


def get_stats(lst, is_sample=False):
    arr = np.array(lst)
    stats = {
        "mean": arr.mean(),
        "min": arr.min(),
        "max": arr.max(),
        "median": np.median(arr),
    }
    if is_sample:
        del stats["median"]
    return stats
