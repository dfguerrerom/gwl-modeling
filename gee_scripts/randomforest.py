from functools import partial
from multiprocessing import Pool
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

    # Best parameters from the hyperparameter tuning
    best_params = {
        "max_depth": 20,
        "min_samples_leaf": 1,
        "min_samples_split": 10,
        "n_estimators": 300,
    }

    # return RandomForestRegressor(
    #     n_estimators=best_params["n_estimators"],
    #     max_depth=best_params["max_depth"],
    #     min_samples_split=best_params["min_samples_split"],
    #     min_samples_leaf=best_params["min_samples_leaf"],
    #     random_state=42,
    #     oob_score=True,
    #     criterion="friedman_mse",
    #     n_jobs=-1,
    # )
    # print("ASDF")

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


# def run_model_for_station(station_id, training_df, variable="gwl_cm"):
#     """"""
#     print(f"Running model for station {station_id}")
#     explans = []

#     # define training subset
#     train_df = training_df[training_df.id != station_id]

#     # define test subset
#     test_df = training_df[training_df.id == station_id]

#     X_train, X_test = train_df[explain_vars], test_df[explain_vars]
#     y_train, y_test = train_df[variable], test_df[variable]

#     regr = get_regressor()

#     regr.fit(X_train, y_train)
#     y_pred_test = regr.predict(X_test)

#     r, p = pearsonr(y_test, y_pred_test)
#     explans.append(r)

#     explans.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))

#     # add correlation of explanatories
#     for expl in temporal_expl:
#         explans.append(test_df[variable].corr(test_df[expl]))

#     return station_id, explans


# def run_randomforest(
#     training_df,
#     variable="gwl_cm",
#     type_="allbutone",
# ):
#     """Run a random forest model on the data."""

#     print(f"total points: {len(training_df)}")
#     print(f"total stations: {len(training_df.id.unique())}")
#     print("Starting random forest model...")

#     row = {}

#     # All but one PHU for training
#     with Pool() as p:
#         func = partial(
#             run_model_for_station, training_df=training_df, variable=variable
#         )
#         results = p.map(func, training_df.id.unique())

#     for station_id, explans in results:
#         row[station_id] = explans

#     return pd.DataFrame.from_dict(row, orient="index")


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
    df: pd.DataFrame,
    variable="gwl_cm",
    iterations=100,
    train_size=0.8,
    explain_vars=explain_vars,
    bootstrap_by=Literal["stations", "observations"],
):
    """Run a random forest model on the data."""

    column = "id" if bootstrap_by == "stations" else "index"
    df = df.copy()
    df.reset_index(inplace=True)

    bootsrap_id = df[column].unique()
    size = int(train_size * len(bootsrap_id))

    print(f"Training with {len(df)} observations")

    r_list, r2_list, rmse_list, samples_train, samples_test = [], [], [], [], []

    i = 0
    while i < iterations:
        train_list = np.random.choice(bootsrap_id, size=size, replace=False)

        gdf_train = df[df[column].isin(train_list)].copy()
        gdf_test = df[~df[column].isin(gdf_train[column].unique())].copy()

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
