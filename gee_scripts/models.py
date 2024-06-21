from functools import partial
from multiprocessing import Pool
import pylab
from typing import Literal, Union
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from IPython.display import display
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .parameters import explain_vars, temporal_expl, biophysical_vars
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

plt.rcParams["figure.figsize"] = (19, 19)


def get_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Get the PCA of the data.

    Args:
        df: a pandas dataframe with stations and biophysical variables
        n_components: the number of components to keep

    Returns:
        a pandas dataframe with the PCA components
    """

    # The biophysical variables are static on each station
    stations_pca = df[biophysical_vars + ["lat", "lon", "id"]].drop_duplicates()

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X = stations_pca[biophysical_vars]
    X = scaler.fit_transform(X)
    X = pca.fit_transform(X)

    for i in range(n_components):
        stations_pca[f"pca_{i}"] = X[:, i]

    return stations_pca


def get_random_forest():
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


def get_gradientboosting():
    """Return a gradient boosting regressor."""

    return GradientBoostingRegressor(
        n_estimators=250,
        min_samples_leaf=1,
        random_state=42,
        criterion="friedman_mse",
    )


def get_linear_regressor():
    """Return a linear regressor."""

    return LinearRegression()


def get_neural_network_model(input_dim=len(explain_vars)):
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_dim,)),
            Dropout(0.1),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(32, activation="relu"),
            Dense(1),  # Output layer for regression; no activation function
        ]
    )
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"]
    )
    return model


def get_regressors(input_dim):
    """Get a list of regressors."""

    return [
        get_random_forest(),
        get_gradientboosting(),
        get_linear_regressor(),
        get_neural_network_model(input_dim),
    ]


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

        regr = get_random_forest()

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

        regr = get_random_forest()
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


import pandas as pd
from typing import Tuple, Literal
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_dataset(
    df: pd.DataFrame,
    by: Literal["year", "month", "station", "observation"],
    n_splits=5,
    test_size=0.2,
    min_test_samples=20,
) -> Tuple[list, list]:
    """Split the dataset into training and test sets by specified 'by' category with approximately 80/20 distribution,
    taking into account categories with limited data.

    Args:
        df (pd.DataFrame): The dataframe containing the data to be split.
        by (Literal): The category by which to split ('year', 'month', or 'station').
        n_splits (int): The number of splits (default 5).
        min_samples_per_category (int): Minimum samples per category to proceed with a split.

    Returns:
        Tuple[list, list]: A tuple containing lists of dataframes representing the training and test splits.
    """

    df = df.copy()

    # Create an unique index column for observations
    df.reset_index(inplace=True)
    df["index"] = df.index

    # Assert that the date column is in datetime format in all cases
    assert df["date"].dtypes == "datetime64[ns]"

    if by == "year":
        df["year"] = df["date"].dt.year
        split_column = "year"

    elif by == "month":
        df["month"] = df["date"].dt.month
        split_column = "month"

    elif by == "observation":
        split_column = "index"

    if by == "station":
        split_column = "id"

    print("Splitting by", by, "with", len(df), "samples")

    train_test_splits = []

    # GroupShuffleSplit instance
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    for train_idx, test_idx in gss.split(df, groups=df.get(split_column)):

        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        print("test", len(test), "train", len(train))
        if len(test) < min_test_samples:

            if len(test) < min_test_samples:
                print("Skipping split due to insufficient test samples")
                continue

        train_test_splits.append((train, test))

    return train_test_splits
