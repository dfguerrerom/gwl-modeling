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
        display(sns.heatmap(stats_df[["r_local"] + temporal_expl], annot=True))

    if type_ == "rmse_local":
        display(
            sns.heatmap(
                stats_df[["rmse_local"] + temporal_expl].sort_values(by="rmse_local"),
                annot=True,
                cmap="YlGnBu",
            )
        )
