
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os


def multi_to_single(columns: pd.MultiIndex) -> pd.Index:
    """
    Creates a single index from a two-level multiindex by combining the levels to a string of the form 'level0-level1'.
    """
    columns_new = []
    for col in list(columns):
        columns_new.append(str(col[0]) + "-" + str(col[1]))
    return columns_new


def rmse(v1, v2):
    sqrt = np.sqrt(np.mean((v1 - v2) ** 2))
    return sqrt


def performances_forecasts(country: str, nruns: int = 100):
    """
    RMSE of the forecasts
    Args:
        country:
        nruns:
    Returns:
    """
    files = []
    path = 'old_predictions/RC'
    for file in os.listdir(path):
        if file.split('_')[0] == country:
            if file.split('_')[-1].split('.')[0]==str(nruns):
                files.append(file)
    rmse_list = []
    for n in np.arange(0, len(files)):
        df = pd.read_csv(path+'/'+files[n])
        rlist = []
        for n in np.arange(2, df.shape[1] - 2, 2):
            rlist.append(rmse(df.iloc[:, n], df.iloc[:, n - 1]))
        rmse_list.append(np.median(rlist))
    return rmse_list


def find_crises(fcs: pd.DataFrame, t: float):
    """
    Finds in the rtm data all the dates for which at least one admin1 is showing a deterioration larger than
    the threshold t.
    Args:
        fcs: dataframe with the real time monitoring data
        t: threshold value
    Returns:
    """
    food_crises = pd.DataFrame()
    date_list = []
    admin_list = []
    for end in fcs.index[::-1]:
        start = end - timedelta(days=60)
        if start > fcs.index[0]:
            diff = fcs.loc[end] - fcs.loc[start]
            if (diff > t).any():
                date_list.append(start)
                admin_list.append((diff > t).to_dict())
    food_crises['dates'] = date_list
    food_crises['adm1_codes'] = admin_list
    return food_crises


def find_hyperparameters(country: str, date: datetime, model='RC'):
    """
    Select the hyperparameters based on the monthly updates. The function is needed when the
    forecasts are not generated at the beginning of the month, and selects the most recent hyperparameter
    update
    Args:
        country: name of the country
        date: first date of the forecasts
        model: name of the model
    Returns:
    """

    path_to_file = f'grid_search_data/best_hyperparameters/HP_{model}_{country}.csv'
    df = pd.read_csv(path_to_file)
    df['split_date'] = pd.to_datetime(df['split_date'])
    df = df.sort_values(by='split_date')
    if date < datetime(2023, 5,1):
        for n in range(len(df)-1):
            if df.loc[n, 'split_date'] <= date <= df.loc[n + 1, 'split_date']:
                row = df.loc[n]
                break
    else:
        row = df.iloc[-1, :]

    hyperparameters = row[["w_in_scale",
                           "differencing",
                           "reg_param",
                           "n_rad",
                           "n_dim",
                           "features"]].to_dict()
    hyperparameters['w_out_fit_flag'] = 'linear_and_square_r'
    hyperparameters['train_sync_steps'] = 50
    hyperparameters['n_avg_deg'] = 8
    hyperparameters['smoothing'] = 10

    return hyperparameters

