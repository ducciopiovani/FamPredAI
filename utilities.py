
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os


def shift_dataframe_by_date(dataframe, target_date):
    """
    Shift each column of the pandas DataFrame so that the last valid index is at least the specified date.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame with a datetime index and numerical columns.
        target_date (str or pd.Timestamp): The desired date as a string in 'yyyy-mm-dd' format or as a pandas Timestamp.

    Returns:
        pandas.DataFrame: The new DataFrame with shifted values.
    """
    # Convert the target_date to a pandas Timestamp object if it's provided as a string.
    if isinstance(target_date, str):
        target_date = pd.Timestamp(target_date)

    # Shift the last valid index for each column.
    shifted_dataframe = dataframe.copy()
    for col in dataframe.columns:
        last_valid_index = dataframe[col].last_valid_index()

        if isinstance(last_valid_index, str):
            last_valid_index = pd.Timestamp(last_valid_index)


        shift = (target_date - last_valid_index).days
        if (target_date - last_valid_index).days > 0:
            shifted_dataframe[col] = dataframe[col].shift(shift)

    return shifted_dataframe


def extrapolate_with_noise(df, freq="daily"):
    if freq == "daily":
        step = pd.DateOffset(days=1)
    elif freq == "monthly":
        step = pd.DateOffset(months=1)
    elif freq == "decade":
        step = pd.DateOffset(days=10)

    df_new = df.copy()
    for col in df.columns:
        df_col = df_new[col].copy()
        scale = np.std(df_col)*0.01
        #df_col = df_col.reset_index().drop('index', 1)

        lvi = df_col.last_valid_index()
        li = df_col.index[-1]

        x = float(df_col[lvi])
        # m = x-float(df_col.iloc[lvi-1])

        for i in pd.date_range(start=lvi + step, end=li, freq=step):
            df_col[i.date()] = x + scale*(np.random.random()-0.5)*2

        df_col = df_col.interpolate()
        df_new[col] = list(df_col)

    return df_new


def shuffle_io(io_data: tuple) -> tuple:
    l = io_data[0].shape[0]
    r = np.arange(l)

    new_input = io_data[0][r]
    new_output = io_data[1][r]

    new_io_data = (new_input, new_output)

    return new_io_data


def smooth_past_data(data, delta_t):
    new_data = data.copy()

    for t in range(len(data)):
        if t >= delta_t:
            new_data[t] = np.nanmean(data[t - delta_t: t+1])
        else:
            new_data[t] = np.nanmean(data[0:t+1])
    #
    return new_data


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


def performances_forecasts(path : str, country: str, nruns: int = 100):
    """
    RMSE of the forecasts
    Args:
        country:
        nruns:
    Returns:
    """
    files = []
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

    path_to_file = f'best_hyperparameters/HP_{model}_{country}.csv'
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

