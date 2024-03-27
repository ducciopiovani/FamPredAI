
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go



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


def performances_forecasts(path : str, country: str, old: bool =False, nruns: int = 100):
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
            if nruns:
                if file.split('_')[-1].split('.')[0]==str(nruns):
                  files.append(file)
            else:
                files.append(file)
    if old:
        rmse_list = []
        for n in np.arange(0, len(files)):
            df = pd.read_csv(path+'/'+files[n])
            rlist = []
            for n in np.arange(2, df.shape[1] - 2, 2):
                rlist.append(rmse(df.iloc[:, n], df.iloc[:, n - 1]))
            rmse_list.append(np.median(rlist))
    else:
        rmse_list = []
        for n in np.arange(0, len(files)):
            df = pd.read_csv(path + '/' + files[n])
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


def merge_predictions_and_rtm(country: str, preds: pd.DataFrame, forecast_window=60):
    """
    Merge data and Predictions
    Args:
        country: Name of the country
        preds: file containing the predictions ( use function forecast)
        forecast_window: the lenght of the forecasts
        show: bollean to show the comparison between data and predictions
    Returns:
    """
    preds = preds.melt(id_vars='date').rename(columns={'variable': 'adm1_code', 'value': 'prediction'})
    preds['adm1_code'] = preds['adm1_code'].astype(int)
    data = pd.read_csv(f"data/{country}/full_timeseries_daily.csv", header=[0, 1], index_col=0)
    data.index.name = 'date'
    data.index = pd.to_datetime(data.index)
    fcs = data['FCS'].rolling('10D').mean()
    fcs = fcs.reset_index().melt(id_vars='date', value_name='data', var_name='adm1_code')
    fcs['adm1_code'] = fcs['adm1_code'].astype(int)

    fcs = fcs.merge(preds, on=['date', 'adm1_code'], how='left')
    fcs = fcs[~fcs.prediction.isnull()]
    return fcs


def plot(data, country, ncols):
    admin1 = pd.read_csv("data/adm1_list.csv")
    admin1 = admin1[admin1.adm0_name == country]
    adm1_list = admin1['adm1_code'].to_list()
    adm1_name = admin1[['adm1_code', 'adm1_name']].set_index('adm1_code').to_dict()['adm1_name']


    nrows = int(np.ceil(len(adm1_list) / ncols))

    f = make_subplots(nrows,
                      ncols,
                      vertical_spacing=0.18,
                      subplot_titles=tuple([adm1_name[a1] for a1 in adm1_list])
                      )

    for num_plot, a in enumerate(adm1_list):
        df = data[data.adm1_code == a].copy()
        col = num_plot % ncols + 1
        row = num_plot // ncols + 1

        # prepare subplot: data, predictions and confidence interval
        f.add_trace(
            go.Scatter(name='data',
                       x=df['date'],
                       y=df['data'],
                       marker_color='blue',
                       showlegend=False),
            row=row, col=col
        )
        f.add_trace(
            go.Scatter(x=df['date'],
                       y=df['prediction'],
                       marker_color='red',
                       showlegend=True if num_plot == 0 else False,
                       ),
            row=row, col=col
        )

    f.update_layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=30,  # top margin
        ),
        height=nrows * 150, width=200 * ncols
    )
    return f


