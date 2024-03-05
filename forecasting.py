
from model import Model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List
from utilities import rmse, find_crises, find_hyperparameters

feature_dict = {"FCS": ["FCS"],
                "FCS+": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "calendar": ["FCS", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "climate": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                             "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                             "log NDVI anomaly"],
                'economics': ["FCS", "rCSI", "Ramadan", "day of the year",
                             "CE official", "CE unofficial","PEWI", "headline inflation", "food inflation"],
                "all":["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                             "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                             "log NDVI anomaly", "CE official", "CE unofficial","PEWI", "headline inflation", "food inflation"]}


def merge_predictions_and_rtm(country: str, preds: pd.DataFrame, forecast_window=60, show=False):
    """
    Merge data and Predictions
    Args:
        country: Name of the country
        preds: file containing the predictions ( use function forecast)
        forecast_window: the lenght of the forecasts
        show: bollean to show the comparison between data and predictions
    Returns:
    """
    train_end_date = preds['date'].min() - timedelta(days=1)

    data = pd.read_csv(f"data/{country}/full_timeseries_daily.csv", header=[0, 1], index_col=0)
    data.index.name = 'date'
    data.index = pd.to_datetime(data.index)
    fcs = data['FCS'].rolling('10D').mean()
    fcs = fcs.reset_index().melt(id_vars='date', value_name='data', var_name='adm1_code')
    fcs['adm1_code'] = fcs['adm1_code'].astype(int)

    fcs = fcs.merge(preds, on=['date', 'adm1_code'], how='left')
    fcs = fcs[~fcs.data.isnull()]
    fcs = fcs[fcs.date > train_end_date - timedelta(days=forecast_window * 5)]
    fcs = fcs[fcs.date < train_end_date + timedelta(days=forecast_window)]

    if not show:
        return fcs
    else:
        admin1 = pd.read_csv("data/adm1_list.csv")
        admin1 = admin1[admin1.adm0_name == 'Nigeria']
        adm1_list = admin1['adm1_code'].to_list()
        adm1_name = admin1[['adm1_code', 'adm1_name']].set_index('adm1_code').to_dict()['adm1_name']

        ncols = 3
        nrows = int(np.ceil(len(adm1_list) / ncols))

        f = make_subplots(nrows,
                          ncols,
                          vertical_spacing=0.18,
                          subplot_titles=tuple([adm1_name[a1] for a1 in adm1_list])
                          )

        for num_plot, a in enumerate(adm1_list):
            df = fcs[fcs.adm1_code == a].copy()
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


def forecast(country: str,
             first_forecast: datetime,
             constants: List[str],
             variables: List[str],
             hyperparameters: dict,
             target: str = 'FCS',
             forecast_window=60,
             runs=20):
    """
    Generate the forecasts
    Args:
        country: name of the forecasts
        first_forecast: date of first forecast
        constants: constants within input variables
        variables: name of the variables
        hyperparameters: dictionary containing hyperparameters
        target: name of the target varible
        forecast_window: length of the forecasts
        runs: number runs
    Returns:
    """

    train_end_date = first_forecast - timedelta(days=1)
    md = Model(country=country,
               forecasting_window=forecast_window,
               target_name=target,
               constants=constants,
               variable_names=variables,
               hyperparameters=hyperparameters
               )

    md.load_data_from_file(train_start_date=datetime(2020, 1, 1),
                           train_end_date=train_end_date)
    md.prepare_data()
    md.run(runs)
    return md.predictions


def forecast_from_file(country: str, model: str, runs: int = 100):
    """
    Generate the forecasts starting from the file with the hyperparameters selected during the grid seach
    Args:
        country: name of the forecasts
        model: name of the model
        runs: number of runs
    Returns:
    """
    path_to_file = f'best_hyperparameters/HP_{model}_{country}.csv'
    df = pd.read_csv(path_to_file)

    all_predictions = []

    for ind, row in df.iterrows():

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

        constants_list = ["Ramadan", "day of the year", "lean season", "rainfall_ndvi_seasonality"]

        target = 'FCS'
        all_variables = pd.read_csv(f"data/{country}/full_timeseries_daily.csv", header=[0,1], index_col=[0])
        all_variables = list(all_variables.melt().variable_0.unique())

        variables = [v for v in feature_dict[hyperparameters['features']] if v in all_variables]
        constants = [t for t in constants_list if t in variables]
        variables = [v for v in variables if v not in constants_list]
        if 'FCS' in variables:
            variables.remove('FCS')
        else:
            print(variables)

        preds = forecast(country=country,
                         variables=variables,
                         constants=constants,
                         hyperparameters=hyperparameters,
                         forecast_window=60,
                         first_forecast=datetime.strptime(row['split_date'], "%Y-%m-%d"),
                         runs=runs
                         )

        data = merge_predictions_and_rtm(country=country,
                                         preds=preds,
                                         forecast_window=60,
                                         show=False)

        data = data[~data['prediction'].isnull()]
        data['split'] = ind+1
        steps = [i for i in range(1, 60)] * data['adm1_code'].nunique()
        data['forecast_step'] = steps

        all_predictions.append(data)
    all_predictions = pd.concat(all_predictions)

    name = f'new_predictions/{country}_RC_{runs}.csv'
    all_predictions.to_csv(name, index=False)
    res = all_predictions.groupby(['adm1_code', 'split']).apply(lambda d: rmse(d['data'], d['prediction'])).reset_index()
    perf = res.groupby('adm1_code')[0].median().median()
    return perf


def early_warning_prototype(country: str):
    """
    Generate and save forecasts for the crises dates
    Args:
        country: Name of the country
    Returns:
    """
    df = pd.read_csv(f'data/{country}/full_timeseries_daily.csv', header=[0,1], index_col=0)
    fcs = df['FCS']
    fcs = fcs[fcs.isnull().sum(1) ==0]
    fcs.index = pd.to_datetime(fcs.index)

    cr = find_crises(fcs, t=0.1)
    cr = cr[(cr['dates']>datetime(2022, 6, 1))&(cr['dates']< datetime(2023,6,1))]
    cr = cr[::-1]

    for ind, row in cr.iterrows():
        date = row['dates']
        hyperparameters = find_hyperparameters(country='Nigeria', date=date, model='RC')
        constants_list = ["Ramadan", "day of the year", "lean season", "rainfall_ndvi_seasonality"]
        target = 'FCS'
        all_variables = pd.read_csv(f"data/{country}/full_timeseries_daily.csv", header=[0, 1], index_col=[0])
        all_variables = list(all_variables.melt().variable_0.unique())
        variables = [v for v in feature_dict[hyperparameters['features']] if v in all_variables]
        constants = [t for t in constants_list if t in variables]
        variables = [v for v in variables if v not in constants_list]
        variables.remove('FCS')
        res = []
        for n in range(0, 5):
            print(n)
            first_forecast = date + timedelta(days=n)
            preds = forecast(country=country,
                             first_forecast=first_forecast,
                             constants=constants,
                             variables=variables,
                             hyperparameters=hyperparameters)
            preds = preds.pivot(index='date',
                                columns="adm1_code",
                                values="prediction"
                                )
            preds['first_forecast_date'] = first_forecast
            res.append(preds)
        res = pd.concat(res)
        res.to_csv("data/{country}_RC_" + date.strftime('%Y-%m-%d') + ".csv")

