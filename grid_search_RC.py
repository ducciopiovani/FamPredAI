from pathlib import Path
import numpy as np
import pandas as pd
from model import Model
import warnings
import time
from datetime import datetime
from sklearn.model_selection import ParameterGrid
warnings.filterwarnings("ignore")


param_grid = {
    'n_rad_range': np.linspace(0.3, 2.1, 10),
    'reg_param_range': [1e-5, 1e-3, 1e-1, 10, 100],
    'w_in_scale_range':  [0.1, 0.3, 0.5, 1.0, 1.5],
    'n_dim_range': [1000],
    'degree_range': [8],
    'features_range': ["FCS", "calendar", "FCS+", "economics", "climate", "all"],
    'differencing_range': [True, False],
    'last_train_date': pd.date_range(start=datetime(2023, 1, 1), end=datetime(2024, 1, 1,), freq='M')
}

feature_dict = {"FCS": ["FCS"],
                "FCS+": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "calendar": ["FCS", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "climate": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                             "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                             "log NDVI anomaly"],
                'economics': ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                             "CE official" "CE unofficial","PEWI", "headline inflation", "food inflation"],
                "all": ['ALL']}


def RMSE(y_pred, y_test, l):
    y_test_df = pd.DataFrame(y_test)
    for col in y_test_df.columns:
        if len(y_test_df[col]) < l:
            return np.array([np.nan] * y_test.shape[1])
    return np.sqrt(np.mean((np.array(y_pred)[:l] - np.array(y_test)[:l]) ** 2))


def run_model(param_grid: dict):

    train_start_date = datetime(2019, 1, 1)

    grid = ParameterGrid(param_grid)
    target = 'FCS'
    for m, hparams in enumerate(grid):
        variables = feature_dict[hparams['features_range']].copy()
        constants = [t for t in ["Ramadan", "day of the year", "lean season", "rainfall_ndvi_seasonality"] if t in variables]
        variables = [v for v in variables if v not in constants]

        md = Model(country='Nigeria',
                   forecasting_window=30,
                   target_name=target,
                   constants=constants,
                   variable_names=variables,
                   hyperparameters=hparams
                   )

        md.load_data_from_file(train_start_date=train_start_date,
                               train_end_date=hparams['last_train_date'])
        md.prepare_data(hparams['smoothing'])
        md.run(20)


run_model(param_grid=param_grid)

