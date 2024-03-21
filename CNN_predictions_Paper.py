import time
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import timedelta
from CNNprototype import CNN_model


countryname = "Yemen"


def check_null(x):
    try:
        x = np.float32(x)
    except:
        x = np.nan
    if np.isinf(x):
        x = np.nan

    return x


feature_dict = {"FCS": ["FCS"],
                "FCS+": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "calendar": ["FCS", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"],
                "climate": ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                             "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                             "log NDVI anomaly"],
                'economics': ["FCS", "rCSI", "Ramadan", "day of the year",
                             "CE official", "CE unofficial","PEWI", "headline inflation", "food inflation"],
                "all":  ["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality",
                         "rainfall", "NDVI", "log rainfall 1 month anomaly", "log rainfall 3 months anomaly",
                         "log NDVI anomaly", "CE official", "CE unofficial","PEWI", "headline inflation",
                         "food inflation"]
                }


def run_model(mydir: str,
              hyperparameters: dict):

    # for simplicity
    hp = hyperparameters.copy()
    data = full_data.copy(deep=True)
    features = [f for f in feature_dict[hp['features']] if f in data.columns]
    new_data = data[features]

    if new_data[:hp["split_date"]].isna().any().any():
        raise ValueError(f"NaN values in training data. Prediction not possible.")

    n_steps_out = 60
    np.random.seed(0)
    model = CNN_model(dense_units=hp['dense_units'],
                      epochs=hp['epochs'],
                      learning_rate=hp['learning_rate'],
                      n_steps_in=hp['n_steps_in'],
                      n_steps_out=hp['n_steps_out'],
                      early_stopping=hp['early_stopping'],
                      kernel_size=hp['kernel_size'],
                      filters=hp['filters'],
                      pool_size=hp['pool_size'],
                      smoothing=hp['smoothing'],
                      layers=hp['layers'],
                      differencing=hp['differencing']
                      )
    t0=time.time()
    counter = 0
    while counter < 100:
        data_index = hp["split_date"]+pd.DateOffset(days=hp["n_steps_out"]+30)
        RMSE_train, RMSE_test, pred = model.test_model(data=new_data[:data_index],
                                                       split_date=hp["split_date"],
                                                       target_column="FCS",
                                                       return_pred=True)
        if (np.isnan(RMSE_test) or RMSE_train > 0.1):
            print(f"NaN result in trial {counter}")
            counter += 1
            if counter < 100:
                print("Try again!")
            else:
                print("Abort!")
        else:
            print(f"success in trial {counter}")
            counter = 100

    t1=time.time()

    trainable_count = model.model.count_params()
    effective_epoch = model.get_final_epoch()

    RMSEres = check_null(RMSE_test)
    timeres = check_null(t1 - t0)
    parameterres = check_null(trainable_count)
    effective_epoch_res = check_null(effective_epoch)

    res = pd.DataFrame()
    for n, a1 in enumerate(data["FCS"].columns):
        res[f"prediction {a1}"] = pred[0, :, n]
    res["RMSE"] = RMSEres
    res["training time"] = timeres
    res["parameter number"] = parameterres
    res["epochs"] = effective_epoch_res

    res.to_csv(f"{mydir}.csv")

    return res


if __name__ == "__main__":

    country = 'Yemen'
    full_data = pd.read_csv(
        Path(f"data/{country}/full_timeseries_daily.csv"),
        index_col=0,
        header=[0, 1],
    )
    dtindex = pd.DatetimeIndex(full_data.index)

    full_data.index = dtindex

    hp_df = pd.read_csv(f"best_hyperparameters/HP_CNN_{country}.csv")
    split_date_list = list(hp_df["split_date"])

    for split_date in ["2022-06-01"]:#split_date_list:

        hp = hp_df.loc[hp_df["split_date"] == split_date]
        hp = hp.iloc[0, :].to_dict()
        run_model(
            mydir=f"testPred/CNN",
            hyperparameters=hp
        )
