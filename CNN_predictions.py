import time

import pandas as pd
from pathlib import Path
import numpy as np

from CNNprototype import CNN_model
from JoschkaCleanCode.settings import countries_dict

countryname = "Yemen"


def run_model(mydir,learning_rate, epochs, split_date, n_steps_in, early_stopping, smoothing, kernel_size,
              filters, pool_size, dense_units, layers, pred_steps, features, differencing):

    data = full_data.copy(deep=True)

    if features == "FCS":
        data = data[["FCS"]]
    elif features == "FCS+":
        data = data[["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"]]
    elif features == "climate":
        data = data[["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality", "rainfall", "NDVI",
                     "log rainfall 1 month anomaly", "log rainfall 3 months anomaly", "log NDVI anomaly"]]
    elif features == "economics":
        new_data = data[["FCS", "rCSI", "Ramadan", "day of the year", "rainfall_ndvi_seasonality"]]
        try:
            new_data["CE official"] = data["CE official"]
        except:
            pass
        try:
            new_data["CE unofficial"] = data["CE unofficial"]
        except:
            pass
        try:
            new_data[pd.MultiIndex.from_product([["PEWI"], data["PEWI"].columns])] = data[
                pd.MultiIndex.from_product([["PEWI"], data["PEWI"].columns])]
        except:
            pass
        try:
            new_data["headline inflation"] = data["headline inflation"]
        except:
            pass
        try:
            new_data["food inflation"] = data["food inflation"]
        except:
            pass
    elif features == "all":
        data = data


    if data[:str(split_date + pd.DateOffset(days=1))].isna().any().any():
        nancolumns = data.columns[data[:str(split_date + pd.DateOffset(days=1))].isna().any().any()].tolist()
        raise ValueError(f"NaN values in training data ({nancolumns}). Prediction not possible.")


    n_steps_out = pred_steps


    np.random.seed(0)
    model = CNN_model(dense_units = dense_units, epochs = epochs, learning_rate = learning_rate, n_steps_in = n_steps_in,
                           n_steps_out = n_steps_out, early_stopping = early_stopping, kernel_size=kernel_size,
                           filters=filters,pool_size=pool_size , smoothing=smoothing, layers=layers, differencing=differencing)
    t0=time.time()
    RMSE_train, RMSE_test, pred = model.test_model(data[:str(split_date+pd.DateOffset(days=n_steps_out+30))],
                                                 split_date=split_date, target_column="FCS",return_pred=True,
                                                 verbose=True)
    t1=time.time()

    trainable_count = model.model.count_params()
    effective_epoch = model.get_final_epoch()

    RMSEres = RMSE_test
    timeres = t1-t0
    parameterres = trainable_count
    effective_epoch_res = effective_epoch

    res = pd.DataFrame()
    for n, a1 in enumerate(data["FCS"].columns):
        res[f"prediction {a1}"] = pred[0, :, n]
        #res[f"target {a1}"] = model.validation_data[1][0, :, n]
    res["RMSE"] = RMSEres
    res["training time"] = timeres
    res["parameter number"] = parameterres
    res["epochs"] = effective_epoch_res

    res.to_csv(f"{mydir}_csv")

    return res


if __name__ == "__main__":
    full_data = pd.read_csv(
        Path(f"../DataTimeSeries/{countryname}/grid_search/full_timeseries_daily.csv"),
        index_col=0,
        header=[0, 1],
    )
    dtindex = pd.DatetimeIndex(full_data.index)

    full_data.index = dtindex

    for i, pred_steps in enumerate([30,60,90]):
        hp_df = pd.read_csv(f"ForecastReports/HP_CNN{countries_dict[countryname]['iso3']}_{pred_steps}_corrected.csv", index_col=0)

        split_date_list = hp_df.index

        l = len(split_date_list)
        print(l)

        for n in range(l-i):
            try:
                split_date = pd.Timestamp(split_date_list[n+1+i])
            except IndexError:
                split_date = pd.Timestamp(split_date_list[n + i]) + pd.DateOffset(months=1)

            learning_rate = hp_df["learning_rate"].iloc[n]
            epochs = hp_df["epochs"].iloc[n]
            n_steps_in = hp_df["n_steps_in"].iloc[n]
            early_stopping = hp_df["early_stopping"].iloc[n]
            smoothing = hp_df["smoothing"].iloc[n]
            kernel_size = int(hp_df["kernel_size"].iloc[n])
            filters = hp_df["filters"].iloc[n]
            pool_size = int(hp_df["pool_size"].iloc[n])
            dense_units = hp_df["dense_units"].iloc[n]
            layers = hp_df["layers"].iloc[n]
            differencing = hp_df["differencing"].iloc[n]
            features = hp_df["features"].iloc[n]

            run_model(
                mydir=f"../Predictions/CNN/{countries_dict[countryname]['iso3']}_{pred_steps}_{str(split_date).split()[0]}",
                learning_rate=learning_rate,
                epochs=epochs,
                n_steps_in=n_steps_in,
                split_date=split_date,
                pred_steps=pred_steps,
                early_stopping=early_stopping,
                smoothing=smoothing,
                kernel_size=kernel_size,
                filters=filters,
                pool_size=pool_size,
                dense_units=dense_units,
                layers=layers,
                differencing=differencing,
                features=features,
            )
