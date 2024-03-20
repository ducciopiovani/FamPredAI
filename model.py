import pandas as pd
from utilities import multi_to_single
from ESNMod import ESNMod
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from typing import Union, Optional, List
from utilities import smooth_past_data


class Model():
    def __init__(self,
                 country: Union[str, int],
                 forecasting_window: int,
                 hyperparameters: dict,
                 variable_names: List[str],
                 constants: List[str],
                 time_granularity: Optional[str] = 'D',
                 target_name: Optional[str] = 'fcs'
                 ):
        """
        Args:
            country: country name/code the model is for
            target_name: str with name of indicator; e.g. 'fcs' (default)
            forecasting_window: size of forecasting window; possible choices: 30, 60, 90
            hyperparameters: dict with list of hyperparameters used for the model
            variable_names: list of varying features used in the model
            constants: list of constant features used in the model
            time_granularity: str with model's time granularity; choices = 'day' (default), 'week', 'dekade', 'month'
        """
        # forecasting parameters
        self.country = country
        admin_info = pd.read_csv("data/adm1_list.csv")
        admin_info = admin_info[admin_info['adm0_name'] == country]
        self.adm1_list = admin_info['adm1_code'].to_list()
        self.adm1_name = admin_info[['adm1_code', 'adm1_name']].set_index('adm1_code').to_dict()['adm1_name']
        self.country_id = admin_info.adm0_code.unique()[0]

        self.target_name = target_name
        self.time_granularity = time_granularity
        self.feature_names = {"variable": variable_names, "constant": constants}
        self.hyperparameters = hyperparameters
        self.forecasting_window = forecasting_window

        self.train_start_date = None
        self.train_end_date = None

        # the data coming from FeatureGenerator
        self.input_data = None
        self.ext_data = None
        # Objects from training and testing
        self.x_train = None
        self.ext_train = None
        self.y_train = None
        self.ext_pred = None
        # storing predictions
        self.predictions = []
        # means and std devs (to reverse the normalization after the predictions)
        self.data_mean = None
        self.data_sigma = None
        # Wrapper of Reservoir computing
        self.esn = None
        self.original_data = None   # saved after eventual smoothing step, before differencing and normalization
        self.confidence_intervals = None

        self.variable_names = variable_names + [target_name]
        self.constants = constants
        self.normalised_columns = None

    def load_data_from_file(self,
                            train_end_date: datetime,
                            train_start_date: Optional[datetime] = None,
                            ):
        path = f"data/{self.country}/full_timeseries_daily.csv"
        data = pd.read_csv(path, header=[0,1],index_col=0)
        data.index = pd.to_datetime(data.index)
        self.input_data = data.loc[train_start_date:train_end_date, self.variable_names].copy()
        self.ext_data = data.loc[train_start_date:train_end_date+timedelta(days=self.forecasting_window),
                        self.constants].copy()
        self.train_end_date = train_end_date
        self.train_start_date = train_start_date

    def prepare_data(self):
        """
        1. from multi-index column to normal + renaming
        2. smoothing all data - ignore
        3. normalised all data - ignore
        """

        no_smoothing_columns = ["Battles", "Violence against civilians",
                                "Remote violence", "day of the year"]
        no_norm_columns = ["Ramadan", "FCS", "rCSI", "rainfall_ndvi_seasonality"]
        data = self.input_data.copy()

        if 'smoothing' in self.hyperparameters.keys():
            data = self._smooth_data(data,
                                     delta_t=self.hyperparameters['smoothing'],
                                     leave_out_columns=no_smoothing_columns)
        self.original_data = data.copy()

        if self.hyperparameters["differencing"]:

            data[self.target_name] = data[self.target_name].diff()

            # If the target is differenced we want to renormalise it
            no_norm_columns.remove(self.target_name)

            # Remove the nan from data and external data
            data = data.iloc[1:, :]
            if self.ext_data is not None:
                self.ext_data = self.ext_data.iloc[1:, :]

        # Normalise the data
        data, data_mean, data_std = self._normalise(data=data,
                                                    leave_out_columns=no_norm_columns)
        self.data_mean = data_mean
        self.data_sigma = data_std

        # rename columns
        data.columns = multi_to_single(data.columns)
        if self.ext_data is not None:
            self.ext_data.columns = multi_to_single(self.ext_data.columns)
        self.input_data = data

    def _smooth_data(self, data: pd.DataFrame, delta_t: int = 10,
                     leave_out_columns: Optional[List] = None) -> pd.DataFrame:
        """
        :param data:
        :param delta_t:
        :return:
        """
        if leave_out_columns is None:
            leave_out_columns = []

        new_data = pd.DataFrame(index=data.index)
        for col in data.columns:
            if col in leave_out_columns:
                new_data[col] = np.array(data[col])
            else:
                new_data[col] = smooth_past_data(np.array(data[col]), delta_t=delta_t)
        new_data.columns = pd.MultiIndex.from_tuples(new_data.columns, names=['Level1', 'Level2'])
        return new_data

    def _normalise(self, data: pd.DataFrame, leave_out_columns: list[str]):
        """
        Normalises the data and ignores the leave_out_columns. For the leave out columns
        a mean of 0 and a std 1 are assigned so that the operation x-mean(x)/std(x) will leave
        the column unchanged.
        """
        mean = data.mean()
        std = data.std()

        for col in data.columns:
            if col[0] in leave_out_columns:
                mean[col] = 0
                std[col] = 1

        mean = np.array(mean)
        std = np.array(std)

        data = data - mean
        data = data / std
        return data, mean, std

    def prepare_x_y_train(self):
        """
        Extract training input and target data from self.data. Target data valid for predicting full input data except
        for ramadan.
        """
        self.x_train = self.input_data[:self.train_end_date].values
        self.y_train = self.input_data.iloc[1:][:self.train_end_date].values
        if self.ext_data is not None:
            self.ext_train = self.ext_data[:self.train_end_date].values
            self.ext_pred = self.ext_data[self.train_end_date:].iloc[1:self.forecasting_window+1].values

    def train_model(self):
        """
        train self.esn on training data obtained from self.data. Network is either created randomly or self._network is
        used, depending on new_network. Assigns self.x_test and self.ramadan_test if test data is available.
        """

        self.esn.create_network(n_dim=self.hyperparameters["n_dim"],
                                n_rad=self.hyperparameters["n_rad"],
                                n_avg_deg=self.hyperparameters["n_avg_deg"]
                                )
        if self.ext_data is not None:
            xtrain = np.append(self.x_train, self.ext_train, 1)
        else:
            xtrain = self.x_train.copy()

        self.esn.train(
            x_train=xtrain,
            y_train=self.y_train,
            sync_steps=self.hyperparameters["train_sync_steps"],
            reg_param=self.hyperparameters["reg_param"],
            w_in_scale=self.hyperparameters["w_in_scale"],
            w_out_fit_flag=self.hyperparameters["w_out_fit_flag"],
            save_r=True,
            w_in_no_update=False)

    def predict(self):
        columns_to_predict = self.x_train.shape[1]

        y_pred = self.esn.predict(
            columns_to_predict=columns_to_predict,
            pred_steps=self.forecasting_window,
            x_external=self.ext_pred
        )
        y_pred = self._reverse_norm(y_pred)
        return y_pred

    def _reverse_norm(self, timeseries: np.array, index=None) -> np.array:
        """
        Function that reverses the normalization of timeseries. May be passed a specific index for data mean and sigma.
        """
        if index is None:
            new_data = (
                    timeseries * self.data_sigma[: timeseries.shape[1]]
                    + self.data_mean[: timeseries.shape[1]]
            )
        else:
            new_data = (
                    timeseries * self.data_sigma[index]
                    + self.data_mean[index]
            )
        return new_data

    def _reverse_differencing(self):
        """
        Function that takes self.predictions and inverts the differencing applied to original data.
        """
        for adm1_code in self.predictions.adm1_code.unique():
            self.predictions.loc[(self.predictions["date"] == self.train_end_date + timedelta(days=1)) &
                                 (self.predictions.adm1_code == adm1_code), "prediction"] += (
                self.original_data.loc[self.train_end_date, self.target_name][str(adm1_code)])
        self.predictions["prediction"] = self.predictions.groupby("adm1_code", dropna=False)["prediction"].cumsum()

    def train_and_predict(self):
        self.train_model()
        return self.predict()

    def run(self,
            runs: int,
            verbose: bool = False,
            training_error: bool = True,
            ):

        # empty dict for results & errors
        adm_list = self.adm1_list
        result = {a1: [] for a1 in adm_list}
        predictions_dict = {a1: [] for a1 in adm_list}

        if training_error:
            training_error_df = pd.DataFrame(columns=["trial", f"adm1_code", "training error"])

        # Define an Echo State Network
        self.esn = ESNMod()
        self.prepare_x_y_train()

        for k in range(runs):
            if verbose:
                print(k)
            np.random.seed(k)
            y_pred = self.train_and_predict()
            pred_df = pd.DataFrame(y_pred, columns=list(self.input_data.columns))
            for a1 in adm_list:
                predictions_dict[a1].append(
                    pred_df[f"{self.target_name}-{a1}"])

            if training_error:
                full_training_error = self.training_error(train_steps=self.x_train.shape[0])
                for i in range(len(adm_list)):
                    training_error_df = training_error_df.append(
                        {
                            "trial": k,
                            f"adm1_code": adm_list[i],
                            "training error target": full_training_error[i],
                        },
                        ignore_index=True,
                    )

        pred_dates = pd.date_range(start=self.train_end_date + timedelta(days=1), periods=self.forecasting_window)

        # The agg results for admin are calculated
        for a1 in adm_list:
            pred_list = predictions_dict[a1]
            median = np.median(pred_list, axis=0)

            a1_pred_dict = {k: p for k, p in enumerate(pred_list)}
            a1_pred_df = pd.DataFrame(a1_pred_dict)
            a1_pred_df.index = pred_dates
            a1_pred_df["prediction"] = median
            a1_pred_df["adm1_code"] = a1
            a1_pred_df["adm1_name"] = self.adm1_name[a1]
            result[a1] = a1_pred_df

        result = pd.concat(result)
        result = result.reset_index()[["level_1", "adm1_code",  "prediction"]].rename(columns={"level_1": "date"})

        self.predictions = result

        if self.hyperparameters['differencing']:
            self._reverse_differencing()

    def get_confidence_intervals(self, runs: int = 20):
        """
        Calculates confidence intervals in real time by computing the RMSE among ~20 runs within the past
        (2 * self.forecasting_window) days and getting standard error of the residuals.
        """
        def get_std_err_from_residuals(df: pd.DataFrame):
            # Get squared diff
            df = df**2
            # Compute RMSE from squared diff (mean and sqrt)
            rmse = np.sqrt(df.mean(axis=1))
            # Get standard error of the residuals
            return rmse / np.sqrt(df.shape[1])

        # Save original train_end_date and get date for window on which testing will be performed to compute errors
        original_end_date = self.train_end_date
        start_train_end_date = original_end_date - timedelta(3*self.forecasting_window)

        # Define steps based on size of self.forecasting_window (i.e. considers number of days, not time granularity)
        steps = int(np.floor(self.forecasting_window/10))

        # Testing errors
        errors_dict = defaultdict(pd.DataFrame)
        for days in range(0, 2*self.forecasting_window-steps, steps):
            # Update train_end_date to start_train_end_date + days
            self.train_end_date = start_train_end_date + timedelta(days)

            # Get predictions for current window
            self.run(runs=runs, training_error=False)
            if self.hyperparameters["differencing"] is True:
                self._reverse_differencing()
            preds = self.predictions.pivot(index="date", columns="adm1_code", values="prediction")

            # Get input data
            input = self.original_data.loc[preds.index, self.target_name]

            # Store residuals (between preds and input)
            residuals = input-preds
            residuals.index = range(self.forecasting_window)
            for col in residuals.columns:
                errors_dict[col] = pd.concat([errors_dict[col], residuals[col]], axis=1)

        # Restore original train end date
        self.train_end_date = original_end_date

        # Compute mean over all days and apply square root to get RMSE
        self.confidence_intervals = pd.DataFrame(columns=[f"adm{self.adm_level}_code", "date", "lower", "upper"])
        dates = pd.date_range(self.train_end_date+timedelta(days=1), periods=self.forecasting_window)
        for k, v in errors_dict.items():
            std_err = get_std_err_from_residuals(v)
            self.confidence_intervals = pd.concat([self.confidence_intervals,
                pd.DataFrame({f"adm{self.adm_level}_code": k, "date": dates,
                "lower": -2*std_err, "upper": 2*std_err})])

    def training_error(self, train_steps: int) -> np.array:
        """
        :param train_steps:
        :return:
        """
        return np.sqrt(
            np.sum(
                (
                        self.esn._w_out @ self.esn._r_train_gen.T
                        - self.y_train[self.hyperparameters["train_sync_steps"]:].T
                )
                ** 2,
                axis=1) / train_steps)


