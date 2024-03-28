from typing import Union, List, Optional
import pandas as pd
from datetime import datetime, timedelta

from keras import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Input, Dense, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from utilities import smooth_past_data, shuffle_io, merge_predictions_and_rtm, feature_dict
import numpy as np
from time import time

import tensorflow_addons as tfa


class LSTMModel():
    def __init__(self,
                 country: str,
                 forecasting_window: int,
                 hyperparameters: dict,
                 time_granularity: Optional[str] = 'D',
                 target_name: Optional[str] = 'FCS'
                 ):
        self.country = country
        # forecasting parameters
        admin_info = pd.read_csv("data/adm1_list.csv")
        admin_info = admin_info[admin_info['adm0_name'] == country]
        self.adm1_list = admin_info['adm1_code'].to_list()
        self.adm1_name = admin_info[['adm1_code', 'adm1_name']].set_index('adm1_code').to_dict()['adm1_name']
        self.country_id = admin_info.adm0_code.unique()[0]

        self.target_name = target_name
        self.time_granularity = time_granularity
        self.hyperparameters = hyperparameters
        self.forecasting_window = forecasting_window

        self.train_start_date = None
        self.train_end_date = None

        # the data coming from FeatureGenerator
        self.input_data = None

        # Objects from training and testing
        self.x_train = None
        self.y_train = None
        self.x_pred = None
        self.training_data = () # tuple that will store input/output data

        # storing predictions
        self.predictions = []

        # Deep Learning Architecture
        self.n_output = None
        self.n_features = None
        self.n_output_internal = None

        self.model = None
        self.scaling = True

    def load_data_from_file(self,
                            train_end_date: datetime,
                            train_start_date: Optional[datetime] = None,
                            ):

        hp = self.hyperparameters
        path = f"data/{self.country}/full_timeseries_daily.csv"
        data = pd.read_csv(path, header=[0, 1], index_col=0)
        features = [f for f in feature_dict[hp['features']] if f in data.columns]
        data = data[features]
        data.index = pd.to_datetime(data.index)
        self.input_data = data.loc[train_start_date:train_end_date].copy()

        self.train_end_date = train_end_date
        self.train_start_date = train_start_date

    @staticmethod
    def _smooth_data(data: pd.DataFrame, delta_t: int = 10,
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

    def prepare_data(self):

        data = self.input_data.copy()
        target = data[self.target_name].copy()
        self.target_columns = target.columns

        hp = self.hyperparameters

        if hp['smoothing'] is not None:
            data = self._smooth_data(data, delta_t=hp['smoothing'])

        if hp['differencing']:
            # differencing the target
            self.seed_value = np.array(target.iloc[-1, :].copy())
            target = target.diff().dropna()

            # differencing the target value in the features
            data[self.target_name] = data[self.target_name].diff()
            if 'rCSI' in data:
                data['rCSI'] = data['rCSI'].diff()
            data = data.iloc[1:, :]
        target = np.array(target)

        if self.scaling:
            scaler = MinMaxScaler(feature_range=(0, 1))
            features = data.loc[:, data.columns.get_level_values(0) != self.target_name]
            # Scale all features
            if features.shape[1] > 0:
                scaled_features = scaler.fit_transform(features)
                data.loc[:, features.columns] = scaled_features

            # Scale the target only if it has been differenced
            if hp['differencing']:
                # Scale the target's past values (feature) and future values (target)
                self.target_scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_target = self.target_scaler.fit_transform(
                    data.loc[:, data.columns.get_level_values(0) == self.target_name])
                data.loc[:, data.columns.get_level_values(0) == self.target_name] = scaled_target
                target = self.target_scaler.transform(target)

        sequences = np.array(data)
        input_data = []
        output_data = []
        # Prepare the input / output structure
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + hp["n_steps_in"]
            out_end_ix = end_ix + self.forecasting_window
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            input_data.append(sequences[i:end_ix, :])
            output_data.append(target[end_ix:out_end_ix, :])

        self.x_train = np.array(input_data)
        self.y_train = np.array(output_data)
        self.training_data = shuffle_io((self.x_train, self.y_train))
        self.x_pred = np.array([sequences[-hp['n_steps_in']:, :]])

    def define_model(self):

        self.set_n_output()

        np.random.seed(0)
        hp = self.hyperparameters
        print(hp)
        if hp["simple"] == False:
            # define model vs 1
            model = Sequential()
            model.add(LSTM(hp['units'], activation='relu', dropout=hp["dropout"], recurrent_dropout=hp["dropout"],
                           input_shape=(hp["n_steps_in"], self.n_features)))
            model.add(RepeatVector(self.forecasting_window))
            model.add(LSTM(hp["units"], activation='relu', dropout=hp["dropout"], recurrent_dropout=hp["dropout"],
                           return_sequences=True))
            model.add(TimeDistributed(Dense(self.n_output)))
            model.compile(optimizer=Adam(learning_rate=hp["learning_rate"]), loss='mse')

        else:
            # define model vs 2
            inputs = Input(shape=(hp["n_steps_in"], self.n_features))
            if hp['dropout'] > 0:
                lstm_out = LSTM(hp["units"],
                                dropout=hp["dropout"],
                                recurrent_dropout=hp["dropout"],
                                return_sequences=True)(inputs)
            else:
                LSTM(hp["units"], return_sequences=True)(inputs)
            outputs = TimeDistributed(Dense(self.n_output))(lstm_out)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=hp["learning_rate"]), loss='mse')

        self.model = model

    def train(self, verbose=False):

        hp = self.hyperparameters
        np.random.seed(0)

        X = self.training_data[0]
        y = self.training_data[1]
        callbacks = []
        timeStopping = tfa.callbacks.TimeStopping(seconds=1800, verbose=1)
        callbacks.append(timeStopping)
        if hp["early_stopping"]:
            earlyStopCallBack = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            callbacks.append(earlyStopCallBack)
        # fit model
        t0 = time()
        self.history = self.model.fit(X, y, validation_split=0.2,
                                      epochs=hp["epochs"],
                                      verbose=verbose,
                                      callbacks=callbacks)
        t1 = time()
        print(f"training took a total of {t1-t0} seconds")

    def set_n_output(self):
        training_target = self.training_data[1]
        training_input = self.training_data[0]
        n_output = training_target.shape[2]

        self.n_output = n_output
        self.n_features = training_input.shape[2]

    def predict(self):

        pred = self.model.predict(self.x_pred, verbose=False)
        pred = pred.reshape(self.x_pred.shape[0], self.forecasting_window, self.n_output)
        pred = pred[0]  # There is only one prediction

        if self.hyperparameters["differencing"]:
            # The inverse transform  is only applied if the target has
            # been differences
            pred = self.target_scaler.inverse_transform(pred)
            pred = np.cumsum(pred, axis=0) + self.seed_value
        self.predictions = pd.DataFrame(pred, columns=self.target_columns)
        return self.predictions


def forecast_from_file(country: str, forecasting_window: int):

    hyperparameters = pd.read_csv(f"best_hyperparameters/HP_LSTM_{country}.csv")

    for ind, row in hyperparameters.iterrows():
        print(row['split_date'])

        hp = row[["units",
                   "learning_rate",
                   "n_steps_in",
                   "early_stopping",
                   "smoothing",
                   "simple",
                   "dropout",
                   "epochs",
                   "differencing",
                   "features"]].to_dict()



        train_end_date = datetime.strptime(row['split_date'], "%Y-%m-%d") - timedelta(days=1)
        lstm = LSTMModel(hyperparameters=hp, country=country, forecasting_window=forecasting_window)
        lstm.load_data_from_file(train_start_date=datetime(2017, 1, 1),
                                 train_end_date=train_end_date)
        lstm.prepare_data()
        lstm.define_model()
        lstm.train(verbose=True)
        dates = pd.date_range(start=train_end_date+timedelta(days=1),
                              end=train_end_date+timedelta(days=forecasting_window))
        predictions = lstm.predict()
        predictions['date'] = pd.to_datetime(dates)

        predictions = predictions.melt(id_vars='date').rename(columns={'variable': 'adm1_code',
                                                                       'value': 'prediction'})

        predictions = merge_predictions_and_rtm(country=country,
                                                preds=predictions)
        predictions.to_csv(f"forecasts/LSTM/{country}_{row['split_date']}.csv")

    return predictions


if __name__ =='__main__':
    forecast_from_file(country='Nigeria', forecasting_window=60)