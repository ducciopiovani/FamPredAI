import numpy as np
import pandas as pd
import time
import warnings

import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from utilities import smooth_past_data, shuffle_io
from sklearn.preprocessing import MinMaxScaler


class KerasModel:
    """
    General class for keras based model for forecasting FCS. This is inherited by actual models like LSTM and CNN.
    """
    def __init__(self, epochs, learning_rate, n_steps_in, n_steps_out, early_stopping, differencing,
                 smoothing=10, scaling=True):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_steps_in = n_steps_in
        self.early_stopping = early_stopping
        self.scaling = scaling
        self.smoothing = smoothing
        self.differencing=differencing

        self.n_steps_out = n_steps_out
        self.n_features = None
        self.n_output = None

        self.model = None
        self.history = None

        self.training_data = None
        self.validation_data = None

        self.seed_value = None
        self.fcs_scaler = None
        self.secondary_scaler = None

        self.flatten = False

    def define_model(self):
        """
        This method should always be overwritten
        """
        warnings.warn("This is not meant as an actual model.")

        np.random.seed(0)

        model = Sequential()
        model.add(Dense(50, input_shape=(self.n_steps_in, self.n_features)))
        model.add(Dense(self.n_output))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        self.model = model

    def train(self, input_data, target_data, verbose=False):
        np.random.seed(0)
        X = input_data
        y = target_data
        callbacks = []

        timeStopping = tfa.callbacks.TimeStopping(seconds=1800, verbose=1)

        callbacks.append(timeStopping)

        if self.early_stopping:
            earlyStopCallBack = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            callbacks.append(earlyStopCallBack)


        # fit model
        t0 = time.time()
        self.history = self.model.fit(X, y, validation_split=0.2, epochs=self.epochs, verbose=verbose, callbacks=callbacks)
        t1 = time.time()
        print(f"training took a total of {t1-t0} seconds")

    def test_model(self,
                   data,
                   split_date,
                   target_column="FCS",
                   verbose=False,
                   return_pred=False):
        data1 = data.copy(deep=True)
        tf.random.set_seed(0)
        training_data = data1[:split_date]
        test_data = data1[split_date-self.n_steps_in + 1:split_date + self.n_steps_out]

        training_input, training_target = self.format_data(training_data,
                                                           n_steps_in=self.n_steps_in,
                                                           n_steps_out=self.n_steps_out,
                                                           target_column=target_column,
                                                           smoothing=self.smoothing,
                                                           scaling=self.scaling,
                                                           set_seed=False)
        test_input, test_target = self.format_data(test_data,
                                                   n_steps_in=self.n_steps_in,
                                                   n_steps_out=self.n_steps_out,
                                                   target_column=target_column,
                                                   smoothing=self.smoothing,
                                                   scaling=self.scaling,
                                                   set_seed=True)
        # This should not make a difference.
        self.training_data = shuffle_io((training_input, training_target))
        self.validation_data = (test_input, test_target)

        self.set_n_output()
        self.define_model()
        self.train(input_data=training_input, target_data=training_target, verbose=verbose)

        pred = self.predict(test_input, seed_value=self.seed_value)

        mse_train = self.evaluate(training_input, training_target, seed_value=0)
        mse_test = self.evaluate(test_input, test_target, seed_value=self.seed_value)

        print(f"training RMSE: {np.sqrt(mse_train)}")
        print(f"testing RMSE: {np.sqrt(mse_test)}")

        if return_pred:
            return np.sqrt(mse_train), np.sqrt(mse_test), pred
        else:
            return np.sqrt(mse_train), np.sqrt(mse_test)

    def set_n_output(self):
        """
        setting the dimensions
        """
        training_target = self.training_data[1]
        training_input = self.training_data[0]

        n_output = training_target.shape[2]
        self.n_output = n_output
        self.n_features = training_input.shape[2]

    def format_data(self,
                    df,
                    n_steps_in,
                    n_steps_out,
                    target_column="FCS",
                    smoothing=10,
                    scaling=True,
                    set_seed=False
                    ):

        new_df = df.copy(deep=True)

        # convert into input/output
        if smoothing:
            new_df[target_column] = smooth_data(data=new_df[target_column], delta_t=smoothing)

        target_df = new_df[target_column].copy(deep=True)

        input_data = []
        output_data = []
        seed_values = []

        if self.differencing:
            original_target = np.array(new_df.loc[:, ["FCS"]].copy(deep=True))
            new_df.loc[:, ["FCS"]] = new_df.loc[:, ["FCS"]].diff().fillna(0)
            if 'rCSI' in new_df.columns:
                new_df.loc[:, ["rCSI"]] = new_df.loc[:, ["rCSI"]].diff().fillna(0)
            target = np.array(target_df.diff().fillna(0))
        else:
            target = np.array(target_df)

        if scaling and list(set(new_df.columns.get_level_values(0))) != ["FCS"]:
            # THis is done because we first apply this to the train data, and the
            # second time we apply it directly to the test but want to use the
            # same scaling conditions
            if self.secondary_scaler is None:
                self.secondary_scaler = MinMaxScaler(feature_range=(0, 1))
                new_df.loc[:, new_df.columns.get_level_values(0) != "FCS"] = self.secondary_scaler.fit_transform(
                    new_df.loc[:, df.columns.get_level_values(0) != "FCS"])
            else:
                new_df.loc[:, new_df.columns.get_level_values(0) != "FCS"] = self.secondary_scaler.transform(
                    new_df.loc[:, df.columns.get_level_values(0) != "FCS"])

        # This takes care of the case we want to scale and difference the target.  This way
        # scaling is only applied if the differing has happened and avoids useless scaling
        # on a target that is already bounded betweem 0 and 1.
        if self.scaling and self.differencing:
            if self.fcs_scaler is None:
                self.fcs_scaler = MinMaxScaler(feature_range=(0, 1))
                new_df.loc[:, new_df.columns.get_level_values(0) == "FCS"] = self.fcs_scaler.fit_transform(
                    new_df.loc[:, df.columns.get_level_values(0) == "FCS"])
            else:
                new_df.loc[:, new_df.columns.get_level_values(0) == "FCS"] = self.fcs_scaler.transform(
                    new_df.loc[:, df.columns.get_level_values(0) == "FCS"])
            target = self.fcs_scaler.transform(target)
        sequences = np.array(new_df) # transorming dataframe to np array

        # its here in cases that the CNN is applied only to one admin1 or to dummy data.
        # this adds a dimension in order to comply with the requiements of  keras.
        if len(target.shape) == 1:
            target = np.expand_dims(target, axis=1)

        # Prepares the Feature / Target Lists
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            input_data.append(sequences[i:end_ix, :])

            # This is required by the CNN that only reads 1d
            if self.flatten:
                output_data.append(target[end_ix:out_end_ix, :].flatten())
            else:
                output_data.append(target[end_ix:out_end_ix, :])

            # Storing the last value of the time series to rebuild it
            # from the outputs. Not for random number generation.
            if self.differencing:
                seed_values.append(original_target[end_ix])

        if self.differencing and set_seed:
            self.seed_value = np.array(seed_values)[0]

        return np.array(input_data), np.array(output_data)

    def predict(self, xinput, seed_value):
        pred = []
        pred0 = self.model.predict(xinput, verbose=False)
        pred0 = pred0.reshape(xinput.shape[0], self.n_steps_out, self.n_output)

        if self.differencing:
            for n in range(len(pred0)):
                pred1 = self.fcs_scaler.inverse_transform(pred0[n])
                pred2 = np.cumsum(pred1, axis=0) + seed_value

                pred.append(pred2)

            return np.array(pred)

        else:
            return pred0

    def evaluate(self, xinput, target, seed_value):
        if self.differencing:
            pred = self.predict(xinput, seed_value)
            target_reshape = target.reshape(xinput.shape[0], self.n_steps_out, self.n_output)
            target0 = []
            for n in range(len(target)):
                target1 = np.cumsum(self.fcs_scaler.inverse_transform(target_reshape[n]),axis=0) + seed_value
                target0.append(target1)
            target0 = np.array(target0)

            MSE = np.mean((pred - target0)**2)

            return MSE

        else:
            return self.model.evaluate(xinput, target)

    def get_final_epoch(self):
        hist = self.history.history['loss']
        n_epochs_best = np.argmin(hist)+1

        return n_epochs_best


def smooth_data(data, delta_t = 10, leave_out_columns = None):
    """
    :param data:
    :param delta_t:
    :return:
    """
    if leave_out_columns is None:
        leave_out_columns=[]

    new_data = pd.DataFrame(index=data.index)
    for col in data.columns:
        if col in leave_out_columns:
            new_data[col] = np.array(data[col])
        else:
            new_data[col] = smooth_past_data(np.array(data[col]), delta_t=delta_t)

    return new_data
