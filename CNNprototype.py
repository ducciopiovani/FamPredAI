import warnings

import numpy as np
import pandas as pd
import time

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras_model import KerasModel


class CNN_model(KerasModel):
    def __init__(self,
                 epochs,
                 learning_rate,
                 n_steps_in,
                 n_steps_out,
                 early_stopping,
                 filters,
                 kernel_size,
                 pool_size,
                 dense_units,
                 layers,
                 differencing,
                 smoothing=10,
                 scaling=True):
        super().__init__(epochs=epochs, learning_rate=learning_rate, n_steps_in=n_steps_in, n_steps_out=n_steps_out,
                       early_stopping=early_stopping, smoothing=smoothing, scaling=scaling, differencing=differencing)

        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.layers=layers
        self.differencing=differencing

        self.n_output_internal = None
        self.n_output = None

        self.flatten = True

    def define_model(self):
        np.random.seed(0)

        model = Sequential()

        # Add Convolutional layers
        model.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='same', input_shape=(self.n_steps_in, self.n_features)))
        model.add(MaxPooling1D(pool_size=self.pool_size, padding="same"))

        if self.layers >=2:
            for n in range(self.layers-1):
                model.add(Conv1D(filters=self.filters, padding='same', kernel_size=self.kernel_size, activation='relu'))
                model.add(MaxPooling1D(pool_size=self.pool_size, padding="same"))

        # Flatten the output for the fully connected layers
        model.add(Flatten())

        # Add Dense layers for regression (output layer with linear activation for forecasting)
        model.add(Dense(self.dense_units, activation='relu'))
        model.add(Dense(self.n_output_internal))

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')  # Use mean squared error for regression tasks

        self.model = model

    def set_n_output(self):
        training_target = self.training_data[1]
        training_input = self.training_data[0]
        n_output = training_target.shape[1]
        self.n_output = int(n_output / self.n_steps_out)

        self.n_features = training_input.shape[2]
        self.n_output_internal = n_output


def plot_prediction(model, x_input, target):
    # demonstrate prediction

    # x_input = test_input_data[30]
    # target = test_target[30,:,:]
    x_input = np.expand_dims(x_input, axis=0)

    yhat = model.predict(x_input, verbose=True)

    for i in range(yhat.shape[2]):
        plt.subplot(5, 5, i + 1)
        plt.plot(range(90), x_input[0, :, i], color="blue")
        plt.plot(range(90, 120), yhat[0, :, i], label="pred", color="red")
        plt.plot(range(90, 120), target[:, i], label="real", color="blue")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    data = pd.read_csv("../DataTimeSeries/Yemen/grid_search/full_timeseries_daily.csv", header=[0, 1], index_col=0)[
           :"2023-05-01"]

    data = data[["FCS", "Ramadan"]]
    data.index = pd.DatetimeIndex(data.index)
    for n, col in enumerate(data.columns):
        data[col] = n

    split_date = pd.Timestamp("2022-06-01")
    epochs = 20
    early_stopping = True
    n_steps_in = 3
    n_steps_out = 3
    target_column = "FCS"
    learning_rate = 0.001
    dense_units = 50
    kernel_size=30
    filters=32
    pool_size=2
    layers=1

    cnn = CNN_model(kernel_size=kernel_size, filters=filters,pool_size=pool_size, dense_units=dense_units, layers=layers, epochs=epochs, early_stopping=early_stopping, n_steps_in=n_steps_in,
                    n_steps_out=n_steps_out, learning_rate=learning_rate)

    cnn.test_model(data=data, split_date=split_date)

    history = cnn.history
    test_input = cnn.validation_data[0]
    test_target = cnn.validation_data[1]

    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.ylim([0, 0.5])

    plt.show()

    plot_prediction(cnn.model, x_input=test_input[30], target=test_target[30, :, :])

    print("Done")
