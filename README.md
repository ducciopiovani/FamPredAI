# FamPredAI
Reservoir Computing for prediction of timeseries of food insecurity.

## Installation process and dependencies
Running Reservoir Computing 
The current repo doesn't require any installation. It can be run directly after installing the required packages using:
```commandline
python -m pip install -r requirements.txt
```
This assumes the version of python used is 3.9. Dependencies can be installed in a virtual/conda environment.

Furthermore, you will need to clone and install the `rescomp` package by running:
```commandline
git clone https://github.com/GLSRC/rescomp.git
cd rescomp
pip install .
```
Complete instructions can be found in [the docs for the package](https://glsrc.github.io/rescomp/installation.html#installation-instructions).
Running Keras Based model (CNN, LSTM)
The rescomp librabry is incompatible with keras because of the required version for nunmpy. To run the deep learning algorithms follow the instructions:
```commandline
python -m pip install -r requirements_deep_learning.txt
```
Once again this assumes the version of python used is 3.9. Dependencies can be installed in a virtual/conda environment.

## Data
Data to train the models can be found for each country in `data/<country>/full_timeseries_daily.csv`.
The [Model class](./model.py) is set up to read from these files when loading the training data.
Therefore, shall the data need to be updated, we suggest keeping the same name and format of the current file.

## Generating forecasts
Reservoir Computing 
To generate forecasts for a chosen country and model, just pass the following arguments to the function `forecast` in [reservoir_computing.py](./reservoir_computing.py):
- country: name of the chosen country;
- first_forecast: first day to be forecasted (NB: can be at max the day following last available day for target data);
- constants: list of discrete/categorical secondary data to be used to train the model;
- variables: list of continuous secondary data to be used to train the model;
- hyperparameters: dict of hyperparameters needed for the model;
- forecast_window: number of days to forecast for.
CNN and LSTM
Follow the steps in the `forecast_from_file` function found in cnn.py and lstm.py

A dictionary with example arguments for Haiti can be imported from the file [parameters.py](./parameters.py), to be passed to this function.

To generate forecasts for monthly test splits, using the best parameters found for the chosen model at each of these,
run the function `forecast_from_file` in the `reservoir_computing.py`, `cnn.py` and `lstm.py` 
- country: name of the chosen country;
- model: name of the model, i.e. `RC`, to be used to read from the correct file of parameters in folder [best_hyperparameters](./best_hyperparameters).
