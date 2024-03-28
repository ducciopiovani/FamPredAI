from datetime import datetime

parameters = {
    "Haiti": {
        "hyperparameters": {
            "n_dim": 1000,
            "n_rad": 1.2,
            "n_avg_deg": 8,
            "reg_param": 10,
            "smoothing": 7,
            "w_in_scale": 0.1,
            "differencing": True,
            "w_out_fit_flag": "linear_and_square_r",
            "train_sync_steps": 50
        },
        "variables": ["rCSI"],
        "constants": ["rainfall_ndvi_seasonality"],
        "forecast_window": 30,
        "first_forecast": datetime(2024,3,1),
        "target": "FCS"
    }
}

