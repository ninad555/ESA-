
import pandas as pd
from DataPreprocessing import get_data, load_and_save
from modeltraining import ModelTraining
import glob
import numpy as np
path = "data/sensors_wise_data"
#
all_files = glob.glob(path + "/*.csv")
#
acc_df = pd.DataFrame(index=['mape', 'me', 'mae',
                   'mpe', 'rmse',
                      'corr', 'minmax'])

for filename in all_files:
    # load data
    data = get_data(filename)
    name = filename[23:-4]

    # Model Training
    training = ModelTraining()
    model = training.create_save_load_model(data=data, model_name=name)
    print("Model Created")

    # Model Forecasting
    fs = training.FutureForecasting(model=model, periods=4320, freq="Min")

    # save the forecasted data to Prediction service/Forecast/ folder
    fs.to_csv("Prediction service/Forecast/{}_forecast.csv".format(name))
    print("\n Forecasted Data is Saved")

    # accuracy metrics
    accuracy = training.accuracy_metrics(model, freq="Min", actual_df=data["y"])
    acc_df[name] = accuracy.values()

    # Creating plots and saving it to Plots Folder
    plots = training.plot_predictions(data, fs, name)

# Save the accuracy_metrics
acc_df.to_csv("Prediction service/Accuracy_metrics.csv")
