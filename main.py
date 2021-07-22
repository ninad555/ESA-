# import pandas as pd
# import numpy as np
import pandas as pd
from DataPreprocessing import get_data, load_and_save
from modeltraining import ModelTraining

import glob
# import numpy as np
#

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.array(np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1))
    minmax = 1 - np.mean(mins / maxs)  # minmax

    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse,
             'corr': corr, 'minmax': minmax})



#
co_data = get_data("sensors_wise_data/co.csv")
print(co_data.co)
training = ModelTraining()
model = training.create_save_load_model(data=co_data, model_name="co_model")
print("Model Created")
fs = training.FutureForecasting(model=model, periods=4320, freq="Min")
print("\n The forecasted Data is ")
print(fs)
acc = training.accuracy_metrics(model=model, freq="Min", actual_df=co_data["y"])
print(acc)
acc_df = pd.DataFrame(index=acc.keys())
acc_df["Co_Data"] = acc.values()
# print(acc_df)
plots = training.plot_predictions(co_data, fs, "co")

# print("saved:", fs.to_csv("Prediction service/Forecast/co_forecast.csv"))
# humidity_data = get_data("sensors_wise_data/humidity.csv")
# temp_data = get_data("sensors_wise_data/temp.csv")
# load_and_save("sensors_wise_data/co.csv", "Prediction service/co.csv")
# load_and_save("sensors_wise_data/humidity.csv", "Prediction service/humidity.csv")
# load_and_save("sensors_wise_data/temp.csv", "Prediction service/temp.csv")
#
# print(len(co_data))
# print(len(humidity_data))
# print(len(temp_data))
# print(temp_data.head())
# # #
# path = "sensors_wise_data"
# # #path = r"C:\Users\kadam\Documents\Data Science\Projects\IoT_Forecasting\sensors_wise_data"
# all_files = glob.glob(path + "/*.csv")
# #
# li = []
# acc_df = pd.DataFrame(index=['mape', 'me', 'mae',
#                      'mpe', 'rmse',
#                      'corr', 'minmax'])
# for filename in all_files:
#
#     data = get_data(filename)
#     name = filename[18:-4]
#     #save_data = load_and_save(filename, "Prediction service/" + name)
#     li.append(name)
#     training = ModelTraining()
#     model = training.create_save_load_model(data=data, model_name=name)
#     print("Model Created")
#     fs = training.FutureForecasting(model=model, periods=4320, freq="Min")
#     print("\n Forecasted Data is Saved")
#    #print(fs.to_csv("Prediction service/Forecast/{}_forecast.csv".format(name)))
#     accuracy = training.accuracy_metrics(model, freq="Min", actual_df=data["y"])
#     acc_df[name] = accuracy.values()
#     plots = training.plot_predictions(data, fs, name)
#
# print(acc_df)
# print(li)

# cols  = ["temp_" + co_data.columns[1:]]
# print(cols)





