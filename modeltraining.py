from fbprophet import Prophet
import pickle
import numpy as np
import plotly as py
import os
import plotly.graph_objects as go
import json
# from application_logging import logger

class ModelTraining:
    """
        This class will train the model from pre-processed data.

        Written By: Ninad y
        Version: 1.0
        Revisions: None

    """
    def __init__(self):
        print("ModelTraining..... ")
        # self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
        # self.log_writer = logger.App_Logger()
        self.path = 'data/sensors_wise_data/co.csv'  #path

    def reformat(self, df):
        """
                         Method Name: reformat
                         Description: This function changes the columns names as per model requirement for a given dataframe.
                         Output: data frame
                         On Failure: Exception

                         Written By: Ninad y
                         Version: 1.0
                         Revisions: None

        """
        self.df = df
        try:
            self.df.columns = ["ds", "y"]
            return df
        except Exception as e:
            print("Error in reformat function as {}".format(e))

    def create_save_load_model(self, data, model_name):
        """
                             Method Name: create_save_load_model
                             Description: This function creates required model for a given data save it to the
                                          assigned location and then load the model for further processing
                             Output: loaded model
                             On Failure: Exception

                             Written By: Ninad
                             Version: 1.0
                             Revisions: None

        """
        self.data = data
        self.model_name = model_name
        try:
            model = Prophet(interval_width=0.95, daily_seasonality=True)
            series = self.reformat(self.data)
            model_fit = model.fit(series)
            # save the model to disk
            filepath = 'Prediction service/models/{}.pkl'.format(self.model_name)
            pickle.dump(model, open(filepath, 'wb'))
            # load the model from disk
            loaded_model = pickle.load(open(filepath, 'rb'))
            return loaded_model
        except Exception as e:
            print("Error in Creating or Loading model as {}".format(e))


    def FutureForecasting(self, model, periods, freq):
        """
                                 Method Name: FutureForecasting
                                 Description: This function does the forecasting of given model for a required period
                                              of time.
                                 Output: Dataframe
                                 On Failure: Exception

                                 Written By: Ninad
                                 Version: 1.0
                                 Revisions: None

        """
        self.model = model
        self.periods = periods
        self.freq = freq
        try:
            forecast = self.model.make_future_dataframe(self.periods, self.freq)
            prediction = model.predict(forecast)
            future_df = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            return future_df
        except Exception as e:
            print("Error in FutureForecasting as {}".format(e))

    def accuracy_metrics(self, model,freq, actual_df):
        """
                                                 Method Name: accuracy_metrics
                                                 Description: This function gives accuracy metrics which involves MAPE,
                                                 ME, MAE, MPE, RMSE, Corr, Min, Max, Minmax
                                                 Output: Dict of accuracy metrics
                                                 On Failure: Exception

                                                 Written By: Ninad
                                                 Version: 1.0
                                                 Revisions: None

        """
        self.model = model
        self.actual_df = actual_df
        self.freq = freq
        try:
            future_forecast = self.FutureForecasting(self.model, periods=0, freq= self.freq)
            forecast_df = future_forecast["yhat"]
            mape = np.mean(np.abs(forecast_df - self.actual_df) / np.abs(self.actual_df))  # MAPE
            me = np.mean(forecast_df - self.actual_df)  # ME
            mae = np.mean(np.abs(forecast_df - self.actual_df))  # MAE
            mpe = np.mean((forecast_df - self.actual_df) / self.actual_df)  # MPE
            rmse = np.mean((forecast_df - self.actual_df) ** 2) ** .5  # RMSE
            corr = np.corrcoef(forecast_df, self.actual_df)[0, 1]  # corr
            mins = np.amin(np.hstack([forecast_df[:, None],
                                      self.actual_df[:, None]]), axis=1)
            maxs = np.amax(np.hstack([forecast_df[:, None],
                                      self.actual_df[:, None]]), axis=1)
            minmax = 1 - np.mean(mins / maxs)  # minmax

            return ({'mape': mape, 'me': me, 'mae': mae,
                     'mpe': mpe, 'rmse': rmse,
                     'corr': corr, 'minmax': minmax})
        except Exception as e:
            print("Error in accuracy metrics as {}".format(e))





    def plot_predictions(self,df,forecast,var):
        """
                                         Method Name: plot_predictions
                                         Description: This function plots the forecasted data Vs raw data
                                         Output: graphJSON
                                         On Failure: Exception

                                         Written By: Ninad
                                         Version: 1.0
                                         Revisions: None

        """
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=df['ds'], y=df["y"],
                                 mode='lines+markers',
                                 name='actual test '+var))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast["yhat"],
                                 mode='lines+markers',
                                 name='Forecasted test'+var))

        fig.update_layout(
            title="Actual Vs Forecasted test "+var,
            xaxis_title="Days ",
            yaxis_title="forecasted "+var,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="RebeccaPurple"
            )
        )
        if not os.path.exists("Prediction service/images"):
            os.mkdir("Prediction service/images")
        fig.write_image("Prediction service/images/{}_fig1.jpeg".format(var))
