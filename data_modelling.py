import pandas as pd
import fbprophet
import pickle
from fbprophet import Prophet
import matplotlib.pyplot as plt
#%matplotlib inline

import plotly.graph_objects as go
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

from application_logging import logger


class DataModelling:

    def __init__(self,path):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
        self.path = 'data/sensors_wise_data/co.csv'  #path

    def read_inputs(self):
        self.df = pd.read_excel(self.path, parse_dates=["Datetime"])
        df_dict ={}
        self.df_co = pd.DataFrame(self.df, columns=['Datetime', 'co'])
        self.df_humidity = pd.DataFrame(self.df, columns=['Datetime', 'humidity'])
        self.df_temp = pd.DataFrame(self.df, columns=['Datetime', 'temp'])
        self.df_temp.columns = ['ds', 'y']
        self.df_co.columns = ['ds', 'y']
        self.df_humidity.columns = ['ds', 'y']
        df_dict['temp'] =self.df_temp
        df_dict['co'] = self.df_co
        df_dict['humidity'] =self.df_humidity
        #df_list =[self.df_temp,self.df_co,self.df_humidity]
        return df_dict

    def split_dataset(self,df):
        # splitting the data set for training and validation
        train = df[(df['ds'] >= '2020-07-12 00:01:00') & (df['ds'] <= '2020-07-16 00:01:00')]
        test = df[(df['ds'] > '2020-07-16 00:01:00')]
    def Fit_Model(self,df):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        # trying to fit the model with confidence interval of 95% on test temperature data
        m_var = Prophet(interval_width=0.95, daily_seasonality=True)
        model = m_var.fit(df)
        return model,m_var

    def predict_future(self,m_var):
        # making future predictions for test dates starting from 2020-07-16 to 2020-07-20
        future = m_var.make_future_dataframe(periods=4320,freq='Min')
        forecast = m_var.predict(future)
        return forecast

    def plot_predictions(self,df,forecast,var):
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

        fig.show()


    def performance_metrics(self,m_var):
        # Prophet includes functionality for time series cross validation to measure forecast error using historical data

        cv_results2 = cross_validation(model=m_var, initial=pd.to_timedelta(1152, unit='Min'),
                                       horizon=pd.to_timedelta(438, unit='Min'))
        df_p = performance_metrics(cv_results2)
        return df_p

    def runModel(self):
        df_dict = self.read_inputs()
        for var,df in df_dict.items():
            #self.split_dataset(df)
            model,m_var =self.Fit_Model(df)
            with open(var+'model.pkl','wb') as filepath:
                pickle.dump(model,filepath)
            forecast = self.predict_future(m_var)
            self.plot_predictions(forecast,var)
            df_p = self.performance_metrics(m_var)
            df_p.to_csv(var+'_modelperfomance_metrics.csv')


if __name__ == "__main__":
    path = 'data_device1.xlsx'
    DM_obj = DataModelling(path)
    DM_obj.runModel()
