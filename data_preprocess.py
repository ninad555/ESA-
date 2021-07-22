import pandas as pd


class DataPreprocessing:
    """
            This class shall  be used to clean and transform the data before training.
            Written By: Akshada
            Version: 3.7.6
            Revisions: None
            """

    def __init__(self, path):
        self.path = path  # path

    def to_datetime(self):
        """
                                        Method Name: to_datetime
                                        Description: This method will convert a timestamp column in data into Date time
                                        and set it as index.
                                        Output: A Dataframe which has Datetime coulmn as index
                                        On Failure: Raise Exception
                                        Written By: Akshada
                                        Version: 3.7.6
                                        Revisions: None
        """

        print("Starting")
        data = pd.read_csv(self.path)
        df = pd.DataFrame(data)
        df["Datetime"] = pd.to_datetime(df['ts'], unit='s')
        df.set_index("Datetime", inplace=True)
        print("Data loaded and converted to datetime ")
        return df

    def group_by_func(self, df):
        """
                                                Method Name: group_by_func
                                                Description:This method will group the data by a given column and
                                                take all groups available and make a separate data frame for each group
                                                Output: A Dataframe based on particular group(i.e device)
                                                On Failure: Raise Exception
                                                Written By: Akshada
                                                Version: 3.7.6
                                                Revisions: None
                """

        self.df = df
        try:

            group = self.df.groupby("device")
            device_1 = group.get_group("00:0f:00:70:91:0a")
            print("Grouping done")
        except Exception as e:
            print("Grouping unsuccessfull" + e)

        return device_1

    def run_preprocessing(self):
        """
                                                Method Name: run_preprocessing
                                                Description:This method will run all methods and create csv files acc to the
                                                sensor data for device 1
                                                Output: CSV files for device_1 and its individual data files of different sensors
                                                On Failure: Raise Exception
                                                Written By: Akshada
                                                Version: 3.7.6
                                                Revisions: None
                """
        df = self.to_datetime()
        df_1 = self.group_by_func(df)
        try:
            df_co = pd.DataFrame(df_1, columns=['co'])
            df_humidity = pd.DataFrame(df_1, columns=['humidity'])
            df_temp = pd.DataFrame(df_1, columns=['temp'])
            df_1.to_csv("Device_1_data.csv")
            df_co.to_csv("co1_data.csv")
            df_humidity.to_csv("humidity1_data.csv")
            df_temp.to_csv("temp1_data.csv")
            print("Files created")
        except:
            print("Files not created")

if __name__ == "__main__":
    path = "iot_telemetry_data.csv"
    obj = DataPreprocessing(path)
    obj.run_preprocessing()
