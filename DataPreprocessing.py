import os
import pandas as pd


def get_data(file_path):
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    return df


def load_and_save(input_path, output_path):
    df = get_data(input_path)
    df.to_csv(output_path, sep=",", index=False)

