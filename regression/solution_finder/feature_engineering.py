import pandas as pd
import numpy as np

def remove_non_numerical(df):
    '''Remove non-numerical columns'''
    return df._get_numeric_data()

def drop_weird_cat(df, error_dictionary):
    dataframe = df.copy()

    for key, value in error_dictionary.items():
        dataframe = dataframe[dataframe[key] != value]
    return dataframe