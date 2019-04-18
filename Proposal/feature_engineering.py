import pandas as pd

def remove_non_numerical(df):
    '''Remove non-numerical columns'''
    return df._get_numeric_data()

def drop_useless_columns(df):
    '''Remove non-numerical columns'''
    dataframe = df.copy()
    del dataframe['ID']
    del dataframe['Z_Revenue']
    del dataframe['Z_CostContact']
    return dataframe

def drop_weird_cat(df, error_dictionary):
    dataframe = df.copy()

    for key, value in error_dictionary.items():
        dataframe = dataframe[dataframe[key] != value]
    return dataframe