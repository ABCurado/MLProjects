import pandas as pd
import numpy as np

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

def partner_binary(df):
    """
    This function creates a Partner column which is binary.
    Someone has a partner or not.
    """
    df["Partner"] = np.where(df["Marital_Status"].isin(["Together", "Married"]), 1, 0)
    return df

def responsiveness_share(df):
    """
    This function creates a new feature responsiveness_share. It is the sum of all Campaign responses divided by its number.
    """
    df["Responsiveness"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]].sum(axis=1) / 5
    return df