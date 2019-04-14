import pandas as pd

def remove_non_numerical(df):
    '''Remove non-numerical columns'''
    return df._get_numeric_data()