import pandas as pd
from sklearn.preprocessing import  OneHotEncoder, LabelEncoder

def convert_to_boolean(df):
    '''
        Converts all possible columns to type np.bool
    '''
    dataframe = df.copy()
    for variable in dataframe:
        if set(dataframe[variable].unique()) == set([0,1]):
            dataframe[variable] = dataframe[variable].astype(bool)
    return dataframe

def one_hot_encoding(df, columns):
    '''
        Performs on hot encoding to the selected columns
    '''
    dataframe = df.copy()
    dataframe = pd.get_dummies(dataframe, columns=columns)
    return dataframe

def ordinal_enconding(df, columns):
    '''
        Performs ordinal encoding to the selected columns
    '''
    dataframe = df.copy()
    # integer encode
    for column in columns:
        label_encoder = LabelEncoder()
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
    return dataframe

def encode_education(df):
    '''
        Encodes the education variable with a hierarchical order
    '''
    dataframe = df.copy()
    education_dict = {
        'Basic' : 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    dataframe["Education"] = dataframe["Education"].apply(lambda edu: education_dict[edu])
    return dataframe

def encode_days_as_costumer(df):
    '''
        Transforms date costumer column from a date to the number of days to the most recent costumer
    '''
    dataframe = df.copy()
    dataframe.Dt_Customer = abs(
        pd.to_datetime(dataframe.Dt_Customer) - pd.to_datetime(dataframe.Dt_Customer).max()
    ).dt.days
    return dataframe

def to_dtype_object(df):
    '''
        Changes columns with 0 or 1 values to an object
    '''
    dataframe = df.copy()
    #df[columns] = df[columns].astype("object")
    dataframe["AcceptedCmp3"] = dataframe["AcceptedCmp3"].astype("object")
    dataframe["AcceptedCmp1"] = dataframe["AcceptedCmp1"].astype("object")
    dataframe["AcceptedCmp2"] = dataframe["AcceptedCmp2"].astype("object")
    dataframe["AcceptedCmp4"] = dataframe["AcceptedCmp4"].astype("object")
    dataframe["AcceptedCmp5"] = dataframe["AcceptedCmp5"].astype("object")
    dataframe["Complain"] = dataframe["Complain"].astype("object")

    return dataframe


