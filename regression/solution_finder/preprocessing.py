import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
from math import sqrt
import feature_engineering
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from scipy import stats

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

def min_max_scale(df, columns=[]):
    scaler = MinMaxScaler()
    if columns == []:
        scaler.fit(df)
    else: 
        scaler.fit(df[columns])

    return scaler

def Min_Max_Train(X_train, X_test):    
    scaler = MinMaxScaler()
    # Only fit the training data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def missing_imputer(df, column, strategy= "median"):
    """
    Imputes the column of a dataframe based on the given strategy ("mean", "median").
    """
    if strategy == "median":
        value = df.median()[column]
    elif strategy == "mean":
        value = df.mean()[column]
    else:
        print("Chose a valid strategy!")

    df.loc[df[column].isna()==True, column] = value
    return df

def outlier_cutoff(df, column, upper_bound):
    """
    This method cuts off outliers smaller than the upper_bound.
    """
    return df.loc[df[column]<upper_bound,:]

def outlier_imputer(df, column, upper_bound, strategy="median"):
    """
    This method imputes outliers based on the given bound and by means of the chosen strategy.
    """
    df.loc[df[column] > upper_bound, column] = np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df[column] = imp.fit_transform(df[column])
    return df

def outlier_value_imputer(df, column, upper_bound, value):
    """
    This method imputes outliers based on the given bound and with a given value.
    """
    df.loc[df[column] > upper_bound, column] = value
    return df

def anomalies_treatment(df, column, anomalies):
    """
    Cuts off the anomalies given.
    """
    return df.loc[~df[column].isin(anomalies), :]


def outlier_IQR(df, columns=["Year_Birth","Income"]):
    """
    outlier deletetion using the IQR, you can choose the variables you want to delete the outliers for by selecting the columns. The default is Year_Birth & Income. Also you can change the quantile values.
    """
    dataframe = df.copy()
    Q1 = dataframe[columns].quantile(0.25)
    Q3 = dataframe[columns].quantile(0.75)
    IQR = Q3 - Q1
    #print(IQR)
    dataframe[columns] = dataframe[columns][~((dataframe < (Q1 - 2 * IQR)) |(dataframe > (Q3 + 2 * IQR))).any(axis=1)]
    dataframe = dataframe.dropna()
    print('Removing outliers using the IQR method with 2 quartiles would lead to a change of data size: ',(dataframe.shape[0] -df.shape[0]) /df.shape[0])
    return dataframe


def outlier_ZSCORE(df, columns=["Year_Birth", "Income"], threshold=3):
    """
    outlier deletion using the Zscore, you can choose which columns you want to apply it on and you can choose which threshold you want to use.
    """
    dataframe = df.copy()
    columns_zscore = []
    for i in dataframe[columns]:
        i_zscore = i + "_zscore"
        columns_zscore.append(str(i_zscore))

        dataframe[i_zscore] = (dataframe[i] - dataframe[i].mean()) / df[i].std(ddof=0)
    for i in dataframe[columns_zscore]:
        dataframe = dataframe[(dataframe[i] < threshold) & (dataframe[i] > -threshold)]
    dataframe = dataframe.drop(columns_zscore, axis=1)
    print('Removing outliers using the ZSCORES method with a threshold of 3 would lead to a change of data size: ',
          (dataframe.shape[0] - df.shape[0]) / df.shape[0])
    return dataframe


## Over and Undersampling Methods

def centroid_undersampling(X, f):
    """
       Implementation of CCMUT (cluster-centroid based Majority under-sampling technique)
       :param X: the df
       :param f: % of undersampling
       :return: X_f a dataframe with undersampled data for both labels
       """
    # get subset of X with label
    X_label = X.loc[X["Response"]==0, :]

    # initialize a new columns
    X_label["distance"] = 0

    # getting cluster-centroids
    cluster_centroid = np.sum(X_label, axis=0) / X_label.shape[0]

    # get the euclidean distance between each sample and the centroid -> sum it afterwards
    for index, row in X_label.iterrows():
        X_label.loc[index, "distance"] = sqrt(sum((cluster_centroid - X_label.loc[index, :]) ** 2))

    # select the highest distances and drop the distances
    X_label.sort_values(by=['distance'], ascending=False, inplace=True)
    X_label.drop("distance", axis=1, inplace=True)

    # concatenate all selected 0 labels with the 1 labels
    X_f = pd.concat([X_label.iloc[:int(f * X_label.shape[0]), :], X.loc[X["Response"]==1, :]])
    # shuffle the dataframe and reset the index
    X_f = X_f.sample(frac=1).reset_index(drop=True)
    return X_f

def random_oversampling(X, ratio, seed):
    """
    Implementation of random_oversampling.
    :param X: df
    :param ratio: ratio for oversampling
    :return: new oversampled dataframe
    """
    # get subset of class 1 -> to be oversampled
    X_label = X.loc[X["Response"]==1, :]
    # oversample randomly based on ratio
    X_label = X_label.sample(frac=ratio, replace=True, random_state=seed)
    # concat with other class
    X_f = pd.concat([X_label, X.loc[X["Response"]==0, :]])
    # shuffle the dataframe and reset the index
    X_f = X_f.sample(frac=1).reset_index(drop=True)
    return X_f


def SMOTE_oversampling(X, y):
    # input DataFrame
    # X →Independent Variable in DataFrame\
    # y →dependent Variable in Pandas DataFrame format
    sm = SMOTE()
    X, y = sm.fit_sample(X, y)
    return X, y


def ADASYN_oversampling(X, y):
    # input DataFrame
    # X →Independent Variable in DataFrame\
    # y →dependent Variable in Pandas DataFrame format
    sm = ADASYN()
    X, y = sm.fit_sample(X, y)
    return (X, y)

def Binning_Features(df, feature, n_bins=10, strategy="quantile", cont_tab=False, target = 'target'):
  
    bindisc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    mnt_bin = bindisc.fit_transform(df[feature].values[:, np.newaxis])
    mnt_bin = pd.Series(mnt_bin[:, 0], index=df.index)
    if cont_tab == True:
        obs_cont_tab = pd.crosstab(mnt_bin, df[target])
        print(obs_cont_tab) 
    df[feature] = mnt_bin
    return df