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
    
def remove_birthyear(df, cutoff):
    """
        Removes all instances with Birth Year < cutoff and returns the dataframe.
    """
    return df.loc[df["Year_Birth"]>cutoff, :]

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
#    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
#    df[column] = imp.fit_transform(df[column])
    return df

def impute_income_KNN(df):
    """
    imputes the missing income values by using a KNN approach, it might yield closer results instead of using the median imputer
    """
    dataframe = df.copy()
    dataframe_c = dataframe.dropna().select_dtypes(include=["number"]).drop(["Response"], axis = 1)
    dataframe_i = dataframe[pd.isnull(dataframe).any(axis=1)].select_dtypes(include=["number"]).drop(["Response"], axis = 1)
    clf = KNeighborsClassifier(3, weights='uniform', metric = 'euclidean')
    trained_model = clf.fit(dataframe_c.drop(["Income"],axis=1), dataframe_c.loc[:,'Income'])
    imputed_values = trained_model.predict(dataframe_i.drop(["Income"], axis=1))
    #print(imputed_values)
    dataframe_i["Income"] = imputed_values
    dataframe_new = pd.concat([dataframe_i, dataframe_c])
    dataframe_new = dataframe_new.sort_index()
    dataframe["Income"] = dataframe_new["Income"]
    return dataframe


def replace_income(df):
    """
    Replaces the 600k income guy with the median.
    """
    df.loc[df["Income"]>600000, "Income"] = df.median()["Income"]
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

def marital_others(df):
    df.loc[df["Marital_Status"] == "YOLO", "Marital_Status"] = "Others"
    df.loc[df["Marital_Status"] == "Absurd", "Marital_Status"] = "Others"
    return df

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

## Mortens Preprocessing Pipeline
##  (Feel Free to use it, if you change it, please let me know)

def morten_preprocessing_pipeline(df):
    """
    One-Version of a Preprocessing Pipeline. Decisions are justified in Data_CLeaning.ipynb.
    """
    df = remove_birthyear(df, 1940)
    df = missing_imputer(df, "Income", "median")
    df = outlier_cutoff(df, "MntSweetProducts", 210)
    df = outlier_cutoff(df, "MntMeatProducts", 1250)
    df = outlier_cutoff(df, "MntGoldProds", 250)
    df = outlier_value_imputer(df, "NumWebPurchases", 11, 11)
    df = outlier_value_imputer(df, "NumCatalogPurchases", 11, 11)
    df = outlier_value_imputer(df, "NumWebVisitsMonth", 9, 9)
    df = anomalies_treatment(df, "Marital_Status", ["YOLO", "Absurd"])
    df = encode_education(df)
    df = feature_engineering.partner_binary(df)
    df = one_hot_encoding(df, columns=["Marital_Status"])
    df = encode_days_as_costumer(df)
    df = feature_engineering.drop_useless_columns(df)
    df = feature_engineering.responsiveness_share(df)
    del df["Complain"]
    return df

## Joris Preprocessing Pipeline
##  (Feel Free to use it, if you change it, please let me know)

def joris_preprocessing_pipeline(df):
    df = impute_income_KNN(df)
    df = feature_engineering.partner_binary(df)
    df = feature_engineering.income_housemember(df)
    df = anomalies_treatment(df, "Marital_Status", ["YOLO", "Absurd"])
    df = one_hot_encoding(df,columns = ["Marital_Status"])
    df = one_hot_encoding(df,columns = ["Education"])
    df = encode_days_as_costumer(df)
    df = feature_engineering.drop_useless_columns(df)
    df = replace_income(df)
    df = feature_engineering.responsiveness_share(df)
    df = feature_engineering.ave_purchase(df)
    df = feature_engineering.income_share(df)
    return df

def bin_it_preprocessing_pipeline(df):
    df = impute_income_KNN(df)
    df = feature_engineering.partner_binary(df)
    df = feature_engineering.income_housemember(df)
    df = anomalies_treatment(df, "Marital_Status", ["YOLO", "Absurd"])
    df = one_hot_encoding(df,columns = ["Marital_Status"])
    df = one_hot_encoding(df,columns = ["Education"])
    df = encode_days_as_costumer(df)
    df = feature_engineering.drop_useless_columns(df)
    df = replace_income(df)
    df = feature_engineering.responsiveness_share(df)
    df = feature_engineering.ave_purchase(df)
    df = feature_engineering.income_share(df)
    df = Binning_Features(df, "Income", n_bins=5)
    df = Binning_Features(df, "MntWines", n_bins=5)
    df = Binning_Features(df, "MntFruits", n_bins=5)
    df = Binning_Features(df, "MntMeatProducts", n_bins=5)
    df = Binning_Features(df, "MntFishProducts", n_bins=5)
    df = Binning_Features(df, "MntSweetProducts", n_bins=5)
    df = Binning_Features(df, "MntGoldProds", n_bins=5)
    return df

def simple_pipeline(df):
    # delete unwanted columns
    df = feature_engineering.drop_useless_columns(df)
    # treatment weird values
    df = marital_others(df)
    df = encode_days_as_costumer(df)
    # check for nan
    df = df.dropna()
    # look at extreme values
    df = one_hot_encoding(df, columns=["Marital_Status"])
    df = one_hot_encoding(df, columns=["Education"])
    # feature engineering

    return df


def chop_off(df):
    # delete unwanted columns
    df = feature_engineering.drop_useless_columns(df)

    # check for nan
    df = df.dropna()
    df = encode_days_as_costumer(df)
    # treatment weird values
    df = anomalies_treatment(df, "Marital_Status", ["YOLO", "Absurd"])
    df = outlier_IQR(df, columns=["Year_Birth", "Income", 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
       'NumWebVisitsMonth', 'Recency'])

    #encoding
    df = one_hot_encoding(df, columns=["Marital_Status"])
    df = one_hot_encoding(df, columns=["Education"])

    # cutoff based on chi-squared test

    return df

def pca_pipeline(df):
    # do fancy pca stuff here
     return df


def box_cox_pipeline(df):
    
    return df

def small_pipeline(df):
    df = impute_income_KNN(df)
    df = df.drop(["Kidhome","Teenhome"], axis=1)
    df = feature_engineering.drop_useless_columns(df)
    df = encode_days_as_costumer(df)

    df = anomalies_treatment(df, "Marital_Status", ["YOLO", "Absurd"])
    df = one_hot_encoding(df,columns = ["Marital_Status"])
    df = one_hot_encoding(df,columns = ["Education"])
    
    return df
                             
def feature_engineered(df):
    # use only feature engineered stuff
    df = feature_engineering.drop_useless_columns(df)
    df = encode_days_as_costumer(df)
    df = feature_engineering.partner_binary(df)
    df = feature_engineering.responsiveness_share(df)
    df = feature_engineering.alcoholic(df)
    df = feature_engineering.income_housemember(df)
    df = feature_engineering.kids_home(df)
    df = feature_engineering.income_share(df)
    df = feature_engineering.veggie(df)
    df = feature_engineering.phd(df)
    df = feature_engineering.ave_purchase(df)
    df =feature_engineering.tutti_frutti(df)
    df = df.drop(columns=["Year_Birth", "Income", 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                                  'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                                  'NumWebVisitsMonth', 'Dt_Customer', 'Recency', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'AcceptedCmp3',
       'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
       'Complain'], axis=1)
    df = outlier_IQR(df, columns=['income_housemember', 'income_share', 'ave_purchase'])
    
    return df


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

def Binning_Features(df, feature="Income", n_bins=10, strategy="quantile", cont_tab=False):
    target = "Response"

    bindisc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    mnt_bin = bindisc.fit_transform(df[feature].values[:, np.newaxis])
    mnt_bin = pd.Series(mnt_bin[:, 0], index=df.index)
    if cont_tab == True:
        obs_cont_tab = pd.crosstab(mnt_bin, df[target])
        print(obs_cont_tab) 
    df[feature] = mnt_bin
    return df