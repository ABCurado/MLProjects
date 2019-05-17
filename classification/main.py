import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils
import preprocessing
#import data_visualization
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.linear_model import *
import feature_engineering
from ML_algorithms import *
import pandas as pd
#from seaborn import countplot
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek


import sys
import os
from subprocess import call

# The model
model = KerasNN_not_fitted(n_layers=15, n_neurons=36,
                           optimizer="adam",init="uniform",loss="mean_squared_error",
                           r_dropout=0.2, input_dim=31)

#The scaler
scaler = RobustScaler()

# The sampler
sampler = RandomOverSampler(random_state=42, ratio=0.35)

# The pipeline
pipeline = preprocessing.morten_preprocessing_pipeline


df_train = utils.get_dataset()
X_test = pd.read_excel("data_files/unseen_students.xlsx")

# Hack to use the same pipeline since test and train were not equal
X_test["Z_Revenue"] = np.zeros
X_test["Z_CostContact"] = np.zeros

df_train = pipeline(df_train)
# Hack so the datasets have the same size
del df_train["Marital_Status_Alone"]
X_test = pipeline(X_test)

# Get the indexes that are not outliers
not_outlier_indexes = [index in X_test.index for index in range(0,7000)]

# Split train
X_train, y_train  = utils.X_y_split(df_train)

# Apply scaler to train and test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Apply oversample to train data
X_train, y_train = sampler.fit_resample(X_train, y_train)            

# Fit the model
model.fit(X_train, y_train, epochs=100, verbose=1)

# Predict results
final_pred = model.predict(X_test)

# Round to 1, 0
preditions = utils.predict_with_threshold(final_pred, 0.5)



final_df = pd.read_excel("data_files/unseen_students.xlsx")

# Set'all predictions to 0
final_df["Response"] = 0
# Set prediction values to current values
final_df["Response"].loc[not_outlier_indexes] = preditions

#export dataset
final_df.to_csv("data_files/labels_group_x.xlsx")

print("Final file outputed to data_files/labels_group_x.xlsx") 
