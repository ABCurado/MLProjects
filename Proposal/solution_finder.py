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
import data_visualization
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import feature_engineering
from ML_algorithms import *
import pandas as pd
from seaborn import countplot
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek


import sys
import os
from subprocess import call

file_name= "LogFiles/" + "results_"+ str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) +"_log.csv"

header_string = "Algorithm,Parameters,Preprocessing Pipeline,Scaling,Sampling,time,precision,recall,result_profit"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")


models = [
    ("MLPClassifier_10_layers", MLPClassifier(hidden_layer_sizes=(10), solver="lbfgs", max_iter=1000, random_state=42)),
    ("MLPClassifier_5_layers", MLPClassifier(hidden_layer_sizes=(5), solver="lbfgs", max_iter=1000, random_state=42)),
    ("KerasNN_not_fitted", KerasNN_not_fitted()),
    ("GaussianNB", GaussianNB()),
    ("MultinomialNB", MultinomialNB()),
    ("ComplementNB", ComplementNB()),
    ("SVC", SVC()),
    ("LinearSVC", LinearSVC()),
    ("LogisticRegression", LogisticRegression()),
    ("SGDClassifier", SGDClassifier()),
    ("KNeighborsClassifier", KNeighborsClassifier()),
    ("DecisionTreeClassifier", DecisionTreeClassifier(criterion="gini", class_weight=None)),
    ("XGBClassifier", XGBClassifier(colsample_by_tree=0.1,
                                   learning_rate=0.89,
                                   max_depth=8,
                                   n_estimators=10000,
                                   eval_metric="auc",
                                   n_jobs=1, silent=0, verbose=0)),
    ("MLPClassifier", MLPClassifier(hidden_layer_sizes=(10), solver="lbfgs", max_iter=1000, random_state=42)),
    ("LinearRegression", LinearRegression()),
    ("KerasNN_3layers" , KerasNN_not_fitted(n_layers=3, init="he_normal")),
    ("KerasNN_6layers" , KerasNN_not_fitted(n_neurons=6, init="he_normal")),
    ("KerasNN_9layers" , KerasNN_not_fitted(n_neurons=9, init="he_normal")),
    ("KerasNN_12layers" , KerasNN_not_fitted(n_neurons=12,init="he_normal"))
]

scalers = [
    ("StandardScaler", StandardScaler()),
    #("RobustScaler", RobustScaler()),
    #("MinMaxScaler", MinMaxScaler()),
    ("None", None)
]

samplers =  [
    #("RandomOverSampler", RandomOverSampler(random_state=42, ratio=0.5)),
    #("TomekLinks", TomekLinks(random_state=42)),
    #("EditedNN", EditedNearestNeighbours(random_state=42, n_neighbors=3)),
    #("SMOTE", SMOTE(random_state=42, ratio=0.5)),
    #("SMOTETomek",SMOTETomek(random_state=42, ratio=0.8))
    ("None", None)
    
]

pre_processing_pipelines = [
    ("Joris_Pipeline", preprocessing.joris_preprocessing_pipeline),
    ("Morten_Pipeline", preprocessing.morten_preprocessing_pipeline),
    ("Bin it!", preprocessing.bin_it_preprocessing_pipeline)

]
seed = [1]


def algo_run(model, pre_processing_pipeline, scaler, sampler, seed):

    start_time = datetime.datetime.now()

    df = utils.get_dataset()
    df = pre_processing_pipeline[1](df)
    X, y = utils.X_y_split(df)
    y_predicted = utils.cross_validation_average_results(
        model[1], X, y, n_splits=5,
        scaler=scaler[1],
        sampling_technique=sampler[1]
    )
    threshold = utils.max_threshold(y_predicted, y, threshold_range=(0.2, 0.6), iterations=1000, visualization=False)
    y_pred = utils.predict_with_threshold(y_predicted, threshold)
    result = utils.profit_share(y_pred, y)
    precision = utils.calculate_precision_score(y_pred, y)
    recall = utils.calculate_recall_score(y_pred, y)
    time_elapsed = datetime.datetime.now() - start_time

    # Create result string
    result_string = ",".join(
        [model[0],
         pre_processing_pipeline[0], scaler[0], sampler[0], str(time_elapsed),str(precision),str(recall), str(result)
         ])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")
    # Output result to terminal
    print(model[0]+": "+str(result))
    if result > 0.6:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!yey!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")


if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[models,pre_processing_pipelines,scalers,samplers, seed]))

    core_count = multiprocessing.cpu_count()
    #print("All possible combinations generated:")
    #print(possible_values)
    print(len(possible_values))
    print("Number of cpu cores: "+str(core_count))
    print()
    print(header_string)

    ####### Magic appens here ########
    pool = multiprocessing.Pool(core_count-1)
    results = pool.starmap(algo_run, possible_values)
