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

import feature_engineering
from ML_algorithms import *
import pandas as pd
from seaborn import countplot
import numpy as np
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")

import sys
import os
from subprocess import call

file_name= "LogFiles/" + "results_antonio_log.csv"

header_string = "Algorithm,Parameters,Preprocessing Pipeline,Scaling,Sampling,time,result_profit"
with open(file_name, "a") as myfile:
    myfile.write(header_string + "\n")


models = [
    ("MLPClassifier_10_layers", MLPClassifier(hidden_layer_sizes=(10), solver="lbfgs", max_iter=1000, random_state=42))

]

scalers = [
    ("MinMaxScaler", MinMaxScaler())
]

samplers =  [
    ("RandomOverSampler", RandomOverSampler(random_state=42, ratio=0.5))
]

pre_processing_pipelines = [
    ("Joris_Pipeline", preprocessing.joris_preprocessing_pipeline)
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

    time_elapsed = datetime.datetime.now() - start_time
    # Create result string
    result_string = ",".join(
        [model[0],
         pre_processing_pipeline[0], scaler[0], sampler[0], str(time_elapsed), str(result)
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
    print("All possible combinations generated:")
    print(possible_values)
    print(len(possible_values))
    print("Number of cpu cores: "+str(core_count))
    print()
    print(header_string)

    ####### Magic appens here ########
    pool = multiprocessing.Pool(core_count-1)
    results = pool.starmap(algo_run, possible_values)
