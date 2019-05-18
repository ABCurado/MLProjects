import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np
import pandas as pd


import utils
import preprocessing
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import *
import feature_engineering
from ML_algorithms import *


file_name= "../log_files/" + "results_"+ str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) +"_log.csv"

header_string = "Seed,Algorithm,Parameters,Preprocessing Pipeline,Scaling,Sampling,time,mean_s_error,explained_variance"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")


models = [
    #Simple Keras NN  
#    ("KerasNN_3neurons" , 'KerasNN_not_fitted(n_neurons=3, init="he_normal")'),
    ("XGBoot", 'XB_Boost()'),
    ("LinearRegression", 'Linear_Regression()'),
    ('GS_GP', 'GS_GP()')
]

scalers = [
    ("StandardScaler", StandardScaler()),
    ("None", None)
]

samplers =  [
    ("None", None)

]

pre_processing_pipelines = [
    ("None", None)
]
seed = [1]


def algo_run(model, pre_processing_pipeline, scaler, sampler, seed):

    start_time = datetime.datetime.now()

    df = utils.get_dataset()
    if pre_processing_pipeline[1] != None:
        df = pre_processing_pipeline[1](df)
     
    X, y = utils.X_y_split(df)
  
    if "Keras" in model[0]:
        model_eval = model[1][:-1]+",input_dim="+str(X.shape[1])+")"
    else:
        model_eval = model[1]
    
    model_eval = eval(model_eval)
    try:
        y_predicted = utils.cross_validation_average_results(
            model_eval, X, y, n_splits=5,
            scaler=scaler[1],
            sampling_technique=sampler[1]
        )
        mean_s_error = utils.calculate_mean_squared_error(y_predicted, y)
        explained_variance = utils.calculate_explained_variance_score(y_predicted, y)
    except ex:
        result = -1
        recall = -1
        precision = -1
        
    time_elapsed = datetime.datetime.now() - start_time

    # Create result string
    result_string = ",".join(
        [str(seed),model[0],
         pre_processing_pipeline[0], scaler[0], sampler[0], str(time_elapsed),str(mean_s_error),str(explained_variance)         ])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")
    # Output result to terminal
    print(model[0]+": "+str(mean_s_error))


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
