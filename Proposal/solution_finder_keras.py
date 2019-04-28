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

file_name= "LogFiles/" + "results_"+ str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) +"_log.csv"

header_string = "Seed,Algorithm,Parameters,Preprocessing Pipeline,Scaling,Sampling,time,precision,recall,result_profit"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")


models = [
        ("KerasNN_6l_12n_or_iu_lm_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="rmsprop",init="uniform",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_3l_6n_oa_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=6, optimizer="adam",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_6l_6n_oa_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=6, optimizer="adam",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_3l_12n_oa_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=12, optimizer="adam",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_6l_12n_oa_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="adam",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_3l_6n_or_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=6, optimizer="rmsprop",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_6l_6n_or_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=6, optimizer="rmsprop",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_3l_12n_or_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=12, optimizer="rmsprop",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_6l_12n_or_ih_lm_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="rmsprop",init="he_normal",loss="mean_squared_error",r_dropout=0.4)'),
    ("KerasNN_3l_6n_oa_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=6, optimizer="adam",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_6n_oa_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=6, optimizer="adam",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_12n_oa_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=12, optimizer="adam",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_12n_oa_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="adam",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_6n_or_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=6, optimizer="rmsprop",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_6n_or_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=6, optimizer="rmsprop",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_12n_or_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=12, optimizer="rmsprop",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_12n_or_iu_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="rmsprop",init="uniform",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_6n_oa_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=6, optimizer="adam",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_6n_oa_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=6, optimizer="adam",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_12n_oa_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=12, optimizer="adam",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_12n_oa_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="adam",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_6n_or_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=6, optimizer="rmsprop",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_6n_or_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=6, optimizer="rmsprop",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_3l_12n_or_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=3, n_neurons=12, optimizer="rmsprop",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)'),
    ("KerasNN_6l_12n_or_ih_lb_d4" , 'KerasNN_not_fitted(n_layers=6, n_neurons=12, optimizer="rmsprop",init="he_normal",loss="binary_crossentropy",r_dropout=0.4)')
]

scalers = [
    ("StandardScaler", StandardScaler()),
    ("RobustScaler", RobustScaler()),
    ("MinMaxScaler", MinMaxScaler()),
    ("Normalizer", Normalizer()),
    ("None", None)
]

samplers =  [
    ("RandomOverSampler_0.2", RandomOverSampler(random_state=42, ratio=0.2)),
    ("RandomOverSampler_0.5", RandomOverSampler(random_state=42, ratio=0.5)),
    ("RandomOverSampler_0.5", RandomOverSampler(random_state=42, ratio=0.35)),
    ("TomekLinks", TomekLinks(random_state=42)),
    ("EditedNN", EditedNearestNeighbours(random_state=42, n_neighbors=3)),
    ("SMOTE", SMOTE(random_state=42, ratio=0.5)),
    ("SMOTETomek",SMOTETomek(random_state=42, ratio=0.8)),
    ("None", None)
    
]

pre_processing_pipelines = [
    ("Joris_Pipeline", preprocessing.joris_preprocessing_pipeline),
    ("Morten_Pipeline", preprocessing.morten_preprocessing_pipeline),
    ("Bin it!", preprocessing.bin_it_preprocessing_pipeline),
    ("simple_pipeline", preprocessing.simple_pipeline),
    ("chop_off", preprocessing.chop_off),
    ("pca_chopoff", preprocessing.pca_chopoff),
    ("box_cox_pipeline", preprocessing.box_cox_pipeline),
    ("feature_engineered", preprocessing.feature_engineered)

]
seed = [1]


def algo_run(model, pre_processing_pipeline, scaler, sampler, seed):

    start_time = datetime.datetime.now()

    df = utils.get_dataset()
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
        threshold = utils.max_threshold(y_predicted, y, threshold_range=(0.2, 0.6), iterations=1000, visualization=False)
        y_pred = utils.predict_with_threshold(y_predicted, threshold)
        result = utils.profit_share(y_pred, y)
        precision = utils.calculate_precision_score(y_pred, y)
        recall = utils.calculate_recall_score(y_pred, y)
    except:
        result = -1
        recall = -1
        precision = -1
        
    time_elapsed = datetime.datetime.now() - start_time

    # Create result string
    result_string = ",".join(
        [str(seed),model[0],
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
