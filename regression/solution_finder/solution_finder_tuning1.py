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
    #("LinearRegression", 'Linear_Regression()'),
    #("DecisionTreeRegressor", 'Decision_Tree(criterion="mse")'),
    #("Gradient_Tree_Boosting","Gradient_Tree_Boosting(verbose=False)"),
    #("Adaptive_Tree_Boosting",'Adaptive_Tree_Boosting(loss="linear")'),
    #("Tree_Bagging", "Tree_Bagging(verbose=False)"),
    #("Random_Tree_Forest", "Random_Tree_Forest(verbose=False)"),
    ('GS_GP_p1000', 'GS_GP(verbose=False)'),
    ('GS_GP_p500', 'GS_GP(verbose=False , population_size=500)'),
    ('GS_GP_p2000', 'GS_GP(verbose=False , population_size=2000)'),
    ('GS_GP_p100', 'GS_GP(verbose=False , population_size=100)'),
    
    ('GS_GP_g20_stop01', 'GS_GP(verbose=False)'),
    ('GS_GP_g40', 'GS_GP(verbose=False, generations=40)'),
    ('GS_GP_g80', 'GS_GP(verbose=False , generations=80)'),
    ('GS_GP_g120', 'GS_GP(verbose=False , generations=120)'),
    ('GS_GP_g10', 'GS_GP(verbose=False , generations=10)'),
    
    ('GS_GP_stop0', 'GS_GP(verbose=False)'),
    ('GS_GP_stop05', 'GS_GP(verbose=False , stopping_criteria=0.05)'),
    ('GS_GP_stop10', 'GS_GP(verbose=False , stopping_criteria=0.10)'),
    ('GS_GP_stop20', 'GS_GP(verbose=False , stopping_criteria=0.20)'),
    ('GS_GP_stop50', 'GS_GP(verbose=False , stopping_criteria=0.50)'),

    
    ('GS_GP_cr_1_1', 'GS_GP(verbose=False , const_range=(-1.0, 1.0))'),
    ('GS_GP_cr_2_2', 'GS_GP(verbose=False , const_range=(-2.0, 2.0))'),
    ('GS_GP_cr_0.5_0.5', 'GS_GP(verbose=False , const_range=(-0.5, 0.5))'),
    ('GS_GP_cr_none', 'GS_GP(verbose=False , const_range=None)'),
    
    ('GS_GP_init_depth_2_6', 'GS_GP(verbose=False , init_depth=(2, 6))'),
    ('GS_GP_init_depth_2_4', 'GS_GP(verbose=False , init_depth=(2, 4))'),
    ('GS_GP_init_depth_2_8', 'GS_GP(verbose=False , init_depth=(2, 8))'),
    ('GS_GP_init_depth_2_3', 'GS_GP(verbose=False , init_depth=(2, 3))'),
    
    ('GS_GP_init_method_hh', 'GS_GP(verbose=False , init_method="half and half")'),
    ('GS_GP_init_method_grow', 'GS_GP(verbose=False , init_method="grow")'),
    ('GS_GP_init_method_full', 'GS_GP(verbose=False , init_method="full")'),
    
    ('GS_GP_function_set_asmd', 'GS_GP(verbose=False , function_set="add", "sub", "mul", "div")'),
    ('GS_GP_function_set_all', 'GS_GP(verbose=False , function_set="add", "sub", "mul", "div", "sqrt", "log","abs","neg","inv","max","min","sin","cos","tan","tanh")'),
    ('GS_GP_function_set_slani', 'GS_GP(verbose=False , function_set="sqrt", "log","abs","neg","inv")'),
    ('GS_GP_function_set_as', 'GS_GP(verbose=False , function_set="add", "sub")'),
    ('GS_GP_function_set_mmsctt', 'GS_GP(verbose=False , function_set="max","min","sin","cos","tan","tanh)'),
    
    ('GS_GP_metric_mae', 'GS_GP(verbose=False , metric="mean absolute error")'),
    ('GS_GP_metric_mse', 'GS_GP(verbose=False , metric="mse")'),
    ('GS_GP_metric_rmse', 'GS_GP(verbose=False , metric="rmse")'),
    ('GS_GP_metric_spearman', 'GS_GP(verbose=False , metric="spearman")'),
    ('GS_GP_metric_pearson', 'GS_GP(verbose=False , metric="pearson")'),
    
    ('GS_GP_parsimony_002', 'GS_GP(verbose=False , parsimony_coefficient=0.002)'),
    
    
    
    
    
    
    
    #('GS_GP', 'GS_GP(verbose=False , population_size=60)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    #('GS_GP', 'GS_GP(verbose=False , population_size=20)'),
    
    #("XGBoost", 'XG_Boost(n_estimators=100)')

]

scalers = [
#    ("StandardScaler", StandardScaler()),
    ("StandardScaler", StandardScaler()),
    ("RobustScaler", RobustScaler()),
    ("None", None)
]

samplers =  [
    ("None", None)

]

pre_processing_pipelines = [
    ("None", None)

]
seed = [0,1]

export_GS_GP_model = False

def algo_run(model, pre_processing_pipeline, scaler, sampler, seed):

    start_time = datetime.datetime.now()

    df = utils.get_dataset()
    if pre_processing_pipeline[1] != None:
        df = pre_processing_pipeline[1](df)
    X, y = utils.X_y_split(df)

    if "Keras" in model[0]:
        model_eval = model[1][:-1]+",input_dim="+str(X.shape[1])+")"
    elif "GP" in model[0]:
        model_eval = model[1][:-1]+",random_state="+str(seed)+",feature_names="+str(list(X.columns))+")"
    elif "Tree" in model[0]:
        model_eval = model[1][:-1]+",random_state="+str(seed)+")"
    elif "XG" in model[0]:
        model_eval = model[1][:-1]+",seed=" + str(seed) + ")"
    else:
        model_eval = model[1]

    model_eval = eval(model_eval)
    try:
        y_predicted = utils.cross_validation_average_results(
            model_eval, X, y, n_splits=5,
            scaler=scaler[1],
            sampling_technique=sampler[1]
        )
        mean_s_error = utils.calculate_mean_absolute_error(y_predicted, y)
        explained_variance = utils.calculate_explained_variance_score(y_predicted, y)
    except:
        mean_s_error = -1
        explained_variance = -1

    time_elapsed = datetime.datetime.now() - start_time

    # Create result string
    result_string = ",".join(
        [str(seed),model[0],
         pre_processing_pipeline[0], scaler[0], sampler[0], str(time_elapsed),str(mean_s_error),str(explained_variance)])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")

    print(model[0]+": "+str(mean_s_error))

    if 'GS_GP' in model[1]:
        idx = model_eval._program.parents['parent_idx']
        fade_nodes = model_eval._program.parents['parent_nodes']
        print(model_eval._programs[-2][idx])


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
