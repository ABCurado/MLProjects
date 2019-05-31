import os
import datetime
import multiprocessing
import itertools

import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import utils
import preprocessing
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import *
import feature_engineering
from ML_algorithms import *


file_name= "../log_files/" + "results_"+ str(datetime.datetime.now().hour) + \
            "_" + str(datetime.datetime.now().minute) +"_log.csv"

header_string = "Seed,Algorithm,time,mean_s_error,explained_variance,final_tree_size"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")


models = [
    # GS_Mutation
    #('GS_GP_mutation01', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
     #('GS_GP_mutation02', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.02, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
     #('GS_GP_mutation05', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.05, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
     #('GS_GP_mutation1', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.1, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    
    # GS_Crossover
    #('GS_GP_crossover0.9', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    #('GS_GP_crossover0.7', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.02, p_gs_crossover=0.7, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    #('GS_GP_crossover0.5', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.05, p_gs_crossover=0.5, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    #('GS_GP_crossover0.3', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.1, p_gs_crossover=0.3, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    
    # Semantic Neighbors
    #('GS_GP_neighbors5', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.25, edv_stopping_criteria=0.0, n_semantic_neighbors=5, semantical_computation=False)'),
    #('GS_GP_neighbors10', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.25, edv_stopping_criteria=0.0, n_semantic_neighbors=10, semantical_computation=False)'),
    #('GS_GP_neighbors20', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.25, edv_stopping_criteria=0.0, n_semantic_neighbors=20)'),
    #('GS_GP_neighbors0', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.25, edv_stopping_criteria=0.0, n_semantic_neighbors=0)')
    
    # TIE Stopping Criteria 
    #('GS_GP_tie15', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.15, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    #('GS_GP_tie25', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.25, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    #('GS_GP_tie35', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.35, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    #('GS_GP_tie45', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.45, edv_stopping_criteria=0.0, n_semantic_neighbors=5)'),
    
    # EDV Stopping Criteria
    ('GS_GP_edv15', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.15, n_semantic_neighbors=5)'),
    ('GS_GP_edv25', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.25, n_semantic_neighbors=5)'),
    ('GS_GP_edv35', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.35, n_semantic_neighbors=5)'),
    ('GS_GP_edv45', 'GS_GP(verbose=False , gsm_ms=-1, p_gs_mutation=0.01, p_gs_crossover=0.9, p_crossover=0.0, p_subtree_mutation=0.00,          p_hoist_mutation=0.0, p_point_mutation=0.0, p_point_replace=0.0, tie_stopping_criteria=0.0, edv_stopping_criteria=0.45, n_semantic_neighbors=5)')

]

seed = list(range(0,1))

export_GS_GP_model = False

def algo_run(seed, model):

    start_time = datetime.datetime.now()

    df = utils.get_dataset()

    X, y = utils.X_y_split(df)

    model_eval = add_seed(model, seed,X)

    model_eval = eval(model_eval)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    model_eval.fit(X_train, y_train)
    y_predicted = model_eval.predict(X_test)

    mean_s_error = utils.calculate_mean_absolute_error(y_predicted, y_test)
    explained_variance = utils.calculate_explained_variance_score(y_predicted, y_test)

    time_elapsed = datetime.datetime.now() - start_time

    # Create result string
    log_parameters = [seed,model[0], time_elapsed, mean_s_error,explained_variance]

    if 'GS_GP' in model[0]:
        #idx = model_eval._program.parents['parent_idx']
        if "p_gs_crossover=0.0" in model[0]:
            fade_nodes = model_eval._program.parents['parent_nodes']
        #log_parameters.append(len(model_eval._programs[-2][idx].program))

    result_string = ",".join([str(value) for value in log_parameters])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")

    print(result_string)



def add_seed(model, seed,X):
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

    return model_eval


if __name__ ==  '__main__':
    possible_values = list(itertools.product(*[seed, models]))

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
