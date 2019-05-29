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

header_string = "Seed,Algorithm,time,mean_s_error,explained_variance"
with open(file_name, "w") as myfile:
    myfile.write(header_string + "\n")


models = [
    #    ("LinearRegression", 'Linear_Regression()'),
    #    ("DecisionTreeRegressor", 'Decision_Tree(criterion="mse")'),
    #    ("Gradient_Tree_Boosting","Gradient_Tree_Boosting(verbose=False)"),
    #    ("Adaptive_Tree_Boosting",'Adaptive_Tree_Boosting(loss="linear")'),
    #    ("Tree_Bagging", "Tree_Bagging(verbose=False)"),
    #    ("Random_Tree_Forest", "Random_Tree_Forest(verbose=False)"),
    #('GS_GP', 'GS_GP(verbose=False ,special_fitness=False, generations=50)'),
    ('GS_GP_genotype_operators', 'GS_GP(verbose=False ,probabilistic_genotype_operators=True, generations=50)'),
 #   ('GS_GP_phenotype_operators', 'GS_GP(verbose=False ,probabilistic_phenotype_operators=True, generations=50)'),
    #('GS_GP_special_fitness', 'GS_GP(verbose=False ,special_fitness=True, generations=50)'),
 #   ("XGBoost", 'XG_Boost(n_estimators=100)')
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
    result_string = ",".join(
        [str(seed),model[0], str(time_elapsed), str(mean_s_error),str(explained_variance)])
    # Write result to a file
    with open(file_name, "a") as myfile:
        myfile.write(result_string + "\n")

    print(str(seed)+" " + model[0]+": "+str(mean_s_error))

    if 'GS_GP' in model[1] and False:
        idx = model_eval._program.parents['parent_idx']
        fade_nodes = model_eval._program.parents['parent_nodes']
        print(model_eval._programs[-2][idx])

def add_seed(model, seed,X):
    if "Keras" in model[0]:
        model_eval = model[1][:-1]+",input_dim="+str(X.shape[1])+")"
    elif "GS_GP" in model[0]:
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
