from keras import models
from keras import layers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import *
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from gplearn_MLAA.genetic import SymbolicRegressor

import numpy as np
import utils

def KerasNN(X_train, y_train, input_dim=32, n_layers=4, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=["accuracy"], random_state=42):
    """
    Keras Neural Network, define the amount of layers you want, which optimizer you want to use and which loss function you want to apply.
    """ 
    np.random.seed(random_state)

    model = models.Sequential()
    model.add(layers.Dense(6, activation="relu", input_dim=input_dim))
    for num in range(n_layers-2):
        model.add(layers.Dense(6, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid", init=init))
    model.compile(optimizer, loss, metrics=metrics)
    
    initial_weights = model.get_weights()
    
    utils.shuffle_weights(model, initial_weights)
    
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

def KerasNN_not_fitted(input_dim=39, n_layers=4, n_neurons=6, r_dropout=0.5, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=["accuracy"], random_state=42):
    """
    Keras Neural Network, define the amount of layers you want, which optimizer you want to use and which loss function you want to apply.
    """ 
    np.random.seed(random_state)
    

    from keras import backend as K
    K.clear_session()


    model = models.Sequential()
    model.add(layers.Dense(n_neurons, activation="relu", input_dim=input_dim))
    model.add(layers.Dropout(r_dropout))
    for num in range(n_layers-2):
        model.add(layers.Dense(n_neurons, activation="relu"))
        model.add(layers.Dropout(r_dropout))
    model.add(layers.Dense(1, activation="sigmoid", init=init))
    model.compile(optimizer, loss, metrics=metrics)
    
    return model


def Linear_Regression():
    linreg = LinearRegression()
    return linreg

def logistic_regression():
    logit_model = LogisticRegression()
    return logit_model

def decision_tree(criterion="gini", class_weight=None):
    tree = DecisionTreeClassifier(criterion=criterion, class_weight=class_weight)
    return tree


def Random_Tree_Forest(verbose = 0, random_state=None):
    rf = RandomForestRegressor(n_estimators="warn", criterion="mse",
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False,
                                n_jobs=None, random_state=random_state, verbose=verbose, warm_start=False)
    return rf

def Gradient_Tree_Boosting(verbose = 0, random_state=None):
    gb = GradientBoostingRegressor(
                loss="ls", learning_rate = 0.1, n_estimators = 100,
                subsample = 1.0, criterion ="friedman_mse", min_samples_split = 2,
                min_samples_leaf = 1, min_weight_fraction_leaf = 0.0, max_depth = 3, min_impurity_decrease = 0.0,
                min_impurity_split = None, init = None, random_state = random_state, max_features = None,
                alpha = 0.9, verbose = verbose, max_leaf_nodes = None, warm_start = False, presort ="auto",
                validation_fraction = 0.1, n_iter_no_change = None, tol = 0.0001)
    return gb

def Adaptive_Tree_Boosting(loss="linear", random_state=None):
    ab = AdaBoostRegressor(base_estimator=None, n_estimators=50,
                           learning_rate=1.0, loss=loss, random_state=random_state)
    return ab

def Tree_Bagging(verbose = 0, random_state=None):
    br = BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
                     bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=random_state,
                     verbose=verbose)
    return br

def Logistic_Regression():
    lr = LogisticRegression()
    return lr

def Stochastic_Gradient_Descent_Regressor():
    sgdc = SGDRegressor()
    return sgdc

def Decision_Tree(criterion='mse', random_state=0):
    dt = DecisionTreeRegressor(criterion=criterion,
                               random_state=random_state)
    return dt

def MLP_Regressor():
    mlp = MLPRegressor(hidden_layer_sizes=(10), solver = "lbfbs", max_iter=1000, random_state=42)
    return mlp

def XG_Boost(n_estimators=100,seed=0):
    xgbc = XGBRegressor(colsample_by_tree=0.1,
                                  learning_rate=0.89,
                                  max_depth=8,
                                  n_estimators=n_estimators,
                                  eval_metric="auc",
                                  n_jobs=-1, silent=0, verbose=1,seed=seed)
    return xgbc

def GS_GP(seed = 0, verbose=False,feature_names=None,population_size=50):
    est_gp = SymbolicRegressor(population_size=population_size,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=verbose,
                               parsimony_coefficient=0.001, init_depth=(2, 6),  random_state=seed,
                               feature_names=feature_names)
    return est_gp
