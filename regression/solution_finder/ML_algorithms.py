from keras import models
from keras import layers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.preprocessing import MinMaxScaler
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


def Gaussian_NB():
    gnb = GaussianNB()
    return gnb

def Multinomial_NB():
    mnb = MultinomialNB()
    return mnb

def Complement_NB():
    cnb = ComplementNB()
    return cnb

def Logistic_Regression():
    lr = LogisticRegression()
    return lr

def Stochastic_Gradient_Descent_Regressor():
    sgdc = SGDRegressor()
    return sgdc

def MLP_Regressor():
    mlp = MLPRegressor(hidden_layer_sizes=(10), solver = "lbfbs", max_iter=1000, random_state=42)
    return mlp

def XB_Boost():
    xgbc = XGBRegressor(colsample_by_tree=0.1,
                                  learning_rate=0.89,
                                  max_depth=8,
                                  n_estimators=100,
                                  eval_metric="auc",
                                  n_jobs=-1, silent=0, verbose=1)
    return xgbc

def GS_GP(seed = 0, verbose=False,feature_names=None,population_size=50):
    est_gp = SymbolicRegressor(population_size=population_size,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=verbose,
                               parsimony_coefficient=0.01, random_state=seed,
                               feature_names=feature_names)
    return est_gp
