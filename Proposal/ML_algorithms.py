from sklearn.linear_model   import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras import models
from keras import layers
import numpy as np
import utils

def logistic_regression(X_train, y_train ):
    logit_model = LogisticRegression()  
    logit_model.fit(X_train, y_train)  
    return logit_model

def decision_tree(X_train, y_train, criterion="gini", class_weight=None):
    tree = DecisionTreeClassifier(criterion=criterion, class_weight=class_weight)
    tree.fit(X_train, y_train)
    return tree

def KNN(X_train, y_train):
    tree = KNeighborsClassifier()
    tree.fit(X_train, y_train)
    return tree

def KerasNN(X_train, X_test, y_train, y_test, input_dim=32, n_layers=4, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=["accuracy"]):
    """
    Keras Neural Network, define the amount of layers you want, which optimizer you want to use and which loss function you want to apply.
    """ 
    np.random.seed(42)

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

