from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from keras import models
from keras import layers
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import utils

def Linear_Regression (X_train, y_train):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    return linreg

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

def KerasNN_not_fitted(input_dim=32, n_layers=4, optimizer="rmsprop", loss="binary_crossentropy", init="uniform", metrics=["accuracy"], random_state=42):
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
    return model

def Gaussian_NB(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    return gnb

def Multinomial_NB(X_train, y_train):
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    return mnb

def Complement_NB(X_train, y_train):
    cnb = ComplementNB()
    cnb.fit(X_train, y_train)
    return cnb

def Support_Vector_Classification(X_train, y_train):
    svc = SVC()
    svc.fit(X_train, y_train)
    return svc

def Linear_Support_Vector_Classification(X_train, y_train):
    lsvc = LinearSVC()
    lsvc.fit(X_train, y_train)
    return lsvc

def Nu_Support_Vector_Classification(X_train, y_train):
    nsvc = NuSVC()
    nsvc.fit(X_train, y_train)
    return nsvc

def Logistic_Regression(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

def Stochastic_Gradient_Descent_Classifier(X_train, y_train):
    sgdc = SGDClassifier()
    sgdc.fit(X_train, y_train)
    return sgdc

def MLP_Classifier(X_train, y_train):
    mlp = MLPClassifier(hidden_layer_sizes=(10), solver = "lbfbs", max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    return mlp

def XB_Boost(X_train, y_train):
    xgbc = XGBClassifier(colsample_by_tree=0.1,
                                  learning_rate=0.89,
                                  max_depth=8,
                               n_estimators=10000,
                                  eval_metric="auc",                                scale_pos_weight=np.sum(y_train==0)/np.sum(y_train==1),
                                  n_jobs=-1, silent=0, verbose=1)
    xgbc.fit(X_train, y_train)
    return xgbc


