import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import KFold
import data_visualization

def get_dataset():
    '''
        Returns the dataset provided for the project as a dataframe
    '''
    return pd.read_excel("data_files/ml_project1_data.xlsx")

def missing_values_reporter(df):
    '''
        Returns the number of missing values in each columns
        Credit to Professor Ilya
    '''
    na_count = df.isna().sum() 
    ser = na_count[na_count > 0]
    return pd.DataFrame({"N missings": ser, "% missings": ser.divide(df.shape[0])})

def data_split(df, test_size=0.33,random_state=42):
    '''
        Selects the outcome variable and calls sklearn trai_test_split
    '''
    y = df["Response"]
    X = df.loc[:, df.columns != "Response"] 
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def simple_train_split(df, test_size=0.33, random_state=42):
    test = df.sample(frac=test_size, random_state=random_state)
    train = df.drop(test.index)
    return train, test

def X_y_split(df):
    '''
        Selects the outcome variable and calls sklearn trai_test_split
    '''
    y = df["Response"]
    X = df.loc[:, df.columns != "Response"] 
    return (X, y)

def calculate_accuracy(y_true, y_pred):
    '''
        Accuracy classification score.

        In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    '''
    return accuracy_score(y_true, y_pred)


def calculate_auc(y_true, y_pred):
    '''
        Compute Area Under the Curve (AUC) using the trapezoidal rule

        This is a general function, given points on a curve. For computing the area under the ROC-curve, see roc_auc_score
    '''
    return roc_auc_score(y_true, y_pred)

def calculate_average_precision_score(y_true, y_pred):
    '''
        Compute average precision (AP) from prediction scores

        AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight
    '''
    return average_precision_score(y_true, y_pred)

def calculate_precision_score(y_true, y_pred):
    '''
       Compute the precision

    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    The best value is 1 and the worst value is 0.
    '''
    return precision_score(y_true, y_pred)

def calculate_recall_score(y_true, y_pred):
    '''
        Compute the recall

        The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

        The best value is 1 and the worst value is 0.
    '''
    return recall_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    '''
        Compute confusion matrix to evaluate the accuracy of a classification

        By definition a confusion matrix
        is such that is equal to the number of observations known to be in group but predicted to be in group
    '''
    return confusion_matrix(y_true, y_pred)

def cross_validation_average_results(model, X, y, n_splits=5):
    '''
        Does cross validation with n_splits and returns an array with y size as predictions.
        !!!!Currently not working with transformations calculated on train data and applied in test data!!!
        
        example with 5 splits:
        
        split 1 -   |||------------
        split 2 -   ---|||---------
        split 3 -   ------|||------
        split 4 -   ---------|||---
        split 5 -   ------------|||
        
        returns     |||||||||||||||  <- which represents the predictions for the whole array
        
    '''
    kf = KFold(n_splits=n_splits)
    predictions = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, _ = y.iloc[train_index], y.iloc[test_index]
        estimator = model(X_train, y_train)
        prediction = estimator.predict(X_test)
        predictions.extend(prediction)
    return np.array(predictions)


def profit_share(y_true, y_pred):
    """
    Computes the profit. For each True/True +8, for each True/False -3 and compares it with the possible profit.
    E.g. 0.26 means, that one got 26% of the max possible profit.
    """
    score = 0
    for i in (y_true - (y_pred * 2)):
        if i == -1:
            score += 8
        elif i == -2:
            score -= 3
    
    if sum(y_true) == 0:
        return 0.00

    return round(score / (sum(y_true) * 8), 2)

def max_threshold(y_pred, y_test, threshold_range = (0.4, 0.6), iterations = 100, visualization=False):
    '''
        For a given continuos predictions array with value [0,1] returns the best threshold to use when categorizint the data
    '''
    profits, thresholds = threshold_optimization(y_pred, y_test, threshold_range, iterations, visualization)
    profits = np.array(profits)
    thresholds = np.array(thresholds)
    if visualization:
        data_visualization.arg_max_plot(thresholds, profits)
    return thresholds[np.argmax(profits)]

def predict_with_threshold(y_pred_cont, threshold):
    '''
        Generates a boolean array with a given continuos array [0,1] and a defining threshold 
    '''
    return [1 if value > threshold else 0 for value in y_pred_cont ]

def threshold_optimization(y_pred_cont, y_test, threshold_range = (0.4, 0.6), iterations = 100, visualization=False):
    '''
        Given a set of treshold boundaries and a iteration number it calculates the profit for each treshold
    '''
    step = (threshold_range[1] - threshold_range[0]) / iterations
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    profits = []
    for threshold in thresholds:
        y_pred = predict_with_threshold(y_pred_cont, threshold)
        
        # Evaluation metric should be dynamic
        profit = profit_share(y_pred, y_test)
        profits.append(profit)
    
    if visualization:
        data_visualization.xy_plot(x=thresholds, y=profits)
    
    return profits, thresholds