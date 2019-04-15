import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

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