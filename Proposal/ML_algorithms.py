from sklearn.linear_model   import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def logistic_regression(X_train, y_train ):
    logit_model = LogisticRegression()  
    logit_model.fit(X_train, y_train)  
    return logit_model

def decision_tree(X_train, y_train):
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    return tree

def KNN(X_train, y_train):
    tree = KNeighborsClassifier()
    tree.fit(X_train, y_train)
    return tree