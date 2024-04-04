import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

def logistic_regression(X_train, X_test, y_train, y_test):
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    prediction = lr_model.predict(X_test)
    accuracy_score(y_test, prediction)
    balanced_accuracy_score(y_test, prediction)
    class_report = classification_report(y_test, prediction)
    print(class_report)
    
    return lr_model

def svc(X_train, X_test, y_train, y_test):
    svc_model = SVC(kernel='linear')
    svc_model.fit(X_train, y_train)
    prediction = svc_model.predict(X_test)
    accuracy_score(y_test, prediction)
    return svc_model

def knn(X_train, X_test, y_train, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    prediction = knn_model.predict(X_test)
    accuracy_score(y_test, prediction)
    return knn_model

def decision_tree(X_train, X_test, y_train, y_test):
    decision_tree_model = tree.DecisionTreeClassifier()
    decision_tree_model.fit(X_train, y_train)
    prediction = decision_tree_model.predict(X_test)
    accuracy_score(y_test, prediction)
    return decision_tree_model

def random_forest(X_train, X_test, y_train, y_test):
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    prediction = random_forest_model.predict(X_test)
    accuracy_score(y_test, prediction)
    return random_forest_model