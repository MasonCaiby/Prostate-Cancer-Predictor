import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import recall_score, precision_score, roc_curve, auc, make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import defaultdict
# from roc import plot_roc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def optimize_model(classifier, param_grid, X_train, X_test, y_train, y_test, scoring):
    GS = GridSearchCV(classifier, param_grid, scoring=scoring, verbose=10, cv=2, refit=False)
    GS.fit(X_train, y_train)
    return GS.cv_results_

def specificity(y_test, y_pred):
    TN = np.sum([(y_pred==0) & (y_test==0)])
    FP = np.sum([(y_pred==1) & (y_test==0)])
    return TN/(TN+FP)

def neg_pred(y_test, y_pred):
    TN = np.sum([(y_pred==0) & (y_test==0)])
    FN = np.sum([(y_pred==0) & (y_test==1)])
    return TN/(TN+FN)


if __name__ == '__main__':
    X = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_train_select.csv', index_col=0)
    X_train = X.fillna(value=0).values
    X_test = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_test_select.csv', index_col=0)
    X_test = X_test.fillna(value=0).values
    y_train = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_train.csv', index_col=0, header=None).values
    y_test = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_test.csv', index_col=0, header=None).values


    scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score), \
               'recall': make_scorer(recall_score), \
               'specificity': make_scorer(specificity, greater_is_better=True), \
               'neg_pred': make_scorer(neg_pred, greater_is_better=True)}

    # for all genes
    tree = DecisionTreeClassifier(class_weight='balanced')

    ada = AdaBoostClassifier()
    param_grid_ada = {'base_estimator': [tree], \
                      'n_estimators': [100, 250, 500], \
                      'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0]}

    rfc = RandomForestClassifier()
    param_grid_rfc = {'n_estimators': [1000, 5000, 10000, 15000], \
                      'n_jobs': [-1], \
                      'max_features': [10, 50, 100, 200], \
                      'max_depth': [3, 4, 5, 6], \
                      'class_weight': ['balanced']}

    svc = SVC()
    param_grid_svc = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], \
                      'gamma': [50, 100, 150, 200], \
                      'class_weight': 'balanced'}



    SGD = SGDClassifier()
    param_grid_sgd = {'loss': ['hinge', 'log', 'modified_huber'], \
                      'alpha': [0.001, 0.01, 0.1, 1.0], \
                      'penalty': ['l1'], \
                      'max_iter': [5, 10, 25, 50, 75], \
                      'class_weight': ['balanced']}

    param_list = [param_grid_ada, param_grid_rfc, param_grid_sgd, param_grid_svc]
    clf_list = [ada, rfc, SGD, svc]

    results_list= []
    for i, classifier in enumerate(clf_list):
        results = optimize_model(clf_list[i], param_list[i], X_train, X_test, y_train, y_test, scoring)
        results_list.append(results)
