import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import recall_score, precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from collections import defaultdict
from roc.py import roc_plot
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler


def logistic_reg (X_train, X_test, y_train, y_test, genes):
    LR = LogisticRegression(penalty='l1')
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    model_score(X_test, y_test, y_pred, 'Logistic Regression', LR)
    params = np.absolute(LR.coef_)
    genes = genes.T
    ind = np.argsort(params)
    top_genes = genes.flatten()[ind.flatten()][-100:]
    return top_genes, y_pred

def stoch_grad (X_train, X_test, y_train, y_test, genes):
    SGD = SGDClassifier(penalty='l1')
    SGD.fit(X_train, y_train)
    y_pred = SGD.predict(X_test)
    model_score(X_test, y_test, y_pred, 'Stochastic Grad', SGD)
    params =np.absolute(SGD.coef_)
    genes = genes.reshape(-1,1)
    genes = genes.T
    ind = np.argsort(params)
    top_genes = genes.flatten()[ind.flatten()][-100:]
    return top_genes, y_pred

def tree_reg (X_train, X_test, y_train, y_test, genes, threshold=0.7):
    RFR = RandomForestRegressor()
    RFR.fit(X_train, y_train)
    y_pred = RFR.predict(X_test)
    y_pred[y_pred>threshold]=1
    y_pred[y_pred<=threshold]=0
    y_pred = y_pred.reshape(-1,1).astype(int)
    model_score(X_test, y_test, y_pred, 'Random Forest', RFR)
    ind = np.argsort(RFR.feature_importances_)
    genes = genes.T
    top_tree_genes = genes.flatten()[ind][-100:]
    return top_tree_genes, y_pred

def adaboost(X_train, X_test, y_train, y_test, genes, threshold=0.7):
    AB = AdaBoostRegressor()
    AB.fit(X_train, y_train)
    y_pred = AB.predict(X_test)
    y_pred[y_pred>threshold]=1
    y_pred[y_pred<=threshold]=0
    y_pred = y_pred.reshape(-1,1).astype(int)
    model_score(X_test, y_test, y_pred, 'Adaboost', AB)
    ind = np.argsort(AB.feature_importances_)
    top_ada_trees = genes[ind][-100:]
    return top_ada_trees, y_pred


def model_score (X_test, y_test, y_pred, name, regressor):
    print ('{} Accuracy/R2: {}'.format(name, regressor.score(X_test, y_test)))
    print ('{} Precision: {}'.format(name, precision_score(y_test, y_pred)))
    print ('{} Recall: {}'.format(name, recall_score(y_test, y_pred)))

def get_gene_list(X):
    genes = X.columns.values
    return genes

def common_genes(log_set, tree_set, ada_set, sgd_trees, max_iter=10):
    gene_set = log_set.intersection(tree_set, ada_set)
    for i in range(max_iter):
        top_log_genes, log_pred = logistic_reg(X_train, X_test, y_train, y_test, genes)
        top_tree_genes, tree_pred = tree_reg(X_train, X_test, y_train, y_test, genes, threshold=threshold)
        top_ada_trees, tree_pred = adaboost(X_train, X_test, y_train, y_test, genes, threshold=threshold)
        top_sgd_trees, sgd_pred = stoch_grad(X_train, X_test, y_train, y_test, genes)

        log_set = set(top_log_genes)
        tree_set = set(top_tree_genes)
        ada_set = set(top_ada_trees)
        sgd_set = set(top_sgd_trees)

        # print (log_set.intersection(tree_set, ada_set, sgd_set))
        gene_set = gene_set.intersection(log_set, tree_set, sgd_set)
    # print ("Gene_set: {}".format(gene_set))
        return gene_set

def common_gene_dict(log_set, tree_set, ada_set, sgd_set):
    com_genes = [common_genes(log_set, tree_set, ada_set, sgd_set) for i in range(10)]
    gene_count = defaultdict(int)
    for s in com_genes:
         for gene in s:
             gene_count[gene]+=1
    return gene_count

if __name__ == '__main__':
    X = pd.read_csv('select_genes1.csv', index_col=0)
    X_data = X.values
    y = pd.read_csv('balanced_labels1.csv', index_col=0)
    y = y['Gleason_binary'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y)

    genes = get_gene_list(X)
    threshold = 0.5

    top_log_genes, log_pred = logistic_reg(X_train, X_test, y_train, y_test, genes)
    top_tree_genes, tree_pred = tree_reg(X_train, X_test, y_train, y_test, genes, threshold=threshold)
    top_ada_trees, tree_pred = adaboost(X_train, X_test, y_train, y_test, genes, threshold=threshold)
    top_sgd_trees, sgd_pred = stoch_grad(X_train, X_test, y_train, y_test, genes)

    log_set = set(top_log_genes)
    tree_set = set(top_tree_genes)
    ada_set = set(top_ada_trees)
    sgd_set = set(top_sgd_trees)

    com_genes = [common_genes(log_set, tree_set, ada_set, sgd_set) for i in range(10)]
    print (com_genes)
    gene_count = defaultdict(int)
    for s in com_genes:
         for gene in s:
             gene_count[gene]+=1
    print(gene_count)
