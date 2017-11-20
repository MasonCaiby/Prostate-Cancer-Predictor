import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import recall_score, precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from collections import defaultdict
from roc import plot_roc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler


def logistic_reg (X_train, X_test, y_train, y_test, genes):
    LR = LogisticRegression(penalty='l2')
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    model_score(X_test, y_test, y_pred, 'Logistic Regression', LR)
    params = np.absolute(LR.coef_)
    genes = genes.T
    ind = np.argsort(params)
    top_genes = genes.flatten()[ind.flatten()][-100:]
    return set(top_genes), y_pred

def stoch_grad (X_train, X_test, y_train, y_test, genes):
    SGD = SGDClassifier(penalty='l2')
    SGD.fit(X_train, y_train)
    y_pred = SGD.predict(X_test)
    model_score(X_test, y_test, y_pred, 'Stochastic Grad', SGD)
    params =np.absolute(SGD.coef_)
    genes = genes.reshape(-1,1)
    genes = genes.T
    ind = np.argsort(params)
    top_genes = genes.flatten()[ind.flatten()][-100:]
    return set(top_genes), y_pred

def tree_reg (X_train, X_test, y_train, y_test, genes, threshold=0.7):
    RFC = RandomForestClassifier()
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)
    y_pred = y_pred.reshape(-1,1)
    model_score(X_test, y_test, y_pred, 'Random Forest', RFC)
    ind = np.argsort(RFC.feature_importances_)
    top_tree_genes = genes[ind][-100:]
    return set(top_tree_genes), y_pred

def adaboost(X_train, X_test, y_train, y_test, genes, threshold=0.7):
    AB = AdaBoostClassifier()
    AB.fit(X_train, y_train)
    y_pred = AB.predict(X_test)
    y_pred = y_pred.reshape(-1,1).astype(int)
    model_score(X_test, y_test, y_pred, 'Adaboost', AB)
    ind = np.argsort(AB.feature_importances_)
    top_ada_trees = genes[ind][-100:]
    return set(top_ada_trees), y_pred


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
        log_set, log_pred = logistic_reg(X_train, X_test, y_train, y_test, genes)
        tree_set, tree_pred = tree_reg(X_train, X_test, y_train, y_test, genes, threshold=threshold)
        ada_set, ada_pred = adaboost(X_train, X_test, y_train, y_test, genes, threshold=threshold)
        sgd_set, sgd_pred = stoch_grad(X_train, X_test, y_train, y_test, genes)

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

    log_set, log_pred = logistic_reg(X_train, X_test, y_train, y_test, genes)
    tree_set, tree_pred = tree_reg(X_train, X_test, y_train, y_test, genes, threshold=threshold)
    ada_set, ada_pred = adaboost(X_train, X_test, y_train, y_test, genes, threshold=threshold)
    sgd_set, sgd_pred = stoch_grad(X_train, X_test, y_train, y_test, genes)

    gene_count = common_gene_dict(log_set, tree_set, ada_set, sgd_set)
    print(gene_count)
