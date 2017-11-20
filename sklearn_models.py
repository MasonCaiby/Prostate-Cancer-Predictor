import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

def logistic_reg (X_train, X_test, y_train, y_test, genes):
    LR = LogisticRegression(penalty='l2')
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    print(y_pred.shape)
    model_score(X_test, y_test, y_pred, 'Logistic Regression', LR)
    params = np.absolute(LR.coef_)
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

def common_genes(log_set, tree_set, ada_set, max_iter=10):
    gene_set = log_set.intersection(tree_set, ada_set)
    for i in range(max_iter):
        top_log_genes, log_pred = logistic_reg(X_train, X_test, y_train, y_test, genes)
        top_tree_genes, tree_pred = tree_reg(X_train, X_test, y_train, y_test, genes, threshold=threshold)
        top_ada_trees, tree_pred = adaboost(X_train, X_test, y_train, y_test, genes, threshold=threshold)
        log_set = set(top_log_genes)
        tree_set = set(top_tree_genes)
        ada_set = set(top_ada_trees)
        print (log_set.intersection(tree_set, ada_set))
        gene_set = gene_set.intersection(log_set, tree_set)
    print ("Gene_set: {}".format(gene_set))


if __name__ == '__main__':
    X = pd.read_csv('select_genes.csv', index_col=0)
    X_data = X.values
    y = pd.read_csv('balanced_labels.csv', index_col=0)
    y = y['Gleason_binary'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y)

    genes = get_gene_list(X)
    threshold = 0.5

    top_log_genes, log_pred = logistic_reg(X_train, X_test, y_train, y_test, genes)
    top_tree_genes, tree_pred = tree_reg(X_train, X_test, y_train, y_test, genes, threshold=threshold)
    top_ada_trees, tree_pred = adaboost(X_train, X_test, y_train, y_test, genes, threshold=threshold)
    # print(top_log_genes, top_tree_genes)
    # gene_list = [element for element in top_log_genes if element in top_tree_genes]
    # print(gene_list)
    log_set = set(top_log_genes)
    tree_set = set(top_tree_genes)
    ada_set = set(top_ada_trees)

    common_genes(log_set, tree_set, ada_set)
