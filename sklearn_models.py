import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

def logistic_reg (X_train, X_test, y_train, y_test, genes):
    LR = LogisticRegression(penalty='l2', class_weight='balanced')
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    model_score(X_test, y_test, y_pred, 'Logistic Regression', LR)
    params = np.absolute(LR.coef_)
    genes = genes.T
    ind = np.argsort(params)
    top_genes = genes.flatten()[ind.flatten()][-100:]
    return top_genes

def tree_reg (X_train, X_test, y_train, y_test, genes, threshold=0.7):
    RFR = RandomForestRegressor()
    RFR.fit(X_train, y_train)
    y_pred = RFR.predict(X_test)
    y_pred[y_pred>0.7]=1
    y_pred[y_pred<=0.7]=0
    y_pred = y_pred.reshape(-1,1).astype(int)
    model_score(X_test, y_test, y_pred, 'Random Forest', RFR)
    ind = np.argsort(RFR.feature_importances_)
    genes = genes.T
    top_tree_genes = genes.flatten()[ind][-100:]
    return top_tree_genes

def model_score (X_test, y_test, y_pred, name, regressor):
    print ('{} Accuracy/R2: {}'.format(name, regressor.score(X_test, y_test)))
    print ('{} Precision: {}'.format(name, precision_score(y_test, y_pred)))
    print ('{} Recall: {}'.format(name, recall_score(y_test, y_pred)))

def get_gene_list(X):
    genes = X.columns.values[1:]
    return genes

if __name__ == '__main__':
    X = pd.read_csv('genes_from_pivot2.csv')
    X_data = X.iloc[:, 1:].values
    y = pd.read_csv('labels_from_pivot2.csv')
    y = y['Gleason_binary'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y)

    genes = get_gene_list(X)

    top_log_genes = logistic_reg(X_train, X_test, y_train, y_test, genes)
    top_tree_genes = tree_reg(X_train, X_test, y_train, y_test, genes, 0.7)

    print(top_log_genes, top_tree_genes)
