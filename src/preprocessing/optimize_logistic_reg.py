import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

'''
Do a train/test split initially
Finds top mutated genes for each patient and creates a union of top genes for all patients
--For all genes and subset genes--
Runs logistic regression optimization
'''

def specificity(y_test, y_pred):
    TN = np.sum([(y_pred==0) & (y_test==0)])
    FP = np.sum([(y_pred==1) & (y_test==0)])
    return TN/(TN+FP)

def neg_pred(y_test, y_pred):
    TN = np.sum([(y_pred==0) & (y_test==0)])
    FN = np.sum([(y_pred==0) & (y_test==1)])
    return TN/(TN+FN)

def model_score (X_test, y_test, y_pred, name, regressor):
    print ('{} Accuracy/R2: {}'.format(name, regressor.score(X_test, y_test)))
    print ('{} Precision: {}'.format(name, precision_score(y_test, y_pred)))
    print ('{} Recall: {}'.format(name, recall_score(y_test, y_pred)))
    print ('{} Specificity (True Neg): {}'.format(name, specificity(y_test, y_pred)))
    print ('{} Negative predict: {}'.format(name, neg_pred(y_test, y_pred)))

def optimize_logistic_regression(C_list, X_train, X_test, y_train, y_test):
    param_dict = defaultdict(int)
    for C in C_list:
        LR = LogisticRegression(penalty='l1', C=C, class_weight='balanced')
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_test)
        model_score(X_test, y_test, y_pred, 'Log', LR)
        log_params = np.absolute(LR.coef_)
        param_dict[C] = len(np.where(log_params>0)[1])
    return param_dict

def subset_features(X_train, X_test, num_genes=100, num_extra_features=5):
    '''
    INPUT: X_train, X_test dataframes
    OUTPUT: subset of X_train and X_test with the top n genes plus extra features
    '''
    x = X_train
    top_hundred_genes = set()
    for i in range(x.shape[0]):
        top_hundred_genes = top_hundred_genes.union(set(np.argsort(x.values[i:i+1, :-num_extra_features]).ravel()[-num_genes:]))
    for n in range(1, num_extra_features+1):
        top_hundred_genes.update({-n})

    top_hundred_genes = np.array(list(top_hundred_genes))
    genes = x.columns.values
    select_genes = genes[top_hundred_genes]

    cols = list(select_genes)
    X_train_select = x[cols]
    X_test_select = X_test[cols]
    return X_train_select, X_test_select

def preprocess_data(X, y):
    y = y.set_index('IndividualID')
    y_data = y['Gleason_binary']
    cols = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y_data)
    ind_train = X_train.index
    ind_test = X_test.index

    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_train_df = pd.DataFrame(data=X_train_scale, columns=cols, index=ind_train)
    X_scale_test= scaler.transform(X_test)
    X_test_df = pd.DataFrame(data=X_test, columns=cols, index=ind_test)
    return (X_train_df, X_test_df, y_train, y_test)

if __name__ == '__main__':
    X = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/genes_from_pivotft.csv', index_col=0)
    y = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/labels_from_pivotft.csv', index_col=0)

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    X_train_select, X_test_select = subset_features(X_train, X_test, num_genes=100, num_extra_features=1)

    C_list = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    print ('Optimal C for all genes')
    print (optimize_logistic_regression(C_list, X_train, X_test, y_train, y_test))
    print ('****************************************')
    print ('Optimal C for select genes')
    print (optimize_logistic_regression(C_list, X_train_select, X_test_select, y_train, y_test))

    # X_train.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_train_all.csv')
    # X_test.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_test_all.csv')
    # y_train.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_train.csv')
    # y_test.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_test.csv')
    # X_train_select.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_train_select.csv')
    # X_test_select.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_test_select.csv')
