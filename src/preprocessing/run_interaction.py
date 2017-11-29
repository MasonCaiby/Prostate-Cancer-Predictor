import pandas as pd
import numpy as np
from interaction_data import InteractionData



if __name__ == '__main__':

    X_train_all = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_train_all.csv', index_col=0)
    X_test_all = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_test_all.csv', index_col=0)
    y_train = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_train.csv', index_col=0, header=None)
    y_test= pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_test.csv', index_col=0, header=None)
    X_train_select = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_train_select.csv', index_col=0)
    X_test_select = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_test_select.csv', index_col=0)


    ID = InteractionData(C=0.05 , num_features=10)
    interaction_train = ID.fit_transform(X_train_all, X_train_select, y_train)
    interaction_test = ID.transform(X_test_all)

    print(interaction_train)
    print('***************')
    print(interaction_test)

    print (ID.subset_union)
