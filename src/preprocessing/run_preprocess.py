import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess_pivot import PreprocessPivot




if __name__ == '__main__':
    X = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/genes_extra_features.csv', index_col=0)
    y = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/labels_from_pivot3.csv', index_col=0)

    PP = PreprocessPivot(X, y)
    all_genes_balanced, balanced_labels = PP.balance_train_data(save_csv=True)
    select_train_genes = PP.select_genes_fit(num_extra_features=2)
    select_test_genes = PP.select_gene_transform_test()

    select_train_genes.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_train_select.csv')
    select_test_genes.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_test_select.csv')

    print ('All genes, balanced: {}'.format(all_genes_balanced.head(5)))
    print ('Balanced labels: {}'.format(balanced_labels.head(5)))
    print ('Select test genes: {}'.format(select_test_genes.head(5)))
    print ('Select train genes: {}'.format(select_train_genes.head()))
