import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

class InteractionData:
    '''
    Instantiate class with :
    C - int, the optimal l1 penalty for logistic Regression
    num_features - int, the number of features to limit gene features to
    Use fit_transform to pass in training DataFrame
    Use transform to transform test data
    '''
    def __init__ (self, C, num_features):
        self.C = C
        self.num_features = num_features


    def lasso_subset(self, X_train, y_train):
        '''
        Performs logisitic regression with strict Lasso penalty to reduce
        feature space
        Called by the fit_transform method
        '''
        genes = X_train.columns.values
        LR = LogisticRegression(penalty='l1', C=self.C, class_weight='balanced')
        LR.fit(X_train, y_train)
        log_params = np.absolute(LR.coef_)
        subset_features = genes[np.argsort(log_params.ravel())].tolist()[-self.num_features:]
        return set(subset_features)

    def interaction_term(self, X_train):
        '''
        Creates interaction terms as the feature space
        Called by the fit_transform method
        '''
        self.ID = PolynomialFeatures(interaction_only=True)
        interaction_train = self.ID.fit(X_train)
        return interaction_train

    def fit_transform(self, X_train_all, X_train_select, y_train):
        '''
        INPUT: X_train data, X_train_select, y_train data
        Finds union of subset genes from all_genes and select_genes
        OUTPUT:  Interaction term data for running in classification models and grid GridSearch
        '''
        subset_all = self.lasso_subset(X_train_all, y_train)
        subset_select = self.lasso_subset(X_train_select, y_train)
        self.subset_union = list(subset_all.union(subset_select))
        X_train = X_train_all[self.subset_union]
        interaction_train = self.interaction_term(X_train)
        return interaction_train

    def transform(self, X_test):
        '''
        INPUT: X_test data
        OUTPUT: X_test data transformed in the same way as X_train (interaction term data)
        '''
        X_test = X_test[self.subset_union]
        interaction_test = self.ID.transform(X_test)
        return interaction_test
