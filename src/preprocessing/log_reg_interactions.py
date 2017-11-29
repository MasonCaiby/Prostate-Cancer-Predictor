import pandas as import pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression


class InteractionData():
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
        LR = LogisticRegression(penalty='l1', C=self.C)
        LR.fit(X_train, y_train)
        log_params = np.absolute(LR.coef_)
        subset_features = genes[np.argsort(log_params.ravel())].tolist()[-self.num_features:]
        return subset_features

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
        subset_all = lasso_subset(X_train_all, y_train)
        subset_select = lasso_subset(X_train_select, y_train)
        self.subset_union = subset_all.union(subset_select)
        X_train = X_train_all[subset_union]
        interaction_train = interaction_term(X_train)
        return interaction_train

    def transform(X_test):
        '''
        INPUT: X_test data
        OUTPUT: X_test data transformed in the same way as X_train (interaction term data)
        '''
        X_test = X_test[self.subset_union]
        interaction_test = self.ID.transform(X_test)
        return interaction_test



    # Found 0.014 to be ideal learning rate in logistic regression--keeps 13 genes, all others go to 0
    # While recall was maximizing recall(65%) then specificity(12.5%), also valid for all genes but
    # Could go up as high as 0.016
    # C_list = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019]
    # param_dict = defaultdict(int)
    # for C in C_list:
        # log_set, log_pred, log_params = logistic_reg(X_train, X_test_scale, y_train, y_test, genes, C=C)
        # param_dict[C] = len(np.where(log_params>0)[1])
