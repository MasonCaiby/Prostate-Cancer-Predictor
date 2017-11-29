import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class PreprocessPivot():
    '''
    Instantiate Preprocess and call the various methods to:
    -train-test split before preprocessing
    -balance the train set, increasing the number in class 0
    -select the top 100 genes from each patient
    -save the csv
    INPUT:
    X - pandas dataframe from datacleaning_pivot script (one hot encoded genes)
    y - pandas dataframe of labels from pivot(can have both 'Gleason_binary' and 'Gleason Score')
    label col - string indicating whether to use 'Gleason_binary' or 'Gleason Score'
    for label
    '''

    def __init__ (self, X, y, label_col='Gleason_binary'):
        self.X = X
        # self.X = self.X.set_index('IndividualID')
        self.y = y
        self.X_data = self.X.values
        self.label_col = label_col
        self.y_data = self.y.set_index('IndividualID')
        self.y_data = self.y_data[self.y_data.index.isin(self.X.index)]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y_data)

    def balance_train_data(self, save_csv=False):
        '''
        Creates balanced training data
        INPUT: boolean to indicate whether or not to save a csv of this data
        '''
        cols = self.X_train.columns
        # Find rows that correspond to 0 binary score and repeat 8 times
        zeros = self.X_train[self.y_train['Gleason_binary'].values==0]
        eight = np.ones(zeros.shape[0])
        eight = eight*8
        eight = eight.astype(int)
        repeat_arr = np.repeat(zeros.values, eight, axis=0)
        repeat_df = pd.DataFrame(repeat_arr, columns=cols)
        self.balanced_X = self.X_train.append(repeat_df)

        #Add repeats to y train
        y_add_arr = np.zeros(zeros.shape[0]*8).reshape(-1,1).astype(int)
        index = repeat_df.index
        y_add = pd.DataFrame(y_add_arr, index=index, columns=['Gleason_binary'])
        self.balanced_labels = self.y_train.append(y_add)

        #If desired, save csv without selecting top 100 genes
        if save_csv == True:
            self.balanced_labels.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/y_train.csv')
            self.balanced_X.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_train_all.csv')
            self.y_test.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/y_test.csv')
            self.X_test.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_test_all.csv')
        return (self.balanced_X, self.balanced_labels)

    def select_genes_fit(self, num_genes=100, num_extra_features=None):
        '''
        Selects the top n mutated genes for each patient in the training data
        '''
        # Determine which set of X to use
        if self.label_col == 'Gleason_binary':
            x = self.balanced_X
        elif self.label_col == 'Gleason Score':
            x = self.X_train

        # Make set of top hundred genes for each patient
        top_hundred_genes = set()
        for i in range(x.shape[0]):
            top_hundred_genes = top_hundred_genes.union(set(np.argsort(x.values[i:i+1, :-num_extra_features]).ravel()[-num_genes:]))
        for n in range(1, num_extra_features+1):
            top_hundred_genes.update({-n})

        top_hundred_genes = np.array(list(top_hundred_genes))
        genes = x.columns.values
        select_genes = genes[top_hundred_genes]

        self.cols = list(select_genes)
        select_df = x[self.cols]
        print (type(select_df))
        return select_df

    def select_gene_transform_test(self):
        '''
        To transform test data using the same top 100 genes as the train datasets
        '''
        return self.X_test[self.cols]
