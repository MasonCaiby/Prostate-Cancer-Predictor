import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


'''
This script
-finds the top hundred genes with the most mutations for each patient
-creates a union of all the top one hundred genes
-creates a data set using the union of genes
-balances classes by duplicating 0 class 8 times each
'''

# genes_from_pivot2 preprocessed using datacleaning_pivot/run_pivot.py scripts
X = pd.read_csv('genes_from_pivot3.csv', index_col=0)
X_data = X.values
y = pd.read_csv('labels_from_pivot3.csv', index_col=0)
y = y.set_index('IndividualID')

# Make set of top hundred genes for each patient
top_hundred_genes = set()
for i in range(X_data.shape[0]):
    top_hundred_genes = top_hundred_genes.union(set(np.argsort(X_data[i])[-100:]))

top_hundred_genes = np.array(list(top_hundred_genes))
genes = X.columns.values
select_genes = genes[top_hundred_genes]

cols = list(select_genes)
select_df = X[cols]

select_df

# Make train/test split first then make balanced classes on train set, leave
# test set unbalanced
X_train, X_test, y_train, y_test = train_test_split(select_df, y)


# Get rows of selectdf that are in class 0 and duplicate 8 times
zeros = X_train[y_train['Gleason_binary'].values==0]
eight = np.ones(zeros.shape[0])
eight = eight*8
eight = eight.astype(int)
repeat_arr = np.repeat(zeros, eight, axis=0)
repeat_df = pd.DataFrame(repeat_arr, columns=cols)
X_train = pd.DataFrame(X_train)
joined = X_train.append(repeat_df)

# Add 0 classes to labels
y_add_arr = np.zeros(zeros.shape[0]).reshape(-1,1).astype(int)
index = repeat_df.index
y_add = pd.DataFrame(y_add_arr, index=index, columns=['Gleason_binary'])
balanced_labels = y.append(y_add)

balanced_labels.to_csv('balanced_labels.csv')
joined.to_csv('select_genes.csv')
