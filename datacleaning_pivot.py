import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.feature import OneHotEncoder, StringIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT


class DataCleaningPivot():
    '''
    Instantiate DataCleaning and call fit_transform in order to preprocess the
    data.  fit_transform will return a tuple of two numpy arrays:
    one containing one hot encoded genes and one containing binary labels
    '''

    def __init__ (self):
        # self.df = df
        # self.columns = columns
        return None

    def create_pivot(self, df):
        id_gene = df.select('IndividualID', 'VEP_GENE')
        pivot = id_gene.groupby('VEP_GENE').pivot('IndividualID').count()
        pandas_df = pivot.toPandas()
        no_nan = pandas_df.fillna(value=0)
        one_hot_genes = no_nan.transpose()
        one_hot_genes = one_hot_genes.rename(columns=one_hot_genes.iloc[0])
        genes = one_hot_genes.iloc[1:, :]
        return genes

    def binary(self, num):
        '''
        Helper function for get_labels
        '''
        if num >= 7:
            return 1
        else:
            return 0

    def get_labels(self, df):
        '''
        Turns Gleanson score >= 7 into 1 and <7 into 0
        Sorts labels to be in alpha-numeric order to match pivot one hot one_hot_encoded
        genes
        '''
        df = df.select('IndividualID', 'Gleason Score')
        df = df.distinct()
        df = df.withColumn('Gleason_int', df['Gleason Score'].cast(IntegerType()))
        my_udf = udf(lambda x: self.binary(x), IntegerType())
        df = df.withColumn('Gleason_binary', my_udf(df['Gleason_int']))
        labels = df.select('IndividualID', 'Gleason_binary')
        sorted_labels = labels.sort('IndividualID')
        sorted_labels = sorted_labels.toPandas()
        return sorted_labels

    def fit_transform(self, df):
        '''
        Calls all data cleaning set to get usable dataframe
        '''
        X = self.create_pivot(df)
        labels = self.get_labels(df)

        return X, labels
