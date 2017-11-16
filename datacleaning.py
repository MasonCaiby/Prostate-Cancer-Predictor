import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.feature import OneHotEncoder, StringIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT


class DataCleaning():
    '''
    Instantiate DataCleaning and call fit_transform in order to preprocess the
    data.  fit_transform will return a tuple of two spark dataframes:
    one containing labels and one containing one hot encoded genes
    '''

    def __init__ (self):
        # self.df = df
        # self.columns = columns
        return None

    def get_columns(self, df, columns):
        '''
        Gets only desired column names
        Be sure to include Gleason Score, IndividualID and VEP_GENE
        '''
        return df.select(*columns)

    def binary(self, num):
        '''
        Helper function for binary_gleason
        '''
        if num >= 7:
            return 1
        else:
            return 0

    def binary_gleason(self, df):
        '''
        Turns Gleanson score >= 7 into 1 and <7 into 0
        '''
        df = df.withColumn('Gleason_int', df['Gleason Score'].cast(IntegerType()))
        my_udf = udf(lambda x: self.binary(x), IntegerType())
        df = df.withColumn('Gleason_binary', my_udf(df['Gleason_int']))
        labels = df.select('IndividualID', 'Gleason_binary')
        labels = labels.distinct()
        df = df.drop('Gleason Score', 'Gleason_int', 'Gleason_binary')

        return labels, df

    # def one_hot_encode(self, df, one_hot_cols):
    #     stages = []
    #     # columns = ['VEP_GENE']
    #     for col in one_hot_cols:
    #         stages.append(StringIndexer(inputCol=col, outputCol=col+'_idx'))
    #         stages.append(OneHotEncoder(inputCol=col+'_idx', outputCol=col+'_oneHot'))
    #
    #     pipeline = Pipeline(stages=stages)
    #     pipe_fit = pipeline.fit(df)
    #     one_hot = pipe_fit.transform(df)
    #     return one_hot


    def one_hot_encode(self, df):
        '''
        OUTPUT:  returns a spark data frame with a column of a list of only
        one_hot_encoded genes for each patient, to one_hot_encode other variables
        another function must be used
        '''
        gene_list = df.groupby('IndividualID').agg(collect_list('VEP_GENE'))
        cv = CountVectorizer(inputCol='collect_list(VEP_GENE)', outputCol='one_hot')
        model = cv.fit(gene_list)
        one_hot_genes = model.transform(gene_list)
        # User defined function to convert sparse vector to dense
        func = udf(lambda vs: Vectors.dense(vs), VectorUDT())
        OH_genes = one_hot_genes.withColumn('one_hot_dense', func(one_hot_genes['one_hot']))
        OH_genes = OH_genes.drop('collect_list(VEP_GENE)', 'one_hot')
        return OH_genes

    def fit_transform(self, df, columns):
        '''
        Calls all data cleaning set to get usable dataframe
        '''
        df = self.get_columns(df, columns)
        labels, df = self.binary_gleason(df)
        df = self.one_hot_encode(df)

        return labels, df
