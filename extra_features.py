import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.feature import OneHotEncoder, StringIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT


class ExtraFeatures():
    '''
    Instantiate ExtraFeatures and call fit_transform, passing the dataframe
    and the columns to include
    The list of features includes:  race and age (days to birth)
    ### add gene_consequence if necessary
    It will return an array with IndividualID and the processed columns
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

    def format_age(self, df):
        df = df.withColumn('age', df['days_to_birth'].cast(IntegerType()))

    def race(self, df):
        stages = []

        stages = [StringIndexer(inputCol='race', outputCol='race'+'_idx') \
                 OneHotEncoder(inputCol='race'+'_idx', outputCol='race'+'_oneHot')]

        pipeline = Pipeline(stages=stages)
        pipe_fit = pipeline.fit(df)
        one_hot = pipe_fit.transform(df)
        return one_hot

    def fit_transform(self, df, columns):
        '''
        Calls all data cleaning set to get usable dataframe
        '''
        columns = columns.append('IndividualID')
        df = self.get_columns(df, columns)
        df = df.distinct()
        df = format_age(df)
        df = race(df)
        return df
