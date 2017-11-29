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


    def impact_count(self, sparkdf):
        imp = sparkdf.select('IndividualID', 'VEP_IMPACT')
        impact_count = imp.groupby('IndividualID').pivot('VEP_IMPACT').count()
        pandas_df = impact_count.toPandas()
        return pandas_df

    def format_age(self, df):
        '''
        INPUT: pandas DataFrame, can input any number of rows
        OUTPUT:  pandas DataFrame
        *** Deal with NaNs
        '''
        df = df.dropna(subset = ['days_to_birth'])
        df['age'] = df.loc['days_to_birth'].astype(int)/-365
        # df = df.drop('days_to_birth')
        return df


    def fit_transform(self, data, columns):
        '''
        Calls all data cleaning set to get usable dataframe
        '''
        # columns = list(columns.insert(0, 'IndividualID'))
        # print (type(columns))
        df = data.select(*columns)
        if 'days_to_birth' in columns and 'VEP_IMPACT' in columns:
            df = df.groupby('IndividualID', 'days_to_birth').pivot('VEP_IMPACT').count()
            df = df.toPandas()
        elif 'VEP_IMPACT' in columns:
            df = self.impact_count(data)
        if 'days_to_birth' in columns:
            df = self.format_age(df)
        return df
