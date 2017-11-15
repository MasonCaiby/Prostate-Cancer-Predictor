import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline


# data = pd.read_csv('PRAD.csv.zip', nrows=100000, compression='infer')
# # type_df = data.dtypes
# subset = data[['IndividualID', 'VEP_GENE', 'Gleason Score']]
# gene_dummies = pd.get_dummies(subset.VEP_GENE)
# gene_encode = subset.join(gene_dummies)
# g_score = gene_encode.groupby('IndividualID')['Gleason Score'].value_counts()
# one_hot_gene = gene_encode.groupby('IndividualID').sum()
# one_hot_gene = np.array(one_hot_gene)

class DataCleaning():
    '''
    INPUT:
    Spark data frame
    Tuple of desired columns
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
        df = df.drop('Gleason Score', 'Gleason_int')
        return df

    def one_hot_encode(self, df, one_hot_cols):
        stages = []
        # columns = ['VEP_GENE']
        for col in one_hot_cols:
            stages.append(StringIndexer(inputCol=col, outputCol=col+'_idx'))
            stages.append(OneHotEncoder(inputCol=col+'_idx', outputCol=col+'_oneHot'))

        pipeline = Pipeline(stages=stages)
        pipe_fit = pipeline.fit(df)
        one_hot = pipe_fit.transform(df)
        return one_hot

    def fit_transform(self, df, columns, one_hot_cols):
        '''
        Calls all data cleaning set to get usable dataframe
        '''
        df = self.get_columns(df, columns)
        df = self.binary_gleason(df)
        df = df.distinct()
        df = self.one_hot_encode(df, one_hot_cols)

        return df
