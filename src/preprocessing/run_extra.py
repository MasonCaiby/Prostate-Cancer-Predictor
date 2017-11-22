import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.feature import OneHotEncoder, StringIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT
from extra_features import ExtraFeatures




if __name__ == '__main__':
    import pyspark as ps
    spark = ps.sql.SparkSession.builder \
        .master('local[6]') \
        .appName('capstone') \
        .getOrCreate()

    df = spark.read.csv('/Users/meghan/DSI/capstone/PRAD.csv.gz', header=True)
    cols = ['IndividualID', 'days_to_birth', 'VEP_IMPACT']
    EF = ExtraFeatures()
    extra_df = EF.fit_transform(data=df, columns=cols)
    extra_df = extra_df.set_index('IndividualID')
    one_hot_genes = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/genes_from_pivot3.csv', index_col=0)
    joined = one_hot_genes.join(extra_df)
    joined = joined.dropna(subset=['age'])
    joined.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/genes_extra_features.csv')
