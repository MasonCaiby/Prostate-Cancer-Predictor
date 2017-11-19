import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.feature import OneHotEncoder, StringIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT
from datacleaning_pivot import DataCleaningPivot

if __name__ == '__main__':
    import pyspark as ps
    spark = ps.sql.SparkSession.builder \
        .master('local[6]') \
        .appName('capstone') \
        .getOrCreate()

    df = spark.read.csv('/Users/meghan/DSI/capstone/PRAD.csv.gz', header=True)
    dcp = DataCleaningPivot()
    genes, labels = dcp.fit_transform(df)
    labels.to_csv('labels_from_pivot2.csv')
    genes.to_csv('genes_from_pivot2.csv')
