import pandas as pd
from pyspark.sql.types import *
from pyspark.sql.functions import udf, collect_list
from pyspark.ml.feature import OneHotEncoder, StringIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT
from datacleaning import DataCleaning

if __name__ == '__main__':
    import pyspark as ps
    spark = ps.sql.SparkSession.builder \
        .master('local[6]') \
        .appName('capstone') \
        .getOrCreate()

    df = spark.read.csv('/Users/meghan/DSI/capstone/PRAD.csv.gz', header=True)
    cols = ('IndividualID', 'VEP_GENE', 'Gleason Score')
    clean = DataCleaning()
    labels, genes = clean.fit_transform(df, cols)
    labels.toPandas().to_csv('labels.csv')
    genes.toPandas().to_csv('genes.csv')
