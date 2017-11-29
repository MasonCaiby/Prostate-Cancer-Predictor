import pandas as pd
from datacleaning_pivot import DataCleaningPivot
from extra_features import ExtraFeatures
from preprocess_pivot import PreprocessPivot



if __name__ == '__main__':
    import pyspark as ps
    spark = ps.sql.SparkSession.builder \
        .master('local[6]') \
        .appName('capstone') \
        .getOrCreate()

    # Read data into Spark dataframe
    df = spark.read.csv('/Users/meghan/DSI/capstone/PRAD.csv.gz', header=True)

    # Make pivot table to one hot encode genes and get labels
    dcp = DataCleaningPivot()
    genes, labels = dcp.fit_transform(df)
    labels = labels.set_index('IndividualID')

    # Add extra features, can add 'days_to_birth' and/or 'VEP_IMPACT', be sure to
    # include 'Individual ID'
    cols = ['IndividualID', 'days_to_birth', 'VEP_IMPACT']
    EF = ExtraFeatures()
    extra_df = EF.fit_transform(data=genes, columns=cols)
    extra_df = extra_df.set_index('IndividualID')

    # Join one hot encoded genes with extra features
    X = genes.join(extra_df)
    X = X.dropna(subset=['age'])

    # Finish preprocessing data by doing a train/test split, balancing classes
    # in train data, selecting top genes and save csvs to run in model

    PP = PreprocessPivot(X, labels)
    all_genes_balanced, balanced_labels = PP.balance_train_data(save_csv=True)
    select_train_genes = PP.select_genes_fit(num_extra_features=len(extra_df.columns)-1)
    select_test_genes = PP.select_gene_transform_test()

    select_train_genes.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_train_select.csv')
    select_test_genes.to_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/balanced_data/X_test_select.csv')

    print ('All genes, balanced: {}'.format(all_genes_balanced.head(5)))
    print ('Balanced labels: {}'.format(balanced_labels.head(5)))
    print ('Select test genes: {}'.format(select_test_genes.head(5)))
    print ('Select train genes: {}'.format(select_train_genes.head()))
