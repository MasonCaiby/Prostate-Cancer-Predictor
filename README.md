# Prostate-Cancer-Predictor
![Digital DNA](http://www.thinkcx.co.uk/wp-content/uploads/2014/07/DigtalDNA.jpg)

* As of October there were 161,360 new cases of prostate cancer and 26,703 people had died from the disease in 2017 1
* 26 % of prostate cancers could not be categorized into molecular subtypes 2
* About 1 in 7 men will be diagnosed with prostate cancer
* In 2006 the total estimated expenditure on prostate cancer in the US was $9.86 billion

#### Need: a way to better predict aggressiveness of cancer in order to make more informed, personalized decisions for treatment which will ultimately improve survivability and decrease unnecessary treatment costs

## The Project
The goal of this project is, in collaboration with Precision Profile, to use genetic data of prostate tumor cells to build a model capable of predicting aggressive and non-aggressive cancer types

Goal 1:  Build a model that will predict the Gleason score (label) of the data using between 2000 and 5000 gene variants
Goal 2:  Apply a Cox proportional hazard model on the data to predict survivability given the genomic data

## Models to consider
Predictive power will be more important that feature importance, however, feature importance would also be nice (and what has been done traditionally)
Initial models to consider:  
* Neural network with aggressive regularization


## The Data
The data consists of 20,218,807 rows and 61 columns.  There are 412 patients, each having ~40,000 to 90,000 rows of data each. The focus of this project was data contained within the VEP_GENE column representing the Ensemble gene containing at least one mutation in the tumor cells of patients.  Patients showed mutations in ~3,000 to 6,000 different genes. 

Data preprocessing was done using pyspark.

A pipeline was built in order to process data sets in the same way (see the datacleaning.py).  These steps are done prior to a train/test split so that genomic data for one patient is not being split.  Preprocessing includes:
* Changing Gleason score to binary (greater than or equal to 7 is 1, less than 7 is 0)
* Combining all of the genes for each patient into a vector 
* One-hot encoding all VEP_GENES using CountVectorizer
* In order to write the data to a csv the Spark dataframe needed to be converted to a Pandas dataframe








