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

## Models to considered
Models were optimized with the grid_search_select.py script
In this study, aggressive cancer was classified as a 1 and non-aggressive as 0.  It is most important to correctly identify aggressive cancer and not mis-classify as non-aggressive, making recall the most important metric.  However, the challenge comes in identifying the under-represented non-aggressive class, making specificity also highly important.

* Neural network with aggressive regularization
* Logistic regression with regularization (both L1 and L2)
* Logisitic regression with gradient descent
* Random forest
* Adaboost
* Support vector classification

## The Data
The data consists of 20,218,807 rows and 61 columns.  There are 412 patients, each having ~40,000 to 90,000 rows of data each. The focus of this project was data contained within the VEP_GENE column representing the Ensemble gene containing at least one mutation in the tumor cells of patients.  Patients showed mutations in ~3,000 to 6,000 different genes. 

Data preprocessing was done using pyspark.

A pipeline was built in order to process data sets in the same way (see the datacleaning.py).  These steps are done prior to a train/test split so that genomic data for one patient is not being split.  Preprocessing includes:
* Filtering mutations when the tumor allele frequency is greater than the normal allele frequency
* Changing Gleason score to binary (greater than or equal to 7 is 1, less than 7 is 0)
* Combining all of the genes for each patient into a vector 
* One-hot encoding all VEP_GENES using a pivot table
* In order to write the data to a csv the Spark dataframe needed to be converted to a Pandas dataframe
* Another script, extra_features.py, allows for the addition of age and VEP_CONSEQUENCE which changes age to years and one hot encodes the consequence


## Preliminary results
Though models are predicting positive, aggressive, cases well, they are still failing to predict the negative, non-aggressive cases with the genomic data alone:

| Model         | Accuracy      | Recall  | Precision | Specificity |
| ------------- |:-------------:| -------:|----------:|------------:|
| Logistic Reg  |88.7           |99.1%    |89.4%      |0.0%         |
| Gradient Desc |89.5%          |100%     |89.5%      |0.0%         |
| Random Forest |89.5%          |100%     |89.5%      |0.0%         |
| **Adaboost**  |**88.7**       |**98.2%**|**90.1%**  |**6.25%**    |


## Reducing feature space
#### PCA also indicated that ~99% of the data can be explained by ~360 gene components


![PCA graph](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/Figure_1.png "PCA")



To reduce the feature space, a logistic regression with a strict lasso penalty was performed.  The optimal penalty was determined by the optimizing_logistic_reg.py script.  This script also creates a union of the top n mutated genes for each patient.
The union of resulting features space from lasso on all genes and lasso on the subset of the gene was used to reduce the final features space. 
Given the nature of genes, there are most likely complicated interactions.  Interaction terms were created, and then Grid Search performed to optimize prediction.



