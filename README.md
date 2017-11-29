# Prostate-Cancer-Predictor
![Digital DNA](http://www.thinkcx.co.uk/wp-content/uploads/2014/07/DigtalDNA.jpg)

* As of October there were 161,360 new cases of prostate cancer and 26,703 people had died from the disease in 2017 <sup>1</sup>
* 26 % of prostate cancers could not be categorized into molecular subtypes <sup>2</sup>
* Approximately 1 in 7 men will be diagnosed with prostate cancer <sup>1</sup>
* In 2006 the total estimated expenditure on prostate cancer in the US was $9.86 billion <sup>3</sup>

#### Need: a way to better predict aggressiveness of cancer in order to make more informed, personalized decisions for treatment which will ultimately improve survivability and decrease unnecessary treatment costs

## The Project
The goal of this project is, in collaboration with Precision Profile, to use genetic data of prostate tumor cells to build a model capable of predicting aggressive and non-aggressive cancer types

Goal 1:  Build a model that will predict the Gleason score (label) of the data using between 2000 and 5000 gene variants
Goal 2:  Identify features associated with predicting class

## Models to considered
In this study, aggressive cancer was classified as a 1 and non-aggressive as 0.  It is most important to correctly identify aggressive cancer and not mis-classify as non-aggressive, making recall the most important metric.  However, the challenge comes in identifying the under-represented non-aggressive class, making specificity also highly important.

* Neural network with aggressive regularization
* Logistic regression with regularization (both L1 and L2)
* Logisitic regression with gradient descent
* Random forest
* Adaboost
* Support vector classification

## The Data
The data consists of 20,218,807 rows and 61 columns.  There are 495 patients, each having ~40,000 to 90,000 associated features. Of those 495 patients, only 45 were classified as having non-aggressive cancer while the remainder were classified as aggressive, presenting a serious challenge of class imbalance. 

Data preprocessing was a significant undertaking utilizing pyspark and pandas, and was carried out in stages.

To create a table of gene counts, a pivot table was utilized and constructed prior to the train/test split so that rows of data belonging to the same patient would not be inadvertently split.  The code can be found in in the [datacleaning_pivot.py](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/src/preprocessing/datacleaning_pivot.py) and does the following:

* Filtering mutations when the tumor allele frequency is greater than the normal allele frequency
* Changing Gleason score to binary (greater than or equal to 7 is 1, less than 7 is 0)
* Combining all of the genes for each patient into a vector 
* One-hot encoding all VEP_GENES using a pivot table
* In order to write the data to a csv the Spark dataframe needed to be converted to a Pandas dataframe
* Another script, [extra_features.py](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/src/preprocessing/extra_features.py), allows for the addition of age and VEP_CONSEQUENCE

## Preliminary results
Though models are predicting positive, aggressive, cases well, they are still failing to predict the negative, non-aggressive cases with the genomic data alone:

| Model         | Accuracy      | Recall  | Precision | Specificity |
| ------------- |:-------------:| -------:|----------:|------------:|
| Logistic Reg  |88.7           |99.1%    |89.4%      |0.0%         |
| Gradient Desc |89.5%          |100%     |89.5%      |0.0%         |
| Random Forest |89.5%          |100%     |89.5%      |0.0%         |
| **Adaboost**  |**88.7**       |**98.2%**|**90.1%**  |**6.25%**    |

Models were optimized with the [grid_search_select.py](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/src/models/grid_search_select.py) but continued to have poor specificity.

![Roc Plot](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/results/roc_plot_filter_allmodels.png)

## Reducing feature space
#### PCA also indicated that ~99% of the data can be explained by ~360 gene components


![PCA graph](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/results/pca.png "PCA")

## Balancing classes
In an effort to balance classes, the [preprocessing_pivot.py](https://github.com/meghan-sloan/Prostate-Cancer-Predictor/blob/master/src/preprocessing/preprocess_pivot.py) duplicates non-aggressive entries.  However, in the grid-search optimization this causes data leakage and instead the class_weight parameter was used on the unbalanced data.

## Conclusions and future directions
Given the nature of the unbalanced classes more data for non-aggressive patients would be ideal
Genomic interactions are incredibly complex.  Continuing to train neural nets in attempt to find complicated interactions may be worth while.  
Additionally, it is logical that the two groups may be similar in their genetic mutations.  More significant differences may be seen in RNA expression or copy number data.


## References
1. American Cancer Society:  Cancer Facts and Figures 2011.  Atlanta, GA: American Cancer Society, 2011
2. https://www.cancer.org/cancer/prostate-cancer/about/key-statistics.html
3. Roehrborn CG, Black LK. ‘The economic burden of prostate cancer.’ BJU International 108(6): 806-813


## Acknowledgements
Thank you to Dave Parkhill, James Costello, PhD, and Joe Kasprzak at Precision Profile for providing data, their collaboration and assistance
Multilayer perceptron code was based on code provided by Frank Burkholder at Galvanize
Thank you to the Taryn Heilman, Jon Courtney and Elliot Cohen as well as fellow Galvanize classmates for recommendations and guidance



