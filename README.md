# Project Title
Feature Importance and Feature Selection with Statistical Methods

## by Fan Li

## Table of Contents
1. [Description](#description)
2. [Workflow](#Workflow)
   1. Importance strategies working directly from the data
       1. [Spearman's rank correlation coefficient](#ssrc)
       2. [Linear coef featimp](#lcf)
       2. [PCA](#pca)
       3. [mRMR](#mrmr)
   2. [Model-based importance strategies](#model)
       1. [Drop column importance](#drp)
       2. [Permutation importance](#per)
7. [Comparing strategies](#com)
8. [Automatic feature selection algorithm](#auto)
9. [Variance and empirical p-values for feature importances](#var)
10. [Wine quality Data(classification)](#class)
11. [Dataset](#Dataset)
12. [About](#About)
13. [References](#ref)

<a name="description"></a>
## Description
For this project, I implemented various feature importance algorithms and explored how to select features in a model. I used two data sets to showcase both the regression and classification scenarios of those algorithms. Further, I compared those algorithms and developed an automatic feature selection algorithm. Finally, to support visual evidence of feature importances, I used bootstrapped strategy and derived statistical results of variance, standard deviation, and p-values of importance of features.


<a name="Workflow"></a>
## Workflow:
> About `noise` column:

For development and demonstration purposes, I will use the `Boston housing` dataset first. Here 'Y' will be the house pricing and 'X' will be the rest of the features.

In order to have a baseline for 'sanity check', I added a Gaussian noise column. Theoretically, if we got any feature's importance below the noise column's importance, those features are not convincing enough to be considered important, and we should be able to drop them just as we can drop the 'noise' column without risk of losing too much (if any) predicting power.

> About `StandardScaler`:

Further, some of the algorithms such as PCA, require data in different dimensions to have the same scale, thus I used sklearn's StandardScaler to transform X first and then fed it into all algorithms so that we can do a fair comparison among the performances.

<a name="ssrc"></a>
##### 1. Spearman's rank correlation coefficient
The simplest technique to identify important regression features is to rank them by their Spearman's rank correlation coefficient; the feature with the largest coefficient is taken to be the most important. This method is measuring single-feature relevance importance and works well for independent features, but suffers in the presence of codependent features. Groups of features with similar relationships to the response variable receive the same or similar ranks, even though just one should be considered important.

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/speaman1.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/spearnman2.png)

<a name="lcf"></a>
##### 2. Linear coef featimp

The fundamental difference between the two correlation coefficients is that the Pearson coefficient works with a linear relationship between the two variables whereas the Spearman coefficient works with monotonic relationships (the variables tend to change together, but not necessarily at a constant rate). The Spearman correlation coefficient is based on the ranked values for each variable rather than the raw data.

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/linear1.png)

<a name="pca"></a>
##### 3. PCA
As a reminder, each principal component is a unit vector that points in the direction of the highest variance (after accounting for the variance captured by earlier principal components, which means PC1 always captures the feature that has the highest variance). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend to be associated with increases in the other. In contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.

> Note: Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. This step was performed during data preprocessing.

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/pca3.png)


<a name="mrmr"></a>
##### 4. mRMR (Spearman's rank coefficient for the function `I`)

In an effort to deal with codependencies, data analysis techniques rank features not just by relevance (correlation with the response variable) but also by low redundancy, the amount of information shared between codependent features, which is the idea behind minimal-redundancy-maximal-relevance (mRMR) - the higher the feature correlated with others, the low the mRMR score will be:

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/mrmr1.png)

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/mrmr2.png)

<a name='drp'></a>
##### 5. Drop column importance
> Procedure:

1. First, we compute the validation metrics for the model trained on all features. This is our baseline.
2. Drop column from the training set, one at a time.
3. Retrain model.
4. Compute validation metric set (OOB score in this case).
5. The importance score is the change in metric

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/drop1.png)

<a name='per'></a>
##### 6. Permutation importance

To work around this codependent problem, we can break the potential connection between features by shuffling the record in each feature. Further, this method is much more efficient than drop column importance since we don't need to retrain the model for each permutation, we just have to re-run the permutated test samples through the already-trained model.

> Procedure:

1. Compute validation metric for a model trained on all features.
2. Permute column in the validation set.
3. Compute validation metric set.
4. The importance score is the change in metric.
> Note: metric used {regression: ![equation](https://latex.codecogs.com/svg.image?R^{2}), classification: accuracy}

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/per1.png)


<a name="com"></a>
## Comparing strategies
According to the three predicting models I tried, PCA underperformed all other algorithms in terms of the number of top k most important features needed. On the other hand, the ‘Permutation’ approach shows a really promising result: even with the same number of topmost important features, permutation always has the lowest MAE.

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/com1.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/com2.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/com3.png)
> Sanity: Compare permutation importance to shap feature importance:


![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/shap.png)

<a name="auto"></a>
## The automatic feature selection algorithm
Recall that permutation only measures the magnitude of metric change, since I added a Gaussian noise column, any column’s feature importance below the noise column, we can just simply drop. Of course, we will drop the noise column as well.

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/select.png)

<a name="var"></a>
## Variance and empirical p-values for feature importances
Here I bootstrapped 1000 times and collected permutation feature importance in these 1000 runs. Barplot shows the error bars with respect to a 95% confidence interval (two standard deviations). But since I ran 1000 bootstrap, the error bars were squashed into very condensed intervals.
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/var1.png)
In order to perform a hypothesis test, we need a null distribution for comparison purposes. To bootstrap, I repeatedly shuffling target y and compute feature importances. Hypothesis:

![equation](https://latex.codecogs.com/svg.image?H_{0}:&space;a&space;=&space;b)

![equation](https://latex.codecogs.com/svg.image?H_{a}:&space;a\neq&space;&space;&space;b)

Where:

* a = calculated permutation importance
* b = random feature importance


![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/var2.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/var3.png)

<a name="class"></a>
## DescrWine quality Data(classification)iption

![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/wine1.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/wine2.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/wine3.png)
![alt test](https://raw.githubusercontent.com/victorlifan/feature_importance/main/img/wine4.png)

<a name="Dataset"></a>
## Dataset

* `Boston housing`: regression case, from shap.datasets.boston()
* `winequality-white`: binary classification case


<a name="About"></a>
## About
+ [`Jupyter Notebook file`](https://github.com/victorlifan/feature_importance/blob/main/featimp.ipynb): workspace where I performed and tested the works.
+ [`featimp.py`](https://github.com/victorlifan/feature_importance/blob/main/featimp.py): modularized support functions
* [`featimp.pdf`](https://github.com/victorlifan/feature_importance/blob/main/featimp.pdf): pdf presentation
+ [`img`](https://github.com/victorlifan/kmeans/tree/main/img): png files were used in this project.

<a name="Software"></a>
## Software used
+ Jupyter Notebook
+ Atom
+ Python 3.9
>   * Numpy
>   * Pandas
>	* scipy
>   * Matplotlib
>   * Seaborn
>   * sklearn
>   * statistics
>   * PIL
>   * tqdm
>	* SHAP
>	* XGBoost


<a name="ref"></a>
## References
* [Clearly explained: Pearson V/S Spearman Correlation Coefficient](https://towardsdatascience.com/clearly-explained-pearson-v-s-spearman-correlation-coefficient-ada2f473b8)
* [Information Driven Healthcare: Machine Learning course](https://www.robots.ox.ac.uk/~davidc/pubs/CDT-B1-Lecture12-MondayWeek2-FeatureSelection2.pdf)
* [Beware Default Random Forest Importances](https://explained.ai/rf-importance/index.html)
