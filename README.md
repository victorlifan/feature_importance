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


<a name="km+"></a>
##### 2. Kmeans ++

> Procedure:

1. Initialize centroids꞉
Randomly initialize k number of data points from the original X data. The number of k depends on how many clusters we want to end up with.
2. Compute distance꞉
Here I used Euclidean distance to measure the distance from each of the remaining data points to each of the centroids we initialized in step 1, assigning each of the remaining data points to the ‘closest ’ centroids.
3. Update centroids꞉
Within each cluster, compute the average distance of all the data points to that centroid FEATURE WISE. This average distance will be the new centroids’ ‘coordinate’ in that cluster. Intuitively speaking, this means we are correcting the centroids to be the ‘center’ of that cluster. This means our final centroids will most likely not be members of the dataset. The reason we picked data points from the dataset as initial centroids is simply to assign a starting point.

4. Reassign data point
Finally, compute distance, reassign data points according to the new centroids we updated in step 3, update centroids. Iterate the above process until the centroids’ ‘coordinates’ don’t change any more.

<a name="app"></a>
##### 3. Applications
*  Synthetic data set

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/1.png)

* Multi-dimension data (Circle data 500*2)

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/multi1.png)
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/multi2.png)

> As you can see, Kmeans performs poorly on disjointed and nested structures. To rescue, I will introduce spectral clustering by using RF and Kmeans together in the Advanced topic section.

* Breast cancer

    * Without scaling X꞉

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/bc1.png)

    * With scaled X
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/bc2.png)

* Image compression
    * Grayscale
        * Original:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/north-africa-1940s-grey.png)
        * Kmeans++ copresion:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/gray_km.png)

    * Color
        * Original:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/parrt-vancouver.jpg)
        * Kmeans++ copresion:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/color_km.jpg)

<a name="rf+km"></a>
##### 4. Advanced topic: RF + Kmeans

> Procedure:

1. RF ‘group’ similar data points.

2. Construct frequency (similarity) matrix

3. Feed similarity matrix to SpectralClustering

> Test on circle data (sklearn vs. RF+Kmeans)

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/vs.png)

<a name='mf'></a>
##### 5. Limitations
* Randomness in RF will sometimes result in unexpected cluster labels (accuracy is not as steady).

* Kmeans++ only considers picking the furthest point to its previous centroid. Take k=3 as an example, as a consequence, the 1st and 3rd controls can sometimes be quite close to each other. Is there a way to consider all the previous centroids and pick the furthest point from all the previous centroids? How do we even define the ‘minimum distance’ since each centroid has its own furthest points, one point can't be the furthest to multiple centroids?


<a name="Dataset"></a>
## Dataset

* `Synthetic data set`: A small synthetic data that has a shape of 16*1
* `Multi-dimension data (Circle data 500*2)`: from sklearn.datasets.make_circles
* `Breast cancer`: from sklearn.datasets.load_breast_cancer
* `north-africa-1940s-grey` and `parrt-vancouver.jpg`: from Professor [Terence Parr](https://en.wikipedia.org/wiki/Terence_Parr)

<a name="summary"></a>


<a name="About"></a>
## About
+ [`Jupyter Notebook file`](https://github.com/victorlifan/kmeans/blob/main/kmeans.ipynb): workspace where I performed and tested the works.
+ [`kmeans.py`](https://github.com/victorlifan/kmeans/blob/main/kmeans.py): modularized support functions
* [`kmeans.pdf`](https://github.com/victorlifan/kmeans/blob/main/kmeans.pdf): pdf presentation
+ [`img`](https://github.com/victorlifan/kmeans/tree/main/img): png files were used in this project.

<a name="Software"></a>
## Software used
+ Jupyter Notebook
+ Atom
+ Python 3.9
>   * Numpy
>   * Pandas
>   * Matplotlib
>   * Seaborn
>   * sklearn
>   * statistics
>   * PIL
>   * tqdm


<a name="ref"></a>
## References
* [K-Means Clustering: From A to Z](https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a)
* [ML | K-means++ Algorithm](https://www.geeksforgeeks.org/ml-k-means-algorithm/)
* [Image Segmentation using K Means Clustering](https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/)
* [Breiman's website](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#prox)
